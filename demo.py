"""Run on a machine with a KoRMo-compatible GPU and installed dependencies."""

import argparse
import sys
from contextlib import nullcontext

from prompting import (english_rewrite_messages, guide_messages, messages, render,
                       require_pinned_revision, translate_korean_answer_messages)
from kormo_compat import fix_causal_mask_keyword
from output_style import (HEADINGS, english_issues, korean_issues,
                          normalize_pronunciation, remove_inline_citations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="KORMo-Team/KORMo-10B-sft")
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face model commit SHA")
    parser.add_argument("--adapter", help="Optional locally trained LoRA adapter directory")
    parser.add_argument("--english-mode", choices=("base", "translate-lora"), default="base",
                        help="Use base English generation or translate a Korean LoRA draft with the base model")
    parser.add_argument("--knowledge", help="Chunk JSONL staged under /local_datasets on this compute node")
    parser.add_argument("--retriever", choices=("e5", "bm25"), default="e5")
    parser.add_argument("--smoke", action="store_true", help="Generate one answer and exit without a web server")
    parser.add_argument("--smoke-suite", action="store_true", help="Test English, Korean, and an English question in one load")
    parser.add_argument("--smoke-student", action="store_true", help="Test a sourced student-life question")
    parser.add_argument("--guide-question", help="Answer one student-life question from --knowledge and exit")
    parser.add_argument("--guide-language", choices=("en", "ko"), default="en")
    parser.add_argument("--compare-adapter", action="store_true", help="Compare base and LoRA answers using one model load")
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio link")
    args = parser.parse_args()
    require_pinned_revision(args.revision)
    if args.compare_adapter and not args.adapter:
        parser.error("--compare-adapter requires --adapter")
    if args.english_mode == "translate-lora" and not args.adapter:
        parser.error("--english-mode translate-lora requires --adapter")
    if args.guide_question and not args.knowledge:
        parser.error("--guide-question requires --knowledge")
    index = None
    if args.knowledge:
        from knowledge import BM25Index, E5Index, guide_source_matches, load_chunks

        chunks = load_chunks(args.knowledge)
        index = E5Index(chunks, cache_path=args.knowledge + ".e5.pt") if args.retriever == "e5" else BM25Index(chunks)

    import gradio as gr
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 필요합니다. 학교 서버에서 실행해 주세요.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    fix_causal_mask_keyword(model)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate_answer(expression: str, situation: str, relationship: str, language: str, task: str,
                        use_knowledge: bool, enforce_style: bool = True) -> str:
        try:
            active_index = index if use_knowledge else None
            if task == "guide" and active_index is None:
                return "Student-life questions need a knowledge index. / 유학생 생활 질문에는 지식 자료가 필요합니다."
            found = active_index.search(expression + " " + situation, limit=5 if task == "guide" else 2) if active_index else None
            if task == "guide" and found:
                found = [row for row in found if guide_source_matches(expression, row)][:1]
            if active_index and not found:
                return "I could not find a relevant source. Please add context or check another source. / 관련 출처를 찾지 못했습니다. 상황을 더 알려주세요."
            if found is not None:
                from knowledge import evidence_for_prompt

                evidence = evidence_for_prompt(found)
            else:
                evidence = None
            conversation = (guide_messages(expression, situation, language, found)
                            if task == "guide" else messages(expression, situation, relationship, language, task, evidence))
        except ValueError as exc:
            return str(exc)

        def infer(turns: list[dict[str, str]], base_only: bool = False, prefix: str = "") -> str:
            prompt = render(tokenizer, turns, True) + prefix
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            adapter_context = model.disable_adapter() if base_only and args.adapter else nullcontext()
            with adapter_context, torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return prefix + tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # The existing dialogue LoRA learned Korean turn continuation, not English explanations.
        base_for_english = bool(args.adapter and language == "en" and enforce_style)

        def finish_english(candidate: str) -> str | None:
            candidate = normalize_pronunciation(candidate, task, expression)
            initial_issues = english_issues(candidate, task)
            if not initial_issues:
                return candidate
            if task in HEADINGS:
                corrected = english_rewrite_messages(conversation, candidate, task)
            else:
                corrected = [dict(turn) for turn in conversation]
                corrected[-1]["content"] += "\nFinal check: Rewrite the answer in English only; do not write Korean prose."
            prefix = HEADINGS[task][0] + "\n" if task in HEADINGS else ""
            revised = normalize_pronunciation(infer(corrected, base_only=True, prefix=prefix), task, expression)
            final_issues = english_issues(revised, task)
            if final_issues:
                if args.smoke_suite:
                    print(f"[English style check] {task}: first={initial_issues}; rewrite={final_issues}", file=sys.stderr)
                    print("[First draft]\n" + candidate + "\n[Rewrite]\n" + revised, file=sys.stderr)
                return None
            return revised

        answer = None
        if language == "en" and enforce_style and args.english_mode == "translate-lora":
            korean_turns = (guide_messages(expression, situation, "ko", found)
                            if task == "guide" else messages(expression, situation, relationship, "ko", task, evidence))
            korean_draft = infer(korean_turns)
            draft_issues = korean_issues(korean_draft, task)
            if args.smoke_suite:
                print("[Korean LoRA draft]\n" + korean_draft, file=sys.stderr)
                if draft_issues:
                    print("[Translation skipped] " + ", ".join(draft_issues), file=sys.stderr)
            if not draft_issues:
                translated = infer(translate_korean_answer_messages(conversation, korean_draft, task), base_only=True)
                answer = finish_english(translated)
                if answer and args.smoke_suite:
                    print("[English path] Korean LoRA draft → base KoRMo translation", file=sys.stderr)
        if answer is None:
            direct = infer(conversation, base_only=base_for_english)
            answer = finish_english(direct) if language == "en" and enforce_style else direct
            if answer is None:
                return "I could not produce a reliable English answer. Please try again with more context."
            if args.smoke_suite and language == "en" and enforce_style:
                print("[English path] direct base KoRMo", file=sys.stderr)
        answer = remove_inline_citations(answer)
        if found:
            from knowledge import source_footer

            answer += source_footer(found)
        return answer

    def explain(expression: str, situation: str, relationship: str, language: str, task: str) -> str:
        return generate_answer(expression, situation, relationship, language, task, True)

    if args.smoke:
        print(explain("반찬은 셀프입니다", "A sign on the wall of a restaurant", "Customer and restaurant", "en", "explain"))
        return

    if args.smoke_suite:
        checks = [
            ("English explanation", "반찬은 셀프입니다", "A sign on the wall of a restaurant", "Customer and restaurant", "en", "explain"),
            ("English indirect reply", "생각해 볼게요", "A friend replies after a weekend invitation", "Friends", "en", "explain"),
            ("Korean explanation", "반찬은 셀프입니다", "식당 벽의 안내문", "손님과 식당", "ko", "explain"),
            ("English question", "How should I greet an older adult in Korea?", "Meeting someone for the first time", "A younger adult and an older adult", "en", "ask"),
        ]
        for label, expression, situation, relationship, language, task in checks:
            print(f"\n=== {label} ===", flush=True)
            print(explain(expression, situation, relationship, language, task), flush=True)
        return

    if args.smoke_student:
        print(explain("외국인등록은 어디서 신청하나요?", "한국에 온 유학생", "", "ko", "guide"))
        return

    if args.guide_question:
        print(explain(args.guide_question, "International student living in Korea", "", args.guide_language, "guide"))
        return

    if args.compare_adapter:
        checks = [
            ("Restaurant sign / 식당 안내", "반찬은 셀프입니다", "A sign in a restaurant", "Customer and restaurant", "en", "explain"),
            ("Indirect reply / 완곡한 답", "생각해 볼게요", "A friend replies to an invitation", "Friends", "en", "explain"),
            ("Greeting / 인사", "How should I greet a professor in Korea?", "Meeting a professor for the first time", "Student and professor", "en", "ask"),
        ]
        if index:
            checks.append(("Student life / 유학생 생활", "외국인등록은 어디서 신청하나요?", "한국에 온 유학생", "", "ko", "guide"))
        for label, expression, situation, relationship, language, task in checks:
            print(f"\n=== {label} ===", flush=True)
            with model.disable_adapter():
                print("--- Base KoRMo ---\n" + generate_answer(expression, situation, relationship, language, task, task == "guide", False), flush=True)
            print("--- KoRMo + dialogue LoRA ---\n" + generate_answer(expression, situation, relationship, language, task, task == "guide", False), flush=True)
        return

    with gr.Blocks(title="말맥 | 한국어 맥락 해설기") as ui:
        gr.Markdown("# 말맥 | Malmaek\nUnderstand Korean expressions in context. 한국어 표현을 상황에 맞게 설명합니다. AI answers may be wrong. Retrieved passages are shown for checking; their presence does not verify every claim in the answer.")
        task = gr.Radio(choices=[("Explain a phrase / 표현 해설", "explain"), ("Ask a Korean phrase / 표현 질문", "ask"), ("Student life / 유학생 생활", "guide")], value="explain", label="Task / 기능")
        expression = gr.Textbox(label="Korean expression or question / 표현 또는 질문", placeholder="생각해 볼게요")
        situation = gr.Textbox(label="Situation / 상황", placeholder="A reply after inviting a friend to a weekend meetup")
        relationship = gr.Textbox(label="Relationship / 상대와의 관계", placeholder="Friends")
        language = gr.Radio(choices=[("English", "en"), ("쉬운 한국어", "ko")], value="en", label="Explanation language / 답변 언어")
        button = gr.Button("Explain / 맥락 살펴보기", variant="primary")
        answer = gr.Markdown()
        button.click(explain, [expression, situation, relationship, language, task], answer)
        gr.Examples(
            [
                ["반찬은 셀프입니다", "A sign on the wall of a restaurant", "Customer and restaurant", "explain"],
                ["생각해 볼게요", "A reply after inviting a friend to a weekend meetup", "Friends", "explain"],
                ["시간 되실 때 확인 부탁드립니다", "An email asking a professor to review a draft", "Student and professor", "explain"],
                ["How should I greet an older adult in Korea?", "Meeting someone for the first time", "A younger adult and an older adult", "ask"],
            ],
            [expression, situation, relationship, task],
        )
    ui.launch(server_name="127.0.0.1", share=args.share)


if __name__ == "__main__":
    main()
