"""Run on a machine with a KoRMo-compatible GPU and installed dependencies."""

import argparse

from prompting import messages, render, require_pinned_revision
from kormo_compat import fix_causal_mask_keyword


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="KORMo-Team/KORMo-10B-sft")
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face model commit SHA")
    parser.add_argument("--adapter", help="Optional locally trained LoRA adapter directory")
    parser.add_argument("--smoke", action="store_true", help="Generate one answer and exit without a web server")
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio link")
    args = parser.parse_args()
    require_pinned_revision(args.revision)

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

    def explain(expression: str, situation: str, relationship: str, language: str) -> str:
        try:
            prompt = render(tokenizer, messages(expression, situation, relationship, language), True)
        except ValueError as exc:
            return str(exc)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if args.smoke:
        print(explain("반찬은 셀프입니다", "A sign on the wall of a restaurant", "Customer and restaurant", "en"))
        return

    with gr.Blocks(title="말맥 | 한국어 맥락 해설기") as ui:
        gr.Markdown("# 말맥 | Malmaek\nUnderstand Korean expressions in context. 한국어 표현을 상황에 맞게 설명합니다. AI answers may be wrong.")
        expression = gr.Textbox(label="Korean expression / 한국어 표현", placeholder="생각해 볼게요")
        situation = gr.Textbox(label="Situation / 상황", placeholder="A reply after inviting a friend to a weekend meetup")
        relationship = gr.Textbox(label="Relationship / 상대와의 관계", placeholder="Friends")
        language = gr.Radio(choices=[("English", "en"), ("쉬운 한국어", "ko")], value="en", label="Explanation language / 답변 언어")
        button = gr.Button("Explain / 맥락 살펴보기", variant="primary")
        answer = gr.Markdown()
        button.click(explain, [expression, situation, relationship, language], answer)
        gr.Examples(
            [
                ["반찬은 셀프입니다", "A sign on the wall of a restaurant", "Customer and restaurant"],
                ["생각해 볼게요", "A reply after inviting a friend to a weekend meetup", "Friends"],
                ["시간 되실 때 확인 부탁드립니다", "An email asking a professor to review a draft", "Student and professor"],
            ],
            [expression, situation, relationship],
        )
    ui.launch(server_name="127.0.0.1", share=args.share)


if __name__ == "__main__":
    main()
