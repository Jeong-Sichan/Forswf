"""Run on a machine with a KoRMo-compatible GPU and installed dependencies."""

import argparse

from prompting import messages, render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="KORMo-Team/KORMo-10B-sft")
    parser.add_argument("--adapter", help="Optional locally trained LoRA adapter directory")
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio link")
    args = parser.parse_args()

    import gradio as gr
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 필요합니다. 학교 서버에서 실행해 주세요.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def explain(expression: str, situation: str, relationship: str) -> str:
        try:
            prompt = render(tokenizer, messages(expression, situation, relationship), True)
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

    with gr.Blocks(title="말맥 | 한국어 맥락 해설기") as ui:
        gr.Markdown("# 말맥\n한국어 표현의 뜻과 상황에 따른 해석을 살펴보세요. AI 답변은 틀릴 수 있습니다.")
        expression = gr.Textbox(label="한국어 표현", placeholder="생각해 볼게요")
        situation = gr.Textbox(label="상황", placeholder="동료에게 주말 모임을 제안했어요")
        relationship = gr.Textbox(label="상대와의 관계", placeholder="직장 동료")
        button = gr.Button("맥락 살펴보기", variant="primary")
        answer = gr.Markdown()
        button.click(explain, [expression, situation, relationship], answer)
        gr.Examples(
            [
                ["반찬은 셀프입니다", "식당 벽의 안내문", "손님과 식당"],
                ["생각해 볼게요", "동료에게 주말 모임을 제안한 뒤 들은 답", "직장 동료"],
                ["시간 되실 때 확인 부탁드립니다", "교수에게 보낼 메일 문장", "학생과 교수"],
            ],
            [expression, situation, relationship],
        )
    ui.launch(server_name="127.0.0.1", share=args.share)


if __name__ == "__main__":
    main()
