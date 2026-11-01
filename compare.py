"""Generate blinded baseline/adapter answers for manual scoring."""

import argparse
import json
from pathlib import Path

from prompting import messages, read_examples, render, require_local_dataset, require_pinned_revision
from kormo_compat import fix_causal_mask_keyword


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="KORMo-Team/KORMo-10B-sft")
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face model commit SHA")
    args = parser.parse_args()

    require_pinned_revision(args.revision)
    require_local_dataset(args.eval)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_examples(args.eval)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 필요합니다")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        ),
    )
    fix_causal_mask_keyword(base)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def generate(row):
        prompt = render(tokenizer, messages(row["expression"], row["situation"], row["relationship"], row["language"], row["task"]), True)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = model.generate(
                **encoded, max_new_tokens=400, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(result[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            with model.disable_adapter():
                baseline = generate(row)
            adapted = generate(row)
            output.write(json.dumps({
                "case": index + 1,
                "expression": row["expression"],
                "situation": row["situation"],
                "relationship": row["relationship"],
                "language": row["language"],
                "task": row["task"],
                "reference": row["answer"],
                "baseline": baseline,
                "adapted": adapted,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
