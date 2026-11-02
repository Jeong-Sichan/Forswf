"""Fill missing English review drafts with KoRMo; never marks them reviewed."""

import argparse
import csv
import re
from pathlib import Path

from kormo_compat import fix_causal_mask_keyword
from prompting import messages, render, require_pinned_revision


ENGLISH_FIELDS = (
    "english_meaning_draft", "english_interpretation_draft",
    "english_response_draft", "english_uncertainty_draft",
)
HEADINGS = (
    "Plain meaning", "Meaning in this situation",
    "What to say or do", "What could change the meaning",
)


def split_answer(answer: str) -> list[str] | None:
    markers = []
    for number, heading in enumerate(HEADINGS, 1):
        found = re.search(rf"(?im)^\s*{number}\.\s*{re.escape(heading)}\s*\n?", answer)
        if not found:
            return None
        markers.append(found)
    if any(markers[i].start() >= markers[i + 1].start() for i in range(3)):
        return None
    parts = [answer[markers[i].end():markers[i + 1].start()].strip() for i in range(3)]
    parts.append(answer[markers[3].end():].strip())
    return parts if all(parts) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Clean 64-case review CSV")
    parser.add_argument("--output", type=Path, required=True, help="New CSV with unreviewed English drafts")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model", default="KORMo-Team/KORMo-10B-sft")
    parser.add_argument("--max-cases", type=int, default=0, help="0 generates for every missing row")
    args = parser.parse_args()
    require_pinned_revision(args.revision)
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a new output path; the original review CSV must be preserved")
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if any(field not in fieldnames for field in ENGLISH_FIELDS):
        parser.error("Review CSV is missing English draft columns")
    if "model_output_draft" not in fieldnames:
        fieldnames.append("model_output_draft")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("KoRMo draft generation requires an allocated GPU compute node")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        ),
    )
    fix_causal_mask_keyword(model)
    model.eval()
    generated = 0
    parsed = 0
    for row in rows:
        if all(row.get(field, "").strip() for field in ENGLISH_FIELDS):
            continue
        if args.max_cases and generated >= args.max_cases:
            break
        prompt = render(tokenizer, messages(
            row["expression"], row["situation"], row["relationship"], "en", "explain"
        ), True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        row["model_output_draft"] = answer
        parts = split_answer(answer)
        if parts:
            for field, part in zip(ENGLISH_FIELDS, parts):
                row[field] = part
            parsed += 1
        generated += 1
        print(f"{row['id']}: {'four draft fields filled' if parts else 'manual split needed'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {generated} unreviewed answers, parsed {parsed}; output: {args.output}")


if __name__ == "__main__":
    main()
