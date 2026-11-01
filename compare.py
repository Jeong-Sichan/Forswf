"""Generate paired base/adapter answers, with and without the same retrieval."""

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
    parser.add_argument("--knowledge", help="Staged chunk JSONL for the same retrieved context in both models")
    parser.add_argument("--retriever", choices=("e5", "bm25"), default="e5")
    args = parser.parse_args()

    require_pinned_revision(args.revision)
    require_local_dataset(args.eval)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_examples(args.eval)
    retrieval_index = None
    if args.knowledge:
        from knowledge import BM25Index, E5Index, load_chunks

        chunks = load_chunks(args.knowledge)
        retrieval_index = E5Index(chunks, cache_path=args.knowledge + ".e5.pt") if args.retriever == "e5" else BM25Index(chunks)
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

    def generate(row, evidence=None):
        prompt = render(tokenizer, messages(row["expression"], row["situation"], row["relationship"], row["language"], row["task"], evidence), True)
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
        for case_number, row in enumerate(rows, 1):
            from knowledge import evidence_for_prompt

            found = retrieval_index.search(row["expression"] + " " + row["situation"]) if retrieval_index else None
            evidence = evidence_for_prompt(found) if found is not None else row.get("evidence")
            with model.disable_adapter():
                baseline = generate(row)
                baseline_rag = generate(row, evidence) if evidence is not None else None
            adapted = generate(row)
            adapted_rag = generate(row, evidence) if evidence is not None else None
            output.write(json.dumps({
                "case": case_number,
                "expression": row["expression"],
                "situation": row["situation"],
                "relationship": row["relationship"],
                "language": row["language"],
                "task": row["task"],
                "reference": row["answer"],
                "baseline": baseline,
                "adapted": adapted,
                "baseline_rag": baseline_rag,
                "adapted_rag": adapted_rag,
                "retrieved_sources": [{"title": source["title"], "url": source["url"], "text": source["text"], "score": source.get("score")} for source in found] if found is not None else None,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
