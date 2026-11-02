"""Pair 64 unreviewed phrase scenarios with candidate sources for human review."""

import argparse
import csv
from pathlib import Path

from data.draft_cases import CASES
from data.draft_english import ENGLISH_DRAFTS
from knowledge import BM25Index, E5Index, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever", choices=("e5", "bm25"), default="e5")
    args = parser.parse_args()
    chunks = load_chunks(args.knowledge)
    index = E5Index(chunks, cache_path=args.knowledge + ".e5.pt") if args.retriever == "e5" else BM25Index(chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "category", "expression", "situation", "relationship", "meaning_draft", "interpretation_draft", "response_draft", "uncertainty_draft", "english_meaning_draft", "english_interpretation_draft", "english_response_draft", "english_uncertainty_draft", "candidate_source_title", "candidate_source_url", "candidate_source_license", "candidate_excerpt", "source_supports_claim", "reviewer", "reviewed"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            results = index.search(case["expression"] + " " + case["situation"], limit=1)
            source = results[0] if results else {}
            english = ENGLISH_DRAFTS.get(case["id"], {})
            writer.writerow({
                "id": case["id"], "category": case["category"],
                "expression": case["expression"], "situation": case["situation"],
                "relationship": case["relationship"], "meaning_draft": case["meaning"],
                "interpretation_draft": case["interpretation"], "response_draft": case["response"],
                "uncertainty_draft": case["uncertainty"], "candidate_source_title": source.get("title", ""),
                "english_meaning_draft": english.get("meaning", ""),
                "english_interpretation_draft": english.get("interpretation", ""),
                "english_response_draft": english.get("response", ""),
                "english_uncertainty_draft": english.get("uncertainty", ""),
                "candidate_source_url": source.get("url", ""), "candidate_source_license": source.get("license", ""),
                "candidate_excerpt": source.get("text", ""), "source_supports_claim": "UNREVIEWED",
                "reviewer": "", "reviewed": "NO",
            })
    print(f"unreviewed review queue: {args.output} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
