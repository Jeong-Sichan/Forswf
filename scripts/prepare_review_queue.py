"""Make a compact human-review sheet for 64 unreviewed phrase scenarios."""

import argparse
import csv

from data.draft_cases import CASES
from data.draft_english import ENGLISH_DRAFTS


def main() -> None:
    parser = argparse.ArgumentParser()
    from pathlib import Path

    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "category", "expression", "situation", "relationship", "meaning_draft", "interpretation_draft", "response_draft", "uncertainty_draft", "english_meaning_draft", "english_interpretation_draft", "english_response_draft", "english_uncertainty_draft", "candidate_source_title", "candidate_source_url", "candidate_source_license", "candidate_excerpt", "source_supports_claim", "reviewer", "reviewed"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            english = ENGLISH_DRAFTS.get(case["id"], {})
            writer.writerow({
                "id": case["id"], "category": case["category"],
                "expression": case["expression"], "situation": case["situation"],
                "relationship": case["relationship"], "meaning_draft": case["meaning"],
                "interpretation_draft": case["interpretation"], "response_draft": case["response"],
                "uncertainty_draft": case["uncertainty"], "candidate_source_title": "",
                "english_meaning_draft": english.get("meaning", ""),
                "english_interpretation_draft": english.get("interpretation", ""),
                "english_response_draft": english.get("response", ""),
                "english_uncertainty_draft": english.get("uncertainty", ""),
                "candidate_source_url": "", "candidate_source_license": "",
                "candidate_excerpt": "", "source_supports_claim": "NO",
                "reviewer": "", "reviewed": "NO",
            })
    print(f"unreviewed review queue: {args.output} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
