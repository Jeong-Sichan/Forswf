"""Create a blinded manual scoring sheet and a separate condition key."""

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Scoring CSV")
    parser.add_argument("--key", type=Path, required=True, help="Keep this mapping away from judges")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.comparison.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or args.output == args.key:
        raise ValueError("Need comparison rows and distinct output/key paths")
    randomizer = random.Random(2604)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.key.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as scoring, args.key.open("w", newline="", encoding="utf-8-sig") as mapping:
        score_writer = csv.DictWriter(scoring, fieldnames=["case", "variant", "language", "expression", "situation", "reference", "retrieved_sources", "answer", "meaning_0_2", "context_0_2", "action_0_2", "grounding_0_2", "notes"])
        key_writer = csv.DictWriter(mapping, fieldnames=["case", "variant", "condition"])
        score_writer.writeheader()
        key_writer.writeheader()
        for row in rows:
            conditions = [name for name in ("baseline", "baseline_rag", "adapted", "adapted_rag") if row.get(name) is not None]
            randomizer.shuffle(conditions)
            sources = "\n\n".join(f"{source['title']} {source['url']}\n{source['text']}" for source in row.get("retrieved_sources") or [])
            for letter, condition in zip("ABCD", conditions):
                score_writer.writerow({
                    "case": row["case"], "variant": letter, "language": row["language"],
                    "expression": row["expression"], "situation": row["situation"],
                    "reference": row["reference"], "retrieved_sources": sources if condition.endswith("rag") else "",
                    "answer": row[condition],
                })
                key_writer.writerow({"case": row["case"], "variant": letter, "condition": condition})
    print(f"blinded scoring sheet: {args.output}; separate key: {args.key}")


if __name__ == "__main__":
    main()
