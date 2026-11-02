"""Export 64 unreviewed draft cases for a clearly labelled QLoRA pilot run."""

import argparse
import csv
import json
import random
import tarfile
from pathlib import Path

from scripts.export_reviewed import FIELDS_EN, convert


def export(source: Path, output: Path) -> tuple[int, int, int]:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 63:
        raise ValueError("시범 학습에는 64개 표현 초안이 필요합니다")
    expressions = [row.get("expression", "").strip() for row in rows]
    if len(set(expressions)) != len(expressions) or any(not item for item in expressions):
        raise ValueError("시범 자료에 중복되거나 빈 표현이 있습니다")
    random.Random(2604).shuffle(rows)
    eval_count = max(1, round(len(rows) * 0.2))
    splits = {"eval": rows[:eval_count], "train": rows[eval_count:]}
    output.mkdir(parents=True, exist_ok=True)
    english_count = 0
    for split, cases in splits.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                # Retrieved Wikipedia snippets were unrelated for many phrases.
                # This pilot learns answer format only; no source-grounding claim.
                case = dict(case, source_supports_claim="NO")
                records = [convert(case, "ko")]
                if all(case.get(field, "").strip() for field in FIELDS_EN):
                    records.append(convert(case, "en"))
                    english_count += 1
                for record in records:
                    record["reviewed"] = False
                    record["pilot_only"] = True
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with tarfile.open(output / "malmaek-pilot.tar.gz", "w:gz") as archive:
        for name in ("train.jsonl", "eval.jsonl"):
            archive.add(output / name, arcname=name)
    return len(splits["train"]), len(splits["eval"]), english_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train, evaluation, english = export(args.draft_csv, args.output_dir)
    print(f"PILOT ONLY, UNREVIEWED: {train} train situations, {evaluation} eval situations, {english} English answers")
    print(f"archive: {args.output_dir / 'malmaek-pilot.tar.gz'}")
