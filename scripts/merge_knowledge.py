"""Merge independently collected source documents into one attributed archive."""

import argparse
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        parser.error("input documents missing; collect them before merging: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    destination = args.output / "documents.jsonl"
    with destination.open("w", encoding="utf-8") as target:
        for path in args.inputs:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record["source_id"]
                if key not in seen:
                    seen.add(key)
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    if not count:
        raise ValueError("No documents to merge")
    with tarfile.open(args.output / "knowledge.tar.gz", "w:gz") as archive:
        archive.add(destination, arcname="documents.jsonl")
    print(f"merged {count} documents: {args.output / 'knowledge.tar.gz'}")


if __name__ == "__main__":
    main()
