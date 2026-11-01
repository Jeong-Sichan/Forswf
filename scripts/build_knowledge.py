"""Chunk staged, attributed documents for lexical retrieval."""

import argparse
import json
from pathlib import Path

from knowledge import chunk_document
from prompting import require_local_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_local_dataset(args.documents)
    require_local_dataset(args.output)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = 0
    documents = 0
    seen = set()
    with Path(args.documents).open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            document = json.loads(line)
            if document["source_id"] in seen:
                continue
            seen.add(document["source_id"])
            documents += 1
            try:
                rows = chunk_document(document)
            except ValueError as exc:
                raise ValueError(f"document line {number}: {exc}") from exc
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunks += 1
    if not chunks:
        raise ValueError("No usable chunks")
    print(f"indexed {documents} documents into {chunks} chunks: {output}")


if __name__ == "__main__":
    main()
