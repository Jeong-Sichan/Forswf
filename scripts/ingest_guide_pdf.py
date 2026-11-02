"""Turn a permitted public guide PDF into page-cited RAG documents.

Run on an allocated Seraph compute node with the PDF under /local_datasets.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from prompting import require_local_dataset


def ingest(pdf_path: Path, output: Path, title: str, source_url: str, license_name: str,
           publisher: str, updated_at: str, language: str) -> tuple[int, int]:
    require_local_dataset(str(pdf_path))
    require_local_dataset(str(output))
    if not source_url.startswith("https://") or not all((title.strip(), license_name.strip(), publisher.strip())):
        raise ValueError("Official source URL, title, publisher, and checked reuse terms are required")
    import fitz

    document_id = hashlib.sha256(source_url.encode()).hexdigest()[:16]
    kept = 0
    skipped = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as pdf, output.open("w", encoding="utf-8") as handle:
        for number, page in enumerate(pdf, 1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            if len(text) < 80:
                skipped += 1
                continue
            handle.write(json.dumps({
                "source_id": f"guide:{document_id}:p{number}",
                "title": f"{title} · {publisher} · p.{number}",
                "url": f"{source_url}#page={number}",
                "license": license_name,
                "language": language,
                "updated_at": updated_at,
                "text": text,
            }, ensure_ascii=False) + "\n")
            kept += 1
    if not kept:
        raise ValueError("No readable page text; the PDF may require OCR")
    return kept, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-url", required=True, help="Public official PDF URL, without a private token")
    parser.add_argument("--license", required=True, help="Checked reuse terms, including their source URL")
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--updated-at", default="", help="Publication or revision date if shown")
    parser.add_argument("--language", choices=("ko", "en"), default="ko")
    args = parser.parse_args()
    kept, skipped = ingest(args.pdf, args.output, args.title, args.source_url,
                           args.license, args.publisher, args.updated_at, args.language)
    print(f"extracted {kept} cited pages; skipped {skipped} pages without readable text: {args.output}")
