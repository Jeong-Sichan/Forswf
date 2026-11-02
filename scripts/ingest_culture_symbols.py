"""Turn the KOGL Type 1 traditional-symbol dictionary CSV into attributed RAG documents.

Input dataset: 국가유산진흥원_전통문화 상징사전 (data.go.kr dataset 15143561).
Download the CSV from the dataset page; this script never downloads or republishes it.
"""

import argparse
import csv
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


DATASET_URL = "https://www.data.go.kr/data/15143561/fileData.do"
LICENSE = "국가유산진흥원 공공누리 제1유형: 출처표시 (https://www.kogl.or.kr/info/licenseType1.do)"
UPDATED_AT = "2025-06-02"
REQUIRED = (
    "연번", "문양형태 대분류", "문양형태 세분류", "원천유물명",
    "유물과 문양에 대한 설명", "문양의 상징적 의미", "재질", "사이트주소",
)


def read_rows(source: Path) -> list[dict[str, str]]:
    raw = source.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Culture-symbol CSV is neither UTF-8 nor CP949")
    reader = csv.DictReader(io.StringIO(text))
    if any(field not in (reader.fieldnames or []) for field in REQUIRED):
        raise ValueError("Culture-symbol CSV columns do not match dataset 15143561")
    rows = list(reader)
    if not rows:
        raise ValueError("Culture-symbol CSV is empty")
    return rows


def convert(row: dict[str, str]) -> dict:
    if any(not row.get(field, "").strip() for field in REQUIRED[:-1]):
        raise ValueError(f"Incomplete culture-symbol record: {row.get('연번', '')}")
    number = row["연번"].strip()
    if not number.isdigit():
        raise ValueError(f"Non-numeric culture-symbol ID: {number}")
    large = row["문양형태 대분류"].strip()
    small = row["문양형태 세분류"].strip()
    artifact = row["원천유물명"].strip()
    material = row["재질"].strip()
    text = (
        f"문양: {small} ({large}). 원천유물: {artifact}. 재질: {material}.\n"
        f"유물과 문양 설명: {row['유물과 문양에 대한 설명'].strip()}\n"
        f"문양의 상징적 의미: {row['문양의 상징적 의미'].strip()}"
    )
    return {
        "source_id": f"publicdata:15143561:{number}",
        "title": f"{small} · {artifact} · 전통문화 상징사전",
        "url": DATASET_URL,
        "license": LICENSE,
        "language": "ko",
        "publisher": "국가유산진흥원",
        "updated_at": UPDATED_AT,
        "retrieved_at": datetime.now(timezone.utc).date().isoformat(),
        "original_url": row["사이트주소"].strip(),
        "text": text,
    }


def ingest(source: Path, output: Path) -> int:
    rows = read_rows(source)
    records = [convert(row) for row in rows]
    ids = [record["source_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate culture-symbol IDs")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "documents.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with tarfile.open(output / "knowledge.tar.gz", "w:gz") as archive:
        archive.add(destination, arcname="documents.jsonl")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Downloaded dataset 15143561 CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = ingest(args.input, args.output)
    print(f"ingested {count} attributed culture-symbol records: {args.output / 'knowledge.tar.gz'}")
