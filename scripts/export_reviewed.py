"""Export human-reviewed CSV cases to a pilot bilingual QLoRA archive."""

import argparse
import csv
import hashlib
import json
import random
import tarfile
from pathlib import Path

from output_style import english_issues, remove_inline_citations


FIELDS_KO = ("meaning_draft", "interpretation_draft", "response_draft", "uncertainty_draft")
FIELDS_EN = ("english_meaning_draft", "english_interpretation_draft", "english_response_draft", "english_uncertainty_draft")
HEADINGS = {
    "ko": ("1. 쉬운 뜻", "2. 상황에 따른 해석", "3. 적절한 말이나 행동", "4. 달라질 수 있는 점"),
    "en": ("1. Plain meaning", "2. Meaning in this situation", "3. What to say or do", "4. What could change the meaning"),
}


def convert(row: dict, language: str) -> dict:
    fields = FIELDS_EN if language == "en" else FIELDS_KO
    parts = [row.get(field, "").strip() for field in fields]
    if not all(parts):
        raise ValueError(f"{row['id']}: {language} 답변 네 칸을 모두 검수·작성해야 합니다")
    support = row.get("source_supports_claim", "").strip().upper()
    if support not in ("YES", "NO"):
        raise ValueError(f"{row['id']}: source_supports_claim에 YES 또는 NO를 적어 주세요")
    evidence = []
    if support == "YES":
        keys = ("candidate_source_title", "candidate_source_url", "candidate_source_license", "candidate_excerpt")
        if any(not row.get(key, "").strip() for key in keys):
            raise ValueError(f"{row['id']}: 근거 문서의 제목·URL·라이선스·본문이 필요합니다")
        url = row["candidate_source_url"].strip()
        excerpt = row["candidate_excerpt"].strip()
        evidence = [{
            "source_id": "review:" + hashlib.sha256((url + excerpt).encode()).hexdigest()[:20],
            "title": row["candidate_source_title"].strip(), "url": url,
            "license": row["candidate_source_license"].strip(), "text": excerpt,
        }]
    elif any("[1]" in part for part in parts):
        raise ValueError(f"{row['id']}: 출처가 부적합하면 {language} 답변의 [1] 인용을 지워 주세요")
    parts = [remove_inline_citations(part) for part in parts]
    answer = "\n".join(f"{heading}\n{part}" for heading, part in zip(HEADINGS[language], parts))
    if language == "en":
        issues = english_issues(answer, "explain")
        if issues:
            raise ValueError(f"{row['id']}: 영어 답변 형식 검토 필요: {', '.join(issues)}")
    return {
        "task": "explain", "expression": row["expression"].strip(),
        "situation": row["situation"].strip(), "relationship": row["relationship"].strip(),
        "language": language, "answer": answer, "evidence": evidence,
        "reviewed": True, "case_id": row["id"].strip(),
    }


def export(source: Path, destination: Path, require_bilingual: bool) -> tuple[int, int]:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviewed = [row for row in rows if row.get("reviewed", "").strip().upper() == "YES"]
    if not reviewed:
        raise ValueError("reviewed=YES인 사례가 없습니다")
    seen = set()
    for row in reviewed:
        expression = row.get("expression", "").strip()
        if not expression or expression in seen or not row.get("reviewer", "").strip():
            raise ValueError(f"중복 표현·빈 표현 또는 검수자 누락: {row.get('id', '')}")
        seen.add(expression)
    random.Random(2604).shuffle(reviewed)
    eval_count = max(1, round(len(reviewed) * 0.2))
    split_rows = {"eval": reviewed[:eval_count], "train": reviewed[eval_count:]}
    destination.mkdir(parents=True, exist_ok=True)
    for split, cases in split_rows.items():
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in cases:
                answer_rows = [convert(row, "ko")]
                english = [row.get(key, "").strip() for key in FIELDS_EN]
                if any(english) or require_bilingual:
                    answer_rows.append(convert(row, "en"))
                for record in answer_rows:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with tarfile.open(destination / "malmaek.tar.gz", "w:gz") as archive:
        for name in ("train.jsonl", "eval.jsonl"):
            archive.add(destination / name, arcname=name)
    return len(split_rows["train"]), len(split_rows["eval"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-bilingual", action="store_true")
    args = parser.parse_args()
    train, evaluation = export(args.reviewed_csv, args.output_dir, args.require_bilingual)
    print(f"exported {train} train situations and {evaluation} held-out eval situations: {args.output_dir / 'malmaek.tar.gz'}")
    if train < 50:
        print("주의: 현재 학습 프로그램은 서로 다른 학습 표현 50개 이상을 요구합니다")
