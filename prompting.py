"""Shared prompt and dataset checks for the KoRMo prototype."""

import json
import re
from pathlib import Path

SYSTEM = """당신은 한국 생활에서 마주치는 짧은 한국어 표현을 설명하는 도우미입니다.
주어진 표현, 상황, 상대와의 관계만 근거로 사용하세요. 문화적 규칙을 단정하거나
상대의 속마음을 확신하지 마세요. 정보가 부족하면 가능한 해석을 나누고 필요한
추가 맥락을 말하세요. 의료·법률·기관 규정의 사실관계는 추측하지 마세요."""

LANGUAGE_RULES = {
    "en": """Answer in clear, simple English. Keep the Korean expression visible. Use exactly these four headings:
1. Plain meaning
2. Meaning in this situation
3. What to say or do
4. What could change the meaning""",
    "ko": """쉬운 한국어로 짧고 구체적으로 답하세요. 아래 네 제목을 순서대로 사용하세요.
1. 쉬운 뜻
2. 상황에 따른 해석
3. 적절한 말이나 행동
4. 달라질 수 있는 점""",
}


def require_local_dataset(path: str) -> None:
    """Prevent training/evaluation from reading the shared Seraph NAS."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path("/local_datasets")):
        raise ValueError(f"학습·평가 데이터는 /local_datasets 아래에 있어야 합니다: {path}")


def require_pinned_revision(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("KoRMo revision에는 모델의 40자리 커밋 SHA가 필요합니다")


def messages(expression: str, situation: str, relationship: str, language: str = "en") -> list[dict[str, str]]:
    if language not in LANGUAGE_RULES:
        raise ValueError("출력 언어는 en 또는 ko여야 합니다")
    expression = expression.strip()
    if not expression or len(expression) > 500:
        raise ValueError("표현은 1~500자로 입력해 주세요.")
    situation = situation.strip()[:1000] or "알 수 없음"
    relationship = relationship.strip()[:200] or "알 수 없음"
    return [
        {"role": "system", "content": SYSTEM + "\n" + LANGUAGE_RULES[language]},
        {"role": "user", "content": f"표현: {expression}\n상황: {situation}\n상대와의 관계: {relationship}\n답변 언어: {language}"},
    ]


def render(tokenizer, conversation: list[dict[str, str]], generation: bool) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=generation
        )
    text = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
    return text + ("\nassistant: " if generation else "")


def read_examples(path: str) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("expression", "situation", "relationship", "language", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{path}:{number}: {field} 값이 필요합니다")
        if row["language"] not in LANGUAGE_RULES:
            raise ValueError(f"{path}:{number}: language는 en 또는 ko여야 합니다")
        key = (row["expression"].strip(), row["situation"].strip(), row["language"])
        if key in seen:
            raise ValueError(f"{path}:{number}: 중복 사례")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: 사례가 없습니다")
    return rows
