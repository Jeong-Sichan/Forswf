"""Shared prompt and dataset checks for the KoRMo prototype."""

import json
import re
from pathlib import Path

SYSTEM = """당신은 한국 생활에서 쓰는 한국어 표현을 안내하는 도우미입니다.
주어진 표현, 상황, 상대와의 관계만 근거로 사용하세요. 문화적 규칙을 단정하거나
상대의 속마음을 확신하지 마세요. 정보가 부족하면 가능한 해석을 나누고 필요한
추가 맥락을 말하세요. 의료·법률·기관 규정의 사실관계는 추측하지 마세요.
입력에 없는 식당 유형, 메뉴, 상대의 동기 같은 세부사항을 만들어 내지 마세요.
실제 방문자가 할 수 있는 자연스러운 행동을 제안하세요."""

SYSTEM_EN = """You explain Korean expressions and everyday situations to newcomers.
Use only the user's phrase, situation, relationship, and supplied excerpts. Do not claim to know
another person's intention or invent details about a restaurant, place, region, or custom.
If context is missing, say what is unknown and suggest one practical question or action.
Do not guess medical, legal, or institutional rules."""

TASK_RULES = {
    "explain": "주어진 한국어 표현을 해당 상황에 맞춰 해설하세요. 뜻이 여러 가지일 수 있으면 구분하세요.",
    "ask": "질문에 맞는 한국어 표현을 한글로 제안하고, 언제 쓰는지 설명하세요. 로마자 표기는 보조로만 쓰세요. 나이만으로 말투를 단정하지 말고 관계와 격식을 고려하세요.",
}

TASK_RULES_EN = {
    "explain": "Explain the Korean phrase in the stated situation. Separate its direct meaning from assumptions about the setting.",
    "ask": "Suggest a Korean phrase in Hangul and explain when to use it. Consider the relationship and formality, not age alone.",
}

SOURCE_RULES_EN = """Use excerpt numbers only for claims directly supported by those excerpts. Put a short citation such as [1] after the supported claim. A source about side dishes does not establish a restaurant's service layout. Do not add source-summary lines under the four headings. If no excerpt supports a cultural claim, leave it out or state that the sources do not establish it. Keep each section to one or two sentences."""
SOURCE_RULES_KO = """발췌문이 직접 뒷받침하는 사실에만 [1]처럼 번호를 붙이세요. 반찬을 설명하는 자료만으로 식당의 실제 운영 방식이나 배치 장소를 단정할 수 없습니다. 네 항목 아래에 출처 요약을 별도 줄로 반복하지 마세요. 근거가 없는 문화적 주장이나 행동 세부사항은 빼거나 확인이 필요하다고 말하세요. 각 항목은 한두 문장으로 쓰세요."""

LANGUAGE_RULES = {
    "en": {
        "explain": """Reply only in clear, simple English, except when quoting Korean expressions. Keep the Korean expression visible. Use exactly these four headings:
1. Plain meaning
2. Meaning in this situation
3. What to say or do
4. What could change the meaning""",
        "ask": """Reply only in clear, simple English, except when quoting Korean expressions. Use exactly these four headings:
1. Korean phrase
2. When to use it
3. What to say or do
4. What may vary""",
    },
    "ko": {
        "explain": """쉬운 한국어로 짧고 구체적으로 답하세요. 아래 네 제목을 순서대로 사용하세요.
1. 쉬운 뜻
2. 상황에 따른 해석
3. 적절한 말이나 행동
4. 달라질 수 있는 점""",
        "ask": """쉬운 한국어로 짧고 구체적으로 답하세요. 아래 네 제목을 순서대로 사용하세요.
1. 추천 표현
2. 사용하는 상황
3. 적절한 말이나 행동
4. 달라질 수 있는 점""",
    },
}


def require_local_dataset(path: str) -> None:
    """Prevent training/evaluation from reading the shared Seraph NAS."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path("/local_datasets")):
        raise ValueError(f"학습·평가 데이터는 /local_datasets 아래에 있어야 합니다: {path}")


def require_pinned_revision(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("KoRMo revision에는 모델의 40자리 커밋 SHA가 필요합니다")


def messages(expression: str, situation: str, relationship: str, language: str = "en", task: str = "explain", evidence: list[dict] | None = None) -> list[dict[str, str]]:
    if language not in LANGUAGE_RULES:
        raise ValueError("출력 언어는 en 또는 ko여야 합니다")
    if task not in TASK_RULES:
        raise ValueError("작업은 explain 또는 ask여야 합니다")
    expression = expression.strip()
    if not expression or len(expression) > 500:
        raise ValueError("표현은 1~500자로 입력해 주세요.")
    situation = situation.strip()[:1000] or "알 수 없음"
    relationship = relationship.strip()[:200] or "알 수 없음"
    label = "표현" if task == "explain" else "질문"
    system = (SYSTEM_EN + "\n" + TASK_RULES_EN[task]) if language == "en" else (SYSTEM + "\n" + TASK_RULES[task])
    user = f"{label}: {expression}\n상황: {situation}\n상대와의 관계: {relationship}\n답변 언어: {language}"
    if evidence is not None:
        system += "\n" + (SOURCE_RULES_EN if language == "en" else SOURCE_RULES_KO)
        if evidence:
            excerpts = []
            for index, item in enumerate(evidence[:3], 1):
                if any(not isinstance(item.get(key), str) or not item[key].strip() for key in ("source_id", "title", "url", "license", "text")):
                    raise ValueError("출처 자료에 source_id/title/url/license/text가 필요합니다")
                excerpts.append(f"[{index}] {item['title']}\nURL: {item['url']}\nExcerpt: {item['text'][:750]}")
            user += "\n\nSource excerpts / 출처 발췌:\n" + "\n\n".join(excerpts)
        else:
            user += "\n\nSource excerpts / 출처 발췌: none found. Do not invent a citation."
    system += "\n" + LANGUAGE_RULES[language][task]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def guide_messages(question: str, situation: str, language: str, evidence: list[dict]) -> list[dict[str, str]]:
    """Ask a student-life question using retrieved, dated official passages."""
    if language not in {"ko", "en"} or not question.strip() or not evidence:
        raise ValueError("Guide questions need a language, question, and retrieved passages")
    language_rule = "Answer in clear English." if language == "en" else "쉬운 한국어로 답하세요."
    system = (
        "You help international students with life in Korea. Answer only from the supplied passages. "
        "Cite each supported claim with [1] or [2]. If the passages do not answer the question, "
        "say so and suggest checking the linked official source. Do not invent procedures, fees, "
        "deadlines, eligibility, or cultural rules. State that procedures may have changed since "
        "the source's reference date. " + language_rule
    )
    excerpts = []
    for number, row in enumerate(evidence[:3], 1):
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("title", "url", "text")):
            raise ValueError("Guide passages need title, URL, and text")
        excerpts.append(
            f"[{number}] {row['title']}\nReference date: {row.get('updated_at') or 'not shown'}\n"
            f"URL: {row['url']}\nExcerpt: {row['text'][:900]}"
        )
    user = f"Question: {question.strip()}\nSituation: {situation.strip() or 'not specified'}\n\n" + "\n\n".join(excerpts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
        row.setdefault("task", "explain")
        if row["task"] not in TASK_RULES:
            raise ValueError(f"{path}:{number}: task는 explain 또는 ask여야 합니다")
        if "evidence" in row:
            if not isinstance(row["evidence"], list):
                raise ValueError(f"{path}:{number}: evidence는 목록이어야 합니다")
            for item in row["evidence"]:
                if not isinstance(item, dict) or any(not isinstance(item.get(field), str) or not item[field].strip() for field in ("source_id", "title", "url", "license", "text")):
                    raise ValueError(f"{path}:{number}: evidence의 출처 정보가 부족합니다")
        key = (row["expression"].strip(), row["situation"].strip(), row["language"], row["task"])
        if key in seen:
            raise ValueError(f"{path}:{number}: 중복 사례")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: 사례가 없습니다")
    return rows
