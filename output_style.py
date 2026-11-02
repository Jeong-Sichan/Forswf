"""Small checks for user-facing English answers from KoRMo."""

import re


HEADINGS = {
    "explain": (
        "1. Plain meaning", "2. Meaning in this situation",
        "3. What to say or do", "4. What could change the meaning",
    ),
    "ask": (
        "1. Korean phrase", "2. When to use it",
        "3. What to say or do", "4. What may vary",
    ),
}

PRONOUNCED_KOREAN = re.compile(
    r"[\"'“‘]?[가-힣][가-힣 ]{0,39}[\"'”’]?\s*\([A-Za-z][A-Za-z\s'-]*\)"
)


def english_issues(answer: str, task: str) -> list[str]:
    issues = []
    if task in HEADINGS:
        positions = [answer.find(heading) for heading in HEADINGS[task]]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            issues.append("missing or out-of-order English sections")
        else:
            for index, position in enumerate(positions):
                start = position + len(HEADINGS[task][index])
                end = positions[index + 1] if index < 3 else len(answer)
                if len(answer[start:end].strip(" \n\t:*#-")) < 8:
                    issues.append("empty English section")
                    break
    hangul = len(re.findall(r"[가-힣]", answer))
    pronounced = PRONOUNCED_KOREAN.findall(answer)
    unpronounced = len(re.findall(r"[가-힣]", PRONOUNCED_KOREAN.sub("", answer)))
    if task in HEADINGS and not hangul:
        issues.append("missing Korean expression")
    if task in HEADINGS and hangul and not pronounced:
        issues.append("missing Latin-letter pronunciation")
    if unpronounced > 8:
        issues.append("Korean prose in English answer")
    return issues


def remove_inline_citations(answer: str) -> str:
    """Keep source attribution in the separate footer, not in answer prose."""
    return re.sub(r"\s*\[\d+\]", "", answer).strip()
