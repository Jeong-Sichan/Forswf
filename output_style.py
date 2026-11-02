"""Small checks for user-facing English answers from KoRMo."""

import re

from pronunciation import romanize


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
READING_AFTER_PHRASE = r"([\"'”’]?)(?:\s*\(([A-Za-z\s'-]+)\))?"


def _replace_reading(match: re.Match[str], phrase: str) -> str:
    closing_quote, existing = match.groups()
    corrected = phrase + closing_quote + " (" + romanize(phrase) + ")"
    if existing and re.search(r"\b(?:I|you|we|it|is|are|the|a|an|to|will|think|means)\b", existing, re.I):
        corrected += " (" + existing + ")"
    return corrected


def normalize_pronunciation(answer: str, task: str, expression: str) -> str:
    """Replace model-written readings for the taught phrase with a consistent reading."""
    if task == "explain":
        phrase = expression.strip().rstrip(".!?。！？ ")
        if not re.search(r"[가-힣]", phrase):
            return answer
        pattern = re.compile(re.escape(phrase) + READING_AFTER_PHRASE)
        replacement = phrase + " (" + romanize(phrase) + ")"
        answer, count = pattern.subn(lambda match: _replace_reading(match, phrase), answer)
        if not count and HEADINGS["explain"][0] in answer:
            answer = answer.replace(HEADINGS["explain"][0], HEADINGS["explain"][0] + "\n" + replacement, 1)
        return answer
    if task == "ask" and HEADINGS["ask"][0] in answer:
        start = answer.find(HEADINGS["ask"][0]) + len(HEADINGS["ask"][0])
        end = answer.find(HEADINGS["ask"][1], start)
        if end < 0:
            end = len(answer)
        section = answer[start:end]
        phrase = re.search(r"[가-힣]+(?: [가-힣]+)*", section)
        if phrase:
            pattern = re.compile(re.escape(phrase.group()) + READING_AFTER_PHRASE)
            section = pattern.sub(lambda match: _replace_reading(match, phrase.group()), section, count=1)
            return answer[:start] + section + answer[end:]
    return answer


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
                section = answer[start:end].strip(" \n\t:*#-")
                if len(section) < 8:
                    issues.append("empty English section")
                    break
                english = PRONOUNCED_KOREAN.sub("", section)
                if len(re.findall(r"[A-Za-z]+", english)) < (3 if index == 0 and task == "explain" else 2):
                    issues.append("section lacks English explanation")
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
