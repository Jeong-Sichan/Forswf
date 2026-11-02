"""Readable, approximate Revised Romanization for short Hangul expressions."""

import re


INITIAL = (
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
    "j", "jj", "ch", "k", "t", "p", "h",
)
VOWEL = (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa",
    "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)
FINAL = (
    "", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "p",
    "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t",
    "k", "t", "p", "t",
)
HANGUL = re.compile(r"[가-힣]+")


def _parts(syllable: str) -> tuple[int, int, int]:
    value = ord(syllable) - 0xAC00
    return value // 588, value % 588 // 28, value % 28


def romanize(text: str) -> str:
    """Romanize Hangul; pronunciation rules are approximate outside common patterns."""
    def word(match: re.Match[str]) -> str:
        syllables = match.group()
        output = []
        for index, syllable in enumerate(syllables):
            initial, vowel, final = _parts(syllable)
            ending = FINAL[final]
            if index + 1 < len(syllables):
                next_initial = _parts(syllables[index + 1])[0]
                if next_initial == 2 and final in (1, 2, 3):
                    ending = "ng"  # 국물 → gungmul
                elif next_initial == 2 and final in (7, 19, 20, 22, 23, 25):
                    ending = "n"
                elif next_initial == 2 and final in (11, 17, 18, 26):
                    ending = "m"  # 입니다 → imnida
            output.append(INITIAL[initial] + VOWEL[vowel] + ending)
        return "".join(output)

    return HANGUL.sub(word, text)
