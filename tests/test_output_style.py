import unittest

from output_style import english_issues, normalize_pronunciation, remove_inline_citations
from prompting import messages
from pronunciation import romanize


class OutputStyleTests(unittest.TestCase):
    def test_short_dialogue_adapter_answer_is_rejected(self):
        answer = "1. Plain meaning: 반찬은 셀프입니다."
        self.assertIn("missing or out-of-order English sections", english_issues(answer, "explain"))

    def test_korean_phrase_alone_is_not_a_plain_meaning(self):
        answer = (
            "1. Plain meaning\n반찬은 셀프입니다 (banchaneun selpeuimnida)\n"
            "2. Meaning in this situation\nThe sign concerns side dishes.\n"
            "3. What to say or do\nAsk the staff if needed.\n"
            "4. What could change the meaning\nThe restaurant setup may vary."
        )
        self.assertIn("section lacks English explanation", english_issues(answer, "explain"))

    def test_korean_prose_under_english_headings_is_rejected(self):
        answer = (
            "1. Plain meaning\n생각해 볼게요는 생각해 보겠습니다의 줄임말입니다.\n"
            "2. Meaning in this situation\n친구에게 말했으니 네라고 답해야 합니다.\n"
            "3. What to say or do\n친구에게 네라고 답하세요.\n"
            "4. What could change the meaning\n상황에 따라 달라집니다."
        )
        self.assertIn("Korean prose in English answer", english_issues(answer, "explain"))

    def test_complete_english_answer_with_pronunciation_is_accepted(self):
        answer = (
            "1. Plain meaning\n\"안녕하세요\" (annyeonghaseyo) is a polite hello.\n"
            "2. Meaning in this situation\nIt is suitable when meeting a professor.\n"
            "3. What to say or do\nSay it with a small nod.\n"
            "4. What could change the meaning\nA more formal event may call for a different greeting."
        )
        self.assertEqual(english_issues(answer, "explain"), [])

    def test_korean_sentence_is_rejected_even_with_english_context(self):
        answer = (
            "1. Plain meaning\n안녕하세요 (annyeonghaseyo) is a polite hello.\n"
            "2. Meaning in this situation\nThis greeting fits a first meeting. "
            "교수님께는 이렇게 인사하는 것이 좋습니다.\n"
            "3. What to say or do\nSay hello with a small nod.\n"
            "4. What could change the meaning\nThe formality of the occasion may matter."
        )
        self.assertIn("Korean prose in English answer", english_issues(answer, "explain"))

    def test_suggested_phrase_is_required(self):
        answer = (
            "1. Korean phrase\nSay hello politely.\n"
            "2. When to use it\nUse it when meeting a professor.\n"
            "3. What to say or do\nA small nod is fine.\n"
            "4. What may vary\nA formal event may call for a different greeting."
        )
        self.assertIn("missing Korean expression", english_issues(answer, "ask"))

    def test_english_prompt_requests_pronunciation_and_citation_free_prose(self):
        turns = messages("반찬은 셀프입니다", "Restaurant sign", "Customer", evidence=[])
        self.assertIn("Latin-letter pronunciation", turns[0]["content"])
        self.assertIn("Do not put bracket citation numbers", turns[0]["content"])

    def test_inline_citation_is_removed(self):
        self.assertEqual(remove_inline_citations("This is self-service. [1]"), "This is self-service.")

    def test_observed_bad_model_pronunciation_is_replaced(self):
        answer = (
            "1. Plain meaning\n반찬은 셀프입니다 (pan-jang-eun seol-peo-im-eo-da) means side dishes are self-service.\n"
            "2. Meaning in this situation\nYou serve yourself.\n"
            "3. What to say or do\nAsk the staff where the side dishes are.\n"
            "4. What could change the meaning\nThe restaurant may use a different setup."
        )
        corrected = normalize_pronunciation(answer, "explain", "반찬은 셀프입니다")
        self.assertIn("반찬은 셀프입니다 (banchaneun selpeuimnida)", corrected)
        self.assertNotIn("pan-jang-eun", corrected)
        self.assertEqual(english_issues(corrected, "explain"), [])

    def test_romanization_for_smoke_phrases(self):
        self.assertEqual(romanize("생각해 볼게요"), "saenggakhae bolgeyo")
        self.assertEqual(romanize("안녕하세요"), "annyeonghaseyo")

    def test_translation_after_quoted_phrase_is_kept(self):
        answer = '1. Plain meaning\n"생각해 볼게요" (I will think about it) means the speaker is deciding.'
        corrected = normalize_pronunciation(answer, "explain", "생각해 볼게요")
        self.assertIn('"생각해 볼게요" (saenggakhae bolgeyo) (I will think about it)', corrected)

    def test_question_phrase_receives_pronunciation(self):
        answer = (
            "1. Korean phrase\n안녕하세요 (wrong-reading) is a polite greeting.\n"
            "2. When to use it\nUse it when meeting someone older.\n"
            "3. What to say or do\nSay hello with a small nod.\n"
            "4. What may vary\nA more formal occasion may call for another phrase."
        )
        corrected = normalize_pronunciation(answer, "ask", "How do I greet an older adult?")
        self.assertIn("안녕하세요 (annyeonghaseyo)", corrected)
        self.assertEqual(english_issues(corrected, "ask"), [])


if __name__ == "__main__":
    unittest.main()
