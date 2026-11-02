import unittest

from output_style import english_issues, remove_inline_citations
from prompting import messages


class OutputStyleTests(unittest.TestCase):
    def test_short_dialogue_adapter_answer_is_rejected(self):
        answer = "1. Plain meaning: 반찬은 셀프입니다."
        self.assertIn("missing or out-of-order English sections", english_issues(answer, "explain"))

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


if __name__ == "__main__":
    unittest.main()
