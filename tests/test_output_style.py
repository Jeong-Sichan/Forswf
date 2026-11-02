import unittest

from output_style import english_issues, korean_issues, normalize_pronunciation, remove_inline_citations
from prompting import english_rewrite_messages, messages, translate_korean_answer_messages
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
        self.assertIn("do not reverse who said the phrase", turns[0]["content"])

    def test_failed_korean_greeting_can_be_rewritten_in_english(self):
        original = messages("How do I greet a professor?", "First meeting", "Student and professor", task="ask")
        rewrite = english_rewrite_messages(original, "1. 한국어 표현: 안녕하세요", "ask")
        self.assertIn("1. Korean phrase", rewrite[0]["content"])
        self.assertIn("4. What may vary", rewrite[0]["content"])
        self.assertIn("안녕하세요", rewrite[1]["content"])
        self.assertIn("How do I greet a professor?", rewrite[1]["content"])

    def test_korean_lora_draft_can_be_translated_without_new_claims(self):
        original = messages("생각해 볼게요", "A friend replies to an invitation", "Friends")
        draft = (
            "1. 쉬운 뜻\n생각한 뒤 답하겠다는 말입니다.\n"
            "2. 상황에 따른 해석\n아직 결정하지 않았습니다.\n"
            "3. 적절한 말이나 행동\n시간을 주고 기다립니다.\n"
            "4. 달라질 수 있는 점\n이 말만으로 수락 여부를 알 수 없습니다."
        )
        self.assertEqual(korean_issues(draft, "explain"), [])
        translated = translate_korean_answer_messages(original, draft, "explain")
        self.assertIn("Do not add facts", translated[0]["content"])
        self.assertIn("1. Plain meaning", translated[0]["content"])
        self.assertIn(draft, translated[1]["content"])

    def test_short_korean_lora_draft_is_rejected(self):
        self.assertIn("Korean draft lacks required sections", korean_issues("안녕하세요라고 인사하세요.", "ask"))

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

    def test_observed_greeting_answer_is_not_discarded(self):
        answer = (
            "1. 안녕하세요 (An-nyeong-ha-se-yo)\n"
            "2. When to use it: Use this phrase when meeting someone for the first time.\n"
            "3. What to say or do: Smile and say ‘안녕하세요’ with a small nod.\n"
            "4. What may vary: The level of formality depends on the context."
        )
        corrected = normalize_pronunciation(answer, "ask", "How should I greet an older adult?")
        self.assertTrue(corrected.startswith("1. Korean phrase\n안녕하세요 (annyeonghaseyo)"))
        self.assertEqual(english_issues(corrected, "ask"), [])

    def test_duplicate_model_reading_is_removed(self):
        answer = (
            "1. Korean phrase\n안녕하세요 (annyeonghaseyo) (An-nyeong-ha-se-yo)\n"
            "2. When to use it\nUse it at a first meeting.\n"
            "3. What to say or do\nSay hello politely.\n"
            "4. What may vary\nA formal event may be different."
        )
        corrected = normalize_pronunciation(answer, "ask", "How should I greet someone?")
        self.assertNotIn("An-nyeong-ha-se-yo", corrected)
        self.assertEqual(english_issues(corrected, "ask"), [])


if __name__ == "__main__":
    unittest.main()
