import unittest
import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

from knowledge import BM25Index, chunk_document, evidence_for_prompt, guide_source_matches, load_chunks, search_guide, source_footer
from prompting import messages
from scripts import collect_wikimedia


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.doc = {
            "source_id": "example:1", "title": "Korean cuisine", "url": "https://example.org/korean-cuisine",
            "license": "CC BY-SA 4.0", "text": "Korean cuisine includes many side dishes called banchan. " * 10,
        }

    def test_chunk_and_retrieve_with_attribution(self):
        chunks = chunk_document(self.doc, size=150, overlap=20)
        self.assertGreater(len(chunks), 1)
        found = BM25Index(chunks).search("Korean banchan", limit=1)
        self.assertEqual(found[0]["source_id"], "example:1")
        self.assertIn(self.doc["url"], source_footer(found))

    def test_prompt_includes_only_supplied_source(self):
        evidence = evidence_for_prompt(chunk_document(self.doc))
        prompt = messages("반찬", "at a restaurant", "customer", evidence=evidence)
        self.assertIn("[1] Korean cuisine", prompt[1]["content"])
        self.assertIn(self.doc["url"], prompt[1]["content"])

    def test_empty_evidence_requires_uncertainty(self):
        prompt = messages("반찬", "unknown", "unknown", evidence=[])
        self.assertIn("none found", prompt[1]["content"])
        self.assertIn("Do not invent source claims", prompt[1]["content"])

    def test_search_cannot_read_nas_data_directly(self):
        with self.assertRaises(ValueError):
            load_chunks("/data/example/knowledge/chunks.jsonl")

    def test_public_transit_guide_matches_question_topic(self):
        passage = {
            "source_id": "easylaw:661:1.2.1",
            "title": "김대리의 안전한 출퇴근길 · 부정승차 하지 않기 · 법제처",
            "text": "버스와 지하철에서 교통카드나 승차권으로 운임을 내야 합니다.",
        }
        self.assertTrue(guide_source_matches("버스 교통카드 없이 타도 되나요?", passage))
        self.assertFalse(guide_source_matches("전세 보증금은 어떻게 하나요?", passage))

    def test_english_lease_question_uses_contract_check_passage(self):
        passage = {
            "source_id": "easylaw:629:2.2.1", "title": "주택임대차 · 임대차계약의 당사자 확인 등 · 법제처",
            "url": "https://example.org/lease", "license": "attribution",
            "text": "계약 전 등기부상 소유자나 적법한 대리인인지 확인해야 합니다. " * 12,
        }
        student = {**passage, "source_id": "easylaw:2853:2.3.1", "title": "외국인유학생 · 외국인등록 및 변경신고 · 법제처",
                   "text": "유학생은 외국인등록과 체류지 변경 신고를 해야 합니다. " * 12}
        chunks = chunk_document(student) + chunk_document(passage)
        index = BM25Index(chunks)
        question = "What should I check before signing a housing lease in Korea?"
        self.assertFalse(guide_source_matches(question, chunks[0]))
        found = search_guide(index, question, "International student living in Korea")
        self.assertEqual(found[0]["source_id"], passage["source_id"])
        self.assertEqual(found[0]["chunk_id"], passage["source_id"] + ":0")

    def test_collector_keeps_license_revision_and_source_url(self):
        def fake_api(_language, params):
            if params.get("meta") == "siteinfo":
                return {"query": {"rightsinfo": {"text": "CC BY-SA", "url": "https://example.org/license"}}}
            self.assertEqual(params["exlimit"], 10)
            self.assertEqual(params["redirects"], 1)
            return {"query": {"pages": [{"pageid": 7, "title": "Korean culture", "fullurl": "https://example.org/page", "extract": self.doc["text"], "revisions": [{"revid": 12, "timestamp": "2026-01-01T00:00:00Z"}]}]}}

        def limited_api(_language, params):
            if params.get("meta") == "siteinfo":
                return {"query": {"rightsinfo": {"text": "CC BY-SA", "url": "https://example.org/license"}}}
            def page(number, extract):
                return {"pageid": number, "title": f"Page {number}", "fullurl": f"https://example.org/{number}", "extract": extract, "revisions": [{"revid": number}]}
            if params["titles"] == "Page 2":
                return {"query": {"pages": [page(2, self.doc["text"])]}}
            return {"query": {"pages": [page(1, self.doc["text"]), page(2, "")]}}

        with tempfile.TemporaryDirectory() as directory, patch.object(collect_wikimedia, "api", side_effect=fake_api):
            collect_wikimedia.collect("en", ["Korean culture"], Path(directory))
            record = json.loads((Path(directory) / "documents.jsonl").read_text())
            self.assertEqual(record["source_id"], "wikipedia:en:7:12")
            self.assertIn("https://example.org/license", record["license"])
            self.assertEqual(record["url"], "https://example.org/page")
            with tarfile.open(Path(directory) / "knowledge.tar.gz") as archive:
                self.assertEqual(archive.getnames(), ["documents.jsonl"])
        with tempfile.TemporaryDirectory() as directory, patch.object(collect_wikimedia, "api", side_effect=limited_api):
            collect_wikimedia.collect("en", ["Page 1", "Page 2"], Path(directory))
            self.assertEqual(len((Path(directory) / "documents.jsonl").read_text().splitlines()), 2)

    def test_category_collection_follows_continuation(self):
        def fake_category_api(_language, params):
            if "cmcontinue" not in params:
                return {"query": {"categorymembers": [{"title": "One"}, {"title": "Two"}]}, "continue": {"cmcontinue": "next"}}
            self.assertEqual(params["cmcontinue"], "next")
            return {"query": {"categorymembers": [{"title": "Three"}]}}

        with patch.object(collect_wikimedia, "api", side_effect=fake_category_api), patch.object(collect_wikimedia.time, "sleep"):
            self.assertEqual(collect_wikimedia.category_titles("en", ["Category:Test"], 3), ["One", "Two", "Three"])

    def test_redirected_titles_are_deduplicated(self):
        def fake_api(_language, params):
            if params.get("meta") == "siteinfo":
                return {"query": {"rightsinfo": {"text": "CC BY-SA", "url": "https://example.org/license"}}}
            self.assertEqual(params["redirects"], 1)
            return {"query": {"pages": [{"pageid": 17, "title": "Korean New Year", "fullurl": "https://example.org/17", "extract": self.doc["text"], "revisions": [{"revid": 18}]}]}}

        with tempfile.TemporaryDirectory() as directory, patch.object(collect_wikimedia, "api", side_effect=fake_api):
            collect_wikimedia.collect("en", ["Seollal", "Korean New Year"], Path(directory))
            self.assertEqual(len((Path(directory) / "documents.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
