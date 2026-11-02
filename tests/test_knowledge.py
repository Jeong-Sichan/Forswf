import unittest
import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

from knowledge import BM25Index, chunk_document, evidence_for_prompt, load_chunks, source_footer
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
        self.assertIn("Do not invent a citation", prompt[1]["content"])

    def test_search_cannot_read_nas_data_directly(self):
        with self.assertRaises(ValueError):
            load_chunks("/data/example/knowledge/chunks.jsonl")

    def test_collector_keeps_license_revision_and_source_url(self):
        def fake_api(_language, params):
            if params.get("meta") == "siteinfo":
                return {"query": {"rightsinfo": {"text": "CC BY-SA", "url": "https://example.org/license"}}}
            self.assertEqual(params["exlimit"], 10)
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


if __name__ == "__main__":
    unittest.main()
