import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from prompting import culture_messages
from scripts.ingest_culture_symbols import ingest


class CultureSymbolsTests(unittest.TestCase):
    def test_cp949_csv_keeps_culture_explanation_and_license(self):
        csv_text = (
            "연번,문양형태 대분류,문양형태 세분류,원천유물명,유물과 문양에 대한 설명,문양의 상징적 의미,재질,사이트주소\n"
            "1,동물문,범문,석호,호랑이 모양을 돌에 새겼다,수호의 의미를 담는다,돌,https://www.kculture.or.kr/example\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "symbols.csv"
            source.write_bytes(csv_text.encode("cp949"))
            output = Path(directory) / "out"
            self.assertEqual(ingest(source, output), 1)
            record = json.loads((output / "documents.jsonl").read_text())
            self.assertEqual(record["source_id"], "publicdata:15143561:1")
            self.assertIn("수호의 의미", record["text"])
            self.assertIn("제1유형", record["license"])
            with tarfile.open(output / "knowledge.tar.gz") as archive:
                self.assertEqual(archive.getnames(), ["documents.jsonl"])
            prompt = culture_messages("What does the tiger motif symbolize?", "At a museum", "en", [record])
            self.assertIn("cultural heritage", prompt[0]["content"])
            self.assertIn("수호의 의미", prompt[1]["content"])


if __name__ == "__main__":
    unittest.main()
