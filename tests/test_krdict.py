import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_krdict import collect, parse_search_xml


XML = '''<channel><total>2</total><start>1</start><num>2</num>
<item><target_code>27955</target_code><word>가공성</word><sup_no>0</sup_no>
<example>나무가 지나치게 단단하면 변형이 어려워 가공성이 떨어진다.</example>
<link>https://krdict.korean.go.kr/kor/dicSearch/SearchView?ParaWordNo=27955</link></item>
<item><target_code>28027</target_code><word>가공하다</word><sup_no>0</sup_no>
<example>이런 나무를 가공한다.</example>
<link>https://krdict.korean.go.kr/kor/dicSearch/SearchView?ParaWordNo=28027</link></item></channel>'''.encode('utf-8')


class KRDictTests(unittest.TestCase):
    def test_parse_search_example_with_source_links(self):
        total, rows = parse_search_xml(XML, '가공', '2026-11-02T00:00:00Z')
        self.assertEqual(total, 2)
        self.assertEqual(rows[0]['source_id'], 'krdict:27955')
        self.assertEqual(rows[0]['word'], '가공성')
        self.assertIn('나무가', rows[0]['example'])
        self.assertTrue(rows[0]['url'].startswith('https://krdict.korean.go.kr/'))
        self.assertFalse(rows[0]['reviewed'])

    def test_collect_writes_candidates_without_key(self):
        with tempfile.TemporaryDirectory() as directory, patch('scripts.collect_krdict.request_search', return_value=XML):
            collect(['가공'], Path(directory), 'secret-test-key', 20)
            content = (Path(directory) / 'search_candidates.jsonl').read_text()
            self.assertNotIn('secret-test-key', content)
            self.assertEqual(len([json.loads(line) for line in content.splitlines()]), 2)


if __name__ == '__main__':
    unittest.main()
