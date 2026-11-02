import unittest

from scripts.collect_easylaw_students import discover_pages, guide_url, parse_page, selected_pages


class EasyLawGuideTests(unittest.TestCase):
    def test_discovers_only_the_selected_guide(self):
        html = (
            '<a href="/CSP/CnpClsMain.laf?csmSeq=629&ccfNo=1&cciNo=1&cnpClsNo=1">rent</a>'
            '<a href="/CSP/CnpClsMain.laf?csmSeq=2853&ccfNo=1&cciNo=1&cnpClsNo=1">student</a>'
        )
        self.assertEqual(discover_pages(html, "629"), [("1", "1", "1")])
        self.assertIn("csmSeq=629", guide_url(("1", "1", "1"), "629"))

    def test_housing_page_keeps_its_own_title_and_source_id(self):
        html = (
            "<div class='cnpClsTitle'>임대차계약 확인</div>"
            "<div id='ovDiv'>임대차계약서를 확인하세요. " + "가" * 100 + "</div>"
            "<!---S:: 기준일 안내 영역--><p>2026년 10월 15일 기준</p>"
        )
        page = parse_page(html, ("1", "1", "1"), "629")
        self.assertEqual(page["source_id"], "easylaw:629:1.1.1")
        self.assertTrue(page["title"].startswith("주택임대차 · "))
        self.assertEqual(page["updated_at"], "2026-10-15")

    def test_commute_guide_excludes_car_and_workplace_sections(self):
        pages = [("1", "2", "1"), ("1", "3", "2"), ("1", "4", "1"), ("2", "2", "1")]
        self.assertEqual(selected_pages("661", pages), pages[:2])
        self.assertEqual(selected_pages("629", pages), pages)


if __name__ == "__main__":
    unittest.main()
