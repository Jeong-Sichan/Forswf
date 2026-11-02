"""Collect selected official Easy Law guides for attributed RAG.

Run on a Seraph compute node. Pages citing another source are set aside for
reuse review because the site's policy excludes third-party copyrighted works.
"""

import argparse
import json
import re
import tarfile
import time
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


BASE = "https://www.easylaw.go.kr"
PATH = "/CSP/CnpClsMain.laf"
COPYRIGHT = "https://www.easylaw.go.kr/CSP/AboutCopyright.laf?topMenu=introUl3"
LICENSE = f"법제처 찾기쉬운 생활법령정보 저작권정책 ({COPYRIGHT})"
PARAMETERS = ("ccfNo", "cciNo", "cnpClsNo")
GUIDES = {
    "2853": "외국인유학생",
    "629": "주택임대차",
    "661": "김대리의 안전한 출퇴근길",
}


def selected_pages(guide_id: str, pages: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Keep the public-transit part of the commute guide, not its car/work sections."""
    if guide_id == "661":
        return [numbers for numbers in pages if numbers[0] == "1" and numbers[1] in {"1", "2", "3"}]
    return pages


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "label"}:
            self.ignored.append(tag)
        if not self.ignored and tag in {"div", "p", "li", "tr", "br", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.ignored and self.ignored[-1] == tag:
            self.ignored.pop()
        elif not self.ignored and tag in {"div", "p", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)

    def text(self):
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def extract_text(fragment: str) -> str:
    parser = VisibleText()
    parser.feed(fragment)
    return parser.text()


def guide_url(numbers: tuple[str, str, str], guide_id: str = "2853") -> str:
    params = {"csmSeq": guide_id, **dict(zip(PARAMETERS, numbers))}
    return BASE + PATH + "?" + urlencode(params)


def discover_pages(html: str, guide_id: str = "2853") -> list[tuple[str, str, str]]:
    found = set()
    for href in re.findall(r"href\s*=\s*['\"]([^'\"]+)['\"]", html, flags=re.I):
        parsed = urlparse(unescape(href))
        query = parse_qs(parsed.query)
        if parsed.path != PATH or query.get("csmSeq") != [guide_id]:
            continue
        if all(len(query.get(key, [])) == 1 and query[key][0].isdigit() for key in PARAMETERS):
            found.add(tuple(query[key][0] for key in PARAMETERS))
    return sorted(found, key=lambda numbers: tuple(map(int, numbers)))


def parse_page(html: str, numbers: tuple[str, str, str], guide_id: str = "2853") -> dict:
    title_match = re.search(r"<div\s+class=['\"]cnpClsTitle['\"]>(.*?)</div>", html, re.I | re.S)
    body_match = re.search(r"<div\s+id=['\"]ovDiv['\"][^>]*>", html, re.I)
    if not title_match or not body_match:
        raise ValueError(f"Missing article title or body: {numbers}")
    end = html.find("<!---S:: 기준일 안내 영역", body_match.end())
    if end < 0:
        raise ValueError(f"Missing end of article: {numbers}")
    date_text = extract_text(html[end:end + 1500])
    date_match = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*기준", date_text)
    if not date_match:
        raise ValueError(f"Missing reference date: {numbers}")
    text = extract_text(html[body_match.end():end])
    if len(text) < 80:
        raise ValueError(f"Article is too short: {numbers}")
    title = extract_text(title_match.group(1))
    return {
        "source_id": f"easylaw:{guide_id}:" + ".".join(numbers),
        "title": f"{GUIDES[guide_id]} · {title} · 법제처",
        "url": guide_url(numbers, guide_id),
        "license": LICENSE,
        "language": "ko",
        "publisher": "법제처",
        "updated_at": "%s-%02d-%02d" % (date_match.group(1), int(date_match.group(2)), int(date_match.group(3))),
        "retrieved_at": datetime.now(timezone.utc).date().isoformat(),
        "text": text,
    }


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Malmaek/1.0 (+https://github.com/Jeong-Sichan/Forswf)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code not in {429, 502, 503, 504} or attempt == 2:
                raise
        except URLError:
            if attempt == 2:
                raise
        time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def collect(output: Path, max_pages: int | None = None, guides: tuple[str, ...] = ("2853",)) -> tuple[int, int]:
    first = ("1", "1", "1")
    if not guides or len(set(guides)) != len(guides) or any(guide_id not in GUIDES for guide_id in guides):
        raise ValueError("Select distinct, supported Easy Law guide IDs")
    if max_pages is not None and max_pages < 1:
        raise ValueError("--max-pages must be positive")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "documents.jsonl"
    review_destination = output / "attribution_review.jsonl"
    kept = 0
    flagged = 0
    with destination.open("w", encoding="utf-8") as target, review_destination.open("w", encoding="utf-8") as review:
        for guide_id in guides:
            first_html = fetch(guide_url(first, guide_id))
            pages = discover_pages(first_html, guide_id)
            if first not in pages or len(pages) < 10:
                raise ValueError(f"Guide {guide_id} navigation changed: found {len(pages)} pages")
            pages = selected_pages(guide_id, pages)
            for index, numbers in enumerate(pages[:max_pages]):
                if index:
                    time.sleep(0.5)
                html = first_html if numbers == first else fetch(guide_url(numbers, guide_id))
                record = parse_page(html, numbers, guide_id)
                if re.search(r"출처\s*[:：]", record["text"]):
                    review.write(json.dumps(record, ensure_ascii=False) + "\n")
                    flagged += 1
                    print(f"review attribution: {record['source_id']} {record['title']}")
                else:
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                    print(f"indexed: {record['source_id']} {record['title']} ({len(record['text'])} chars)")
    if not kept:
        raise ValueError("No pages passed the third-party attribution screen")
    with tarfile.open(output / "knowledge.tar.gz", "w:gz") as archive:
        archive.add(destination, arcname="documents.jsonl")
    return kept, flagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, help="Fetch only the first N pages for a trial")
    parser.add_argument("--guides", nargs="+", choices=sorted(GUIDES), default=["2853"],
                        help="Easy Law guide IDs: 2853 students, 629 housing leases, 661 transport")
    args = parser.parse_args()
    kept, flagged = collect(args.output, args.max_pages, tuple(args.guides))
    print(f"collected {kept} attributed Easy Law pages; {flagged} need reuse review")
    print(f"RAG archive: {args.output / 'knowledge.tar.gz'}")
