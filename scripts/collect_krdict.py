"""Collect Korean Basic Dictionary search *candidates*, not approved RAG documents.

The documented search response supplies a headword, example and source link.
Definitions and reuse terms require separate verification before indexing.
Run on an allocated compute node, with KRDICT_API_KEY in the environment.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = "https://krdict.korean.go.kr/api/search"


def parse_search_xml(payload: bytes, query: str, retrieved_at: str) -> tuple[int, list[dict]]:
    root = ET.fromstring(payload)
    channel = root if root.tag == "channel" else root.find("channel")
    if channel is None:
        raise ValueError(f"Unexpected KRDict XML root: {root.tag}")
    total = int(channel.findtext("total", "0"))
    records = []
    for item in channel.findall("item"):
        code = (item.findtext("target_code") or "").strip()
        word = (item.findtext("word") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not code or not word or not link.startswith("https://krdict.korean.go.kr/"):
            continue
        records.append({
            "source_id": f"krdict:{code}",
            "query": query,
            "target_code": code,
            "word": word,
            "sup_no": (item.findtext("sup_no") or "").strip(),
            "example": (item.findtext("example") or "").strip(),
            "url": link,
            "retrieved_at": retrieved_at,
            "reviewed": False,
        })
    return total, records


def request_search(key: str, query: str, start: int) -> bytes:
    url = ENDPOINT + "?" + urllib.parse.urlencode({"key": key, "q": query, "start": start})
    request = urllib.request.Request(url, headers={"User-Agent": "MalmaekResearchPrototype/0.1 (https://github.com/Jeong-Sichan/Forswf)"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        # Do not print the request URL: it contains the private API key.
        raise RuntimeError(f"KRDict HTTP {error.code} for query {query!r}") from None
    except urllib.error.URLError:
        raise RuntimeError(f"KRDict connection failed for query {query!r}") from None


def collect(queries: list[str], output: Path, key: str, max_results: int = 20) -> int:
    if not key or not 1 <= max_results <= 1000:
        raise ValueError("A key and max_results between 1 and 1000 are required")
    retrieved_at = datetime.now(timezone.utc).isoformat()
    records = {}
    for query in queries:
        start = 1
        while start <= max_results:
            total, page = parse_search_xml(request_search(key, query, start), query, retrieved_at)
            for record in page:
                records.setdefault(record["source_id"], record)
            if not page or start + len(page) > total or start + len(page) > max_results:
                break
            start += len(page)
            time.sleep(0.2)
        print(f"{query}: total={total}, fetched through result {min(start + len(page) - 1, max_results)}")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "search_candidates.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for record in records.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"collected {len(records)} distinct unreviewed search candidates: {target}")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True, help="UTF-8 headwords, one per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-results-per-query", type=int, default=20)
    args = parser.parse_args()
    key = os.environ.get("KRDICT_API_KEY", "")
    if not key:
        parser.error("Set KRDICT_API_KEY privately in the server shell; never put it on the command line")
    queries = list(dict.fromkeys(line.strip() for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")))
    if not queries:
        parser.error("No queries supplied")
    collect(queries, args.output, key, args.max_results_per_query)
