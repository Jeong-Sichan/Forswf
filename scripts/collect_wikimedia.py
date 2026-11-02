"""Collect attributed Wikimedia extracts. Run on a compute node or personal PC."""

import argparse
import json
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


USER_AGENT = "MalmaekResearchPrototype/0.1 (student project; https://github.com/Jeong-Sichan/Forswf)"


def api(language: str, params: dict) -> dict:
    endpoint = f"https://{language}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"format": "json", "formatversion": 2, **params})
    request = urllib.request.Request(endpoint, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def category_titles(language: str, names: list[str], per_category: int) -> list[str]:
    if per_category < 1:
        raise ValueError("max-per-category must be positive")
    titles = []
    for name in names:
        collected = []
        continuation = None
        while len(collected) < per_category:
            params = {
                "action": "query", "list": "categorymembers", "cmtitle": name,
                "cmtype": "page", "cmnamespace": 0,
                "cmlimit": min(500, per_category - len(collected)),
            }
            if continuation:
                params["cmcontinue"] = continuation
            result = api(language, params)
            members = [item["title"] for item in result["query"]["categorymembers"]]
            collected.extend(members)
            continuation = result.get("continue", {}).get("cmcontinue")
            if not continuation or not members:
                break
            time.sleep(0.2)
        print(f"{name}: {len(collected)} article titles")
        titles.extend(collected)
    return titles


def collect(language: str, titles: list[str], output: Path) -> None:
    rights = api(language, {"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"})["query"]["rightsinfo"]
    if not rights.get("url") or not rights.get("text"):
        raise ValueError("site license was not returned")
    license_name = rights["text"] + " (" + rights["url"] + ")"
    records = []
    for offset in range(0, len(titles), 10):
        params = {
            "action": "query", "prop": "extracts|info|revisions", "explaintext": 1,
            "exlimit": 10, "inprop": "url", "rvprop": "timestamp|ids",
            "titles": "|".join(titles[offset:offset + 10]),
        }
        result = api(language, params)
        for page in result["query"]["pages"]:
            if not page.get("missing") and not page.get("extract", "").strip():
                # A single-page retry also handles servers that cap multi-page extracts.
                single = api(language, dict(params, titles=page["title"], exlimit=1))
                page = single["query"]["pages"][0]
            if page.get("missing") or not page.get("extract", "").strip() or not page.get("fullurl"):
                print("skipped:", page.get("title"))
                continue
            revision = page.get("revisions", [{}])[0]
            records.append({
                "source_id": f"wikipedia:{language}:{page['pageid']}:{revision.get('revid', '')}",
                "title": page["title"], "url": page["fullurl"], "license": license_name,
                "language": language, "updated_at": revision.get("timestamp", ""),
                "text": page["extract"],
            })
        time.sleep(0.2)
    output.mkdir(parents=True, exist_ok=True)
    document_file = output / "documents.jsonl"
    with document_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with tarfile.open(output / "knowledge.tar.gz", "w:gz") as archive:
        archive.add(document_file, arcname="documents.jsonl")
    print(f"collected {len(records)} attributed documents; archive: {output / 'knowledge.tar.gz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--titles", type=Path, required=True, help="UTF-8 text, one page title per line")
    parser.add_argument("--categories", type=Path, help="Optional category names, one per line")
    parser.add_argument("--max-per-category", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    titles = [line.strip() for line in args.titles.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if args.categories:
        names = [line.strip() for line in args.categories.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
        titles.extend(category_titles(args.language, names, args.max_per_category))
    if not titles:
        raise ValueError("No titles supplied")
    collect(args.language, list(dict.fromkeys(titles)), args.output)
