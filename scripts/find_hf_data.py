"""List Hugging Face dataset metadata only; no dataset files are downloaded.

Run on an allocated Seraph compute node, not the login node.
"""

from huggingface_hub import HfApi


QUERIES = (
    "Korean idiom",
    "Korean conversation",
    "Korean dialogue",
    "Korean culture",
    "Korean etiquette",
    "Korean daily life",
    "Korean heritage",
    "Korean instruction",
    "한국어 관용구",
    "한국어 대화",
    "한국 문화",
    "생활 한국어",
)


def main() -> None:
    api = HfApi()
    found = set()
    for query in QUERIES:
        print(f"\n## {query}")
        try:
            candidates = api.list_datasets(search=query, limit=15)
            for info in candidates:
                if info.id in found:
                    continue
                found.add(info.id)
                tags = getattr(info, "tags", None) or []
                licenses = [tag.removeprefix("license:") for tag in tags if tag.startswith("license:")]
                print(f"{info.id}\tlicense={','.join(licenses) or 'unknown'}\tprivate={info.private}\tgated={getattr(info, 'gated', None)}")
        except Exception as exc:
            print(f"metadata search failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
