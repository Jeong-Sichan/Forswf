"""Inspect retrieved passages before trusting them in the demo."""

import argparse

from knowledge import BM25Index, E5Index, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--retriever", choices=("e5", "bm25"), default="e5")
    args = parser.parse_args()
    chunks = load_chunks(args.knowledge)
    index = E5Index(chunks, cache_path=args.knowledge + ".e5.pt") if args.retriever == "e5" else BM25Index(chunks)
    for number, item in enumerate(index.search(args.query), 1):
        print(f"[{number}] score={item['score']} {item['title']}\n{item['url']}\n{item['text'][:750]}\n")


if __name__ == "__main__":
    main()
