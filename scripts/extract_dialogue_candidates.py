"""Sample contextual dialogue candidates from an approved local corpus export.

This creates review candidates, not QLoRA answers. It preserves conversation
boundaries and anonymizes speaker labels. Run only on a compute node with the
source corpus unpacked under /local_datasets.
"""

import argparse
import json
import random
import re
from pathlib import Path

from prompting import require_local_dataset


PRIVATE_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:01[016789])[ -]?\d{3,4}[ -]?\d{4}")


def conversations(value):
    if isinstance(value, dict):
        utterances = value.get("utterance")
        if isinstance(utterances, list) and utterances and all(isinstance(item, dict) for item in utterances):
            yield utterances
            return
        for child in value.values():
            yield from conversations(child)
    elif isinstance(value, list):
        for child in value:
            yield from conversations(child)


def windows(utterances: list[dict], size: int, stride: int):
    clean = []
    speaker_names = {}
    for item in utterances:
        form = item.get("form")
        if not isinstance(form, str):
            clean.append(None)
            continue
        form = " ".join(form.split())
        if not 2 <= len(form) <= 500 or PRIVATE_PATTERN.search(form):
            clean.append(None)
            continue
        speaker = str(item.get("speaker_id", "unknown"))
        speaker_names.setdefault(speaker, f"S{len(speaker_names) + 1}")
        clean.append({"speaker": speaker_names[speaker], "text": form, "utterance_id": str(item.get("id", ""))})
    for start in range(0, max(0, len(clean) - size + 1), stride):
        window = clean[start:start + size]
        if any(turn is None for turn in window):
            continue
        if len({turn["speaker"] for turn in window}) >= 2:
            yield window


def extract(input_dir: Path, output: Path, max_cases: int, size: int, stride: int) -> tuple[int, int, int]:
    require_local_dataset(str(input_dir))
    require_local_dataset(str(output))
    if not 2 <= size <= 8 or stride < 1 or max_cases < 1:
        raise ValueError("window must be 2-8, stride and max-cases must be positive")
    files = sorted(input_dir.rglob("*.json"))
    if not files:
        raise ValueError("No JSON files found under the local corpus directory")
    sample = []
    rng = random.Random(2604)
    found_conversations = 0
    found_windows = 0
    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            document = json.load(handle)
        for conversation in conversations(document):
            found_conversations += 1
            for window in windows(conversation, size, stride):
                found_windows += 1
                record = {
                    "source_file": str(path.relative_to(input_dir)),
                    "conversation_number": found_conversations,
                    "turns": window,
                    "reviewed": False,
                    "answer": None,
                }
                if len(sample) < max_cases:
                    sample.append(record)
                else:
                    replacement = rng.randrange(found_windows)
                    if replacement < max_cases:
                        sample[replacement] = record
    if not found_conversations or not found_windows:
        raise ValueError("No usable dialogue windows found; inspect the corpus JSON schema and speaker IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in sample:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(files), found_conversations, found_windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=3000)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    files, chats, candidates = extract(args.input_dir, args.output, args.max_cases, args.window, args.stride)
    print(f"read {files} JSON files, found {chats} conversations and {candidates} windows")
    print(f"sampled up to {args.max_cases} unreviewed candidates: {args.output}")
