"""Build grouped dialogue-continuation SFT data from local AI Hub JSON files."""

import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path

from prompting import require_local_dataset
from scripts.extract_dialogue_candidates import conversations, windows


def build(source: Path, output: Path, maximum: int) -> tuple[int, int, int]:
    require_local_dataset(str(source))
    require_local_dataset(str(output))
    if maximum < 1000:
        raise ValueError("At least 1000 dialogue examples are required for this run")
    paths = sorted(source.rglob("*.json"))
    if not paths:
        raise ValueError("No JSON files found. Unpack the corpus under /local_datasets first")
    chosen = []
    observed = 0
    rng = random.Random(2604)
    for path in paths:
        with path.open(encoding="utf-8-sig") as handle:
            document = json.load(handle)
        session = document.get("multisessionInfo", {}) if isinstance(document, dict) else {}
        session_id = session.get("multisessionID") if isinstance(session, dict) else None
        group = hashlib.sha256(str(session_id or path.relative_to(source)).encode()).hexdigest()[:20]
        for utterances in conversations(document):
            for block in windows(utterances, 4, 2):
                context, target = block[:-1], block[-1]
                if sum(len(turn["text"]) for turn in block) > 1000:
                    continue
                prompt = "Continue this Korean conversation with one natural line from the indicated speaker.\n" \
                         + "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in context) \
                         + f"\nNext speaker: {target['speaker']}"
                example = {"prompt": prompt, "answer": target["text"],
                           "source_group": group, "training_task": "dialogue_continuation"}
                observed += 1
                if len(chosen) < maximum:
                    chosen.append(example)
                else:
                    slot = rng.randrange(observed)
                    if slot < maximum:
                        chosen[slot] = example
    if observed < 1000:
        raise ValueError(f"Only {observed} usable dialogue examples found; check the JSON schema")
    output.mkdir(parents=True, exist_ok=True)
    group_sizes = {}
    for row in chosen:
        group_sizes[row["source_group"]] = group_sizes.get(row["source_group"], 0) + 1
    groups = sorted(group_sizes)
    rng.shuffle(groups)
    eval_groups = set()
    eval_size = 0
    for group in groups:
        if eval_size >= max(1, round(len(chosen) * 0.1)):
            break
        eval_groups.add(group)
        eval_size += group_sizes[group]
    counts = {"train": 0, "eval": 0}
    with (output / "train.jsonl").open("w", encoding="utf-8") as train, (output / "eval.jsonl").open("w", encoding="utf-8") as evaluation:
        for row in chosen:
            split = "eval" if row["source_group"] in eval_groups else "train"
            (evaluation if split == "eval" else train).write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    if min(counts.values()) == 0:
        raise ValueError("Grouped split produced an empty train or eval set")
    with tarfile.open(output / "aihub-dialogue-sft.tar.gz", "w:gz") as archive:
        for name in ("train.jsonl", "eval.jsonl"):
            archive.add(output / name, arcname=name)
    return observed, counts["train"], counts["eval"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Unpacked AI Hub JSON directory under /local_datasets")
    parser.add_argument("--output", type=Path, required=True, help="Output directory under /local_datasets")
    parser.add_argument("--max-cases", type=int, default=5000)
    args = parser.parse_args()
    observed, train, evaluation = build(args.source, args.output, args.max_cases)
    print(f"found {observed} usable windows; sampled {train} train and {evaluation} grouped eval rows")
    print(f"archive: {args.output / 'aihub-dialogue-sft.tar.gz'}")
