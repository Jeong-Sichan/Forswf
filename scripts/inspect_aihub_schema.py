"""Print AI Hub JSON structure without exposing conversation or persona values."""

import argparse
import json
from pathlib import Path

from prompting import require_local_dataset


PRIVATE_BRANCHES = {"participantsInfo", "personaInfo"}


def describe(value, label: str, depth: int, max_depth: int, lines: list[str]) -> None:
    if depth > max_depth or len(lines) >= 120:
        return
    if isinstance(value, dict):
        lines.append(f"{label or '<root>'}: object ({len(value)} keys)")
        for key, child in value.items():
            path = f"{label}.{key}" if label else key
            if key in PRIVATE_BRANCHES:
                lines.append(f"{path}: object (contents omitted)")
            else:
                describe(child, path, depth + 1, max_depth, lines)
    elif isinstance(value, list):
        lines.append(f"{label}: list ({len(value)} items)")
        if value:
            describe(value[0], label + "[0]", depth + 1, max_depth, lines)
    else:
        lines.append(f"{label}: {type(value).__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True, help="One unpacked JSON file under /local_datasets")
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()
    require_local_dataset(str(args.file))
    with args.file.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    result = []
    describe(document, "", 0, args.max_depth, result)
    print("\n".join(result))
