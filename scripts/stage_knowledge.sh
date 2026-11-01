#!/usr/bin/env bash
# Run only inside an allocated Seraph compute node.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'usage: stage_knowledge.sh /data/.../knowledge.tar.gz /local_datasets/.../knowledge' >&2
  exit 2
fi
archive=$1
destination=$2
case "$archive" in /data/*) ;; *) echo 'archive must be on /data' >&2; exit 2 ;; esac
case "$destination" in /local_datasets/*) ;; *) echo 'destination must be under /local_datasets' >&2; exit 2 ;; esac
if [[ ! -f "$archive" || -e "$destination" ]]; then
  echo 'archive missing or destination exists' >&2
  exit 2
fi
mkdir -p "$destination"
cp "$archive" "$destination/knowledge.tar.gz"
if [[ "$(tar -tzf "$destination/knowledge.tar.gz")" != documents.jsonl ]]; then
  echo 'archive must contain exactly documents.jsonl' >&2
  exit 2
fi
tar -xzf "$destination/knowledge.tar.gz" -C "$destination" --no-same-owner --no-same-permissions
test -f "$destination/documents.jsonl"
PYTHONPATH="$(dirname "$(dirname "$0")")" python "$(dirname "$0")/build_knowledge.py" --documents "$destination/documents.jsonl" --output "$destination/chunks.jsonl"
