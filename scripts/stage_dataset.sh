#!/usr/bin/env bash
# Run inside an allocated Seraph compute node, never on the login node.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo 'usage: stage_dataset.sh /data/.../dataset.tar.gz /local_datasets/.../data' >&2
  exit 2
fi
archive=$1
destination=$2
case "$archive" in /data/*) ;; *) echo 'archive must be on /data' >&2; exit 2 ;; esac
case "$destination" in /local_datasets/*) ;; *) echo 'destination must be under /local_datasets' >&2; exit 2 ;; esac
if [[ ! -f "$archive" ]]; then
  echo "archive missing: $archive" >&2
  exit 2
fi
if [[ -e "$destination" ]]; then
  echo "destination already exists; choose a new run directory: $destination" >&2
  exit 2
fi

mkdir -p "$destination"
cp "$archive" "$destination/dataset.tar.gz"
tar -xzf "$destination/dataset.tar.gz" -C "$destination" --no-same-owner --no-same-permissions
test -f "$destination/train.jsonl"
test -f "$destination/eval.jsonl"
echo "staged dataset: $destination"
