#!/usr/bin/env bash
# Submit from the project directory on the login node after activating your environment:
# sbatch -p <your_batch_partition> scripts/train.slurm.sh /data/$USER/datasets/malmaek.tar.gz
#SBATCH --job-name=malmaek-qlora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'usage: sbatch -p <partition> scripts/train.slurm.sh /data/.../dataset.tar.gz' >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'this script must run as a Slurm job' >&2
  exit 2
fi
repository=${SLURM_SUBMIT_DIR:?}
local_root="/local_datasets/${USER}/malmaek-${SLURM_JOB_ID}"
archive=$1

bash "$repository/scripts/stage_dataset.sh" "$archive" "$local_root/data"
export HF_HOME="$local_root/hf-cache"
mkdir -p "$HF_HOME"
python "$repository/validate_data.py" --train "$local_root/data/train.jsonl" --eval "$local_root/data/eval.jsonl"
python "$repository/train_qlora.py" --train "$local_root/data/train.jsonl" --output "$local_root/adapter"
python "$repository/compare.py" --eval "$local_root/data/eval.jsonl" --adapter "$local_root/adapter" --output "$local_root/comparison.jsonl"

# Return only compact artifacts to NAS. Keep large downloads and training files local.
tar -czf "$local_root/adapter.tar.gz" -C "$local_root" adapter
mkdir -p "$repository/outputs"
cp "$local_root/adapter.tar.gz" "$repository/outputs/adapter-${SLURM_JOB_ID}.tar.gz"
cp "$local_root/comparison.jsonl" "$repository/outputs/comparison-${SLURM_JOB_ID}.jsonl"
echo "adapter: $repository/outputs/adapter-${SLURM_JOB_ID}.tar.gz"
echo "comparison: $repository/outputs/comparison-${SLURM_JOB_ID}.jsonl"
