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

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'usage: sbatch -p <partition> scripts/train.slurm.sh /data/.../dataset.tar.gz [/data/.../knowledge.tar.gz]' >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'this script must run as a Slurm job' >&2
  exit 2
fi
repository=${SLURM_SUBMIT_DIR:?}
local_root="/local_datasets/${USER}/malmaek-${SLURM_JOB_ID}"
archive=$1
revision=${KORMO_REVISION:?Set KORMO_REVISION to the Hugging Face model commit SHA before sbatch}
model_root="/data/${USER}/models/malmaek"
adapter_dir="$model_root/adapter-${SLURM_JOB_ID}"

bash "$repository/scripts/stage_dataset.sh" "$archive" "$local_root/data"
export HF_HOME="$model_root/hf-cache"
mkdir -p "$HF_HOME" "$repository/outputs"
validate_args=(--train "$local_root/data/train.jsonl" --eval "$local_root/data/eval.jsonl")
if [[ "${MALMAEK_REQUIRE_EVIDENCE:-0}" == 1 ]]; then
  validate_args+=(--grounded)
fi
python "$repository/validate_data.py" "${validate_args[@]}"
train_args=(--train "$local_root/data/train.jsonl" --output "$adapter_dir" --revision "$revision")
if [[ "${MALMAEK_REQUIRE_EVIDENCE:-0}" == 1 ]]; then
  train_args+=(--require-evidence)
fi
compare_args=(--eval "$local_root/data/eval.jsonl" --adapter "$adapter_dir" --output "$repository/outputs/comparison-${SLURM_JOB_ID}.jsonl" --revision "$revision")
if [[ $# -eq 2 ]]; then
  bash "$repository/scripts/stage_knowledge.sh" "$2" "$local_root/knowledge"
  compare_args+=(--knowledge "$local_root/knowledge/chunks.jsonl")
fi
python "$repository/train_qlora.py" "${train_args[@]}"
python "$repository/compare.py" "${compare_args[@]}"

# Training data stays local to the compute node; model files stay in /data.
echo "adapter: $adapter_dir"
echo "comparison: $repository/outputs/comparison-${SLURM_JOB_ID}.jsonl"
