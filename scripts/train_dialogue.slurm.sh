#!/usr/bin/env bash
# Submit from the project directory:
# sbatch -p <batch_partition> scripts/train_dialogue.slurm.sh /data/$USER/datasets/aihub-dialogue-sft.tar.gz
#SBATCH --job-name=malmaek-dialogue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dialogue-%j.out
set -euo pipefail

if [[ $# -ne 1 || -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'usage: sbatch -p <partition> scripts/train_dialogue.slurm.sh /data/.../aihub-dialogue-sft.tar.gz' >&2
  exit 2
fi
repository=${SLURM_SUBMIT_DIR:?}
local_data="/local_datasets/${USER}/malmaek-dialogue-${SLURM_JOB_ID}/data"
model_root="/data/${USER}/models/malmaek"
revision=${KORMO_REVISION:?Export the pinned KoRMo commit SHA before sbatch}
source "/data/${USER}/envs/malmaek/bin/activate"
cd "$repository"

bash "$repository/scripts/stage_dataset.sh" "$1" "$local_data"
export HF_HOME="$model_root/hf-cache"
python "$repository/train_qlora.py" --format dialogue \
  --train "$local_data/train.jsonl" --eval "$local_data/eval.jsonl" \
  --output "$model_root/dialogue-adapter-${SLURM_JOB_ID}" \
  --revision "$revision" --epochs 1
echo "dialogue adapter: $model_root/dialogue-adapter-${SLURM_JOB_ID}"
