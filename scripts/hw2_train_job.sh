#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

require_env MODE
require_env DATASET
require_env CONFIG
require_env EXP_NAME
require_env REPO_ROOT
require_env ENV_NAME

case "${DATASET}" in
  modelnet40)
    DATA_PATH="${REPO_ROOT}/data/modelnet40_normal_resampled"
    ;;
  scanobjectnn)
    DATA_PATH="${REPO_ROOT}/data/scanobjectnn_eval"
    ;;
  *)
    echo "Unsupported DATASET: ${DATASET}" >&2
    exit 2
    ;;
esac

if [ ! -d "${REPO_ROOT}" ]; then
  echo "Missing repo root: ${REPO_ROOT}" >&2
  exit 2
fi

if [ ! -d "${DATA_PATH}" ]; then
  echo "Missing dataset directory: ${DATA_PATH}" >&2
  exit 2
fi

# `conda activate` sources shell snippets that read unset vars in this env.
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . "${HOME}/miniconda/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda initialization script not found" >&2
  exit 2
fi

conda activate "${ENV_NAME}"
set -u

cd "${REPO_ROOT}"
mkdir -p "exp/${DATASET}/${EXP_NAME}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

OPTIONS=("save_path=exp/${DATASET}/${EXP_NAME}")

if [ "${MODE}" = "smoke" ]; then
  OPTIONS+=(
    "epoch=1"
    "eval_epoch=1"
    "batch_size=8"
    "batch_size_val=8"
  )
elif [ "${MODE}" != "full" ]; then
  echo "Unsupported MODE: ${MODE}" >&2
  exit 2
fi

echo "HOSTNAME=$(hostname)"
echo "MODE=${MODE}"
echo "DATASET=${DATASET}"
echo "CONFIG=${CONFIG}"
echo "EXP_NAME=${EXP_NAME}"
echo "ENV_NAME=${ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python tools/train.py \
  --config-file "configs/${DATASET}/${CONFIG}.py" \
  --num-gpus 1 \
  --options "${OPTIONS[@]}"
