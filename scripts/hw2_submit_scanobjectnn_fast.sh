#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/gridview/slurm/bin:${PATH}"

TARGET="${1:-both}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="${REPO_ROOT}/scripts/hw2_train_job.sh"
ENV_NAME="${ENV_NAME:-pointcept-torch2.5.0-cu12.4}"
LOG_DIR="${LOG_DIR:-/public/home/xinwuye/slurm}"
PARTITION="${PARTITION:-gpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
GPU_REQUEST="${GPU_REQUEST:-gpu:1}"
EXTRA_OPTIONS="${EXTRA_OPTIONS:-epoch=20 eval_epoch=20 batch_size=64 batch_size_val=64}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found on PATH" >&2
  exit 2
fi

if [ ! -x "${WORKER}" ]; then
  echo "Worker script is not executable: ${WORKER}" >&2
  exit 2
fi

if [ ! -d "${REPO_ROOT}/data/scanobjectnn_eval" ]; then
  echo "Missing dataset directory: ${REPO_ROOT}/data/scanobjectnn_eval" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

submit_job() {
  local config="$1"
  local name="$2"

  sbatch --parsable \
    --partition="${PARTITION}" \
    --gres="${GPU_REQUEST}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    -J "${name}" \
    -o "${LOG_DIR}/%x-%j.out" \
    --export=ALL,MODE=full,DATASET=scanobjectnn,CONFIG="${config}",EXP_NAME="${name}",REPO_ROOT="${REPO_ROOT}",ENV_NAME="${ENV_NAME}",EXTRA_OPTIONS="${EXTRA_OPTIONS}" \
    "${WORKER}"
}

case "${TARGET}" in
  both)
    echo "scanobjectnn baseline $(submit_job cls-ptv3-v1m1-0-medium-baseline hw2-so-baseline-fast)"
    echo "scanobjectnn gahs $(submit_job cls-ptv3-v1m1-1-medium-gahs hw2-so-gahs-fast)"
    ;;
  baseline)
    echo "scanobjectnn baseline $(submit_job cls-ptv3-v1m1-0-medium-baseline hw2-so-baseline-fast)"
    ;;
  gahs)
    echo "scanobjectnn gahs $(submit_job cls-ptv3-v1m1-1-medium-gahs hw2-so-gahs-fast)"
    ;;
  *)
    echo "Usage: $0 [both|baseline|gahs]" >&2
    exit 2
    ;;
esac
