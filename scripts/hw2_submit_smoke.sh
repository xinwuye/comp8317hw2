#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/gridview/slurm/bin:${PATH}"

TARGET="${1:-all}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="${REPO_ROOT}/scripts/hw2_train_job.sh"
ENV_NAME="${ENV_NAME:-pointcept-torch2.5.0-cu12.4}"
LOG_DIR="${LOG_DIR:-/public/home/xinwuye/slurm}"
PARTITION="${PARTITION:-gpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
GPU_REQUEST="${GPU_REQUEST:-gpu:1}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found on PATH" >&2
  exit 2
fi

if [ ! -x "${WORKER}" ]; then
  echo "Worker script is not executable: ${WORKER}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

submit_job() {
  local dataset="$1"
  local config="$2"
  local name="$3"

  local expected_path
  case "${dataset}" in
    modelnet40)
      expected_path="${REPO_ROOT}/data/modelnet40_normal_resampled"
      ;;
    scanobjectnn)
      expected_path="${REPO_ROOT}/data/scanobjectnn_eval"
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      exit 2
      ;;
  esac

  if [ ! -d "${expected_path}" ]; then
    echo "Missing dataset directory: ${expected_path}" >&2
    exit 2
  fi

  sbatch --parsable \
    --partition="${PARTITION}" \
    --gres="${GPU_REQUEST}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    -J "${name}" \
    -o "${LOG_DIR}/%x-%j.out" \
    --export=ALL,MODE=smoke,DATASET="${dataset}",CONFIG="${config}",EXP_NAME="${name}",REPO_ROOT="${REPO_ROOT}",ENV_NAME="${ENV_NAME}" \
    "${WORKER}"
}

case "${TARGET}" in
  all)
    echo "modelnet40 baseline $(submit_job modelnet40 cls-ptv3-v1m1-1-medium-baseline hw2-mn-baseline-smoke)"
    echo "modelnet40 gahs $(submit_job modelnet40 cls-ptv3-v1m1-2-medium-gahs hw2-mn-gahs-smoke)"
    echo "scanobjectnn baseline $(submit_job scanobjectnn cls-ptv3-v1m1-0-medium-baseline hw2-so-baseline-smoke)"
    echo "scanobjectnn gahs $(submit_job scanobjectnn cls-ptv3-v1m1-1-medium-gahs hw2-so-gahs-smoke)"
    ;;
  modelnet40)
    echo "modelnet40 baseline $(submit_job modelnet40 cls-ptv3-v1m1-1-medium-baseline hw2-mn-baseline-smoke)"
    echo "modelnet40 gahs $(submit_job modelnet40 cls-ptv3-v1m1-2-medium-gahs hw2-mn-gahs-smoke)"
    ;;
  scanobjectnn)
    echo "scanobjectnn baseline $(submit_job scanobjectnn cls-ptv3-v1m1-0-medium-baseline hw2-so-baseline-smoke)"
    echo "scanobjectnn gahs $(submit_job scanobjectnn cls-ptv3-v1m1-1-medium-gahs hw2-so-gahs-smoke)"
    ;;
  *)
    echo "Usage: $0 [all|modelnet40|scanobjectnn]" >&2
    exit 2
    ;;
esac
