#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/eval_jfcrl_unseen_j2.sh <checkpoint.pt> [n_episodes]" >&2
  exit 2
fi

CKPT="$1"
N_EPISODES="${2:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/_environment.sh
source "$SCRIPT_DIR/_environment.sh"
repair_activate_environment
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ ! -f "$CKPT" ]]; then
  echo "ABORT: checkpoint not found: $CKPT" >&2
  exit 2
fi

STEM="$(basename "$CKPT" .pt)"
RUN_DIR="$(basename "$(dirname "$CKPT")")"
OUTDIR="rollouts_${RUN_DIR}_${STEM}_unseen_j2"
LOGFILE="eval_${RUN_DIR}_${STEM}_unseen_j2.log"

python rl/eval_joint_factorized_multitask.py \
  --ckpt "$CKPT" \
  --conditions 2 \
  --policies zero ckpt \
  --held_out \
  --n_episodes "$N_EPISODES" \
  --outdir "$OUTDIR" \
  |& tee "$LOGFILE"
