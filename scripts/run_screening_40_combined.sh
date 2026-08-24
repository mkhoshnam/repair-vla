#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/_environment.sh
source "$SCRIPT_DIR/_environment.sh"
repair_activate_environment

export CUDA_VISIBLE_DEVICES=${GPU:-1}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

N=${N:-5}
OUT="40_task_screening_combinedvla"

mkdir -p "$OUT"

SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

for SUITE in "${SUITES[@]}"; do
    echo
    echo "============================================================"
    echo "COMBINED-VLA SCREENING: $SUITE"
    echo "Checkpoint: moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    echo "Episodes/cell: $N"
    echo "Output: $OUT"
    echo "============================================================"

    python scripts/screen_faults_combined.py \
        --suite "$SUITE" \
        --n_episodes "$N" \
        --joints 0 1 2 3 4 5 6 \
        --out "$OUT"
done
