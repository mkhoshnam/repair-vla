#!/usr/bin/env bash
# JFCRL-v2: held-out-blind screen-aware curriculum, A/B capability-gate test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# Set SCREEN to a raw screening log to regenerate a curriculum. When it is
# unset, the tracked held-out-blind curriculum is used as-is.
SCREEN="${SCREEN:-}"
VLA="${VLA:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
SEED="${SEED:-7}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
MODE="${1:-diag}"

cd "$REPO"
# shellcheck source=scripts/_environment.sh
source "$SCRIPT_DIR/_environment.sh"
repair_activate_environment
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
mkdir -p configs logs runs

case "$MODE" in
  diag)
    CFG=configs/curriculum_diag6.json
    # 40k transitions/task: close to the exposure where the successful 5-task
    # run already moved several 0%-baseline held-out cells.
    STEPS=${STEPS:-240000}
    N_EVAL_STATES=${N_EVAL_STATES:-10}  # preserve the existing 40/10 split
    EVAL_EPISODES=${EVAL_EPISODES:-10}
    EXTRA_SELECT=(--tasks
      libero_spatial:0 libero_goal:6 libero_object:6
      libero_goal:0 libero_10:3 libero_object:3)
    MIN_CELLS=1
    EVAL_CONDITIONS=(healthy 2)
    ;;
  main)
    CFG=configs/curriculum_main.json
    STEPS=${STEPS:-1400000}             # 50k transitions/task over 28 tasks
    N_EVAL_STATES=${N_EVAL_STATES:-20} # final runs: 30 train / 20 held out
    EVAL_EPISODES=${EVAL_EPISODES:-20}
    EXTRA_SELECT=()
    MIN_CELLS=1
    EVAL_CONDITIONS=(healthy 2)
    ;;
  main13)
    CFG=configs/curriculum_main13.json
    STEPS=${STEPS:-600000}
    N_EVAL_STATES=${N_EVAL_STATES:-10}
    EVAL_EPISODES=${EVAL_EPISODES:-10}
    EXTRA_SELECT=()
    MIN_CELLS=2
    EVAL_CONDITIONS=(healthy 2)
    ;;
  *) echo "usage: $0 {diag|main|main13}" >&2; exit 2 ;;
esac

if [[ -n "$SCREEN" ]]; then
  if [[ ! -f "$SCREEN" ]]; then
    echo "ABORT: SCREEN log not found: $SCREEN" >&2
    exit 2
  fi
  python rl/screen_to_curriculum.py \
    --log "$SCREEN" \
    --train_joints 0 4 5 6 \
    --heldout 2 \
    --min_train_cells "$MIN_CELLS" \
    "${EXTRA_SELECT[@]}" \
    --out "$CFG"
elif [[ ! -f "$CFG" ]]; then
  echo "ABORT: tracked curriculum not found: $CFG" >&2
  echo "Provide SCREEN=/path/to/screening.log to regenerate it." >&2
  exit 2
fi

echo
echo "=== $MODE: ${STEPS} steps/arm, n_eval_states=$N_EVAL_STATES ==="
python - "$CFG" "$STEPS" <<'PY'
import json,sys
cfg,steps=sys.argv[1],int(sys.argv[2])
n=len(json.load(open(cfg))["tasks"])
print(f"=== {n} tasks; target {steps//n:,} transitions/task ===")
PY
read -r -p "continue? [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0

launch () {
  local ARM=$1 GPU=$2
  local RUN="runs/jfcrl_v2_${MODE}_${ARM}_seed${SEED}"
  echo "[launch] arm=$ARM physical_gpu=$GPU -> $RUN" >&2
  CUDA_VISIBLE_DEVICES="$GPU" python -u rl/train_joint_factorized_multitask_sac.py \
    --curriculum "$CFG" \
    --shared_vla_checkpoint "$VLA" \
    --arm "$ARM" \
    --sampler_mode generation \
    --total_steps "$STEPS" \
    --n_eval_states "$N_EVAL_STATES" \
    --seed "$SEED" \
    --device cuda:0 \
    --gamma 0.99 --n_step 3 --residual_scale 0.1 \
    --encoder_q_weight 0.05 \
    --logdir "$RUN" \
    --wandb \
    > "logs/jfcrl_v2_${MODE}_${ARM}_seed${SEED}.log" 2>&1 &
  LAST_PID=$!
}

launch none "$GPU_A"
PID_A=$LAST_PID
launch gate "$GPU_B"
PID_B=$LAST_PID
status=0
wait "$PID_A" || status=1
wait "$PID_B" || status=1
if [[ $status -ne 0 ]]; then
  echo "training failed; inspect logs/jfcrl_v2_${MODE}_*_seed${SEED}.log" >&2
  exit 1
fi

echo "--- training done; deterministic healthy + globally unseen j2 evaluation ---"
for ARM in none gate; do
  RUN="runs/jfcrl_v2_${MODE}_${ARM}_seed${SEED}"
  CUDA_VISIBLE_DEVICES="$GPU_A" python -u rl/eval_joint_factorized_multitask.py \
    --ckpt "$RUN/ckpt_${STEPS}.pt" \
    --curriculum "$CFG" \
    --policies zero ckpt \
    --conditions "${EVAL_CONDITIONS[@]}" \
    --n_episodes "$EVAL_EPISODES" \
    --n_eval_states "$N_EVAL_STATES" \
    --seed "$SEED" \
    --device cuda:0 \
    --held_out \
    --outdir "$RUN/heldout_j2" \
    | tee "logs/eval_jfcrl_v2_${MODE}_${ARM}_seed${SEED}.txt"
done

cat <<'TXT'

Decision rule for the 6-task diagnostic (declared before looking):
  B is useful only if it reduces/avoids the unseen-j2 regressions on
  goal:0, libero_10:3, and object:3 while preserving most of A's unseen-j2
  gains on spatial:0, goal:6, and object:6.

Read the gate per TASK x CONDITION, not only per condition. A useful gate can
be low on an impaired joint when the base VLA already copes, and high when
compensation is actually needed.
TXT
