#!/usr/bin/env bash
set -euo pipefail

# Three-suite launcher for the unseen-j0 JFCRL experiment.
#
# Examples:
#   bash scripts/run_jfcrl_unseen_j0_3suite.sh list
#   bash scripts/run_jfcrl_unseen_j0_3suite.sh 3 --print-only
#   bash scripts/run_jfcrl_unseen_j0_3suite.sh all --dry-run
#   bash scripts/run_jfcrl_unseen_j0_3suite.sh all
#
# A number N means: train on the first N tasks in
# configs/jfcrl_unseen_j0_3suite.tsv.  "all" means all screened tasks.
# The tracked tasks span three LIBERO suites. j0 is globally excluded from
# training; the policy trains on j4, j6, and healthy episodes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_FILE="$REPO_ROOT/configs/jfcrl_unseen_j0_3suite.tsv"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_jfcrl_unseen_j0_3suite.sh <N|all|list> [--dry-run|--print-only]

Examples:
  bash scripts/run_jfcrl_unseen_j0_3suite.sh list
  bash scripts/run_jfcrl_unseen_j0_3suite.sh 3 --print-only
  bash scripts/run_jfcrl_unseen_j0_3suite.sh all --dry-run
  bash scripts/run_jfcrl_unseen_j0_3suite.sh all

Optional environment overrides:
  GPU=1 SEED=7 STEPS_PER_TASK=50000 bash scripts/run_jfcrl_unseen_j0_3suite.sh all
  RUNNAME=my_run bash scripts/run_jfcrl_unseen_j0_3suite.sh all
  VENV=/path/to/venv bash scripts/run_jfcrl_unseen_j0_3suite.sh all
USAGE
}

if [[ ! -f "$TASK_FILE" ]]; then
  echo "ABORT: missing task pool: $TASK_FILE" >&2
  exit 2
fi

SELECTOR="${1:-}"
MODE="${2:-}"
if [[ -z "$SELECTOR" ]]; then
  usage
  exit 2
fi
if [[ -n "$MODE" && "$MODE" != "--dry-run" && "$MODE" != "--print-only" ]]; then
  echo "ABORT: unknown second argument: $MODE" >&2
  usage
  exit 2
fi
if [[ $# -gt 2 ]]; then
  echo "ABORT: too many arguments" >&2
  usage
  exit 2
fi

# Load the ordered screened task pool.  Only task_key is used for training;
# screening success rates remain documentation and never enter the learner.
mapfile -t ALL_TASKS < <(awk -F'\t' 'NR>1 && $2!="" {print $2}' "$TASK_FILE")
TOTAL_AVAILABLE=${#ALL_TASKS[@]}

print_pool() {
  printf '%-4s %-20s %-8s %-8s %-8s %-8s %s\n' "#" "task" "healthy" "j0" "j6" "j2" "description"
  printf '%s\n' "$(printf '=%.0s' {1..120})"
  awk -F'\t' 'NR>1 {printf "%-4s %-20s %-8s %-8s %-8s %-8s %s\n", $1,$2,$4,$5,$6,$7,$3}' "$TASK_FILE"
}

if [[ "$SELECTOR" == "list" ]]; then
  print_pool
  exit 0
fi

if [[ "$SELECTOR" == "all" ]]; then
  N_TASKS=$TOTAL_AVAILABLE
elif [[ "$SELECTOR" =~ ^[0-9]+$ ]]; then
  N_TASKS=$SELECTOR
else
  echo "ABORT: selector must be an integer, 'all', or 'list'; got '$SELECTOR'" >&2
  usage
  exit 2
fi

if (( N_TASKS < 1 || N_TASKS > TOTAL_AVAILABLE )); then
  echo "ABORT: choose 1..$TOTAL_AVAILABLE or 'all'; got $N_TASKS" >&2
  exit 2
fi

TASKS=("${ALL_TASKS[@]:0:N_TASKS}")

# Equal target ENV-STEP exposure.  This is transition-balanced rather than
# episode-balanced because LIBERO suites have different horizons.
P=$(python3 - <<PY
n=$N_TASKS
print(f"{1.0/n:.12f}")
PY
)
TASK_PROBS=()
for ((i=0; i<N_TASKS; i++)); do TASK_PROBS+=("$P"); done

GPU=${GPU:-1}
SEED=${SEED:-7}
STEPS_PER_TASK=${STEPS_PER_TASK:-50000}
STEPS=${STEPS:-$((STEPS_PER_TASK * N_TASKS))}
SHARED_VLA_CHECKPOINT=${SHARED_VLA_CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}
RUNNAME=${RUNNAME:-jfcrl_unseen_j0_3suite_seed${SEED}}
RUNDIR=${RUNDIR:-runs/$RUNNAME}
LOGFILE=${LOGFILE:-${RUNNAME}.log}

print_selection() {
  echo "=============================================================="
  echo "JFCRL screened-task selection"
  echo "=============================================================="
  echo "selector:          $SELECTOR"
  echo "number of tasks:   $N_TASKS / $TOTAL_AVAILABLE"
  echo "shared VLA:        $SHARED_VLA_CHECKPOINT"
  echo "seen train faults: j0=45%, j6=45%, healthy=10%"
  echo "held-out joint:    j2 (globally excluded from training)"
  echo "steps:             $STEPS (~$STEPS_PER_TASK per task)"
  echo "seed:              $SEED"
  echo "GPU:               $GPU"
  echo "run dir:           $RUNDIR"
  echo
  echo "selected tasks:"
  for ((i=0; i<N_TASKS; i++)); do
    printf '  %2d. %s\n' "$((i+1))" "${TASKS[$i]}"
  done
  echo "=============================================================="
}

print_selection
if [[ "$MODE" == "--print-only" ]]; then
  exit 0
fi

cd "$REPO_ROOT"
# shellcheck source=scripts/_environment.sh
source "$SCRIPT_DIR/_environment.sh"
repair_activate_environment

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "$MODE" != "--dry-run" ]] && compgen -G "$RUNDIR/ckpt_*.pt" >/dev/null; then
  echo "ABORT: $RUNDIR already contains checkpoints." >&2
  echo "Set RUNNAME to a new name or move the existing run first." >&2
  exit 1
fi

COMMON=(
  --tasks "${TASKS[@]}"
  --task_probs "${TASK_PROBS[@]}"
  --shared_vla_checkpoint "$SHARED_VLA_CHECKPOINT"
  --joint_pool 4 6 healthy
  --fault_probs 0.45 0.45 0.10
  --heldout_conditions 0
  --n_eval_states 10
  --context_len 16
  --temporal_hidden 128
  --cap_dim 32
  --z_dim 64
  --transformer_layers 2
  --transformer_heads 4
  --transformer_ffn 256
  --encoder_lr 1e-4
  --lambda_joint 1.0
  --lambda_eef 1.0
  --lambda_kin 0.25
  --encoder_q_weight 0.05
  --residual_scale 0.1
  --alpha_init 0.01
  --utd 4
  --n_step 3
  --seed "$SEED"
  --logdir "$RUNDIR"
)

if [[ "$MODE" == "--dry-run" ]]; then
  echo "=== DRY RUN: $N_TASKS tasks, one shared frozen VLA ==="
  python rl/train_joint_factorized_multitask_sac.py "${COMMON[@]}" --dry_run
  exit 0
fi

echo "=== TRAINING START ==="
python rl/train_joint_factorized_multitask_sac.py \
  "${COMMON[@]}" \
  --total_steps "$STEPS" \
  --wandb |& tee "$LOGFILE"

echo
echo "Training finished."
echo "Checkpoint: $RUNDIR/ckpt_${STEPS}.pt"
echo "NOTE: this experiment holds out j0. Use an unseen-j0 evaluator; do not use eval_jfcrl_unseen_j2.sh."
