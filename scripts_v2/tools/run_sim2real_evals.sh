#!/usr/bin/env bash
# Run the OmniReset state sim2real eval suite sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-uwlab}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/paresh/cage/UWLab/logs/rsl_rl/omnireset_peg_insertion/2026-04-09_14-50-19}"
LOGGER="${LOGGER:-wandb}"

# Pass the checkpoint directory as the first arg. Falls back to the default above.
#   bash run_sim2real_evals.sh                              # default dir
#   bash run_sim2real_evals.sh /path/to/checkpoint/dir      # use this dir
if [[ $# -gt 0 ]]; then
  CHECKPOINT_DIR="$1"
fi

PYTHON=(conda run -n "$CONDA_ENV" python)
EVAL_SCRIPT="scripts_v2/tools/eval_policy.py"
TASK_PREFIX="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-EvalCfg"

# label|task|extra args
# Every eval runs at --action_noise 0 by default — action noise is its own
# isolated axis (entries below) and should not contaminate other variants.
EVALS=(
  "V0 baseline|${TASK_PREFIX}V0|"
  "V1 OOD OSC gains|${TASK_PREFIX}V1|"
  "V2 robot-side OOD|${TASK_PREFIX}V2|"
  "V3 object-side OOD|${TASK_PREFIX}V3|"
  "V4 friction point|${TASK_PREFIX}V4|"
  "V5 peg mass point|${TASK_PREFIX}V5|"
  "V6 hole-pose perception noise default|${TASK_PREFIX}V6|"
  "V7 grasp offset default|${TASK_PREFIX}V7|"
  "V8 realistic combo|${TASK_PREFIX}V8|"
  "V0 action delay 1|${TASK_PREFIX}V0|--action_delay_steps 1"
  "V0 action delay 2|${TASK_PREFIX}V0|--action_delay_steps 2"
  "V0 action delay 3|${TASK_PREFIX}V0|--action_delay_steps 3"
  "V0 action noise 1|${TASK_PREFIX}V0|--action_noise 1"
  "V0 action noise 2|${TASK_PREFIX}V0|--action_noise 2"
  "V0 action noise 3|${TASK_PREFIX}V0|--action_noise 3"
  "V8 realistic combo + action delay 1|${TASK_PREFIX}V8|--action_delay_steps 1"
  "V8 realistic combo + action noise 2|${TASK_PREFIX}V8|--action_noise 2"
  "V8 realistic combo + delay 1 + action noise 2 (all axes)|${TASK_PREFIX}V8|--action_delay_steps 1 --action_noise 2"
)

echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Logger: $LOGGER"
echo ""
echo "Eval queue:"
for i in "${!EVALS[@]}"; do
  IFS="|" read -r label task extra <<<"${EVALS[$i]}"
  printf "  %2d. %s  (%s %s)\n" "$((i + 1))" "$label" "$task" "$extra"
done

for i in "${!EVALS[@]}"; do
  IFS="|" read -r label task extra <<<"${EVALS[$i]}"
  extra_args=()
  if [[ -n "$extra" ]]; then
    read -r -a extra_args <<<"$extra"
  fi

  echo ""
  echo "================================================================================"
  echo "Eval $((i + 1))/${#EVALS[@]}: $label"
  echo "Task: $task"
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    echo "Extra args: ${extra_args[*]}"
  fi
  echo "================================================================================"

  "${PYTHON[@]}" "$EVAL_SCRIPT" \
    --task "$task" \
    --checkpoint "$CHECKPOINT_DIR" \
    --num_envs 100 \
    --num_trajectories 100 \
    --headless \
    --logger "$LOGGER" \
    --action_noise 0 \
    "${extra_args[@]}"
done

echo ""
echo "All sim2real evals finished."
