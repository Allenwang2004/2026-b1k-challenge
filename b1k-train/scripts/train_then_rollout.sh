#!/usr/bin/env bash
# Wait for a training run to finish, publish its final checkpoint, then evaluate it in OmniGibson.
#
# It is the hand-off between the two watchers: wait_for_gpu_and_train.sh produces a checkpoint under
# outputs/checkpoints/<CONFIG>/<EXP_NAME>/<step>, and wait_for_gpu_and_rollout.sh wants one under
# evaluation/behavior_checkpoints/. This moves it (so the 44 GB lives with the other checkpoints
# rather than inside the repo tree) and starts the rollout with the matching policy config.
#
#   cd b1k-train && setsid nohup scripts/train_then_rollout.sh > outputs/logs/train_then_rollout.log 2>&1 &
#   tail -f outputs/logs/train_then_rollout.log
#   kill "$(cat outputs/train_then_rollout.pid)"     # stop waiting (does not touch the training)
#
# Nothing here kills or restarts the training: if it fails, or produces no final checkpoint, this
# script says so and exits without starting a rollout.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/logs
echo $$ > outputs/train_then_rollout.pid
trap '[ "$(cat outputs/train_then_rollout.pid 2>/dev/null)" = "$$" ] && rm -f outputs/train_then_rollout.pid; exit 130' INT TERM

CONFIG="${CONFIG:-pi_behavior_b1k_bddl_ckpt2}"
EXP_NAME="${EXP_NAME:-bddl_ckpt2_task1}"
FINAL_STEP="${FINAL_STEP:-19999}"          # num_train_steps - 1, what train.py saves last
CKPT_NAME="${CKPT_NAME:-$EXP_NAME}"        # -> behavior_checkpoints/<CKPT_NAME>/<step>
TASKS="${TASKS:-1}"
INSTANCES="${INSTANCES:-0-9}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d)}"
TEAM="${TEAM:-$EXP_NAME}"
POLL_SEC="${POLL_SEC:-120}"

SRC_DIR="outputs/checkpoints/${CONFIG}/${EXP_NAME}"
DEST_DIR="$EVAL_ROOT/behavior_checkpoints/${CKPT_NAME}"

log() { echo "$(date '+%F %T') $*"; }

# ---- 1. wait for the training to stop running -------------------------------------------------
log "waiting for training to finish (config=${CONFIG} exp=${EXP_NAME}, final step ${FINAL_STEP})"
while pgrep -f "scripts/train.py ${CONFIG}" >/dev/null || [ -f outputs/watcher.pid ]; do
    sleep "$POLL_SEC"
done
log "training process is gone"

# ---- 2. check it actually produced the final checkpoint ---------------------------------------
if [ ! -d "$SRC_DIR/$FINAL_STEP" ]; then
    have=$(ls "$SRC_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')
    log "no checkpoint ${FINAL_STEP} in ${SRC_DIR} (present: ${have:-none}). Training did not finish; not evaluating."
    log "To evaluate an earlier checkpoint anyway: FINAL_STEP=<step> $0"
    rm -f outputs/train_then_rollout.pid
    exit 1
fi
if [ -e "$SRC_DIR/$FINAL_STEP.orbax-checkpoint-tmp-0" ]; then
    log "checkpoint ${FINAL_STEP} still has a temp directory beside it; the save was interrupted. Not evaluating."
    rm -f outputs/train_then_rollout.pid
    exit 1
fi

# ---- 3. publish it to behavior_checkpoints/ ---------------------------------------------------
mkdir -p "$DEST_DIR"
if [ -d "$DEST_DIR/$FINAL_STEP" ]; then
    log "${DEST_DIR}/${FINAL_STEP} already exists; leaving both copies alone and evaluating the published one"
else
    log "moving ${SRC_DIR}/${FINAL_STEP} -> ${DEST_DIR}/${FINAL_STEP} ($(du -sh "$SRC_DIR/$FINAL_STEP" | cut -f1))"
    mv "$SRC_DIR/$FINAL_STEP" "$DEST_DIR/$FINAL_STEP" || { log "move failed"; rm -f outputs/train_then_rollout.pid; exit 1; }
    # keep --resume working from the training tree
    rmdir "$SRC_DIR" 2>/dev/null && ln -s "$(realpath --relative-to="$(dirname "$SRC_DIR")" "$DEST_DIR")" "$SRC_DIR" \
        && log "left a symlink at ${SRC_DIR}"
fi
CKPT="$DEST_DIR/$FINAL_STEP"
log "checkpoint: ${CKPT} ($(du -sh "$CKPT" | cut -f1)), assets: $(ls "$CKPT/assets" 2>/dev/null | tr '\n' ' ')"

# ---- 4. roll it out ---------------------------------------------------------------------------
log "starting rollout: tasks=${TASKS} instances=${INSTANCES} run=${RUN_NAME}"
CKPT="$CKPT" POLICY_CONFIG="$CONFIG" RUN_NAME="$RUN_NAME" TEAM="$TEAM" \
TASKS="$TASKS" INSTANCES="$INSTANCES" RESTART_TRAINING_WATCHER=0 \
SERVE_SCRIPT="${SERVE_SCRIPT:-serve_ilia_logged.py}" \
    scripts/wait_for_gpu_and_rollout.sh
rc=$?
log "rollout finished with exit ${rc}; results in $EVAL_ROOT/eval_runs/${RUN_NAME}"
rm -f outputs/train_then_rollout.pid
exit "$rc"
