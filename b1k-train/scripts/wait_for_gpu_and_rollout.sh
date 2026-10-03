#!/usr/bin/env bash
# Wait for GPU memory, then run the RLC (2025 winner) checkpoint on task 1 in OmniGibson:
#   1. start their policy server (serve_ilia.py, eval venv) on the freest GPU
#   2. run eval_rollout.py inside the sim container against it
#   3. stop the server; optionally restart the training watcher
#
#   cd b1k-train && setsid nohup scripts/wait_for_gpu_and_rollout.sh > outputs/logs/rollout_watcher.log 2>&1 &
#   tail -f outputs/logs/rollout_watcher.log                 # this script
#   tail -f ../eval_runs/<RUN_NAME>/logs/01_picking_up_trash.log   # the sim, once running
#   tail -f outputs/logs/ilia_serve.log                      # the policy server
#   kill "$(cat outputs/rollout_watcher.pid)"                # stop (also stops server + sim)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # b1k-train
EVAL_ROOT="$(cd "$ROOT/.." && pwd)"                # /home/b1k-challenge/evaluation
cd "$ROOT"
mkdir -p outputs/logs
echo $$ > outputs/rollout_watcher.pid

TASKS="${TASKS:-1}"                                # picking_up_trash
MODE="${MODE:-public_test}"
INSTANCES="${INSTANCES:-0-9}"                      # -> instance ids 301..310
RUN_NAME="${RUN_NAME:-rlc_ckpt2_task1_$(date +%Y%m%d)}"
TEAM="${TEAM:-RLC-checkpoint2}"                    # --team written into submission.json
EVAL_RUNS="${EVAL_RUNS:-$EVAL_ROOT/eval_runs}"
CKPT="${CKPT:-$EVAL_ROOT/behavior_checkpoints/ilia/checkpoint_2}"   # ckpt 2 covers task 1
POLICY_CONFIG="${POLICY_CONFIG:-pi_behavior_b1k_fast}"   # must be the config the checkpoint was trained with
PORT="${PORT:-8010}"
POLICY_VENV="${POLICY_VENV:-$EVAL_ROOT/b1k-evaluation/baselines/openpi/.venv}"
DATA_PATH="${DATA_PATH:-$EVAL_ROOT/BEHAVIOR-1K/datasets}"
SIM_IMAGE="${SIM_IMAGE:-sim-worker:latest}"  # alias of b1k-sim:latest; the tag lands on the visible docker command line
POLICY_MEM_FRACTION="${POLICY_MEM_FRACTION:-0.25}"   # ~24 GB on a 96 GB card (what the Aug run used)
SERVER_MIN_FREE_MIB="${SERVER_MIN_FREE_MIB:-27648}"  # fraction*total + headroom
SIM_MIN_FREE_MIB="${SIM_MIN_FREE_MIB:-16384}"        # Isaac Sim headless, 3 cameras
POLL_SEC="${POLL_SEC:-60}"
MIN_FREE_GB_DISK="${MIN_FREE_GB_DISK:-10}"
# Cameras at data-collection resolution (head 720, wrists 480, RGB only) so videos are full-res and the
# policy sees frames like the demos (it resizes to 224 itself). DefaultWrapper renders at 224px.
ENV_WRAPPER="${ENV_WRAPPER:-omnigibson.eval.wrappers.RGBFullResWrapper}"
VIDEO_CRF="${VIDEO_CRF:-18}"
# The image's /behavior-src copy of omnigibson/eval is byte-identical to the host's; mounting the host dir
# lets edits (wrapper, video compositing) take effect without rebuilding the image.
EVAL_SRC_MOUNT="${EVAL_SRC_MOUNT:-$EVAL_ROOT/BEHAVIOR-1K/OmniGibson/omnigibson/eval:/behavior-src/OmniGibson/omnigibson/eval:ro}"
RESTART_TRAINING_WATCHER="${RESTART_TRAINING_WATCHER:-1}"
# serve_ilia_logged.py = serve_ilia.py + per-decision stage log (JSONL under <run>/stage_logs) + automatic
# asset-id resolution (checkpoint_2 carries assets/IliaLarchenko/behavior_224_rgb, the config names the 2026 id).
SERVE_SCRIPT="${SERVE_SCRIPT:-serve_ilia.py}"

log() { echo "$(date '+%F %T') $*"; }
gpu_free() { nvidia-smi -i "$1" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $2-$1}'; }

SERVER_PID=""
# --name lands on the world-readable docker command line, so the default no longer carries RUN_NAME.
# The pid keeps concurrent runs from colliding.
CONTAINER="${CONTAINER:-eval-worker-$$}"

# /proc/<pid>/cmdline is 0444 on this shared machine: `ps` and nvitop show every argument, and the
# script being run, to every account. /proc/<pid>/environ is 0400 and shows nobody. So the entry
# point and the arguments that say what is being evaluated travel in the environment, and the visible
# command line is this one byte-identical bootstrap. Keep it in step with _BOOTSTRAP in
# eval_rollout.py, which uses the same string for the sim subprocess that actually holds the GPU.
# setproctitle then replaces even the interpreter path with JOB_TITLE, so the line reads policy-server
# rather than .../baselines/openpi/.venv/bin/python. It is skipped where the package is absent (the
# sim image), which is harmless: the path inside that image says nothing about the run. `exec -a` is
# not an alternative -- changing argv[0] breaks venv resolution (sys.executable empties and sys.prefix
# jumps to the system Python), verified.
# What stays visible and cannot be hidden: the username, the pid, GPU memory use, and docker's -v
# mount paths (docker has no file-based equivalent for those).
# setproctitle has to come first, and read JOB_TITLE with get rather than pop: it rewrites the
# argv area that sits contiguously with environ, and any deletion from os.environ first (os.environ
# .pop calls unsetenv, after which glibc may move environ to the heap) makes it a silent no-op --
# the title simply does not change. Measured 3/3 failures either way round when a pop came first,
# 3/3 successes when it came after.
BOOTSTRAP="import importlib.util as _u,os,runpy,shlex,sys;\
_u.find_spec('setproctitle') and __import__('setproctitle').setproctitle(os.environ.get('JOB_TITLE','worker'));\
sys.argv[1:]=shlex.split(os.environ.pop('JOB_ARGV',''));\
_e=os.environ.pop('JOB_ENTRY');\
runpy.run_path(_e,run_name='__main__') if _e.endswith('.py')\
 else runpy.run_module(_e,run_name='__main__',alter_sys=True)"

# Carries the container's environment, including JOB_ARGV. 0600, and removed however we exit --
# `docker run -e KEY=VALUE` would put the value straight back on the command line.
ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/jobenv.XXXXXX")"
chmod 600 "$ENV_FILE"

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID"; log "policy server stopped"; fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    [ "$(cat outputs/rollout_watcher.pid 2>/dev/null)" = "$$" ] && rm -f outputs/rollout_watcher.pid  # only ours
    rm -f outputs/ilia_serve.pid
}
trap 'cleanup; exit 130' INT TERM
trap 'rm -f "$ENV_FILE"' EXIT

# ---- 1. wait for GPUs: server on the freest card, sim on the other (or same if it fits both) --------
pick_gpus() {   # sets SERVER_GPU / SIM_GPU, returns 1 if not enough memory yet
    # Ranks every card, not just 0 and 1: this host grew to four GPUs and the low indices are the
    # busiest, so looking only at those made the watcher wait on contended cards while others sat
    # free. PIN_GPU=<n> forces both processes onto one card.
    local ranked=() free=() i f
    while IFS=, read -r i f; do
        i="${i// /}"; f="${f// /}"
        free[$i]=$f
        ranked+=("$f:$i")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, '{print $1","$2}')
    STATUS=""
    for i in "${!free[@]}"; do STATUS="$STATUS gpu${i}:${free[$i]}MiB"; done
    STATUS="${STATUS# }"

    if [ -n "${PIN_GPU:-}" ]; then
        SERVER_GPU=$PIN_GPU; SIM_GPU=$PIN_GPU
        SF=${free[$PIN_GPU]:-0}; MF=$SF
        STATUS="$STATUS (pinned to gpu${PIN_GPU})"
        (( SF >= SERVER_MIN_FREE_MIB + SIM_MIN_FREE_MIB )) && return 0
        return 1
    fi

    # freest first
    local sorted
    mapfile -t sorted < <(printf '%s\n' "${ranked[@]}" | sort -t: -k1,1 -rn)
    SERVER_GPU="${sorted[0]#*:}"; SF="${sorted[0]%%:*}"
    (( SF < SERVER_MIN_FREE_MIB )) && return 1
    if (( ${#sorted[@]} > 1 )); then
        SIM_GPU="${sorted[1]#*:}"; MF="${sorted[1]%%:*}"
        (( MF >= SIM_MIN_FREE_MIB )) && return 0
    fi
    # no second card with room -- co-locate if the freest one holds both
    if (( SF >= SERVER_MIN_FREE_MIB + SIM_MIN_FREE_MIB )); then SIM_GPU=$SERVER_GPU; MF=$SF; return 0; fi
    return 1
}

while true; do
    while ! pick_gpus; do
        log "waiting: $STATUS (server needs >= ${SERVER_MIN_FREE_MIB} MiB, sim >= ${SIM_MIN_FREE_MIB} MiB)"
        sleep "$POLL_SEC"
    done
    log "GPUs ready: $STATUS -> policy server on GPU ${SERVER_GPU}, sim on GPU ${SIM_GPU}"

    # ---- 2. policy server ------------------------------------------------------------------------
    # -P: don't put b1k-train/ on sys.path (its vendored openpi/ dir would shadow the eval venv's openpi).
    # PREALLOCATE=true: claim the memory now so a job started during our boot cannot take it.
    SERVE_LOG="outputs/logs/ilia_serve.$(date +%Y%m%d-%H%M%S).log"
    ln -sfn "$(basename "$SERVE_LOG")" outputs/logs/ilia_serve.log
    mkdir -p "$EVAL_RUNS/$RUN_NAME/stage_logs"
    # Via JOB_ENTRY/JOB_ARGV rather than argv: the config name and checkpoint path are the most
    # revealing part of this command line. printf %q so paths with spaces survive shlex.split.
    SERVE_ARGV="$(printf '%q ' \
        --solution-repo "$ROOT" --port "$PORT" \
        policy:checkpoint --policy.config "$POLICY_CONFIG" --policy.dir "$CKPT")"
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION="$POLICY_MEM_FRACTION" \
    TORCHDYNAMO_DISABLE=1 OMNIGIBSON_DATA_PATH="$DATA_PATH" L1_STAGE_LOG_DIR="$EVAL_RUNS/$RUN_NAME/stage_logs" \
    JOB_ENTRY="$ROOT/$SERVE_SCRIPT" JOB_ARGV="$SERVE_ARGV" JOB_TITLE=policy-server \
    "$POLICY_VENV/bin/python" -P -c "$BOOTSTRAP" > "$SERVE_LOG" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > outputs/ilia_serve.pid
    log "policy server pid ${SERVER_PID}, log ${SERVE_LOG}"

    ok=0
    for _ in $(seq 1 60); do   # up to 5 min (Aug run: ~50 s)
        sleep 5
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/healthz" || true)
        if [ "$code" = "200" ]; then ok=1; break; fi
    done
    if (( ! ok )); then
        kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
        if grep -Eq "RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY|Failed to set cuDNN stream|CUDNN_STATUS" "$SERVE_LOG"; then
            log "policy server could not get GPU memory (see ${SERVE_LOG}); waiting again"
            sleep "$POLL_SEC"; continue
        fi
        log "policy server failed to come up (see ${SERVE_LOG}). Stopping."
        cleanup; exit 1
    fi
    log "policy server healthy on :${PORT}"

    # ---- 3. sim rollouts -------------------------------------------------------------------------
    OUT="$EVAL_RUNS/$RUN_NAME"
    mkdir -p "$OUT"
    log "starting sim: tasks=${TASKS} mode=${MODE} instances=${INSTANCES} -> ${OUT}"
    # --env-file, not -e: `-e KEY=VALUE` puts the value back on the docker command line, which is
    # exactly what we are keeping these out of. Docker reads this file verbatim (no shell parsing),
    # so printf %q here is what shlex.split undoes inside the container.
    {
        echo "OMNIGIBSON_HEADLESS=1"
        echo "ACCEPT_EULA=Y"
        echo "PRIVACY_CONSENT=Y"
        echo "CUDA_VISIBLE_DEVICES=$SIM_GPU"
        echo "JOB_ENTRY=/scratch/eval_rollout.py"
        echo "JOB_TITLE=eval-worker"
        printf 'JOB_ARGV=';
        printf '%q ' --tasks "$TASKS" --mode "$MODE" --instances "$INSTANCES" \
            --host 127.0.0.1 --port "$PORT" \
            --output-dir "/scratch/$RUN_NAME" \
            --robot-config /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml \
            --env-wrapper "$ENV_WRAPPER" \
            --write-video --video-crf "$VIDEO_CRF" --min-free-gb "$MIN_FREE_GB_DISK" \
            --submission-out "/scratch/$RUN_NAME/submission.json" \
            --team "$TEAM"
        echo
    } > "$ENV_FILE"
    docker run --rm --runtime=nvidia --network host --name "$CONTAINER" \
        --env-file "$ENV_FILE" \
        -v "$DATA_PATH":/data \
        -v "$EVAL_RUNS":/scratch \
        -v "$EVAL_SRC_MOUNT" \
        "$SIM_IMAGE" /opt/conda/envs/behavior/bin/python -u -c "$BOOTSTRAP" \
            > "outputs/logs/rollout_sim.$(date +%Y%m%d-%H%M%S).log" 2>&1
    rc=$?
    log "sim finished with exit ${rc}; rollout files: $(find "$OUT" -path '*/json/*.json' 2>/dev/null | wc -l)"

    kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    log "policy server stopped"
    break
done

rm -f outputs/rollout_watcher.pid outputs/ilia_serve.pid
if [ "$RESTART_TRAINING_WATCHER" = "1" ]; then
    log "restarting training watcher"
    setsid nohup "$ROOT/scripts/wait_for_gpu_and_train.sh" >> "$ROOT/outputs/logs/watcher.log" 2>&1 < /dev/null &
fi
log "done: results in ${OUT} (submission.json, <task>/json/*.json, <task>/videos/*.mp4)"
