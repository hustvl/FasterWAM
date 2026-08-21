#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  monitor_checkpoint_qdel.sh \
    --job-id JOB_ID \
    --run-dir RUN_DIR \
    --training-log TRAINING_LOG \
    --target-step STEP \
    [--expected-project PROJECT] \
    [--poll-seconds SECONDS] \
    [--pid-file PATH] \
    [--check-once] \
    [--dry-run]

The job is deleted only after all of the following are true:
  1. weights/step_NNNNNN.pt exists and is non-empty;
  2. state/step_NNNNNN/trainer_state.json records the target global step;
  3. the training log contains the completed [ckpt] entry for that step;
  4. the PBS job still belongs to the current user and expected project.
EOF
}

log() {
  printf '%s [checkpoint-monitor] %s\n' "$(date -Is)" "$*"
}

JOB_ID=""
RUN_DIR=""
TRAINING_LOG=""
TARGET_STEP=""
EXPECTED_PROJECT="gcg51472"
POLL_SECONDS=15
PID_FILE=""
CHECK_ONCE=false
DRY_RUN=false

while (($# > 0)); do
  case "$1" in
    --job-id)
      JOB_ID="${2:?Missing value for --job-id}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="${2:?Missing value for --run-dir}"
      shift 2
      ;;
    --training-log)
      TRAINING_LOG="${2:?Missing value for --training-log}"
      shift 2
      ;;
    --target-step)
      TARGET_STEP="${2:?Missing value for --target-step}"
      shift 2
      ;;
    --expected-project)
      EXPECTED_PROJECT="${2:?Missing value for --expected-project}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:?Missing value for --poll-seconds}"
      shift 2
      ;;
    --pid-file)
      PID_FILE="${2:?Missing value for --pid-file}"
      shift 2
      ;;
    --check-once)
      CHECK_ONCE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${JOB_ID}" || -z "${RUN_DIR}" || -z "${TRAINING_LOG}" || -z "${TARGET_STEP}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${TARGET_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--target-step must be a positive integer: ${TARGET_STEP}" >&2
  exit 2
fi
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--poll-seconds must be a positive integer: ${POLL_SECONDS}" >&2
  exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory does not exist: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${TRAINING_LOG}" ]]; then
  echo "Training log does not exist: ${TRAINING_LOG}" >&2
  exit 2
fi

STEP_TAG=$(printf 'step_%06d' "${TARGET_STEP}")
WEIGHTS_PATH="${RUN_DIR}/checkpoints/weights/${STEP_TAG}.pt"
STATE_DIR="${RUN_DIR}/checkpoints/state/${STEP_TAG}"
TRAINER_STATE_PATH="${STATE_DIR}/trainer_state.json"

cleanup() {
  if [[ -n "${PID_FILE}" && -f "${PID_FILE}" ]]; then
    if [[ "$(<"${PID_FILE}")" == "$$" ]]; then
      rm -f -- "${PID_FILE}"
    fi
  fi
}
trap cleanup EXIT
trap 'log "monitor interrupted"; exit 130' INT TERM

if [[ -n "${PID_FILE}" ]]; then
  mkdir -p -- "$(dirname "${PID_FILE}")"
  if [[ -s "${PID_FILE}" ]]; then
    existing_pid=$(<"${PID_FILE}")
    if [[ "${existing_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      echo "A monitor is already running with PID ${existing_pid}: ${PID_FILE}" >&2
      exit 2
    fi
  fi
  printf '%s\n' "$$" > "${PID_FILE}"
fi

get_job_info() {
  qstat -xf "${JOB_ID}" 2>/dev/null || true
}

job_field() {
  local info="$1"
  local field="$2"
  awk -F= -v field="${field}" '
    $1 ~ "^[[:space:]]*" field "[[:space:]]*$" {
      value=$2
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      print value
      exit
    }
  ' <<<"${info}"
}

validate_job_identity() {
  local info="$1"
  local owner
  local project
  owner=$(job_field "${info}" "Job_Owner")
  project=$(job_field "${info}" "project")
  owner="${owner%%@*}"

  if [[ -z "${owner}" || "${owner}" != "${USER}" ]]; then
    log "refusing to monitor job with owner=${owner:-unknown}; expected ${USER}"
    return 1
  fi
  if [[ -z "${project}" || "${project}" != "${EXPECTED_PROJECT}" ]]; then
    log "refusing to monitor job with project=${project:-unknown}; expected ${EXPECTED_PROJECT}"
    return 1
  fi
}

checkpoint_complete() {
  [[ -s "${WEIGHTS_PATH}" ]] || return 1
  [[ -s "${TRAINER_STATE_PATH}" ]] || return 1
  grep -Eq '"global_step"[[:space:]]*:[[:space:]]*'"${TARGET_STEP}"'([,}]|$)' \
    "${TRAINER_STATE_PATH}" || return 1
  grep -Fq "[ckpt] step=${TARGET_STEP}" "${TRAINING_LOG}" || return 1
}

initial_info=$(get_job_info)
if [[ -z "${initial_info}" ]]; then
  log "job not found: ${JOB_ID}"
  exit 2
fi
validate_job_identity "${initial_info}"

log "started job=${JOB_ID} project=${EXPECTED_PROJECT} target=${STEP_TAG} poll=${POLL_SECONDS}s dry_run=${DRY_RUN}"
log "run_dir=${RUN_DIR}"
log "waiting for weights=${WEIGHTS_PATH}, state=${TRAINER_STATE_PATH}, and completed checkpoint log entry"

missing_polls=0
while true; do
  info=$(get_job_info)
  if [[ -z "${info}" ]]; then
    missing_polls=$((missing_polls + 1))
    if ((missing_polls >= 3)); then
      log "job was missing for ${missing_polls} consecutive polls; exiting without qdel"
      exit 1
    fi
  else
    missing_polls=0
    state=$(job_field "${info}" "job_state")
    if [[ "${state}" == "F" ]]; then
      log "job finished before target checkpoint; exiting without qdel"
      exit 1
    fi
  fi

  if checkpoint_complete; then
    info=$(get_job_info)
    validate_job_identity "${info}"
    state=$(job_field "${info}" "job_state")
    if [[ "${state}" != "R" && "${state}" != "Q" ]]; then
      log "checkpoint is complete, but job state=${state:-unknown}; exiting without qdel"
      exit 1
    fi

    log "verified completed checkpoint ${STEP_TAG}; weights_bytes=$(stat -c %s "${WEIGHTS_PATH}")"
    if [[ "${DRY_RUN}" == "true" ]]; then
      log "dry-run enabled; would execute: qdel ${JOB_ID}"
      exit 0
    fi

    qdel "${JOB_ID}"
    log "qdel submitted for ${JOB_ID} after verified checkpoint ${STEP_TAG}"
    exit 0
  fi

  if [[ "${CHECK_ONCE}" == "true" ]]; then
    log "target checkpoint is not complete"
    exit 3
  fi
  sleep "${POLL_SECONDS}"
done
