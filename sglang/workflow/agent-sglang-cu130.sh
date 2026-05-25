#!/usr/bin/env bash
# Agent/debug helper functions. Human operators do not need this file.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "This script is meant to be sourced after env-sglang-cu130.sh:" >&2
  echo "  source /root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh" >&2
  echo "  source /root/autodl-tmp/public/iws/scripts/agent-sglang-cu130.sh" >&2
  exit 2
fi

sglang_env_check() {
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "VENV_ROOT=$VENV_ROOT"
  echo "CACHE_ROOT=$CACHE_ROOT"
  echo "CUDA_ROOT=$CUDA_ROOT"
  echo "CUDA_LIB_PATH=$CUDA_LIB_PATH"
  echo "LD_PRELOAD=${LD_PRELOAD-}"
  command -v python
  python -m pip check
  python - <<'PY'
import importlib.metadata as md
for p in ["torch", "triton", "flashinfer-python", "sglang-kernel", "nvidia-cuda-nvcc", "nvidia-cuda-cccl"]:
    print(p, md.version(p))
PY
}

_sglang_arg_port() {
  local previous=""
  for arg in "$@"; do
    if [ "$previous" = "--port" ]; then
      echo "$arg"
      return 0
    fi
    case "$arg" in
      --port=*)
        echo "${arg#--port=}"
        return 0
        ;;
    esac
    previous="$arg"
  done
  echo "$SGLANG_PORT"
}

_sglang_start_usage() {
  echo "Usage: sglang_start <sglang.launch_server args...>" >&2
  echo "Example: sglang_start --model-path \"\$MODEL_PATH\" --host 0.0.0.0 --port 30000 --dp-size 2" >&2
  echo "For launch_server help, use: sglang_launch --help" >&2
}

_sglang_has_help_arg() {
  for arg in "$@"; do
    case "$arg" in
      -h|--help) return 0 ;;
    esac
  done
  return 1
}

sglang_launch() {
  if [ "$#" -eq 0 ]; then
    echo "Usage: sglang_launch <sglang.launch_server args...>" >&2
    return 2
  fi
  cd "$PROJECT_ROOT" || return
  python -m sglang.launch_server "$@"
}

sglang_launch_qwen() {
  sglang_launch --model-path "$MODEL_PATH" --host "$SGLANG_HOST" --port "$SGLANG_PORT" "$@"
}

sglang_start() {
  if [ "$#" -eq 0 ]; then
    _sglang_start_usage
    return 2
  fi
  if _sglang_has_help_arg "$@"; then
    _sglang_start_usage
    return 0
  fi
  local log pid port
  cd "$PROJECT_ROOT" || return
  port="$(_sglang_arg_port "$@")"
  log="$LOG_ROOT/server-cu130-custom-$(date +%Y%m%d-%H%M%S)-port${port}.log"
  /usr/bin/nohup python -m sglang.launch_server "$@" > "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$STATUS_ROOT/sglang-server.pid"
  echo "$log" > "$STATUS_ROOT/sglang-server.log"
  echo "$port" > "$STATUS_ROOT/sglang-server.port"
  echo "PID=$pid"
  echo "PORT=$port"
  echo "LOG=$log"
}

sglang_start_qwen() {
  sglang_start --model-path "$MODEL_PATH" --host "$SGLANG_HOST" --port "$SGLANG_PORT" "$@"
}

sglang_start_dp2_ft() {
  sglang_start_qwen \
    --dp-size 2 \
    --attention-backend triton \
    --moe-runner-backend triton \
    --sampling-backend pytorch \
    --enable-fault-tolerance \
    "$@"
}

sglang_status() {
  local pid=""
  pid="$(cat "$STATUS_ROOT/sglang-server.pid" 2>/dev/null || true)"
  echo "PID=${pid:-none}"
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    ps -p "$pid" -o pid,ppid,etime,stat,cmd
  fi
  curl -fsS --max-time 5 "http://127.0.0.1:${SGLANG_PORT}/fault_tolerance/status" 2>/dev/null || true
  echo
}

sglang_stop() {
  local pid=""
  pid="$(cat "$STATUS_ROOT/sglang-server.pid" 2>/dev/null || true)"
  if [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
    echo "No recorded running SGLang process."
    return 0
  fi
  ps -p "$pid" -o pid,ppid,etime,stat,cmd
  kill -TERM "$pid" 2>/dev/null || true
  sleep 5
  if ps -p "$pid" >/dev/null 2>&1; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

sglang_tail() {
  local log=""
  log="$(cat "$STATUS_ROOT/sglang-server.log" 2>/dev/null || true)"
  if [ -z "$log" ]; then
    echo "No recorded log path."
    return 1
  fi
  tail -f "$log"
}
