#!/usr/bin/env bash
# Human-facing SGLang cu130 environment.
# Source this file, then run `python -m sglang.launch_server ...` directly.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "This script is meant to be sourced:" >&2
  echo "  source /root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh" >&2
  echo "For an interactive shell, run:" >&2
  echo "  bash /root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh" >&2
  exit 2
fi

export IWS_HOME=/root/autodl-tmp/public/iws
case "$IWS_HOME" in
  /root/autodl-tmp/public/iws) ;;
  *) echo "Refusing unexpected IWS_HOME=$IWS_HOME" >&2; return 2 ;;
esac

export PROJECT_ROOT="$IWS_HOME/projects/sglang"
export VENV_ROOT="$IWS_HOME/venvs/sglang-py312-cu130"
export CACHE_ROOT="$IWS_HOME/caches/sglang-cu130"
export CUDA_ROOT="$IWS_HOME/toolchains/cuda-cu130"
export MODEL_PATH=/root/autodl-tmp/models/Qwen3-30B-A3B
export SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
export SGLANG_PORT="${SGLANG_PORT:-30000}"

export HOME="$IWS_HOME/home"
export XDG_CACHE_HOME="$CACHE_ROOT"
export XDG_CONFIG_HOME="$CACHE_ROOT/config"
export XDG_DATA_HOME="$CACHE_ROOT/data"
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"

export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"

export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/nv"
export FLASHINFER_CACHE_DIR="$CACHE_ROOT/flashinfer"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer_workspace"
export FLASHINFER_CUBIN_DIR="$CACHE_ROOT/flashinfer/cubins"
export VLLM_NO_USAGE_STATS=1
export VLLM_CONFIG_ROOT="$CACHE_ROOT/vllm"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export DO_NOT_TRACK=1
export CCACHE_DIR="$CACHE_ROOT/ccache"
export BUILD_CACHE_ROOT="$CACHE_ROOT/build"
export LOG_ROOT="$IWS_HOME/logs/sglang"
export STATUS_ROOT="$IWS_HOME/status"

export CUDNN_LIB="$VENV_ROOT/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDA_LIB_PATH="$CUDA_ROOT/lib64"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDNN_LIB"
unset LD_PRELOAD

export PATH="$VENV_ROOT/bin:$CUDA_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$PROJECT_ROOT/python"
export VIRTUAL_ENV="$VENV_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p \
  "$HOME" "$PROJECT_ROOT" "$VENV_ROOT" "$CACHE_ROOT" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
  "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$FLASHINFER_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" \
  "$FLASHINFER_CUBIN_DIR" "$VLLM_CONFIG_ROOT" "$VLLM_CACHE_ROOT" "$CCACHE_DIR" \
  "$BUILD_CACHE_ROOT" "$LOG_ROOT" "$STATUS_ROOT"
