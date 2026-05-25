# SGLang 手工操作手册

更新时间：2026-05-21

这份文档给“已经有环境的人”使用：如何登录、确认环境、启动/停止 SGLang、验证服务、排查是否污染 base CUDA。

## 当前可用环境

```text
服务器:      region-42.seetacloud.com:38651
IWS 根目录:  /root/autodl-tmp/public/iws
源码仓库:    /root/autodl-tmp/public/iws/projects/sglang
虚拟环境:    /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130
缓存目录:    /root/autodl-tmp/public/iws/caches/sglang-cu130
CUDA wrapper: /root/autodl-tmp/public/iws/toolchains/cuda-cu130
模型路径:    /root/autodl-tmp/models/Qwen3-30B-A3B
当前 commit: 8a6c8d74b
```

当前验证通过的服务：

```text
PID: 118217
port: 30000
log: /root/autodl-tmp/public/iws/logs/sglang/server-firstft-human-enter-20260521-154352.log
```

说明：旧服务 `108493` 已停止。当前服务是在 `8a6c8d74b first FT` 上，用人工入口 `enter-sglang-cu130.sh` 进入环境后直接执行 `python -m sglang.launch_server ...` 启动的。

## 登录

Windows 本机推荐：

```powershell
ssh -i C:\Users\59699\Documents\sglang\.ssh\sglang_autodl_git_key `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  -o StrictHostKeyChecking=accept-new `
  -p 38651 root@region-42.seetacloud.com
```

不要把 `passwd.txt` 的内容复制进命令或文档。

## 快速确认

人工操作推荐先进入 cu130 环境：

```bash
bash /root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh
```

进入后会自动：

```text
source /root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh
cd /root/autodl-tmp/public/iws/projects/sglang
清空 LD_PRELOAD
设置 venv PATH、PYTHONPATH、CUDA_HOME、CUDA_PATH、CUDA_LIB_PATH、LD_LIBRARY_PATH
设置所有 cache/tmp/HF/Torch/Triton 路径到 /root/autodl-tmp/public/iws/caches/sglang-cu130
```

进入后不需要再感知 venv、CUDA、cache 等环境变量，直接输入 `python -m sglang.launch_server ...` 即可。

如果只想在当前 shell 里加载环境而不新开交互 shell：

```bash
source /root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh
```

```bash
BASE=/root/autodl-tmp/public/iws
VENV=$BASE/venvs/sglang-py312-cu130
PROJECT=$BASE/projects/sglang
CACHE=$BASE/caches/sglang-cu130

cd "$PROJECT"
git log -1 --oneline
git status --short

PIP_CACHE_DIR=$CACHE/pip PIP_CONFIG_FILE=/dev/null $VENV/bin/python -m pip check

$VENV/bin/python - <<'PY'
import importlib.metadata as md
for p in ["torch", "triton", "flashinfer-python", "sglang-kernel", "nvidia-cuda-nvcc", "nvidia-cuda-cccl"]:
    print(p, md.version(p))
PY
```

当前应看到：

```text
8a6c8d74b first FT
No broken requirements found.
torch 2.11.0+cu130
triton 3.6.0
flashinfer-python 0.6.11.post1
sglang-kernel 0.4.2.post1
nvidia-cuda-nvcc 13.0.88
nvidia-cuda-cccl 13.0.85
```

## 启动 SGLang

进入环境后，环境变量已经准备好，你可以直接使用原生启动命令，不需要再在命令前面绑一大串环境变量：

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --dp-size 2 \
  --attention-backend triton \
  --moe-runner-backend triton \
  --sampling-backend pytorch \
  --enable-fault-tolerance
```

复现当前里程碑的最短路径：

```bash
bash /root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$SGLANG_HOST" \
  --port "$SGLANG_PORT" \
  --dp-size 2 \
  --attention-backend triton \
  --moe-runner-backend triton \
  --sampling-backend pytorch \
  --enable-fault-tolerance
```

下面是 agent/自动化使用的等价完整后台启动写法，适合审计环境变量，不是人工日常入口：

```bash
BASE=/root/autodl-tmp/public/iws
VENV=$BASE/venvs/sglang-py312-cu130
PROJECT=$BASE/projects/sglang
CACHE=$BASE/caches/sglang-cu130
STATUS=$BASE/status
LOGDIR=$BASE/logs/sglang
CUDA_ROOT=$BASE/toolchains/cuda-cu130
CUDNN_LIB=$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
MODEL=/root/autodl-tmp/models/Qwen3-30B-A3B
PORT=30000

mkdir -p "$STATUS" "$LOGDIR" "$CACHE/tmp" "$CACHE/config" "$CACHE/data" \
  "$CACHE/pip" "$CACHE/huggingface" "$CACHE/huggingface/hub" \
  "$CACHE/huggingface/transformers" "$CACHE/torch" \
  "$CACHE/torch_extensions" "$CACHE/triton" "$CACHE/nv" "$BASE/home"

LOG="$LOGDIR/server-cu130-dp2-triton-ft-cleanenv-$(date +%Y%m%d-%H%M%S).log"

/usr/bin/nohup /usr/bin/env -i \
  HOME=$BASE/home USER=root LOGNAME=root SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  VIRTUAL_ENV=$VENV \
  PATH=$VENV/bin:$CUDA_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONPATH=$PROJECT/python \
  XDG_CACHE_HOME=$CACHE XDG_CONFIG_HOME=$CACHE/config XDG_DATA_HOME=$CACHE/data \
  TMPDIR=$CACHE/tmp TEMP=$CACHE/tmp TMP=$CACHE/tmp \
  PIP_CACHE_DIR=$CACHE/pip PIP_CONFIG_FILE=/dev/null \
  HF_HOME=$CACHE/huggingface HF_HUB_CACHE=$CACHE/huggingface/hub TRANSFORMERS_CACHE=$CACHE/huggingface/transformers \
  TORCH_HOME=$CACHE/torch TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions TRITON_CACHE_DIR=$CACHE/triton CUDA_CACHE_PATH=$CACHE/nv \
  CUDA_HOME=$CUDA_ROOT CUDA_PATH=$CUDA_ROOT CUDA_LIB_PATH=$CUDA_ROOT/lib64 LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUDNN_LIB \
  CUDA_VISIBLE_DEVICES=0,1 \
  $VENV/bin/python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dp-size 2 \
    --attention-backend triton \
    --moe-runner-backend triton \
    --sampling-backend pytorch \
    --enable-fault-tolerance \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$STATUS/sglang-server.pid"
echo "$LOG" > "$STATUS/sglang-server.log"
echo "$PORT" > "$STATUS/sglang-server.port"
echo "PID=$PID"
echo "PORT=$PORT"
echo "LOG=$LOG"
```

关键点：

```text
CUDA_HOME      = /root/autodl-tmp/public/iws/toolchains/cuda-cu130
CUDA_PATH      = /root/autodl-tmp/public/iws/toolchains/cuda-cu130
CUDA_LIB_PATH  = /root/autodl-tmp/public/iws/toolchains/cuda-cu130/lib64
LD_LIBRARY_PATH= wrapper lib64 + venv cudnn lib
```

`CUDA_LIB_PATH` 不能省，否则 FlashInfer 可能加载 `/usr/local/cuda-12.8` 的 `libcudart.so.12`。

## 等待启动

```bash
tail -f "$(cat /root/autodl-tmp/public/iws/status/sglang-server.log)"
```

看到以下日志后服务可用：

```text
The server is fired up and ready to roll!
```

上一次验证启动耗时约 85 秒。

## API 验证

```bash
curl -fsS http://127.0.0.1:30000/get_model_info

curl -fsS http://127.0.0.1:30000/fault_tolerance/status

curl -fsS http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, my name is","sampling_params":{"max_new_tokens":8,"temperature":0}}'

curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4,"temperature":0}'
```

`/fault_tolerance/status` 当前期望：

```text
enabled=true
state=RUNNING
topology.dp_size=2
accepting_requests=true
```

注意：当前 `sentinels={}` 和 `components={}` 仍为空，服务可以推理，但 FT 注册语义还需要继续 review。

## 检查进程和端口

```bash
IWS_HOME=/root/autodl-tmp/public/iws
PID=$(cat "$IWS_HOME/status/sglang-server.pid" 2>/dev/null)
PORT=$(cat "$IWS_HOME/status/sglang-server.port" 2>/dev/null)
LOG=$(cat "$IWS_HOME/status/sglang-server.log" 2>/dev/null)

echo "PID=$PID"
echo "PORT=$PORT"
echo "LOG=$LOG"

ps -p "$PID" -o pid,ppid,etime,stat,cmd
ss -ltnp | grep ":$PORT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

当前 dp=2 服务大约占用 GPU 0/1 各 86GB 显存。

## 检查 CUDA/cuDNN 隔离

```bash
PID=$(cat /root/autodl-tmp/public/iws/status/sglang-server.pid)

echo "--- env ---"
tr '\0' '\n' < /proc/$PID/environ | grep -E '^(VIRTUAL_ENV|PYTHONPATH|PATH|LD_|CUDA|HF_HOME|TRITON_CACHE_DIR|TORCH_EXTENSIONS_DIR)=' | sort

echo "--- loaded cuda libs ---"
grep -E 'libcudart|libcudnn|libcublas|libnvrtc|libcuda' /proc/$PID/maps | awk '{print $NF}' | sort -u

echo "--- unexpected base/system cuda runtime libs ---"
grep -E '/root/miniconda3|/usr/local/cuda' /proc/$PID/maps | grep -E 'libcudart|libcudnn|libcublas|libnvrtc' || echo "no base/system CUDA runtime libs"
```

正确结果：

```text
libcudart.so.13 来自 /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130/...
libcudnn.so.9   来自 /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130/...
libcublas.so.13 来自 /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130/...
libnvrtc.so.13  来自 /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130/...
没有 /root/miniconda3 或 /usr/local/cuda 下的 CUDA runtime/cuDNN/cuBLAS/NVRTC
```

`/usr/lib/x86_64-linux-gnu/libcuda.so.*` 是宿主机 driver，出现是正常的。

## 停止服务

```bash
IWS_HOME=/root/autodl-tmp/public/iws
PID=$(cat "$IWS_HOME/status/sglang-server.pid" 2>/dev/null)

ps -p "$PID" -o pid,ppid,etime,stat,cmd
kill -TERM "$PID"
sleep 5

if ps -p "$PID" >/dev/null 2>&1; then
  kill -KILL "$PID"
fi
```

确认停止：

```bash
PORT=$(cat /root/autodl-tmp/public/iws/status/sglang-server.port 2>/dev/null)
ps -p "$PID" -o pid,ppid,etime,stat,cmd
ss -ltnp | grep ":$PORT" || echo "port stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

## 其他人使用时的最短路径

已经安装好环境时，只需要：

```bash
ssh -i <key> -p 38651 root@region-42.seetacloud.com
bash /root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh
python -m sglang.launch_server --model-path "$MODEL_PATH" --host "$SGLANG_HOST" --port "$SGLANG_PORT" --dp-size 2
```

不要用旧的 `/root/autodl-tmp/public/iws/scripts/env-sglang.sh` 作为默认入口，因为旧脚本仍指向 `sglang-py312` 旧环境。

## 旧环境说明

旧 venv：

```text
/root/autodl-tmp/public/iws/venvs/sglang-py312
```

旧环境曾依赖这些兼容开关：

```text
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
SGLANG_TRITON_DISABLE_PDL=1
--disable-cuda-graph
--disable-piecewise-cuda-graph
--disable-overlap-schedule
```

当前 fresh cu130 环境已经不需要这些开关。后续调试应以 `sglang-py312-cu130` 为准。
