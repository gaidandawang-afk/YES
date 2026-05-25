# SGLang IWS 部署 Runbook

更新时间：2026-05-21

这份文档记录 AutoDL 服务器上当前已经验证通过的 SGLang 工作流。目标是让后来的人能重新搭建、核验、启动服务，同时不污染 base 环境、不把大文件写到根分区。

## 当前结论

当前可用环境：

```text
IWS 根目录:  /root/autodl-tmp/public/iws
源码仓库:    /root/autodl-tmp/public/iws/projects/sglang
虚拟环境:    /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130
缓存目录:    /root/autodl-tmp/public/iws/caches/sglang-cu130
CUDA wrapper: /root/autodl-tmp/public/iws/toolchains/cuda-cu130
模型路径:    /root/autodl-tmp/models/Qwen3-30B-A3B
当前 commit: 8a6c8d74b first FT
```

已经验证：

```text
dp_size=2
--enable-fault-tolerance
--attention-backend triton
--moe-runner-backend triton
--sampling-backend pytorch
```

不再需要旧环境里的这些 workaround：

```text
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
SGLANG_TRITON_DISABLE_PDL
--disable-cuda-graph
--disable-piecewise-cuda-graph
--disable-overlap-schedule
```

`first FT` 之后曾临时加过三个环境兼容/debug patch：

```text
116dc1d02 Handle missing FlashInfer MXInt4 symbols
494733722 Defer unavailable FlashInfer FP8 import
1bfd903ba Allow disabling Triton attention PDL
```

fresh cu130 venv 已验证不需要这些 patch，服务器仓库和本地 clone 当前都已回到 `8a6c8d74b first FT`，便于后续专注 review FT 本身。

## 硬性约束

服务器根分区非常小，当前 `/` 约 99% 使用率，只有约 448M 空闲。所有 venv、pip cache、HF cache、Triton cache、JIT cache、日志、临时目录都必须放到 `/root/autodl-tmp/public/iws` 下。

禁止事项：

```text
1. 不要在 base/root miniconda 环境里 pip install。
2. 不要创建带 --system-site-packages 的 venv。
3. 不要把大文件写到 /root、/tmp、/home 或系统 Python 环境。
4. 不要继承 root shell 里的 LD_PRELOAD=/root/miniconda3/...。
5. 不要用没有检查过的 rm -rf 清理路径。
6. 如果必须改源码才能继续，先记录原因并停下来确认。
```

允许的系统依赖来源：

```text
/root/miniconda3/bin/python 只作为 venv 的 Python 解释器来源。
/usr/lib/x86_64-linux-gnu/libcuda.so.* 作为宿主机 NVIDIA driver，正常且不可避免。
```

不允许依赖的运行时来源：

```text
/root/miniconda3/lib/python3.12/site-packages/*
/root/miniconda3/targets/x86_64-linux/lib/libcudart*
/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart*
```

## SSH 连接

Windows 本机已有 key：

```text
C:\Users\59699\Documents\sglang\.ssh\sglang_autodl_git_key
```

推荐命令：

```powershell
ssh -i C:\Users\59699\Documents\sglang\.ssh\sglang_autodl_git_key `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  -o StrictHostKeyChecking=accept-new `
  -p 38651 root@region-42.seetacloud.com
```

不要把 `passwd.txt` 里的内容写入文档、命令历史或日志。

## 只读预检

进入服务器后先做只读检查：

```bash
set -u
hostname
whoami
pwd
df -h / /root/autodl-tmp
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

cd /root/autodl-tmp/public/iws/projects/sglang
git rev-parse --show-toplevel
git rev-parse --short HEAD
git log -1 --oneline
git status --short
```

当前已确认形态：

```text
/:                30G, 99% used, about 448M free
/root/autodl-tmp: 432G, about 319G free
GPU:              NVIDIA H20
repo HEAD:         8a6c8d74b
```

## 目录布局

当前建议布局：

```text
/root/autodl-tmp/public/iws/
  projects/
    sglang/
  venvs/
    sglang-py312-cu130/
  caches/
    sglang-cu130/
      pip/
      huggingface/
      torch/
      torch_extensions/
      triton/
      nv/
      tmp/
      config/
      data/
  toolchains/
    cuda-cu130/
  logs/
    sglang/
  status/
  home/
```

## 环境变量模板

安装和启动前都使用这一组路径。它的要点是清理继承环境，并把所有 cache 和 tmp 指到 IWS 下：

```bash
BASE=/root/autodl-tmp/public/iws
VENV=$BASE/venvs/sglang-py312-cu130
PROJECT=$BASE/projects/sglang
CACHE=$BASE/caches/sglang-cu130
CUDA_ROOT=$BASE/toolchains/cuda-cu130
CUDNN_LIB=$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib

mkdir -p "$BASE/home" "$CACHE/tmp" "$CACHE/config" "$CACHE/data" \
  "$CACHE/pip" "$CACHE/huggingface" "$CACHE/huggingface/hub" \
  "$CACHE/huggingface/transformers" "$CACHE/torch" \
  "$CACHE/torch_extensions" "$CACHE/triton" "$CACHE/nv" \
  "$BASE/logs/sglang" "$BASE/status"
```

启动服务时使用 `/usr/bin/env -i` 显式列出环境变量，不要直接 `source activate` 后继承 root shell。

## 人工入口脚本

服务器上已放置新的 cu130 入口脚本：

```text
/root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh
/root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh
/root/autodl-tmp/public/iws/scripts/bashrc-sglang-cu130
```

人工登录后推荐执行：

```bash
bash /root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh
```

它会进入一个已经配置好的交互 shell，并自动：

```text
cd /root/autodl-tmp/public/iws/projects/sglang
清空 LD_PRELOAD
设置 PATH 指向 sglang-py312-cu130 和 cuda-cu130/bin
设置 PYTHONPATH 指向 SGLang 源码 python 目录
设置 CUDA_HOME/CUDA_PATH/CUDA_LIB_PATH/LD_LIBRARY_PATH
设置所有 cache/tmp 路径到 /root/autodl-tmp/public/iws/caches/sglang-cu130
```

进入后人工不需要使用任何 helper 函数，直接运行原生命令：

```bash
python -m sglang.launch_server --model-path "$MODEL_PATH" --host "$SGLANG_HOST" --port "$SGLANG_PORT" <其他参数...>
```

agent/debug helper 单独放在：

```text
/root/autodl-tmp/public/iws/scripts/agent-sglang-cu130.sh
```

人工入口不会 source 这个 helper 文件，避免把人工操作和自动化操作混在一起。

如果只想在当前 shell 中加载环境：

```bash
source /root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh
```

## 重新安装虚拟环境

只有在明确需要重装时执行。先停服务，再确认路径：

```bash
BASE=/root/autodl-tmp/public/iws
VENV=$BASE/venvs/sglang-py312-cu130
case "$VENV" in
  /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130) ;;
  *) echo "bad VENV=$VENV"; exit 2 ;;
esac
```

创建干净 venv：

```bash
rm -rf "$VENV"
/root/miniconda3/bin/python -m venv "$VENV"

grep -n 'include-system-site-packages' "$VENV/pyvenv.cfg"
```

必须看到：

```text
include-system-site-packages = false
```

注意：`pyvenv.cfg` 里的 `home = /root/miniconda3/bin` 是正常的，它表示解释器来源。隔离判断看的是 `include-system-site-packages=false`、`site-packages` 路径和运行时动态库路径。

## 安装 Python 包

所有 pip 命令都显式设置 cache 和 config：

```bash
BASE=/root/autodl-tmp/public/iws
VENV=$BASE/venvs/sglang-py312-cu130
CACHE=$BASE/caches/sglang-cu130
mkdir -p "$CACHE/pip" "$CACHE/tmp"
```

安装 cu130 PyTorch 层：

```bash
PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp TEMP=$CACHE/tmp TMP=$CACHE/tmp PIP_CONFIG_FILE=/dev/null \
  $VENV/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple \
  torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 torchcodec==0.11.1

PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp TEMP=$CACHE/tmp TMP=$CACHE/tmp PIP_CONFIG_FILE=/dev/null \
  $VENV/bin/python -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple \
  torch_c_dlpack_ext==0.1.5
```

安装 SGLang kernel：

```bash
PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp TEMP=$CACHE/tmp TMP=$CACHE/tmp PIP_CONFIG_FILE=/dev/null \
  $VENV/bin/python -m pip install \
  --force-reinstall --no-deps \
  -i https://mirrors.aliyun.com/pypi/simple \
  sglang-kernel==0.4.2.post1
```

安装 SGLang runtime 依赖。当前已生成的依赖文件：

```text
/root/autodl-tmp/public/iws/status/sglang-cu130-runtime-reqs.txt
```

安装：

```bash
PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp TEMP=$CACHE/tmp TMP=$CACHE/tmp PIP_CONFIG_FILE=/dev/null \
  $VENV/bin/python -m pip install \
  -r /root/autodl-tmp/public/iws/status/sglang-cu130-runtime-reqs.txt
```

补齐 CUDA 13 JIT 编译链：

```bash
PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp TEMP=$CACHE/tmp TMP=$CACHE/tmp PIP_CONFIG_FILE=/dev/null \
  $VENV/bin/python -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-cuda-cccl==13.0.85

PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp TEMP=$CACHE/tmp TMP=$CACHE/tmp PIP_CONFIG_FILE=/dev/null \
  $VENV/bin/python -m pip install --force-reinstall \
  -i https://mirrors.aliyun.com/pypi/simple \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88
```

最终核验：

```bash
PIP_CACHE_DIR=$CACHE/pip PIP_CONFIG_FILE=/dev/null $VENV/bin/python -m pip check

$VENV/bin/python - <<'PY'
import importlib.metadata as md
for p in [
    "torch", "torch_c_dlpack_ext", "triton",
    "flashinfer-python", "flashinfer-cubin", "sglang-kernel",
    "nvidia-cuda-nvcc", "nvidia-cuda-cccl", "nvidia-nvvm",
    "nvidia-cuda-crt", "nvidia-cudnn-cu13", "transformers",
]:
    print(p, md.version(p))
PY
```

当前版本应为：

```text
torch 2.11.0+cu130
torch_c_dlpack_ext 0.1.5
triton 3.6.0
flashinfer-python 0.6.11.post1
flashinfer-cubin 0.6.11.post1
sglang-kernel 0.4.2.post1
nvidia-cuda-nvcc 13.0.88
nvidia-cuda-cccl 13.0.85
nvidia-nvvm 13.0.88
nvidia-cuda-crt 13.0.88
nvidia-cudnn-cu13 9.19.0.56
transformers 5.6.0
```

## 创建 CUDA wrapper

PyPI CUDA wheel 的目录是 `nvidia/cu13/lib`，但 SGLang/tvm_ffi JIT 链接命令会找传统 CUDA 布局 `CUDA_HOME/lib64`，并使用 `-lcudart`。因此需要 wrapper：

```bash
BASE=/root/autodl-tmp/public/iws
VENV=$BASE/venvs/sglang-py312-cu130
REAL=$VENV/lib/python3.12/site-packages/nvidia/cu13
WRAP=$BASE/toolchains/cuda-cu130

rm -rf "$WRAP"
mkdir -p "$WRAP" "$WRAP/lib64"
ln -s "$REAL/bin" "$WRAP/bin"
ln -s "$REAL/include" "$WRAP/include"
ln -s "$REAL/lib" "$WRAP/lib"
for f in "$REAL/lib"/*; do
  ln -sfn "$f" "$WRAP/lib64/$(basename "$f")"
done
ln -sfn "$REAL/lib/libcudart.so.13" "$WRAP/lib64/libcudart.so"

"$WRAP/bin/nvcc" --version
ls -l "$WRAP/lib64"/libcudart.so*
```

## 启动服务

启动命令详见 `sglang_manual_ops.md`。人工开发时推荐先执行：

```bash
bash /root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh
```

之后直接输入原生命令：

```bash
python -m sglang.launch_server --model-path "$MODEL_PATH" --host 0.0.0.0 --port 30000 <其他参数...>
```

核心要求：

```text
使用 /usr/bin/env -i
设置 CUDA_HOME/CUDA_PATH/CUDA_LIB_PATH 到 /root/autodl-tmp/public/iws/toolchains/cuda-cu130
设置 LD_LIBRARY_PATH 到 wrapper lib64 和 venv cudnn lib
设置 PYTHONPATH 到 /root/autodl-tmp/public/iws/projects/sglang/python
设置所有 cache/tmp 到 /root/autodl-tmp/public/iws/caches/sglang-cu130
```

`CUDA_LIB_PATH` 不能漏。漏掉后 `flashinfer.jit` 会默认检查 `/usr/local/cuda/targets/x86_64-linux/lib`，从而把系统 CUDA 12 runtime 拉进进程。

## 验证服务

```bash
curl -fsS http://127.0.0.1:30000/get_model_info
curl -fsS http://127.0.0.1:30000/fault_tolerance/status
curl -fsS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello, my name is","sampling_params":{"max_new_tokens":8,"temperature":0}}'
```

当前成功样例：

```text
PID: 118217
log: /root/autodl-tmp/public/iws/logs/sglang/server-firstft-human-enter-20260521-154352.log
ready_after: 85s
/get_model_info: 200
/fault_tolerance/status: enabled=true, state=RUNNING, dp_size=2, accepting_requests=true
/generate: 200
```

这次验证前已停止旧服务 `108493`。当前服务是通过 `enter-sglang-cu130.sh` 进入人工环境后，直接执行 `python -m sglang.launch_server ...` 拉起的，不依赖 agent helper 函数。

## 验证 CUDA 隔离

```bash
PID=$(cat /root/autodl-tmp/public/iws/status/sglang-server.pid)

tr '\0' '\n' < /proc/$PID/environ | grep -E '^(VIRTUAL_ENV|PYTHONPATH|PATH|LD_|CUDA|HF_HOME|TRITON_CACHE_DIR|TORCH_EXTENSIONS_DIR)=' | sort

grep -E 'libcudart|libcudnn|libcublas|libnvrtc|libcuda' /proc/$PID/maps | awk '{print $NF}' | sort -u

grep -E '/root/miniconda3|/usr/local/cuda' /proc/$PID/maps | grep -E 'libcudart|libcudnn|libcublas|libnvrtc' || echo "no base/system CUDA runtime libs"
```

最终应看到：

```text
libcudart.so.13 来自 venv nvidia/cu13/lib
libcudnn.so.9 来自 venv nvidia/cudnn/lib
libcublas.so.13 来自 venv nvidia/cu13/lib
libnvrtc.so.13 来自 venv nvidia/cu13/lib
没有 /root/miniconda3 或 /usr/local/cuda 下的 CUDA runtime/cuDNN/cuBLAS/NVRTC
```

`/usr/lib/x86_64-linux-gnu/libcuda.so.*` 是宿主机 driver，出现是正常的。

## 常见失败与处理

`nvcc: not found`：

```text
缺 nvidia-cuda-nvcc。安装 nvidia-cuda-nvcc==13.0.88。
```

`fatal error: nv/target: No such file or directory`：

```text
缺 CUDA CCCL 头文件。安装 nvidia-cuda-cccl==13.0.85。
```

`Unsupported .version 9.2; current version is '9.0'`：

```text
nvvm/crt 与 nvcc/ptxas 版本不一致。将 nvidia-nvvm 和 nvidia-cuda-crt force reinstall 到 13.0.88。
```

`/usr/bin/ld: cannot find -lcudart`：

```text
CUDA wrapper 的 lib64 缺 unversioned libcudart.so。创建 lib64/libcudart.so -> venv lib/libcudart.so.13。
```

进程 maps 里出现 `/usr/local/cuda-12.8/.../libcudart.so.12`：

```text
启动时漏了 CUDA_LIB_PATH。设置 CUDA_LIB_PATH=/root/autodl-tmp/public/iws/toolchains/cuda-cu130/lib64。
```

进程 maps 里出现 `/root/miniconda3/.../libcudart.so.13`：

```text
继承了 base LD_PRELOAD 或 LD_LIBRARY_PATH。使用 /usr/bin/env -i 重启。
```

## 仍需关注

`/fault_tolerance/status` 当前返回：

```text
enabled=true
state=RUNNING
dp_size=2
accepting_requests=true
sentinels={}
components={}
```

服务可用，但 `sentinels/components` 为空对 FT 语义可疑，后续 review `first FT` 时应继续追。
