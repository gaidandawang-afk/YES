# SGLang first FT / cu130 环境踩坑复盘

更新时间：2026-05-21

这份文档覆盖记录到目前为止踩过的坑、原因判断、最终解决方案和仍需关注的问题。主线是：围绕 `first FT` commit，最终在 AutoDL 服务器上用 fresh cu130 venv 成功拉起 SGLang `dp=2` + FT。

## 当前里程碑

服务器当前状态：

```text
服务器仓库: /root/autodl-tmp/public/iws/projects/sglang
当前 commit: 8a6c8d74b first FT
模型路径: /root/autodl-tmp/models/Qwen3-30B-A3B
服务 PID: 118217
服务端口: 30000
日志: /root/autodl-tmp/public/iws/logs/sglang/server-firstft-human-enter-20260521-154352.log
```

本次最终校验前已停止旧服务 `108493`。当前服务是从 `enter-sglang-cu130.sh` 进入人工环境后，直接输入 `python -m sglang.launch_server ...` 启动的，因此能证明人工入口脚本本身有效。

已验证接口：

```text
GET  /get_model_info              200
GET  /fault_tolerance/status      200
POST /generate                    200
POST /v1/chat/completions         200
```

`/fault_tolerance/status` 当前关键字段：

```text
enabled=true
state=RUNNING
topology.dp_size=2
accepting_requests=true
sentinels={}
components={}
```

服务已经能正常推理，但 `sentinels/components` 为空仍然可疑，后续 review FT 代码时要继续追。

`first FT` 之后曾叠过的三个临时 patch 已从当前工作状态中移除：

```text
116dc1d02 Handle missing FlashInfer MXInt4 symbols
494733722 Defer unavailable FlashInfer FP8 import
1bfd903ba Allow disabling Triton attention PDL
```

fresh cu130 venv 中对应 FlashInfer 符号和 `launch_server --help` 均已验证正常，这些 patch 不再必要；其中还改动了非 FT 场景代码，继续保留会干扰后续 review。

## 最终正确工作流

当前正确环境：

```text
venv:       /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130
cache:      /root/autodl-tmp/public/iws/caches/sglang-cu130
toolchain:  /root/autodl-tmp/public/iws/toolchains/cuda-cu130
project:    /root/autodl-tmp/public/iws/projects/sglang
```

核心原则：

```text
1. 不继承 base site-packages。
2. 不在 /root/miniconda3 里安装或改包。
3. 不继承 base LD_PRELOAD。
4. 不让 FlashInfer 或 JIT 编译链回落到 /usr/local/cuda-12.8。
5. 所有缓存、临时文件、日志都在 /root/autodl-tmp/public/iws 下。
6. 启动服务用 /usr/bin/env -i 列出需要的环境变量。
```

最终关键版本：

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

## 坑 1：差点继承 base 环境

现象：

第一次创建新 venv 时用了 `--system-site-packages`，这会把 `/root/miniconda3` 的包暴露给新 venv。

为什么不合理：

用户明确要求重新创建一个符合当前 SGLang 要求的虚拟环境，不继承任何环境。继承 base 会让问题变得不可复现：到底是新 venv 的包工作，还是 base 的包工作，很难判断。

处理：

```text
删除错误 venv。
重新执行 /root/miniconda3/bin/python -m venv /root/autodl-tmp/public/iws/venvs/sglang-py312-cu130。
确认 pyvenv.cfg 中 include-system-site-packages = false。
```

注意：

`venv` 由 `/root/miniconda3/bin/python` 创建，所以 Python 标准库、`libpython`、`lib-dynload` 路径仍可能显示 `/root/miniconda3`。这不是继承三方包。真正要看的是：

```text
site-packages 是否只在新 venv 下
CUDA/cuDNN/cuBLAS/NVRTC 是否从新 venv 加载
LD_PRELOAD/LD_LIBRARY_PATH 是否干净
```

## 坑 2：base shell 自带 LD_PRELOAD

现象：

远端 root shell 环境里有：

```text
LD_PRELOAD=/root/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.13
```

如果直接在这个 shell 里启动服务，即使 venv 是新的，也可能把 base 的 CUDA runtime preload 进来。

处理：

最终启动使用：

```bash
/usr/bin/env -i ...
```

明确列出 `PATH`、`PYTHONPATH`、`CUDA_HOME`、`CUDA_PATH`、`CUDA_LIB_PATH`、`LD_LIBRARY_PATH`、cache/tmp/HF/Torch/Triton 变量，不继承父 shell。

## 坑 3：只装 runtime 不够，SGLang JIT 需要 nvcc

现象：

clean-env 启动时，SGLang/tvm_ffi 编译 fused rope JIT kernel，报错：

```text
/root/.../nvidia/cu13/bin/nvcc: not found
ninja exited with status 127
```

原因：

PyTorch/cu130 runtime 包能跑 torch，但 SGLang 的 JIT kernel 还需要 CUDA 编译器。

处理：

安装：

```bash
nvidia-cuda-nvcc==13.0.88
```

Aliyun 镜像有正确包名 `nvidia-cuda-nvcc`。之前误查 `nvidia-cuda-nvcc-cu13` 只看到一个无用占位版本。

## 坑 4：nvcc 有了但缺 `<nv/target>`

现象：

补了 nvcc 后，编译失败：

```text
fatal error: nv/target: No such file or directory
```

原因：

缺 CUDA C++ Core Libraries / CCCL 头文件。CUDA 13 的 `cuda_fp16.h` 等头文件会 include `<nv/target>`。

处理：

安装：

```bash
nvidia-cuda-cccl==13.0.85
```

安装后新 venv 中出现：

```text
/root/autodl-tmp/public/iws/venvs/sglang-py312-cu130/lib/python3.12/site-packages/nvidia/cu13/include/nv/target
```

## 坑 5：nvvm/crt 自动拉到 13.2，导致 PTX 版本不匹配

现象：

编译时出现：

```text
ptxas fatal: Unsupported .version 9.2; current version is '9.0'
```

原因：

`nvidia-cuda-nvcc==13.0.88` 安装时自动拉了：

```text
nvidia-nvvm 13.2.78
nvidia-cuda-crt 13.2.78
```

导致前端生成 PTX 9.2，而实际 `ptxas 13.0.88` 只支持 PTX 9.0。

处理：

强制对齐：

```bash
pip install --force-reinstall nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88
```

最终编译链：

```text
nvidia-cuda-nvcc 13.0.88
nvidia-nvvm 13.0.88
nvidia-cuda-crt 13.0.88
nvidia-cuda-cccl 13.0.85
```

## 坑 6：PyPI CUDA wheel 没有传统 `lib64/libcudart.so`

现象：

JIT 编译已经能生成 object，但链接时报错：

```text
/usr/bin/ld: cannot find -lcudart
```

原因：

PyPI CUDA 13 wheel 把库放在：

```text
.../site-packages/nvidia/cu13/lib
```

但 tvm_ffi/JIT 链接命令按传统 CUDA layout 找：

```text
CUDA_HOME/lib64
```

并使用无版本名：

```text
-lcudart
```

处理：

创建 wrapper：

```text
/root/autodl-tmp/public/iws/toolchains/cuda-cu130/bin     -> venv nvidia/cu13/bin
/root/autodl-tmp/public/iws/toolchains/cuda-cu130/include -> venv nvidia/cu13/include
/root/autodl-tmp/public/iws/toolchains/cuda-cu130/lib     -> venv nvidia/cu13/lib
/root/autodl-tmp/public/iws/toolchains/cuda-cu130/lib64   -> wrapper links to venv nvidia/cu13/lib
```

并补：

```text
lib64/libcudart.so -> venv nvidia/cu13/lib/libcudart.so.13
```

这没有修改 package 本体，也没有碰 base。

## 坑 7：FlashInfer 默认从 `/usr/local/cuda` 加载 CUDA 12 runtime

现象：

服务能启动，但 `/proc/$PID/maps` 里仍有：

```text
/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12.8.90
```

原因：

`flashinfer/jit/__init__.py` 逻辑：

```text
CUDA_LIB_PATH 默认 /usr/local/cuda/targets/x86_64-linux/lib/
如果存在 libcudart.so.12，就 ctypes.CDLL(..., RTLD_GLOBAL)
```

服务器的 `/usr/local/cuda` 指向 CUDA 12.8，于是 FlashInfer 把 CUDA 12 runtime 拉进来了。

处理：

启动时设置：

```text
CUDA_LIB_PATH=/root/autodl-tmp/public/iws/toolchains/cuda-cu130/lib64
```

最终验证：

```text
没有 /usr/local/cuda 下的 libcudart/cuDNN/cuBLAS/NVRTC
CUDA runtime/cuDNN/cuBLAS/NVRTC 都来自新 venv
```

## 坑 8：老环境能跑不代表环境合理

现象：

旧 venv 曾经在加 workaround 后能启动，但依赖链混乱：

```text
torch 2.8/cu128-ish
Triton 3.4
sgl-kernel 0.3.17
需要 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
需要禁用 cuda graph / piecewise cuda graph / overlap schedule
```

这只能说明找到了一条能跑通的兼容路径，不代表环境符合当前 SGLang 要求。

最终取舍：

废弃旧环境作为默认路径，改用 fresh cu130 venv。新环境不需要旧 workaround，能跑 CUDA graph/piecewise graph，并能使用 `sglang-kernel==0.4.2.post1`。

## 坑 9：根分区太小，安装必须全程控 cache/tmp

现象：

服务器 `/` 约 99% 使用率，只有约 448M 空闲。任何 pip cache、临时编译文件、HF cache 写到默认位置都可能把根分区打满。

处理：

所有安装和启动都设置：

```text
HOME=/root/autodl-tmp/public/iws/home
XDG_CACHE_HOME=/root/autodl-tmp/public/iws/caches/sglang-cu130
TMPDIR=/root/autodl-tmp/public/iws/caches/sglang-cu130/tmp
PIP_CACHE_DIR=/root/autodl-tmp/public/iws/caches/sglang-cu130/pip
HF_HOME=/root/autodl-tmp/public/iws/caches/sglang-cu130/huggingface
TORCH_EXTENSIONS_DIR=/root/autodl-tmp/public/iws/caches/sglang-cu130/torch_extensions
TRITON_CACHE_DIR=/root/autodl-tmp/public/iws/caches/sglang-cu130/triton
CUDA_CACHE_PATH=/root/autodl-tmp/public/iws/caches/sglang-cu130/nv
```

安装后空间：

```text
/                  30G used 99%, 448M free
/root/autodl-tmp   432G used 27%, 319G free
venv size          9.4G
cache size         4.1G
```

## 坑 10：PyPI 直连很慢，镜像源更可靠

现象：

`nvidia-cuda-nvcc==13.0.88` 先从 PyPI 直连安装，十几分钟仍未结束。

处理：

确认 Aliyun 镜像有正确包名后，停止慢进程，改用：

```text
https://mirrors.aliyun.com/pypi/simple
```

安装 37MB + 64MB 的相关 wheel 约 50 秒完成。

## 坑 11：push 到服务器不等于服务器工作区自动更新

现象：

本地 push 到：

```text
ssh://root@region-42.seetacloud.com:38651/root/autodl-tmp/public/iws/projects/sglang
```

可以成功，但服务器 repo 是非 bare 工作区。push 更新 refs，不会自动让正在运行的 working tree 切到新 commit。

处理：

正确流程：

```text
本地创建 codex/<task> 分支
本地 commit
push 到 origin refs/heads/codex/<task>
SSH 到服务器
确认 git status --short 干净
git switch codex/<task> 或 git switch --detach <commit>
按 clean-env 命令重启服务
```

## 坑 12：repo-local SSH wrapper 会影响同仓库内所有 SSH remote

当前本地 repo 配置：

```text
core.sshCommand=C:/Users/59699/Documents/sglang/.ssh/sglang_autodl_git_ssh.cmd
```

这让 AutoDL push 可用，但如果在同一个 repo 添加：

```text
git@github.com:gaidandawang-afk/sglang.git
```

GitHub SSH 也会使用这把 AutoDL key。它不会强行连到 AutoDL host，但 GitHub 未必接受这把 key。

建议：

```text
GitHub remote 优先用 HTTPS。
如果必须用 SSH，写 host-aware wrapper 或单次命令使用 GIT_SSH_COMMAND。
```

## first FT 相关代码风险

这些不是环境坑，而是 review `first FT` 时发现的功能风险，仍未因为环境跑通而消失。

### 1. Retry 没有 sentinel 也可能返回成功

`SentinelManager.retry()` 通过 `_issue_command()` 向已注册 sentinel 发命令。如果没有 sentinel 注册，目标列表为空，而 `_all_success([])` 会被视为成功。

风险：

```text
retry 可能没有恢复任何 scheduler，却把状态设回 RUNNING 并重新放开 admission。
```

### 2. fault 后状态可能卡在 `ABORTING_COMM`

fault 上报后，manager 设置状态为 `ABORTING_COMM` 并发出 `HARD_ABORT_COMM`，但异步 command result 没有驱动全局状态转到稳定态。

风险：

```text
/fault_tolerance/status 可能长时间停在 ABORTING_COMM。
```

### 3. heartbeat timeout 可能误杀正常长耗时 forward

scheduler heartbeat 只在 `recv_requests()` 开头 feed。长 prefill、CUDA graph capture、大 batch 或阻塞 forward 超过默认 10 秒时，可能被 sentinel 误判为 stall。

风险：

```text
FT 在正常推理中误触发 hard pause / abort communicator。
```

### 4. 当前 dp=2 status 里 sentinel/component 为空

环境最终跑通后仍看到：

```text
sentinels={}
components={}
```

风险：

```text
FT API 显示 enabled/RUNNING，但实际 sentinel 注册可能没有发生或没有被 manager 观测到。
这会和 retry 空目标成功的问题叠加。
```

## 当前建议

后续继续任务时按这个顺序走：

```text
1. 以 fresh cu130 venv 为唯一默认环境。
2. 人工登录后先执行 bash /root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh。
3. 日常开发改参数时，直接用 python -m sglang.launch_server ...。
4. agent/debug helper 单独放在 agent-sglang-cu130.sh，不进入人工 shell。
5. 如需手动改参数，先 source /root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh。
6. 每次启动后检查 /fault_tolerance/status、/generate 和 CUDA maps。
7. 代码修改在本地 F:\theend\repo\sglang-autodl 完成。
8. push 到 AutoDL 分支后，SSH 到服务器显式切换 commit。
9. 继续 review/fix FT 时，优先处理 sentinel 注册为空和 retry 空目标成功。
```

新增人工入口脚本：

```text
/root/autodl-tmp/public/iws/scripts/env-sglang-cu130.sh
/root/autodl-tmp/public/iws/scripts/enter-sglang-cu130.sh
/root/autodl-tmp/public/iws/scripts/bashrc-sglang-cu130
```

相关文档：

```text
C:\Users\59699\Documents\sglang\sglang_iws_deploy_runbook.md
C:\Users\59699\Documents\sglang\sglang_manual_ops.md
C:\Users\59699\Documents\sglang\sglang_git_push_workflow.md
```
