SGLang 的做法不是像 `vllm-ascend` 那样单独做一个推理引擎插件仓库，而是：

**主仓库内置 Ascend/NPU backend，运行时主链路不分叉；外部只依赖 NPU runtime/kernel 包。**

也就是没有单独的 `NPUWorker` / `NPUModelRunner` 体系。SGLang 仍然使用同一套 `Scheduler -> TpModelWorker -> ModelRunner -> ForwardBatch` 推理链路，只在 backend、算子、KV cache、graph、通信、量化等层面根据 `is_npu()` / `--device npu` 切到 Ascend 实现。

**核心接入方式**
- 设备识别：`is_npu()` 检测 `torch.npu`，设备类型进入 `npu` 分支。
- 参数默认值：`ServerArgs._handle_npu_backends()` 调用 [utils.py](D:/Workspace/codex/projects/sglang/sglang/python/sglang/srt/hardware_backend/npu/utils.py:46)，强制 `attention_backend/prefill/decode = "ascend"`，默认 `page_size=128`，按 NPU 显存设置 `chunked_prefill_size` / `cuda_graph_max_bs`，并禁用 custom all-reduce。
- backend 初始化：[init_npu_backend](D:/Workspace/codex/projects/sglang/sglang/python/sglang/srt/hardware_backend/npu/utils.py:92) 直接导入 `torch_npu`、`sgl_kernel_npu`、`torch_npu.contrib.transfer_to_npu`，设置 `torch_npu.npu.config.allow_internal_format=True`，关闭 `jit_compile`。
- Attention backend：`ATTENTION_BACKENDS["ascend"]` 注册到 [attention_registry.py](D:/Workspace/codex/projects/sglang/sglang/python/sglang/srt/layers/attention/attention_registry.py:55)，实际实现是 `AscendAttnBackend`，调用 `torch_npu` / `torch.ops.npu` / `torch_npu.atb` 的 FIA、paged attention、sparse attention、ring MLA 等算子。
- Graph：仍走 `ModelRunner.init_device_graphs()`，但 NPU 分支选择 [NPUGraphRunner](D:/Workspace/codex/projects/sglang/sglang/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py:73)，底层用 `torch.npu.NPUGraph()`。
- KV cache：`ModelRunnerKVCacheMixin` 在 NPU 下替换为 `NPUMHATokenToKVPool` / `NPUMLATokenToKVPool` / `NPUPagedTokenToKVPoolAllocator`，写 KV 用 `_npu_reshape_and_cache`、`npu_scatter_nd_update_` 和 `sgl_kernel_npu` allocator。
- 通信：`parallel_state.py` 对 `npu` 默认用 `hccl`，并创建 [NpuCommunicator](D:/Workspace/codex/projects/sglang/sglang/python/sglang/srt/distributed/device_communicators/npu_communicator.py:13)，支持 HCCL all-reduce/all-gather，以及 `npu_dynamic_quant` 后的 quant all-reduce。
- 算子分发：`MultiPlatformOp.dispatch_forward()` 会选 `forward_npu`，所以 activation、RMSNorm、RoPE、MoE、LoRA、sampling 等组件在类内部切到 NPU 算子。
- 安装形态：没有 `sglang-ascend` 这种引擎插件包；Ascend 安装使用 `python/pyproject_npu.toml`，但 `torch_npu`、CANN、`triton-ascend`、`sgl_kernel_npu`、可选 `memfabric-hybrid` 仍需按 Ascend 文档/镜像准备。

**和 vLLM-Ascend 的关键区别**
- vLLM：独立 `vllm-ascend` 包，通过插件注册替换 Worker/Runner/ops。
- SGLang：Ascend 是主仓库一等 backend，核心 Worker/Runner 不继承分叉；通过 `is_npu()` 分支、backend registry、`hardware_backend/npu/*` 和外部 NPU kernel 包完成适配。
- 外部包角色不同：`sgl_kernel_npu` 更像算子/kernel 包，不是完整推理引擎插件。MindSpore 模型另有 `sgl-mindspore`，但那是模型实现扩展，不是 SRT Ascend 运行时的主要适配方式。

结论：

**1. Ascend 不是完全靠额外 fork 跑的，而是已经进入 SGLang 主仓的内置后端；但高性能算子/依赖有独立包。**

SGLang 官方文档已经有独立的 **Ascend NPUs** 硬件页，源码安装路径也是直接克隆 `sgl-project/sglang`，然后用 `python/pyproject_npu.toml` 替换默认 `pyproject.toml`，再 `pip install -e python[all_npu]`。运行时也直接用 `python3 -m sglang.launch_server ... --device npu --attention-backend ascend`。这说明 Ascend 支持是 **主仓内置路径**，不是必须依赖一个外部 `sglang-ascend` fork。([SGLang文档][1])

但是，Ascend 的算子库不是全塞在 SGLang 主仓里。官方还有独立的 **sgl-kernel-npu**，它被描述为 SGLang 面向 Ascend NPU 的官方 kernel library，包含 NPU 推理 kernels 以及 DeepEP-Ascend 这类 EP 通信加速组件。([GitHub][2]) 文档里也明确列出 CANN、`torch_npu`、`triton-ascend`、`memfabric-hybrid`、SGLang NPU Kernels、DeepEP-compatible Library 等依赖。([SGLang文档][1])

所以可以理解成：

```text
sgl-project/sglang 主仓
  ├── SRT 调度器 / model runner / memory pool / attention registry 等通用框架
  ├── sglang/srt/hardware_backend/npu/...   # Ascend 专用后端代码
  ├── pyproject_npu.toml                    # NPU 安装依赖入口
  └── 通过 --device npu / --attention-backend ascend 激活

sgl-project/sgl-kernel-npu 独立包
  ├── Ascend attention / norm / activation / LoRA / MoE kernels
  └── DeepEP-Ascend / EP 通信相关能力
```

**2. 不同 platform 目前是“混合机制”：主仓内置后端仍大量靠 `is_npu()` / `is_cuda()` 分支，外部硬件才走 plugin/current_platform。**

现在 SGLang 不是完全统一成一个干净的 `Platform` 多态抽象。官方 plugin 文档说得比较直接：plugin 系统目前主要面向 **out-of-tree hardware platforms**，而主仓内置的 CUDA / ROCm / NPU / XPU 等路径仍然继续使用现有的 `is_cuda()`、`is_npu()` 等工具函数；后续目标才是把分散的 `if device == "cuda" ... elif device == "npu"` 逐步迁移到统一的 platform interface。([SGLang文档][3])

从代码形态看，大致有四类分发：

第一类是 **CLI/config 显式选择**。例如 Ascend 启动时常见参数是：

```bash
python3 -m sglang.launch_server \
  --model-path ... \
  --device npu \
  --attention-backend ascend \
  --sampling-backend ascend
```

PD disaggregation 场景还会用 `--disaggregation-transfer-backend ascend`。这些参数决定了 device、attention backend、transfer backend 等路径。([SGLang文档][1])

第二类是 **`is_npu()` / `_is_npu` 这种 import-time guard**。例如 `model_runner.py` 会导入 `NPUGraphRunner`，然后通过 `_is_npu = is_npu()` 判断是否初始化 NPU backend；如果是 out-of-tree platform，才调用 `current_platform.init_backend()`。([GitHub][4])

第三类是 **registry/factory 分发**。例如 graph runner 创建时，OOT platform 先走 `current_platform.get_graph_runner_cls()`；否则内置映射里 `"npu"` 对应 `NPUGraphRunner`，`"cpu"` 对应 `CPUGraphRunner`，默认 CUDA graph runner。([GitHub][4])

第四类是 **算子级 multi-platform dispatch**。`MultiPlatformOp` 里有 `forward_cuda()`、`forward_npu()`、`forward_hip()`、`forward_xpu()`、`forward_cpu()` 等方法。OOT platform 会先查 `current_platform.get_dispatch_key_name()` 和 OOT registry；内置平台则继续按 `_is_cuda / _is_hip / _is_npu / _is_xpu / _is_musa` 做 if/elif 分发。([GitHub][5])

简化成架构图就是：

```text
启动参数 / 环境
  ├── --device npu
  ├── --attention-backend ascend
  ├── --sampling-backend ascend
  └── --disaggregation-transfer-backend ascend
          │
          ▼
SGLang SRT 通用框架
  ├── Scheduler / TpModelWorker / ModelRunner
  ├── Attention registry
  ├── Memory pool / KV cache pool
  ├── Distributed parallel state
  └── Quantization / MoE / sampling
          │
          ├── 内置平台路径
          │     ├── is_cuda()
          │     ├── is_hip()
          │     ├── is_npu()
          │     ├── is_xpu()
          │     └── if/elif + registry/factory
          │
          └── OOT plugin 路径
                ├── entry_points("sglang.srt.platforms")
                ├── current_platform
                ├── SGLANG_PLATFORM 选择平台
                └── platform method / hook / registry
```

**所以对你的两个问题可以更精确地回答：**

| 问题                 | 答案                                                                                                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ascend 是额外代码仓还是内置？ | **主流程内置在 SGLang 主仓**，有 `hardware_backend/npu`、`pyproject_npu.toml`、`--device npu`、`--attention-backend ascend` 等官方路径；但 NPU 高性能 kernel、DeepEP-Ascend 等算子/通信库在 **sgl-kernel-npu** 等独立依赖包里。      |
| 不同 platform 怎么区分？  | **当前内置平台主要还是 `is_xxx()` + if/elif + backend registry/factory 混合分发**。SGLang 已有 `current_platform` / plugin system，但官方文档明确说它当前主要服务 OOT 平台，内置 CUDA/ROCm/NPU/XPU 还没有完全迁移到统一多态 platform interface。 |

一句话总结：**SGLang 对 Ascend 的兼容不是外置补丁式 fork，而是“主仓内置后端 + 独立 NPU kernel 包”；平台抽象仍处于过渡态，内置 NPU 路径大量保留 `is_npu()` 分支，OOT 硬件才更依赖 plugin/current_platform 机制。**

[1]: https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu "SGLang installation with NPUs support - SGLang Documentation"
[2]: https://github.com/sgl-project/sgl-kernel-npu?utm_source=chatgpt.com "sgl-project/sgl-kernel-npu: SGLang kernel library for NPU"
[3]: https://docs.sglang.io/docs/hardware-platforms/plugin "SGLang Plugin System - SGLang Documentation"
[4]: https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/model_executor/model_runner.py "raw.githubusercontent.com"
[5]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/utils/multi_platform.py "sglang/python/sglang/srt/layers/utils/multi_platform.py at main · sgl-project/sglang · GitHub"
