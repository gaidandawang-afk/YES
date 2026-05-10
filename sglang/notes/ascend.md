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
