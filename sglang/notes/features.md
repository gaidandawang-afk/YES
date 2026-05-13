# AsyncScheduling

## vllm

`async_scheduling` 解决的是 vLLM V1 decode 主循环里的“token gap”问题：同步模式下每一步都要按顺序完成 `schedule -> execute_model -> sample/copy token -> scheduler.update_from_output -> 下一次 schedule`。当 decode step 很短、并发 batch 较大或 GPU 很快时，CPU 调度、Python/GIL、采样结果从 GPU 拷回 CPU、调度器状态更新会让两次 GPU decode 之间出现空档，GPU 利用率下降。官方配置说明也直接写明它用于避免 GPU utilization gaps，从而改善 latency/throughput；早期设计 issue 里提到 Llama2-7B、bs=256、ShareGPT 场景下两次 decode 间 gap 约 5-6ms，希望压到 200-300us 量级。

它不是“HTTP/AsyncLLMEngine 的异步请求能力”，而是 engine 内部 CPU 调度与 GPU 执行的流水化。

**适用场景**

更适合：

- 高并发 decode，尤其每步 decode 计算较短、CPU 调度占比明显的场景。
- chunked prefill + decode 混合调度时，希望减少模型执行批次之间的空窗。
- 当前本地代码里，显式开启时只允许部分 executor：`uni`、`mp`、`external_launcher`；不支持 `pipeline_parallel_size > 1`。
- speculative decoding 方面，本地代码已允许 EAGLE/MTP 类型；vLLM Ascend 文档也说明 v0.12.0rc1 起适配了 EAGLE async scheduler。但旧版 vLLM 文档仍写过 spec decoding 不支持，所以要以当前代码为准。

**核心实现**

1. 配置切换 scheduler

`SchedulerConfig.async_scheduling` 打开后，`get_scheduler_cls()` 返回 `AsyncScheduler`，否则返回普通 `Scheduler`。本地位置：  
[vllm/config/scheduler.py](M:/sync-to-yellow/vllm-blue/vllm/config/scheduler.py:133)

2. executor 允许并发两个 batch

`UniProcExecutor.max_concurrent_batches` 在 async scheduling 下返回 `2`，否则 `1`。EngineCore 看到 `max_concurrent_batches > 1` 就启用 `batch_queue`。  
[vllm/v1/executor/uniproc_executor.py](M:/sync-to-yellow/vllm-blue/vllm/v1/executor/uniproc_executor.py:59)  
[vllm/v1/engine/core.py](M:/sync-to-yellow/vllm-blue/vllm/v1/engine/core.py:178)

3. EngineCore 改成“先发下一批，再等上一批”

普通 `step()` 是调度后立刻等模型结果；`step_with_batch_queue()` 则先尝试 schedule 新 batch、非阻塞执行模型/采样，把 future 放进队列；只有队列满或没有新请求可调度时，才 `future.result()` 取最老结果并更新 scheduler。  
[vllm/v1/engine/core.py](M:/sync-to-yellow/vllm-blue/vllm/v1/engine/core.py:378)

这就是流水线：GPU 正在跑 batch N 时，CPU 已经可以准备 batch N+1。

4. 用 placeholder 解决“下一步调度时还不知道上一步 token”的依赖

异步调度最大的难点是：下一步 schedule 时，上一步采样 token 还没回到 scheduler。`AsyncScheduler._update_after_schedule()` 会在请求即将产生新 token 时增加 `request.num_output_placeholders`，把未来 token 先占位计入长度/KV 资源。等真实 token 回来后，`_update_request_with_output()` 再扣掉 placeholder，并更新 KV cache bookkeeping。  
[vllm/v1/core/sched/async_scheduler.py](M:/sync-to-yellow/vllm-blue/vllm/v1/core/sched/async_scheduler.py:11)

5. worker 侧避免 GPU token 先回 CPU 再发回 GPU

decode 下一步的输入 token 本质上就是上一步采样 token。async 模式下，GPU model runner 会把 `sampler_output.sampled_token_ids` 缓存在 GPU 上的 `prev_sampled_token_ids`，下一次 `_prepare_input_ids()` 直接在 GPU 上 copy/scatter 到 `input_ids`，避免调度器拿到 CPU token 后再传回 GPU。  
[vllm/v1/worker/gpu_model_runner.py](M:/sync-to-yellow/vllm-blue/vllm/v1/worker/gpu_model_runner.py:1185)  
[vllm/v1/worker/gpu_model_runner.py](M:/sync-to-yellow/vllm-blue/vllm/v1/worker/gpu_model_runner.py:2682)

6. 输出拷回 CPU 也异步化

`AsyncGPUModelRunnerOutput` 在单独 CUDA stream 上把 sampled tokens/logprobs 非阻塞拷到 CPU，并记录 event。`get_output()` 只有在 scheduler 真要消费输出时才 synchronize。  
[vllm/v1/worker/gpu_model_runner.py](M:/sync-to-yellow/vllm-blue/vllm/v1/worker/gpu_model_runner.py:190)

7. structured output 要特殊处理

如果有结构化输出 grammar，并且还存在 placeholder，则 grammar bitmask 依赖真实 token，不能提前采样。代码用 `pending_structured_output_tokens` 标记这种情况，在 `step_with_batch_queue()` 中延迟 `sample_tokens()`，等上一批输出处理完再做。  
[vllm/v1/core/sched/async_scheduler.py](M:/sync-to-yellow/vllm-blue/vllm/v1/core/sched/async_scheduler.py:18)  
[vllm/v1/engine/core.py](M:/sync-to-yellow/vllm-blue/vllm/v1/engine/core.py:418)

**一句话概括**

`async_scheduling` 的本质是把“调度下一步”和“等待上一步模型输出”解耦，用 batch queue、future、token placeholder、GPU 侧 sampled-token 缓存、异步 CPU copy 来隐藏 CPU 调度和同步开销，从而减少 decode step 之间的 GPU 空闲时间。

参考资料：vLLM scheduler 配置文档说明 async scheduling 用于减少 GPU utilization gaps（https://docs.vllm.ai/en/v0.12.0/api/vllm/config/scheduler/），以及 vLLM async scheduling 设计 issue 中对 token gap 和实现思路的描述（https://github.com/vllm-project/vllm/issues/10634）。

## SGLang

SGLang 里的 `overlap` 和 vLLM `async_scheduling` 目标相同：减少 decode step 之间 CPU 调度、采样输出同步、状态更新造成的 GPU 空档。但实现方式不完全一样。

**一句话对齐**

vLLM async scheduling 是：

```text
CPU 提前 schedule batch N+1
GPU 还在跑 batch N
用 placeholder 解决 “N 的 token 还没回来但 N+1 要先排队”
```

SGLang overlap schedule 也是这个思想，但它的 placeholder 更“GPU 化”：

```text
batch N 输出 token 不急着回 CPU
先写进 GPU FutureMap
下一轮 batch N+1 的 input_ids 里先放一个负数 future index
真正 forward 前，在 GPU 上把 future index 替换成真实 token id
```

对应代码主线：

- 开关：`enable_overlap = not disable_overlap_schedule`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:329)

- loop 选择：非 PP、非 disagg 时走 `event_loop_overlap()`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:3444)

- 普通 loop：`schedule -> run_batch -> process_result`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1286)

- overlap loop：`schedule 当前 batch -> launch 当前 GPU forward -> process 上一个 batch result -> 必要时 sample 当前 batch`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1314)

**SGLang overlap schedule 的时间线**

普通同步模式：

```text
step N:
  recv/process requests
  schedule batch N
  GPU forward batch N
  copy token/logprob to CPU
  scheduler/process_result 更新 request、KV bookkeeping、输出
  schedule batch N+1
```

overlap 模式：

```text
iteration N:
  CPU schedule batch N
  GPU forward batch N 被 enqueue 到 forward_stream
  result_queue 保存 batch N 的 result 句柄
  CPU 处理 batch N-1 的结果
  如果需要，补做 batch N 的 delayed sample

iteration N+1:
  CPU 在 batch N 结果完全处理前，先 schedule batch N+1
  batch N+1 的 input_ids 里可能包含 “future token”
  GPU forward batch N+1 前，把 future token 替换成 batch N 真实采样 token
```

这里关键不是 Python `Future`，而是 CUDA stream + GPU buffer：

```text
batch N sampled_token
  -> FutureMap.token_ids_buf[future_index]   # GPU 上
batch N+1 input_ids = -future_index          # 负数代表还未解析的 future token
  -> forward 前 resolve_future()
  -> input_ids 变成真实 token id
```

相关代码：

- 分配 future index：`future_map.alloc_future_indices(bs)`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2619)

- forward 前解析 future token：`future_map.resolve_future(model_worker_batch)`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2623)

- forward 后把真实 token 存回 FutureMap：`future_map.store_to_map(...)`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2632)

- `batch.output_ids = -future_indices.indices`，让调度器先拿 future placeholder 往下走  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2669)

- FutureMap 负数替换逻辑  
  [overlap_utils.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/overlap_utils.py:21)

**和 vLLM async_scheduling 的关键差异**

vLLM 更像：

```text
EngineCore 维护 batch_queue
executor 支持 max_concurrent_batches = 2
future.result() 延迟等待
scheduler 用 num_output_placeholders 占长度/KV
worker 缓存 sampled token
```

SGLang 更像：

```text
Scheduler 自己维护 result_queue
CUDA forward_stream 异步 enqueue 当前 batch
CPU schedule_stream 继续做调度/处理上批结果
FutureMap 用 GPU buffer 保存未来 token
input_ids 里用负数 future index 表示依赖
```

所以 SGLang 的 overlap 不只是“CPU future 队列”，而是把 token 依赖也搬到了 GPU 侧解决，减少：

```text
GPU sampled token -> CPU scheduler -> 再传回 GPU input_ids
```

这条往返。

**输出拷贝也异步化**

SGLang overlap 下，`GenerationBatchResult.copy_to_cpu()` 会用 `non_blocking=True` 把 next token、logprob、hidden states 等拷到 CPU，并记录 `copy_done` event。真正处理结果时才 synchronize。

代码：

- 异步 D2H copy：  
  [utils.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/utils.py:52)

- 处理结果前等待 copy 完成：  
  [scheduler_output_processor_mixin.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:129)

**grammar / structured output 的特殊点**

SGLang 也有类似 vLLM “structured output 不能盲目提前采样”的处理。

`event_loop_overlap()` 里注释写得很直接：当前 batch 的 sample 可能依赖上一个 batch 的结果，比如 grammar，所以要在上一个 batch 处理完之后再 `launch_batch_sample_if_needed()`。

```text
forward 当前 batch
process 上一个 batch result
再 sample 当前 batch
```

对应：

- `launch_batch_sample_if_needed()`  
  [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2731)

**SGLang 还有更深一层：TBO / SBO**

这点是 SGLang 和 vLLM async scheduling 最大的不同。vLLM async scheduling 主要解决 engine 主循环 token gap；SGLang 还做模型内部的 overlap，尤其针对 MoE。

SGLang overlap 可以分三层看：

```text
1. overlap schedule
   CPU 调度 / 结果处理 与 GPU forward 重叠

2. TBO: two batch overlap
   把一个 batch 切成 A/B 两个 micro-batch，在 layer 内按 stage 交错执行

3. SBO: single batch overlap
   一个 batch 内，把 MoE dispatch/combine/shared expert/down GEMM 等用 stream/event overlap
```

TBO 代码入口：

- `model_forward_maybe_tbo(...)`  
  [two_batch_overlap.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/batch_overlap/two_batch_overlap.py:819)

它大概做：

```text
原 batch
  -> split 成 micro-batch A / B
  -> A 先跑若干 stage
  -> A、B 交错跑 stage
  -> B 补齐剩余 stage
  -> merge 输出
```

核心执行器：

```text
execute_overlapped_operations(
  inputs_arr=[A, B],
  operations_arr=[ops, ops],
  delta_stages=[0, delta]
)
```

对应：

- [operations.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/batch_overlap/operations.py:30)

以 DeepSeek MoE decode 为例，它把一层拆成类似：

```text
comm_prepare_attn
attn_prepare
yield
attn_core
comm_prepare_mlp
gate
select_experts
yield
dispatch_a
shared_experts
yield
dispatch_b
experts
combine_a
yield
combine_b
yield
output
comm_postprocess_layer
```

代码：

- [operations_strategy.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/batch_overlap/operations_strategy.py:92)

这里的 `YieldOperation` 就是人为插入的“交错点”。

**SBO 从集合通信角度看**

SBO 主要是 MoE 内部：

```text
hidden states
  -> router/topk
  -> dispatch all-to-all
  -> expert compute
  -> combine all-to-all
  -> output/down/shared expert
```

SGLang 会尝试让这些阶段重叠，例如：

```text
dispatch 过程中跑 shared expert
combine 过程中跑 down GEMM
combine 和 shared expert 使用不同 CUDA stream
用 event 控制依赖
用 SM 数量切分通信/计算资源
```

入口之一：

- [single_batch_overlap.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/batch_overlap/single_batch_overlap.py:81)

DeepSeek MoE 里能看到：

- dispatch hook / combine hook  
  [deepseek_v2.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/models/deepseek_v2.py:785)

**所以结论是**

vLLM `async_scheduling` 主要是 engine 级别的 decode 流水化；SGLang 的 `overlap` 包含 engine 级别的 overlap schedule，但还进一步深入到 MoE forward 内部。

可以这样记：

```text
vLLM async_scheduling:
  隐藏 CPU scheduler / output sync gap

SGLang overlap schedule:
  隐藏 CPU scheduler / output sync gap
  用 FutureMap 在 GPU 上传递下一步 token

SGLang TBO:
  把 batch 切成两个 micro-batch，在模型层内部交错执行

SGLang SBO:
  在单个 MoE batch 内 overlap dispatch/combine/shared expert/GEMM
```

如果你是读 SGLang 代码，我建议先按这个顺序看：

```text
scheduler.event_loop_overlap
  -> scheduler.run_batch overlap 分支
  -> FutureMap
  -> GenerationBatchResult.copy_to_cpu
  -> scheduler_output_processor_mixin
  -> two_batch_overlap.py
  -> operations_strategy.py
  -> single_batch_overlap.py
  -> 具体 MoE model，如 deepseek_v2.py
```

# Overall

sglang中对应关系大概如下：

| vLLM 特性 | SGLang 对应/相近特性 | 是否等价 |
|---|---|---|
| `async_scheduling` | overlap scheduler / batch overlap / spec overlap | 不完全等价 |
| external launcher | 内置 multiprocessing launcher，另有 Ray / torchrun/checkpoint-engine 场景 | 不同抽象 |
| hybrid KV cache manager | hybrid SWA memory、Mamba cache、hybrid attention backend | 目标相近，实现不同 |
| dual batch overlap | `--enable-two-batch-overlap` / `--enable-single-batch-overlap` | 很接近 |
| disaggregated prefill | SGLang PD disaggregation | 都有，SGLang 这块很重 |
| KV offload / external KV | SGLang HiCache / storage backend | SGLang 很突出 |

**1. async scheduling**
vLLM 的 `async_scheduling` 是 SchedulerConfig 里的开关，官方说它用于避免 GPU utilization gap，改善 latency/throughput；启用后会选 `AsyncScheduler`。见 vLLM docs：`async_scheduling` 说明和 `get_scheduler_cls()` 分支在官方 API 文档里有写。

SGLang 没有同名 `--async-scheduling`，但有几类相近机制：

```text
普通 scheduler:
  event_loop_normal

CPU/GPU overlap:
  event_loop_overlap

PP:
  event_loop_pp

Spec V2 overlap:
  SGLANG_ENABLE_SPEC_V2

MoE overlap:
  two-batch overlap / single-batch overlap
```

你可以把 SGLang 的 `disable_overlap_schedule=False` 默认路径理解成：**调度/CPU 后处理尽量和 GPU forward 重叠**。它不是 vLLM 那个 AsyncScheduler 类，但目标类似：减少 GPU 等 CPU scheduler 的空洞。

SGLang 相关参数在本地：

```text
server_args.py:
  disable_overlap_schedule
  enable_two_batch_overlap
  enable_single_batch_overlap
```

**2. external launcher**
如果你说的是 vLLM 的 `entrypoints.launcher` 或外部进程管理方式，SGLang 的默认路线不同。

SGLang 默认是：

```text
python -m sglang.launch_server
  -> 主进程 HTTP / TokenizerManager
  -> 自己 fork Scheduler / Detokenizer / Worker 子进程
```

也就是说 SGLang 的 runtime launcher 是内置在 `Engine._launch_subprocesses` / `_launch_scheduler_processes` 里的，不需要单独外部 launcher 管每个 worker。

但 SGLang 也有几类“外部启动”相关能力：

```text
--use-ray
  用 Ray actor 启动 scheduler/worker

多节点:
  --dist-init-addr / node_rank / nnodes

checkpoint engine:
  文档里有 torchrun 方式，用独立 checkpoint workers 加载/分发权重

K8s / PD deployment:
  有 prefill worker / decode worker / router 分角色部署
```

所以结论是：**SGLang 没有完全同名的 external launcher 抽象；默认更偏一体化 launcher，但 Ray、多节点、checkpoint engine、PD 部署能覆盖类似运维需求。**

**3. hybrid KV cache / hybrid model**
vLLM 的 hybrid KV cache manager 是针对“同一个模型里有多种 KV cache 需求”的管理器，比如 full attention + sliding window/Mamba。官方文档说它会按 KV cache group 选择 coordinator，`HybridKVCacheCoordinator` 处理 full attention + 另一类 efficient attention 的情况。

SGLang 有相近但实现不同的东西：

```text
--swa-full-tokens-ratio
--disable-hybrid-swa-memory
```

用于 SWA/local attention 和 full attention 混合模型的 KV 内存比例/布局。比如 Llama4 文档里明确提到 hybrid kv cache，可以通过 `--swa-full-tokens-ratio` 调整 SWA layer KV tokens / full layer KV tokens ratio。

还有 Mamba/SSM 相关：

```text
--max-mamba-cache-size
--mamba-ssm-dtype
--mamba-full-memory-ratio
--mamba-scheduler-strategy
```

以及 hybrid attention backend：

```text
prefill attention backend 和 decode attention backend 可以不同
python/sglang/srt/layers/attention/hybrid_attn_backend.py
```

所以这里的对应关系是：

```text
vLLM:
  HybridKVCacheCoordinator / KVCacheGroup

SGLang:
  SWA hybrid memory + Mamba cache + hybrid attention backend
```

目标相近：**不要所有层都按 full attention 最大 KV 规格分配**。但 SGLang 不是照搬 vLLM 的 KVCacheCoordinator 结构。

**4. SGLang 比较突出的独有/强特性**
就你现在看的 MoE/并行方向，SGLang 有几块很值得单独看：

```text
DP Attention
  attention 用 DP/CP/TP，MoE 用 MOE_DP/EP/MOE_TP，通信域按层重解释

PD disaggregation
  prefill worker / decode worker 分离，配 Mooncake/NIXL/MORI 等 KV transfer

HiCache
  GPU KV + host memory + storage backend，包括 hf3fs/mooncake/eic/dynamic backend

MoE EP overlap
  two-batch overlap / single-batch overlap，dispatch/combine/GEMM overlap

Piecewise CUDA graph
  针对 prefill/decode 分段 capture，减少 launch overhead

Radix cache / chunked prefix cache
  prefix cache 是 SGLang 的核心路径之一
```

如果按你当前的阅读重点，我建议优先对齐这几条：

```text
vLLM async_scheduling
  -> SGLang event_loop_overlap / TBO / SBO

vLLM hybrid KV cache manager
  -> SGLang hybrid SWA memory + Mamba cache

vLLM external launcher
  -> SGLang internal multiprocessing launcher + Ray/PD/K8s/checkpoint engine

vLLM EP / disagg
  -> SGLang DP attention + DeepEP/Mooncake/NIXL/MORI + PD disaggregation + HiCache
```

参考源：
- vLLM SchedulerConfig `async_scheduling` 和 `disable_hybrid_kv_cache_manager`：https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM Hybrid KV Cache Manager：https://docs.vllm.ai/en/v0.12.0/design/hybrid_kv_cache_manager/
- vLLM launcher API：https://docs.vllm.ai/en/stable/api/vllm/entrypoints/launcher/