# SGLang Fault-Tolerance Retry 需求文档

## 0. 文档信息

- 文档目标：定义 SGLang fault-tolerance retry 能力的需求边界、控制模型和验收标准。
- 核心设计：主进程一个全局 `SentinelManager`，每个 scheduler 进程一个 `FaultSentinel` 常驻控制线程。
- 本轮范围：聚焦 fault 后挂起、hard pause、通信域重建和 retry 恢复；`scale_down` 作为后续扩展能力保留，不在本文档展开。
- 参考材料：SGLang 当前架构与 pause/continue 代码、`output/` 下现有 RFC、vLLM `ClientSentinel / EngineCoreSentinel / WorkerSentinel` 实现。

## 1. 背景

SGLang 当前关键子进程遇到未捕获异常时，通常通过 `SIGQUIT` 通知父进程并终止整棵进程树。这个 fail-stop 兜底简单，但对在线服务不够友好：外部平台缺少结构化状态，也无法在所有 rank 仍存活的情况下尝试 retry。

本轮设计把 fault 处理改为“先纳入容错框架，再由控制面决定”。当 FT 开启后，scheduler 相关 fault 不再默认触发 fail-stop，而是进入容错状态机：冻结入口、挂起 scheduler、必要时 hard-abort 通信域、暴露状态，并等待 retry 或后续管理动作。

## 2. 关键需求变化

过去的保守设计只捕获显式 `RecoverableFault`，未知异常仍 fail-stop。新的需求是：

```text
FT enabled 时，scheduler 执行路径中的异常都先进入 fault-tolerance framework。
默认行为是挂起和暴露状态，而不是立即 kill process tree。
```

这包括：

- scheduler Python exception。
- communication/collective timeout 或 communicator failure。
- scheduler main loop 检测到不可继续执行的 batch/runtime 状态。
- 上层平台主动下发 hard pause。
- FaultSentinel 检测到 scheduler heartbeat stall。

对于 native crash、进程直接退出、CUDA context fatal 等无法在当前进程内捕获的情况，主进程也不应在 FT enabled 时自动 kill 全部进程；它应记录组件 `EXITED/UNRESPONSIVE` 状态并保持管理面可查询，由外部平台决定 terminate 或后续恢复策略。

## 3. 为什么需要 Sentinel

vLLM 新增大量 Sentinel 的原因是主执行线程可能卡在 collective 或 busy loop 中。如果 pause/retry 命令也走主执行线程，故障时命令可能永远无法被处理。

SGLang 默认拓扑不同：`TpModelWorker / ModelRunner` 通常在 scheduler 进程内部，不需要拆出 vLLM 的三层 Sentinel。SGLang 只需要两层：

| 层级 | 组件 | 位置 | 职责 |
| --- | --- | --- | --- |
| 全局控制层 | `SentinelManager` | HTTP/Tokenizer 主进程 | 状态机、API、middleware、admission gate、向 scheduler sentinels 下发命令。 |
| scheduler 控制层 | `FaultSentinel` | 每个 scheduler 进程内的常驻控制线程 | out-of-band 接收 pause/retry，报告 fault，abort 通信域，唤醒或协调 scheduler main loop 执行 reinit。 |

这个设计等价于把 vLLM 的 `ClientSentinel` 映射为 `SentinelManager`，把 `EngineCoreSentinel + WorkerSentinel` 在 SGLang 默认拓扑下合并为 scheduler-local `FaultSentinel`。

## 4. 当前 SGLang 可复用基础

- TokenizerManager 已有 `is_pause` / `is_pause_cond`，可作为 admission gate。
- Scheduler 已有 `_engine_paused`。
- `PauseGenerationReqInput` 支持 `abort`、`in_place`、`retract`。
- `ContinueGenerationReqInput` 可恢复 scheduler event loop。
- `parallel_state.py` 已有 `destroy_model_parallel()`、`destroy_distributed_environment()`、`cleanup_dist_env_and_memory()`。
- ModelRunner 初始化路径已调用 `init_distributed_environment()`、`initialize_model_parallel()`、`initialize_dp_attention()`，可被封装为 same-topology reinit callback。

需要补齐：

- scheduler 进程内 `FaultSentinel` 常驻控制线程。
- SentinelManager 与 FaultSentinel 的 out-of-band command channel。
- PyNccl `ncclCommAbort` wrapper。
- ProcessGroup abort/destroy helper。
- ModelRunner group rebind、CUDA graph invalidation/recapture。
- fault 后 scheduler main loop 的 parked/recovery wait point。

## 5. 总体目标

本轮 retry 能力必须满足：

1. FT 默认关闭；开启后 fault 进入容错框架。
2. 所有 scheduler 运行异常先被包装成 `FaultEvent`，不直接 fail-stop。
3. 上层 hard pause 命令即使 scheduler main loop 卡住，也能被 FaultSentinel 处理。
4. fault 后冻结新请求入口。
5. scheduler 停止继续 schedule/run batch。
6. 通信相关故障或 hard pause 需要 abort/destroy 旧通信域。
7. retry 使用相同 topology 重建 distributed environment、ProcessGroup、communicator。
8. 重建后执行 health collective。
9. retry 成功后恢复 scheduler 和 admission gate。
10. retry 失败时保持 fault 状态可查询，或按配置 terminate。

## 6. 非本轮展开的能力

本文档只展开 retry path。以下能力作为后续扩展点保留：

- scale_down。
- rank replacement。
- world size / parallel size 变化。
- expert migration。
- 跨实例请求重放。
- 外部持久化请求队列。

这些能力会复用 `SentinelManager` 状态机和 `FaultSentinel` command channel，但需要单独设计 topology 变更、请求重分配和 group 重新规划。

## 7. 状态模型

### 7.1 全局状态

| 状态 | 含义 | 普通请求 |
| --- | --- | --- |
| `RUNNING` | 正常运行。 | 放行 |
| `FAULT_DETECTED` | 已收到 fault event。 | 503 |
| `PAUSING` | 正在冻结入口并暂停 scheduler。 | 503 |
| `ABORTING_COMM` | 正在 abort/destroy 通信域。 | 503 |
| `COMM_ABORTED` | 旧通信域已拆除，等待 retry。 | 503 |
| `PAUSED` | 已暂停，不一定拆通信域。 | 503 |
| `RECOVERING` | 正在 retry reinit。 | 503 |
| `WAITING_OPERATOR` | fault 已被框架接管，但当前需要外部动作。 | 503 |
| `TERMINATING` | 正在终止实例。 | 503 或连接断开 |

### 7.2 组件状态

| 状态 | 含义 |
| --- | --- |
| `HEALTHY` | 正常。 |
| `FAULTED` | 报告 fault。 |
| `PAUSED` | 已暂停。 |
| `COMM_ABORTING` | 正在拆通信域。 |
| `COMM_ABORTED` | 通信域已拆。 |
| `RECOVERING` | 正在恢复。 |
| `UNRESPONSIVE` | control channel 或 heartbeat 超时。 |
| `EXITED` | 进程退出。 |
| `WAITING_OPERATOR` | 等待外部决策。 |

## 8. Fault 分类

| Fault | 来源 | 框架行为 |
| --- | --- | --- |
| Python exception | scheduler main loop wrapper | 捕获、报告、挂起，不 fail-stop。 |
| Communication exception | collective / communicator / ProcessGroup | 报告、hard pause、abort comm。 |
| Manual hard pause | `/fault_tolerance/apply` | 冻结入口、out-of-band hard pause。 |
| Heartbeat stall | FaultSentinel 监控 scheduler main loop | 报告 stall，尝试 hard abort。 |
| Process exit | 子进程监控 | 标记 EXITED，管理面保持可查。 |
| Native crash / CUDA fatal | 进程或 runtime 无法安全继续 | 标记 fault；retry 是否可执行由状态和 health check 决定。 |

重点：fault 是否最终可以 retry，不在捕获阶段决定。捕获阶段只负责把 fault 纳入框架；retry 阶段根据 rank 存活、通信域状态、reinit 结果和 health check 决定是否恢复。

## 9. 用户故事

### 9.1 平台主动 hard pause

```http
POST /fault_tolerance/apply
{
  "fault_tolerance_instruction": "pause",
  "fault_tolerance_timeout": 30,
  "fault_tolerance_params": {"mode": "retract", "hard": true}
}
```

预期：

1. SentinelManager 冻结 admission。
2. SentinelManager 向所有 FaultSentinel 下发 hard pause。
3. FaultSentinel 即使 scheduler main loop 卡住，也能尝试 abort communicator/process group。
4. 状态进入 `COMM_ABORTED` 或 `WAITING_OPERATOR`。

### 9.2 scheduler 抛出任意异常

```text
scheduler main loop exception
  -> wrapper captures exception
  -> local FaultSentinel receives FaultEvent
  -> SentinelManager state RUNNING -> FAULT_DETECTED
  -> freeze admission
  -> hard pause all scheduler sentinels
  -> state COMM_ABORTED / WAITING_OPERATOR
```

预期：异常不直接触发 `SIGQUIT -> kill_process_tree`。

### 9.3 retry

```http
POST /fault_tolerance/apply
{
  "fault_tolerance_instruction": "retry",
  "fault_tolerance_timeout": 60,
  "fault_tolerance_params": {
    "reinit_distributed": true,
    "clear_running_batch": true,
    "recapture_cuda_graph": true
  }
}
```

预期：

1. SentinelManager 校验当前状态。
2. 向 FaultSentinel 下发 retry。
3. FaultSentinel 协调 scheduler main loop 在 safe point 执行 reinit。
4. 通信域、group、CUDA graph、ModelRunner 引用重建。
5. health collective 成功。
6. admission gate 打开，状态回到 `RUNNING`。

## 10. API 需求

### 10.1 Status

```http
GET /fault_tolerance/status
```

返回字段至少包括：

- `enabled`
- `state`
- `epoch`
- `accepting_requests`
- `topology`
- `sentinels`
- `components`
- `last_fault`
- `last_apply_result`

### 10.2 Apply

```http
POST /fault_tolerance/apply
```

本轮展开：

- `pause`
- `retry`
- `terminate`

请求体沿用 vLLM 兼容字段：

```json
{
  "fault_tolerance_instruction": "retry",
  "fault_tolerance_timeout": 60,
  "fault_tolerance_params": {}
}
```

为后续扩展预留 instruction namespace，但本文档只定义 retry 行为。

## 11. Retry 需求

retry 必须执行：

1. 关闭普通请求入口。
2. 确认所有 FaultSentinel 可达，或把不可达组件记录到 status。
3. 确认旧通信域已 abort/destroy；未完成则先执行 hard abort。
4. 清理 running batch、batch queue、chunked req、unsafe forward state。
5. same-topology reinit distributed env。
6. 重建 TP/PP/DP/EP/attention/MoE groups。
7. rebind ModelRunner 和 group-dependent objects。
8. invalidate/recapture CUDA graph。
9. 执行 health collective。
10. scheduler main loop 退出 parked state。
11. admission gate 打开。

retry 失败时必须：

- 保留管理面可用。
- 记录失败阶段、组件、rank、traceback。
- 状态进入 `WAITING_OPERATOR` 或保持 `COMM_ABORTED`。
- 根据配置决定是否自动 terminate。

## 12. Middleware 需求

非 `RUNNING` 状态下普通推理请求返回 503，body 包含当前 fault state。放行：

- `/fault_tolerance/status`
- `/fault_tolerance/apply`
- health endpoint
- metrics endpoint
- 必要 admin endpoint

## 13. 验收标准

### 13.1 控制面

- FT 开启后 status 返回 `RUNNING`。
- 非 RUNNING 状态普通请求返回 503。
- pause/retry/terminate API 可用。
- SentinelManager 能看到所有 scheduler FaultSentinel 心跳。

### 13.2 fault 捕获

- scheduler 任意 Python exception 被包装为 FaultEvent。
- exception 后 scheduler 不触发默认 fail-stop。
- status 暴露 exception type、message、traceback 摘要。
- FaultSentinel heartbeat stall 能生成 fault event。

### 13.3 hard pause

- 上层 hard pause 在 scheduler main loop 正常、异常、或卡住时都能到达 FaultSentinel。
- FaultSentinel 能 disable/abort/destroy communicator 和 ProcessGroup。
- abort 超时后状态保持可查询。

### 13.4 retry

- retry 能重建同 topology distributed env。
- retry 后 health collective 成功。
- retry 后普通推理恢复。
- retry 失败不导致管理面消失。

### 13.5 回归

- FT disabled 时现有行为保持不变。
- 现有 `/pause_generation` 和 `/continue_generation` 兼容。
- 权重更新相关 pause/retract 测试不回归。

## 14. 结论

SGLang 本轮 retry 容错应采用两层 Sentinel：主进程全局 `SentinelManager` 和每个 scheduler 进程内的 `FaultSentinel`。所有 scheduler 相关 fault 先被框架接管并挂起，不再默认 fail-stop。`FaultSentinel` 提供 out-of-band pause/abort 能力，`SentinelManager` 提供全局状态与 API，二者共同完成 hard pause、通信域重建和 retry 恢复。