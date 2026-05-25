# SGLang Fault-Tolerance 开发 Spec v2

## 0. 结论和基线

本文档定义 SGLang FT v2 的最小实现方案。v2 不应继续基于 `codex/first-ft-sentinel-registration` 开发；该分支中的 `8a6c8d74` 和 `03034c16` 是 v1 实现，生产代码新增超过 2000 行，并包含 v2 不需要的外部 pause instruction、sentinel heartbeat、拓扑快照、复杂内部状态机和 terminate 路径。

v2 开发应从去掉这两个提交的 clean main 开始，v1 代码只作为参考材料。首个社区 PR 的生产代码目标控制在 1000 行以内，优先实现清晰的外部语义和最小恢复闭环。

总体原则：

- 对外只暴露 `GET /fault_tolerance/status` 和 `POST /fault_tolerance/apply`。
- `apply` 只支持 `retry` 和 `scale_down`；`pause` 不是 FT instruction。
- 故障后的暂停是内部 pause-on-error 行为，尽量复用现有 `/pause_generation`、tokenizer pause gate 和 scheduler `_engine_paused`。
- 不引入 v1 的 `SentinelManager`、`FaultSentinel`、独立 ROUTER/DEALER heartbeat channel 或 topology/status debug API。

## 1. Clean Main 可复用能力

v2 的实现锚点来自 clean main 已有能力：

| 位置 | 复用点 |
| --- | --- |
| `entrypoints/http_server.py` | 新增 FT status/apply HTTP API，并挂载 admission middleware。 |
| `managers/tokenizer_manager.py` | 复用 `is_pause`、`is_pause_cond`、`pause_generation()`、`continue_generation()` 和已有 scheduler 发送通道。 |
| `managers/multi_tokenizer_mixin.py` | 复用多 tokenizer worker 下的 pause/continue broadcast。 |
| `managers/data_parallel_controller.py` | 复用 control message 转发能力，补充按 active rank 集合发送 FT control request。 |
| `managers/scheduler.py` | 复用 `_engine_paused`、control request dispatcher、主循环 safe point 和 runtime state 清理点。 |
| `model_executor/model_runner.py` | 只增加最小 reinit/health-check helper，不复制 v1 callback 体系。 |
| `distributed/parallel_state.py` | 复用 `cleanup_dist_env_and_memory()`、已有 process group 初始化和 communicator cleanup 能力。 |
| `elastic_ep/elastic_ep.py` | 复用 `ElasticEPStateManager.active_ranks`、`sync_active_to_cpu()`、`snapshot_active_to_last()`。 |
| `utils/watchdog.py` | 复用 `SubprocessWatchdog` 做进程退出检测，不另建 scheduler heartbeat thread。 |

新增代码建议集中在一个轻量模块，例如 `python/sglang/srt/fault_tolerance/controller.py`，以及少量现有文件接入点。不要创建 v1 风格的多文件 FT 子系统，除非后续 PR 证明确有必要。

## 2. Public API 和配置

### 2.1 配置

新增字段：

```python
fault_tolerance_on_error_strategy: Literal["pause", "continue"] = "pause"
```

CLI：

```text
--fault-tolerance-on-error-strategy {pause,continue}
```

语义：

| 值 | 行为 |
| --- | --- |
| `pause` | 故障后关闭普通请求入口，并把 active ranks 标为 `paused`。 |
| `continue` | backend 支持续推时隔离故障 rank 并保持其他 rank 服务；否则回退到 `pause`。 |

保留或新增 `--enable-fault-tolerance` 作为总开关。其他 v1 配置，例如 default pause mode、hard pause on fault、shutdown on FT failure，不进入 v2 首个 PR。

### 2.2 HTTP API

`GET /fault_tolerance/status` 返回：

```json
{
  "ranks": [
    {"rank": 0, "state": "healthy"},
    {"rank": 1, "state": "dead"}
  ]
}
```

`POST /fault_tolerance/apply` 请求：

```json
{
  "fault_tolerance_instruction": "scale_down",
  "fault_tolerance_timeout": 60,
  "fault_tolerance_params": {"ranks": [1]}
}
```

响应：

```json
{
  "success": true,
  "message": "scale_down succeeded",
  "ranks": [
    {"rank": 0, "state": "healthy"},
    {"rank": 1, "state": "dead"}
  ]
}
```

HTTP 状态码：

| 场景 | 状态码 |
| --- | --- |
| status | 200 |
| retry/scale_down 成功 | 200 |
| unsupported instruction | 400 |
| 参数错误 | 400 |
| retry/scale_down 执行失败 | 500 |
| 普通请求因 FT 暂停被拒绝 | 503 |

## 3. 状态模型

新增对外状态：

```python
class RankState(str, Enum):
    HEALTHY = "healthy"
    DEAD = "dead"
    PAUSED = "paused"
```

新增轻量 controller：

```python
class FaultToleranceController:
    rank_states: dict[int, RankState]
    isolated_ranks: set[int]
    accepting_requests: bool
```

rank id 使用 SGLang 启动时的原始 scheduler global rank。建议新增统一 helper：

```python
def scheduler_global_rank(dp_rank, pp_rank, tp_rank, attn_cp_rank, pp_size, tp_size, attn_cp_size):
    return (((dp_rank * pp_size + pp_rank) * tp_size + tp_rank) * max(1, attn_cp_size) + attn_cp_rank)
```

rank 初始化数量：

```python
num_ranks = server_args.dp_size * server_args.pp_size * server_args.tp_size * max(1, server_args.attn_cp_size)
```

没有 attn-CP 时 `server_args.attn_cp_size` 默认为 1。所有 status 输出按 rank 升序排序。

状态规则：

- 初始化成功后，所有 rank 为 `healthy`。
- 进程退出或控制通道确认不可达的 rank 为 `dead`。
- pause-on-error 下，仍可控但不可服务的 active ranks 为 `paused`。
- `scale_down` 指定的 rank 立即加入 `isolated_ranks`，对外显示为 `dead`。
- `retry` 成功后 active ranks 变为 `healthy`，不改变 `isolated_ranks`。
- `scale_down` 成功后 active ranks 变为 `healthy`，isolated ranks 保持 `dead`。

controller 不暴露 internal phase、topology、traceback、epoch 或 command result details。

## 4. Admission

在 HTTP 层新增轻量 admission middleware。它只拦截普通推理入口，不拦截管理和健康检查入口。

放行路径：

- `/fault_tolerance/status`
- `/fault_tolerance/apply`
- `/health`
- `/health_generate`
- `/metrics`
- `/ping`

判定逻辑：

```python
def should_accept_requests(self) -> bool:
    active = [
        state for rank, state in self.rank_states.items()
        if rank not in self.isolated_ranks
    ]
    return bool(active) and self.accepting_requests and all(
        state == RankState.HEALTHY for state in active
    )
```

拒绝响应：

```json
{
  "error": {
    "type": "fault_tolerance_unavailable",
    "message": "SGLang engine is paused by fault tolerance."
  }
}
```

同时设置 tokenizer 的 `is_pause=True`，让已经进入 tokenizer gate 的新请求等待或被 middleware 提前拒绝。恢复时设置 `is_pause=False` 并通知 `is_pause_cond`。

## 5. 故障接管

### 5.1 进程退出

复用 `SubprocessWatchdog` 的进程轮询，不另建 v1 scheduler heartbeat。watchdog 发现 scheduler 或 detokenizer 非正常退出时：

1. 调用 controller 的同步故障回调。
2. 将对应 rank 标为 `dead`；如果无法定位具体 rank，则将所有 active ranks 标为 `paused`。
3. 执行 pause-on-error，关闭普通请求入口。
4. 不再触发 `SIGQUIT` 直接杀掉整实例，除非 FT 未启用或 controller 回调不存在。

### 5.2 Scheduler 异常

在 scheduler 主循环外层只增加最小 try/except。捕获不可继续执行的异常时：

1. best-effort 向 tokenizer/main process 发送一个 fault notification。
2. 设置本 scheduler `_engine_paused=True`。
3. 进入 parked loop，只处理控制类请求。

如果 fault notification 发送失败，scheduler 保持 paused，等待上层通过进程状态或 health 检查发现异常。

### 5.3 pause-on-error

pause 策略流程：

```python
def pause_on_error(fault_rank: int | None):
    accepting_requests = False
    if fault_rank is not None and is_process_dead(fault_rank):
        rank_states[fault_rank] = RankState.DEAD
    for rank in active_ranks():
        if rank_states[rank] != RankState.DEAD:
            rank_states[rank] = RankState.PAUSED
    set_tokenizer_pause(True)
```

不调用外部 `/pause_generation` API，也不支持 FT `pause` instruction。内部可以直接复用 tokenizer/scheduler 的 pause helpers。

### 5.4 continue-on-error

`continue` 只在 backend 明确支持 active-rank 隔离时生效：

```python
def supports_fault_tolerance_continue(server_args) -> bool:
    return getattr(server_args, "elastic_ep_backend", None) in {"mooncake", "nixl"}
```

成功路径：

1. 故障 rank 加入 `isolated_ranks` 并标为 `dead`。
2. 更新 `ElasticEPStateManager.active_ranks`。
3. 调用 backend member refresh hook。
4. active ranks 保持 `healthy`，admission 保持打开。

失败或 unsupported：

1. 记录日志。
2. 回退到 pause-on-error。

## 6. Control Request

不要引入 v1 独立 sentinel channel。新增一个走现有 scheduler input 通道的控制请求：

```python
@dataclass
class FaultToleranceControlReqInput:
    request_id: str
    action: Literal["prepare_retry", "reinit", "health_check", "resume", "isolate"]
    target_ranks: list[int]
    active_ranks: list[int]
    isolated_ranks: list[int]
    params: dict[str, Any]

@dataclass
class FaultToleranceControlReqOutput:
    request_id: str
    rank: int
    success: bool
    message: str
```

调度规则：

- tokenizer/main process 发送 control request。
- DP controller 只把 request 转发给 `target_ranks` 对应的 worker leader。
- scheduler group leader 接收后，通过现有 broadcast 机制分发给同组 TP/PP/CP ranks。
- controller 只等待 `target_ranks` 的输出；isolated/dead ranks 不参与成功判定。

首个 PR 不需要 v1 那种 command history、per-command details 或 heartbeat result table。

## 7. Retry

`retry` 用于不改变 active rank 集合的恢复。

manager 流程：

```python
async def retry(timeout: int | None, params: dict[str, Any]) -> dict[str, Any]:
    freeze_admission()
    targets = active_ranks()
    mark_targets_paused(targets)

    send_control("prepare_retry", targets, params={"clear_runtime": True})
    send_control("reinit", targets, params=params)
    send_control("health_check", targets, params={})
    send_control("resume", targets, params=params)

    if all_targets_success:
        mark_targets_healthy(targets)
        open_admission()
    else:
        mark_targets_paused(targets)
        freeze_admission()
```

约束：

- `retry` 不修改 `isolated_ranks`。
- 存在 dead 但未 isolated 的必要 rank 时，retry 应失败并保持 paused。
- 如果所有 active ranks 已经 healthy，可以返回 no-op 成功，或执行一次 health check 后成功。
- `retry` 必须丢弃本实例 in-flight 请求。

## 8. Scale Down

### 8.1 参数解析

```python
def parse_scale_down_ranks(params: dict[str, Any]) -> list[int]:
    for key in ("ranks", "rank", "isolate_ranks", "isolate_rank"):
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, int):
            return [value]
        return [int(x) for x in value]
    raise ValueError("scale_down requires params.ranks")
```

校验：

- unknown rank 返回 400。
- 隔离全部 rank 返回 400，message 为 `scale_down leaves no active rank`。
- 所有 rank 都是 `healthy` 且 admission open 时返回失败，message 为 `scale_down requires a paused engine`。
- 重复隔离同一 rank 是 no-op 成功。

### 8.2 Manager 流程

```python
async def scale_down(timeout: int | None, params: dict[str, Any]) -> dict[str, Any]:
    to_isolate = set(parse_scale_down_ranks(params))
    freeze_admission()
    isolated_ranks.update(to_isolate)
    mark_targets_dead(to_isolate)
    targets = active_ranks()
    mark_targets_paused(targets)

    best_effort_send_control("isolate", to_isolate)

    scale_params = {
        **params,
        "active_ranks": sorted(targets),
        "isolated_ranks": sorted(isolated_ranks),
        "clear_runtime": True,
    }
    send_control("prepare_retry", targets, scale_params)
    send_control("reinit", targets, scale_params)
    send_control("health_check", targets, scale_params)
    send_control("resume", targets, scale_params)

    if all_targets_success:
        mark_targets_healthy(targets)
        open_admission()
    else:
        mark_targets_paused(targets)
        freeze_admission()
```

关键点：

- 被隔离 rank 可达时 best-effort 收到 `isolate`，不可达时直接继续。
- 被隔离 rank 不参与后续 command result 聚合。
- `scale_down` 的成败只由 active rank recovery 和 health check 决定。
- `scale_down` 可以隔离当前状态为 `dead`、`paused` 或 `healthy` 的 rank。

## 9. Scheduler 行为

新增 control request handler，不引入独立 recovery queue。scheduler 主循环已经在 `_engine_paused=True` 时继续接收控制类请求，应复用该路径。

`prepare_retry`：

```python
self._engine_paused = True
self.fault_tolerance_cleanup_runtime_state()
```

`isolate`：

```python
self._engine_paused = True
self.fault_tolerance_cleanup_runtime_state()
self.disable_or_abort_current_communicators_best_effort()
```

`reinit`：

- same-rank retry：清理并重建原通信域。
- scale_down：读取 `active_ranks`、`isolated_ranks`，优先调用 backend active-mask 路径；如果当前模型/并行配置不支持该 active set，返回失败。

`health_check`：

- 在 active rank 通信域内执行轻量 collective 或 backend health hook。
- isolated rank 不参与。

`resume`：

```python
if previous_steps_success:
    self._engine_paused = False
```

## 10. 通信域恢复

首个 PR 优先实现 backend active-mask 路径，避免一次性重写通用 ProcessGroup compact-rank rebuild。

### 10.1 Active-mask backend

适用于 mooncake/nixl 或已有 elastic EP active-rank mask 的 backend：

```python
def apply_active_rank_mask(active_ranks: list[int]) -> None:
    state = ElasticEPStateManager.instance()
    if state is None:
        raise RuntimeError("Elastic EP state is not initialized.")
    state.active_ranks.zero_()
    state.active_ranks[active_ranks] = 1
    state.snapshot_active_to_last()
    state.sync_active_to_cpu()
    refresh_backend_members()
```

`refresh_backend_members()` 封装 mooncake/nixl 具体 hook，避免 HTTP/controller 直接依赖 backend 实现。

### 10.2 Same-rank retry

same-rank retry 可复用：

- `cleanup_dist_env_and_memory()`
- `init_distributed_environment(...)`
- `initialize_model_parallel(...)`
- `initialize_dp_attention(...)`
- invalidate CUDA graph
- 必要时 recapture CUDA graph

### 10.3 Generic compact rebuild

通用 NCCL/ProcessGroup compact-rank rebuild 不进入首个 PR，除非该 PR 仍能保持代码量目标。若 backend 不支持 active-mask，而当前 active rank set 需要 compact rebuild，`scale_down` 应返回失败并保持 paused，status 继续可查询。

## 11. In-flight 清理

`retry` 和 `scale_down` 都必须丢弃本实例所有 in-flight 请求。

最小 scheduler 清理：

```python
def fault_tolerance_cleanup_runtime_state(self):
    self.cur_batch = None
    self.last_batch = None
    self.chunked_req = None
    if hasattr(self, "result_queue"):
        self.result_queue.clear()
    if hasattr(self, "waiting_queue"):
        self.waiting_queue.clear()
    if hasattr(self, "running_batch"):
        self.running_batch.clear()
```

需要补齐对 tokenizer/detokenizer 的失败通知：

- running/streaming 请求返回 interrupted/error。
- waiting queue 请求返回 503 或内部 interrupted。
- overlap result queue 结果直接丢弃。
- speculative、chunked prefill、overlap schedule 的临时引用必须清空。

不保存请求重放状态；请求重放由上层服务处理。

## 12. 测试计划

单元测试：

- status 只返回 `ranks`，状态值只允许 `healthy`、`dead`、`paused`。
- `apply pause` 返回 unsupported。
- `retry` 不修改 `isolated_ranks`。
- `scale_down` 可隔离 `dead`、`paused`、`healthy` rank。
- unknown rank、隔离全部 rank、正常服务中 scale_down 的错误响应。
- isolated rank control timeout 不影响 active rank 成功判定。
- unsupported backend 下 `continue` 回退到 pause。

控制流测试：

- tokenizer pause gate 被 FT pause-on-error 关闭，恢复后通知 `is_pause_cond`。
- DP controller 只向 active/target ranks 转发 FT control request。
- scheduler paused loop 仍能处理 FT control request。
- retry/scale_down 都调用 in-flight cleanup。

集成测试：

- 单机 TP：注入 scheduler exception，status 显示 paused，retry 恢复 healthy。
- rank 退出：status 显示 dead，scale_down 隔离该 rank，active ranks 恢复 healthy。
- paused 状态下隔离可达 healthy rank，目标 rank 变 dead，其余 rank healthy。
- mooncake/nixl + `continue`：backend 隔离成功时故障 rank dead，其他 rank healthy。
- backend continue 失败：回退 paused，普通请求返回 503。

## 13. PR 拆分

| PR | 内容 |
| --- | --- |
| PR1 | 从 clean main 增加轻量 controller、public status/apply、admission、rank state 和 unsupported pause。 |
| PR2 | pause-on-error 接入 subprocess/scheduler fault，复用 tokenizer pause gate 和 scheduler parked loop。 |
| PR3 | retry control request、same-rank reinit、health check、in-flight cleanup。 |
| PR4 | scale_down control request、isolated set、active-rank result 聚合、backend active-mask 路径。 |
| PR5 | continue strategy、mooncake/nixl 集成、E2E 和回归测试。 |

首个社区 PR 如果需要进一步压缩，应优先保留 public status/apply、admission 和状态模型，把真实 distributed recovery 放到后续 PR。

## 14. 完成定义

v2 完成时应满足：

1. 开发基线来自 clean main，不依赖 v1 sentinel package。
2. status 只返回每个 rank 的 `healthy`、`dead`、`paused`。
3. apply 只支持 `retry` 和 `scale_down`。
4. FT API 不支持 `pause` instruction。
5. pause-on-error 能关闭普通请求入口，并保持 FT API、health、metrics 可用。
6. `retry` 丢弃 in-flight 请求，恢复 active ranks 到 `healthy`。
7. `scale_down` 隔离指定 rank，指定 rank 对外显示为 `dead`。
8. `scale_down` 不等待 isolated/dead rank 的控制通道响应。
9. active-mask backend 支持的场景下，scale_down 后其余 active ranks 可恢复服务。
10. 失败后 status 仍可查询，上层服务可以继续 retry、scale_down 或替换实例。
