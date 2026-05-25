# SGLang Fault-Tolerance 开发 Spec v2

## 0. 目标

本文档定义 SGLang FT v2 的最小实现方案。对外只暴露 rank 状态查询、`retry` 和 `scale_down`；故障后的暂停是内部 pause-on-error 行为。实现尽量复用当前 FT 模块、scheduler pause 机制、distributed recovery helper 和 ZMQ control channel。

总体结构：

```text
Main process
  SentinelManager
  FaultToleranceAdmissionMiddleware
  HTTP /fault_tolerance/status
  HTTP /fault_tolerance/apply

Scheduler process
  FaultSentinel control thread
  scheduler main loop recovery queue
  DistributedRecoveryManager
```

## 1. 代码锚点

| 文件 | v2 职责 |
| --- | --- |
| `python/sglang/srt/fault_tolerance/state.py` | 新增对外 `RankState`，保留内部 fault event 和 sentinel status。 |
| `python/sglang/srt/fault_tolerance/command.py` | 复用现有 command schema，新增隔离 rank 的可选 command。 |
| `python/sglang/srt/fault_tolerance/manager.py` | 收敛对外 API，维护 rank 状态、isolated set、retry 和 scale_down。 |
| `python/sglang/srt/fault_tolerance/sentinel.py` | 复用 scheduler-local control thread，处理 isolate 和 active-rank recovery。 |
| `python/sglang/srt/fault_tolerance/distributed_recovery.py` | 复用 abort/reinit/health check，支持 active rank 参数。 |
| `python/sglang/srt/fault_tolerance/middleware.py` | 根据 public rank state 决定普通请求是否放行。 |
| `python/sglang/srt/managers/scheduler.py` | 复用 recovery queue 和 main-loop safe point，强化 in-flight 清理。 |
| `python/sglang/srt/model_executor/model_runner.py` | 复用 FT reinit callbacks，支持 active rank rebuild。 |
| `python/sglang/srt/distributed/parallel_state.py` | 提供 active rank 通信域重建 helper。 |
| `python/sglang/srt/server_args.py` | 新增故障后策略配置。 |
| `python/sglang/srt/entrypoints/http_server.py` | status 简化，apply 支持 `retry` 和 `scale_down`。 |

## 2. 配置项

新增：

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
| `pause` | 故障后自动暂停 active ranks，等待上层调用 `retry` 或 `scale_down`。 |
| `continue` | 若 backend 支持 active-rank 隔离续推，则故障 rank 记为 `dead`，其他 rank 保持 `healthy`；否则回退为 `pause`。 |

后端能力判断建议：

```python
def supports_fault_tolerance_continue(server_args) -> bool:
    backends = {
        getattr(server_args, "moe_a2a_backend", None),
        getattr(server_args, "elastic_ep_backend", None),
    }
    return bool(backends & {"mooncake", "nixl"})
```

`continue` 策略只决定故障后的默认动作，不改变 `retry` 和 `scale_down` 的 API 语义。

## 3. Public RankState

在 `state.py` 增加对外状态：

```python
class RankState(str, Enum):
    HEALTHY = "healthy"
    DEAD = "dead"
    PAUSED = "paused"
```

`SentinelManager` 增加：

```python
self.rank_states: dict[int, RankState]
self.isolated_ranks: set[int]
```

初始化：

```python
def _init_rank_states(self) -> None:
    for rank in self._all_scheduler_ids_locked():
        self.rank_states[rank] = RankState.HEALTHY
```

public status 只从 `rank_states` 生成，不直接暴露内部 `FaultToleranceState` 或 `ComponentState`。

内部状态到 public 状态的映射：

| 内部状态 | public 状态 |
| --- | --- |
| `ComponentState.HEALTHY`、recovery 成功 | `healthy` |
| `ComponentState.EXITED`、`UNRESPONSIVE`、isolated rank | `dead` |
| 其他可控但不可服务状态 | `paused` |

## 4. Status API

`http_server.py`：

```python
@app.get("/fault_tolerance/status")
async def fault_tolerance_status(request: Request):
    manager = _get_sentinel_manager()
    if manager is None:
        return ORJSONResponse(content={"ranks": []})
    return ORJSONResponse(content=manager.get_public_status())
```

`SentinelManager`：

```python
def get_public_status(self) -> dict[str, Any]:
    with self._lock:
        return {
            "ranks": [
                {"rank": rank, "state": self.rank_states[rank].value}
                for rank in sorted(self.rank_states)
            ]
        }
```

`get_status()` 可以作为内部调试方法保留，但 FT HTTP status 使用 `get_public_status()`。

## 5. Apply API

请求 schema 复用现有 `FaultToleranceApplyRequest`：

```python
class FaultToleranceApplyRequest(BaseModel):
    fault_tolerance_instruction: str
    fault_tolerance_timeout: Optional[int] = None
    fault_tolerance_params: Dict[str, Any] = Field(default_factory=dict)
```

`SentinelManager.apply()`：

```python
async def apply(self, instruction: str, timeout: int | None, params: dict[str, Any]):
    instruction = instruction.lower()
    if instruction == "retry":
        result = await self.retry(timeout=timeout, params=params or {})
    elif instruction == "scale_down":
        result = await self.scale_down(timeout=timeout, params=params or {})
    else:
        result = {
            "success": False,
            "message": f"Unsupported fault tolerance instruction: {instruction}",
            "ranks": self._public_rank_list_locked(),
        }
    return result
```

`pause` 不作为 FT instruction。收到 `pause` 时返回 unsupported。

响应 helper：

```python
def _apply_result(self, success: bool, message: str) -> dict[str, Any]:
    return {
        "success": success,
        "message": message,
        "ranks": self._public_rank_list_locked(),
    }
```

## 6. 故障接管

### 6.1 Scheduler exception

`scheduler.py` 已有 `_run_event_loop_with_fault_tolerance()` 和 `enter_fault_parked_state()`。v2 保留该结构。

故障事件处理：

```python
def report_fault_sync(self, event: FaultEvent) -> None:
    if not self.enabled:
        return

    fault_rank = self._rank_from_event(event)
    if self._should_continue_on_error(event):
        self._mark_rank_dead_locked(fault_rank)
        self._open_admission_if_active_ranks_healthy_locked()
        return

    self._pause_on_error_locked(event)
```

### 6.2 pause-on-error

```python
def _pause_on_error_locked(self, event: FaultEvent) -> None:
    fault_rank = self._rank_from_event(event)
    if event.fault_type in ("process_exit", "heartbeat_stall", "native_crash"):
        self._mark_rank_dead_locked(fault_rank)

    for rank in self._active_ranks_locked():
        if self.rank_states.get(rank) != RankState.DEAD:
            self.rank_states[rank] = RankState.PAUSED

    self._freeze_admission_locked()
    self._broadcast_hard_abort_best_effort(reason="fault")
```

说明：

- 对外状态只显示 `paused` 和 `dead`。
- hard abort 是 best-effort，用于打断不可信通信域。
- 上层服务通过 status 选择 `retry` 或 `scale_down`。

### 6.3 continue-on-error

```python
def _should_continue_on_error(self, event: FaultEvent) -> bool:
    return (
        self.server_args.fault_tolerance_on_error_strategy == "continue"
        and supports_fault_tolerance_continue(self.server_args)
        and self._backend_reports_or_accepts_isolation(event)
    )
```

continue 成功路径：

1. fault rank 标记为 `dead`。
2. backend 更新 active rank mask。
3. active rank 保持 `healthy`。
4. admission 保持打开。

continue 失败路径：

1. active rank 标记为 `paused`。
2. admission 关闭。
3. 进入 pause-on-error 等待上层决策。

## 7. Command 设计

复用现有 ZMQ ROUTER/DEALER：

```text
SentinelManager ROUTER
  -> FaultSentinel DEALER per scheduler
```

现有命令继续使用：

- `PREPARE_RETRY`
- `HARD_ABORT_COMM`
- `RETRY_REINIT`
- `HEALTH_CHECK`
- `RESUME`

新增可选命令：

```python
class SentinelCommandType(str, Enum):
    ISOLATE_RANK = "isolate_rank"
```

`ISOLATE_RANK` 只发送给被 scale_down 指定且当前仍可达的 rank。该命令失败不阻塞 scale_down。

`RETRY_REINIT` 增加参数：

```json
{
  "scale_down": true,
  "active_ranks": [0, 2, 4, 5],
  "isolated_ranks": [1, 3]
}
```

当未传 `active_ranks` 时，行为等价于使用原始完整 rank 集合。

## 8. Retry 实现

### 8.1 Manager 流程

```python
async def retry(self, timeout: int, params: dict[str, Any]) -> dict[str, Any]:
    timeout = timeout or self.server_args.fault_tolerance_recovery_timeout_sec
    with self._lock:
        self._freeze_admission_locked()
        target_ranks = self._all_scheduler_ids_minus_isolated_locked()
        self._mark_targets_paused_locked(target_ranks)

    results = []
    results += self._issue_command(PREPARE_RETRY, timeout, params, target_ranks)
    results += self._issue_command(HARD_ABORT_COMM, abort_timeout, params, target_ranks)
    results += self._issue_command(RETRY_REINIT, timeout, params, target_ranks)
    results += self._issue_command(HEALTH_CHECK, timeout, params, target_ranks)

    success = self._all_success_for_targets(results, target_ranks)
    if success:
        results += self._issue_command(RESUME, timeout, params, target_ranks)
        success = self._all_success_for_targets(results, target_ranks)

    with self._lock:
        if success:
            self._mark_targets_healthy_locked(target_ranks)
            self._open_admission_locked()
        else:
            self._mark_targets_paused_locked(target_ranks)
            self._freeze_admission_locked()
        return self._apply_result(success, "retry succeeded" if success else "retry failed")
```

`retry` 不修改 `isolated_ranks`。如果 dead rank 没有被隔离，仍属于 `target_ranks`，command 会超时或失败，从而让 retry 失败。

### 8.2 Scheduler 流程

当前 `fault_tolerance_submit_command()` 和 `_fault_tolerance_process_command_on_main_loop()` 可复用。

`PREPARE_RETRY`：

```python
self.fault_tolerance_prepare_retry({"clear_running_batch": True})
```

`RETRY_REINIT`：

```python
recovery.cleanup_distributed()
recovery.reinit_distributed_on_main_thread(cmd.params)
```

`HEALTH_CHECK`：

```python
recovery.health_check(cmd.timeout_sec)
```

`RESUME`：

```python
if self.fault_tolerance_recovery_ok:
    self._engine_paused = False
```

## 9. Scale Down 实现

### 9.1 参数解析

```python
def _parse_scale_down_ranks(params: dict[str, Any]) -> list[int]:
    for key in ("ranks", "rank", "isolate_ranks", "isolate_rank"):
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, int):
            return [value]
        return [int(x) for x in value]
    raise ValueError("scale_down requires params.ranks")
```

### 9.2 Manager 流程

```python
async def scale_down(self, timeout: int, params: dict[str, Any]) -> dict[str, Any]:
    timeout = timeout or self.server_args.fault_tolerance_recovery_timeout_sec
    isolated = set(self._parse_scale_down_ranks(params))

    with self._lock:
        if self._is_accepting_requests_locked() and self._all_ranks_healthy_locked():
            return self._apply_result(False, "scale_down requires a paused engine")
        self._freeze_admission_locked()
        self.isolated_ranks.update(isolated)
        for rank in isolated:
            self.rank_states[rank] = RankState.DEAD
        active = self._active_ranks_locked()
        self._mark_targets_paused_locked(active)

    if not active:
        return self._apply_result(False, "scale_down leaves no active rank")

    self._best_effort_isolate_ranks(isolated, timeout_sec=min(timeout, 5))

    scale_params = dict(params)
    scale_params.update(
        {
            "scale_down": True,
            "active_ranks": sorted(active),
            "isolated_ranks": sorted(self.isolated_ranks),
            "clear_running_batch": True,
        }
    )

    results = []
    results += self._issue_command(PREPARE_RETRY, timeout, scale_params, active)
    results += self._issue_command(HARD_ABORT_COMM, abort_timeout, scale_params, active)
    results += self._issue_command(RETRY_REINIT, timeout, scale_params, active)
    results += self._issue_command(HEALTH_CHECK, timeout, scale_params, active)

    success = self._all_success_for_targets(results, active)
    if success:
        results += self._issue_command(RESUME, timeout, scale_params, active)
        success = self._all_success_for_targets(results, active)

    with self._lock:
        if success:
            self._mark_targets_healthy_locked(active)
            for rank in self.isolated_ranks:
                self.rank_states[rank] = RankState.DEAD
            self._open_admission_locked()
        else:
            self._mark_targets_paused_locked(active)
            self._freeze_admission_locked()
        return self._apply_result(
            success,
            "scale_down succeeded" if success else "scale_down failed",
        )
```

关键点：

- 被隔离 rank 不参与后续 command result 聚合。
- 被隔离 rank 可达时发送 `ISOLATE_RANK`，不可达时直接继续。
- active rank 的恢复路径与 retry 相同。
- `scale_down` 的最终成功由 active rank health check 决定。

### 9.3 FaultSentinel isolate

```python
if cmd.command == SentinelCommandType.ISOLATE_RANK:
    self.scheduler._engine_paused = True
    self.state = ComponentState.WAITING_OPERATOR
    self.recovery.disable_communicators()
    return self._result(cmd, True, "Rank isolated.")
```

隔离命令不要求当前 rank 退出进程。上层服务可以在实例级决定是否回收该进程。

## 10. Active Rank 通信域重建

`DistributedRecoveryManager.reinit_distributed_on_main_thread()` 读取：

```python
active_ranks = params.get("active_ranks")
isolated_ranks = params.get("isolated_ranks", [])
scale_down = params.get("scale_down", False)
```

建议增加轻量 plan：

```python
@dataclass
class ActiveRankPlan:
    original_world_size: int
    active_ranks: list[int]
    isolated_ranks: list[int]
    compact_rank: dict[int, int]

    def compact(self, global_rank: int) -> int:
        return self.compact_rank[global_rank]
```

构造：

```python
def build_active_rank_plan(original_world_size: int, active_ranks: list[int]) -> ActiveRankPlan:
    active = sorted(set(active_ranks))
    return ActiveRankPlan(
        original_world_size=original_world_size,
        active_ranks=active,
        isolated_ranks=[r for r in range(original_world_size) if r not in active],
        compact_rank={rank: idx for idx, rank in enumerate(active)},
    )
```

### 10.1 Backend active-mask 路径

mooncake/nixl 这类 backend 优先使用 active-rank mask：

```python
def apply_active_rank_mask(active_ranks: list[int]) -> None:
    state = ElasticEPStateManager.instance()
    if state is None:
        return
    state.active_ranks.zero_()
    state.active_ranks[active_ranks] = 1
    state.snapshot_active_to_last()
    state.sync_active_to_cpu()
```

之后调用 backend 的 group/member refresh hook，例如：

```python
EPBuffer._buffer.update_ep_member()
```

具体 backend hook 封装在 `distributed_recovery.py`，避免 FT manager 直接依赖 mooncake/nixl 实现细节。

### 10.2 Generic rebuild 路径

当 backend 需要重建 ProcessGroup：

1. 使用 active rank plan 计算当前进程的 compact rank。
2. `cleanup_dist_env_and_memory()` 清理当前通信环境。
3. 以 `world_size=len(active_ranks)` 和 `rank=compact_rank[old_global_rank]` 初始化新 default group。
4. 按 active rank 过滤原 group ranks，重新创建 TP/PP/DP/EP/CP/MoE groups。
5. 调用 `ModelRunner.fault_tolerance_rebind_distributed_groups()`。
6. invalidate CUDA graph。
7. 按配置 recapture CUDA graph。

如果当前模型/并行配置无法在 active rank set 上形成合法 group，reinit 返回失败，`scale_down` 保持 paused。

## 11. ModelRunner 修改

保留现有 callbacks：

```python
fault_tolerance_prepare_reinit()
fault_tolerance_reinit_distributed(params)
fault_tolerance_rebind_distributed_groups()
fault_tolerance_invalidate_cuda_graphs()
fault_tolerance_recapture_cuda_graphs()
fault_tolerance_health_check()
```

`fault_tolerance_reinit_distributed(params)` 增加 active-rank 分支：

```python
def fault_tolerance_reinit_distributed(self, params: dict[str, Any]) -> None:
    active_ranks = params.get("active_ranks")
    if active_ranks is not None:
        self._fault_tolerance_reinit_active_ranks(active_ranks, params)
    else:
        self._fault_tolerance_reinit_same_ranks(params)
```

health check：

- same-rank retry：对当前通信域执行 all-reduce。
- scale_down：只在 active rank 通信域内执行 health check。
- isolated rank 不参与 health check。

## 12. Scheduler in-flight 清理

`retry` 和 `scale_down` 都必须清理本实例 in-flight 请求。

在 `scheduler.py` 强化：

```python
def fault_tolerance_cleanup_runtime_state(self):
    self.cur_batch = None
    self.last_batch = None
    self.chunked_req = None
    if hasattr(self, "result_queue"):
        self.result_queue.clear()
    self.waiting_queue.clear()
    self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
```

需要补齐的通知：

1. 对已进入 streaming 的请求，向 tokenizer/detokenizer 返回 interrupted/error。
2. 对 waiting queue 中尚未执行的请求，返回 503 或内部 interrupted。
3. 对 overlap result queue 丢弃结果，避免恢复后处理故障前 batch。
4. 清理 speculative、chunked prefill、overlap schedule 的临时引用。

不在 SGLang 内保存请求重放状态。

## 13. Middleware

`FaultToleranceAdmissionMiddleware` 不读取内部 `FaultToleranceState`，改为读取 public rank state。

```python
def should_accept_requests(self) -> bool:
    active = [
        state for rank, state in self.rank_states.items()
        if rank not in self.isolated_ranks
    ]
    return bool(active) and all(state == RankState.HEALTHY for state in active)
```

普通请求拦截：

```json
{
  "error": {
    "type": "fault_tolerance_unavailable",
    "message": "SGLang engine is paused by fault tolerance."
  }
}
```

放行：

- `/fault_tolerance/status`
- `/fault_tolerance/apply`
- `/health`
- `/health_generate`
- `/metrics`
- `/ping`

## 14. HTTP 状态码

建议：

| 场景 | 状态码 |
| --- | --- |
| status | 200 |
| retry/scale_down 成功 | 200 |
| unsupported instruction | 400 |
| 参数错误 | 400 |
| retry/scale_down 执行失败 | 500 |
| 普通请求在 paused 时被拒绝 | 503 |

## 15. 幂等性

`scale_down`：

- 重复隔离同一 rank：成功或 no-op 成功。
- 隔离 unknown rank：400。
- 隔离所有 rank：500 或 400，message 为 no active rank。
- 在 paused 上下文中隔离可达 healthy rank：允许，best-effort 通知该 rank isolate。
- 所有 rank 都 healthy 且 admission 打开时调用：失败，message 为 scale_down requires a paused engine。

`retry`：

- 全部 active rank healthy 时可 no-op 成功，也可执行一次轻量 health check 后成功。
- 存在 dead 且未 isolated 的 rank 时，应失败并保持 paused。
- 存在 isolated rank 时，不等待 isolated rank command result。

## 16. 测试计划

### 16.1 单元测试

新增或调整：

```text
test/srt/fault_tolerance/test_public_status.py
test/srt/fault_tolerance/test_apply_dispatch.py
test/srt/fault_tolerance/test_on_error_strategy.py
test/srt/fault_tolerance/test_scale_down_manager.py
test/srt/fault_tolerance/test_middleware_rank_state.py
```

覆盖：

- status 只返回 `ranks`。
- rank state 只允许 `healthy`、`dead`、`paused`。
- `apply pause` 返回 unsupported。
- `retry` 不修改 isolated set。
- `scale_down` 将指定 rank 标为 `dead`。
- `scale_down` 不等待 isolated rank result。
- continue strategy backend 不支持时回退 pause。

### 16.2 Mock recovery 测试

- active rank 全部成功：scale_down 成功。
- active rank 任一 reinit 失败：scale_down 失败，active rank 为 `paused`。
- isolated rank command timeout：scale_down 仍继续。
- retry 遇到未隔离 dead rank：失败。
- repeated scale_down 同一 rank：幂等。

### 16.3 集成测试

- TP/DP 单机：注入 scheduler exception，status 显示 paused，retry 恢复 healthy。
- rank 退出：status 显示 dead，scale_down 隔离该 rank，active rank health check 成功。
- paused 上下文中指定 healthy rank scale_down：该 rank 变 dead，其他 rank 恢复 healthy。
- mooncake backend + `pause`：故障后进入 paused。
- mooncake backend + `continue`：backend 隔离成功时故障 rank dead，其他 rank healthy。
- backend continue 失败：回退 paused，普通请求 503。

## 17. PR 拆分建议

| PR | 内容 |
| --- | --- |
| PR1 | `RankState`、public status、middleware rank-state 化、apply instruction 收敛。 |
| PR2 | `fault_tolerance_on_error_strategy`、pause-on-error/continue-on-error 分支。 |
| PR3 | `scale_down` manager flow、rank isolation、active target command 聚合。 |
| PR4 | active rank distributed recovery、backend active-mask hook、health check。 |
| PR5 | scheduler in-flight 清理补齐、E2E 和回归测试。 |

## 18. 完成定义

完成时应满足：

1. FT status 只返回每个 rank 的 `healthy`、`dead`、`paused`。
2. FT apply 只支持 `retry` 和 `scale_down`。
3. 故障后 pause-on-error 能暂停普通请求入口。
4. `fault_tolerance_on_error_strategy=continue` 能在支持 backend 上保留健康 rank 服务能力。
5. `retry` 丢弃 in-flight 请求，重建通信域，成功后 active rank 为 `healthy`。
6. `scale_down` 隔离指定 rank，指定 rank 显示为 `dead`。
7. `scale_down` 不强依赖被隔离 rank 当前状态或控制通道响应。
8. `scale_down` 成功后未隔离 rank 为 `healthy`，普通推理恢复。
9. 失败后 status 仍可查询，上层服务可以继续决策。
