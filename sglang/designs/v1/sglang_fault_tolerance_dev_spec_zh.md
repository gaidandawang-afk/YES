# SGLang Fault-Tolerance Retry 开发 Spec

## 0. 目标

本文档定义 SGLang retry 容错能力的可开发方案。核心架构为：

```text
Main Process: SentinelManager
Scheduler Process: FaultSentinel control thread + scheduler main loop wrapper
```

本轮实现目标：FT enabled 后，scheduler 相关 fault 统一进入容错框架，系统挂起并保持管理面可用；上层或自动流程可执行 hard pause、通信域 abort/destroy、same-topology reinit 和 retry。本文档只展开 retry path，`scale_down` 保留为后续扩展能力。

## 1. 关键结论

1. 不新增 SGLang 常驻进程。
2. 每个 scheduler 进程新增一个常驻 `FaultSentinel` 控制线程。
3. 主进程新增一个全局 `SentinelManager` 对象。
4. 不再要求 fault 预先分类为 recoverable；scheduler 执行异常先全部包装为 `FaultEvent`。
5. 是否能 retry 在 recovery 阶段判断。
6. hard pause / comm abort 必须走 out-of-band control channel，不能依赖 scheduler main loop。
7. reinit / ModelRunner rebind / CUDA graph recapture 应由 scheduler main loop 在 parked safe point 执行，避免跨线程修改运行时对象。

## 2. 当前代码锚点

| 文件 | 当前职责 | 改造点 |
| --- | --- | --- |
| `python/sglang/srt/server_args.py` | 服务参数。 | 新增 FT / sentinel / retry 配置。 |
| `python/sglang/srt/entrypoints/http_server.py` | FastAPI route。 | 新增 `/fault_tolerance/*` API 与 middleware。 |
| `python/sglang/srt/entrypoints/engine.py` | 子进程启动、fail-stop handler。 | 初始化 SentinelManager；FT enabled 时调整子进程 fault 处理策略。 |
| `python/sglang/srt/managers/tokenizer_manager.py` | admission、pause/continue。 | 接入 SentinelManager，处理 FaultEvent。 |
| `python/sglang/srt/managers/scheduler.py` | scheduler event loop。 | 增加 loop wrapper、parked recovery point、FaultSentinel 初始化。 |
| `python/sglang/srt/managers/tp_worker.py` | TpModelWorker holder。 | 暴露 distributed recovery callbacks。 |
| `python/sglang/srt/model_executor/model_runner.py` | distributed init、model、graph。 | 增加 prepare/reinit/rebind/health check 方法。 |
| `python/sglang/srt/distributed/parallel_state.py` | group lifecycle。 | 增加 group 枚举、safe abort/destroy helper。 |
| `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py` | NCCL ctypes wrapper。 | 增加 `ncclCommAbort`。 |
| `python/sglang/srt/distributed/device_communicators/pynccl.py` | PyNccl communicator。 | 增加 `abort()`。 |
| `python/sglang/srt/managers/io_struct.py` | IPC struct。 | 新增 FaultEvent、sentinel command/result。 |

## 3. 新增模块

```text
python/sglang/srt/fault_tolerance/
  __init__.py
  state.py
  manager.py
  sentinel.py
  command.py
  middleware.py
  distributed_recovery.py
  exceptions.py
```

模块职责：

| 模块 | 职责 |
| --- | --- |
| `state.py` | 状态枚举、FaultEvent、snapshot、state store。 |
| `manager.py` | 主进程 `SentinelManager`。 |
| `sentinel.py` | scheduler 进程内 `FaultSentinel` 控制线程。 |
| `command.py` | command/result schema。 |
| `middleware.py` | 非 RUNNING 503。 |
| `distributed_recovery.py` | abort/reinit/rebind/health check helper。 |
| `exceptions.py` | FT 内部异常类型；不是 fault 捕获白名单。 |

## 4. 配置项

在 `ServerArgs` 增加：

```python
@dataclass
class ServerArgs:
    enable_fault_tolerance: bool = False
    fault_tolerance_recovery_timeout_sec: int = 60
    fault_tolerance_comm_abort_timeout_sec: int = 30
    fault_tolerance_sentinel_cmd_timeout_sec: int = 10
    fault_tolerance_sentinel_heartbeat_interval_sec: float = 1.0
    fault_tolerance_sentinel_heartbeat_timeout_sec: float = 10.0
    fault_tolerance_default_pause_mode: str = "retract"
    fault_tolerance_hard_pause_on_fault: bool = True
    fault_tolerance_reinit_dist_on_retry: bool = True
    shutdown_on_fault_tolerance_failure: bool = False
```

默认说明：

- FT disabled 时现有 fail-stop 行为不变。
- FT enabled 时 `shutdown_on_fault_tolerance_failure=False` 更符合“fault 后挂住并等待管理面”的需求。
- 生产可按运维策略设置为 True，让 retry/abort 失败时自动 terminate。

## 5. 状态与数据结构

### 5.1 状态枚举

```python
class FaultToleranceState(str, Enum):
    RUNNING = "RUNNING"
    FAULT_DETECTED = "FAULT_DETECTED"
    PAUSING = "PAUSING"
    ABORTING_COMM = "ABORTING_COMM"
    COMM_ABORTED = "COMM_ABORTED"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    WAITING_OPERATOR = "WAITING_OPERATOR"
    TERMINATING = "TERMINATING"

class ComponentState(str, Enum):
    HEALTHY = "HEALTHY"
    FAULTED = "FAULTED"
    PAUSED = "PAUSED"
    COMM_ABORTING = "COMM_ABORTING"
    COMM_ABORTED = "COMM_ABORTED"
    RECOVERING = "RECOVERING"
    UNRESPONSIVE = "UNRESPONSIVE"
    EXITED = "EXITED"
    WAITING_OPERATOR = "WAITING_OPERATOR"
```

### 5.2 FaultEvent

```python
@dataclass
class FaultEvent:
    event_id: str
    timestamp: float
    origin: str
    scheduler_id: int | None
    rank: int | None
    fault_type: str
    exception_type: str | None
    message: str
    traceback: str | None
    requires_hard_pause: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
```

注意：`requires_hard_pause` 是处理建议，不是 recoverable 判定。所有 scheduler exception 都可以生成 event；retry 是否可执行由后续阶段判断。

### 5.3 Sentinel 状态

```python
@dataclass
class SentinelStatus:
    scheduler_id: int
    pid: int
    state: ComponentState
    last_heartbeat_ts: float
    last_command_id: str | None
    last_fault_event_id: str | None
    details: dict[str, Any] = field(default_factory=dict)
```

## 6. Command schema

```python
class SentinelCommandType(str, Enum):
    PAUSE = "pause"
    HARD_ABORT_COMM = "hard_abort_comm"
    PREPARE_RETRY = "prepare_retry"
    RETRY_REINIT = "retry_reinit"
    HEALTH_CHECK = "health_check"
    RESUME = "resume"
    TERMINATE = "terminate"

@dataclass
class SentinelCommand:
    command_id: str
    command: SentinelCommandType
    epoch: int
    timeout_sec: int
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class SentinelCommandResult:
    command_id: str
    scheduler_id: int
    success: bool
    state: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
```

## 7. SentinelManager

### 7.1 职责

`SentinelManager` 位于主进程，负责：

- 持有全局状态机。
- 注册和跟踪 scheduler `FaultSentinel`。
- 接收 fault event。
- 冻结/打开 TokenizerManager admission。
- 聚合 sentinel command result。
- 提供 `/fault_tolerance/status` 和 `/fault_tolerance/apply`。
- 非 RUNNING 状态配合 middleware 返回 503。
- 在 FT enabled 时阻止默认 child fault 直接 kill 全实例。

### 7.2 API

```python
class SentinelManager:
    def __init__(self, server_args, tokenizer_manager, terminate_callback): ...
    def register_sentinel(self, scheduler_id: int, address: str, pid: int) -> None: ...
    def get_status(self) -> dict[str, Any]: ...
    async def report_fault(self, event: FaultEvent) -> None: ...
    async def apply(self, instruction: str, timeout: int | None, params: dict[str, Any]) -> dict[str, Any]: ...
    async def pause(self, hard: bool, mode: str, timeout: int, reason: str) -> dict[str, Any]: ...
    async def retry(self, timeout: int, params: dict[str, Any]) -> dict[str, Any]: ...
    async def terminate(self, reason: str) -> dict[str, Any]: ...
```

### 7.3 Fault 处理策略

```python
async def report_fault(self, event):
    record_event(event)
    transition(FAULT_DETECTED)
    freeze_admission()
    if event.requires_hard_pause or server_args.fault_tolerance_hard_pause_on_fault:
        await pause(hard=True, mode=default_pause_mode, reason="fault")
    else:
        await pause(hard=False, mode=default_pause_mode, reason="fault")
```

report fault 必须幂等：同一 epoch 内重复 fault 不重复触发多轮 abort。

## 8. FaultSentinel

### 8.1 位置

每个 scheduler 进程内创建一个 `FaultSentinel` 常驻控制线程：

```text
Scheduler process
  - scheduler main loop thread/process main thread
  - FaultSentinel control thread
```

### 8.2 职责

- 监听 SentinelManager 的 out-of-band command。
- 向 SentinelManager 发送 heartbeat。
- 接收 scheduler main loop wrapper 报告的 exception event。
- 在 main loop 卡住时执行 hard abort communicator/process group。
- 把 retry reinit 请求投递给 scheduler main loop 的 parked recovery point。
- 汇总 local recovery result 并回传 SentinelManager。

### 8.3 为什么是线程

FaultSentinel 必须独立于 scheduler main loop，因为 main loop 可能：

- 卡在 collective。
- 卡在 model forward。
- 卡在某个阻塞 I/O。
- 已经进入 exception parked state。

如果 command 只走 scheduler 普通输入队列，hard pause 无法保证被执行。

### 8.4 API

```python
class FaultSentinel:
    def __init__(self, scheduler, tp_worker, model_runner, manager_addr, scheduler_id): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def report_fault_from_main_loop(self, event: FaultEvent) -> None: ...
    def handle_command(self, cmd: SentinelCommand) -> SentinelCommandResult: ...
    def hard_abort_comm(self, timeout: int) -> SentinelCommandResult: ...
    def request_retry_reinit_on_main_loop(self, cmd: SentinelCommand) -> SentinelCommandResult: ...
```

## 9. Scheduler main loop wrapper

### 9.1 包装所有异常

FT enabled 时，scheduler event loop 不能让 exception 直接冒泡到 `run_scheduler_process` 的 fail-stop handler。需要在 loop 外层或每轮 iteration 外层包装：

```python
while True:
    try:
        scheduler.run_one_iteration()
    except BaseException as exc:
        if not server_args.enable_fault_tolerance:
            raise
        event = FaultEvent.from_exception(exc, origin="scheduler", scheduler_id=self.scheduler_id)
        self._engine_paused = True
        self.fault_sentinel.report_fault_from_main_loop(event)
        self.enter_fault_parked_state(event)
```

说明：

- 捕获面应覆盖 `Exception`，并谨慎处理 `BaseException` 中的 `KeyboardInterrupt/SystemExit`。系统退出类事件应记录后交给 SentinelManager 决策。
- 不要吞掉进程级 fatal；如果进程直接崩溃，主进程通过 exit monitor 记录 `EXITED`。

### 9.2 Parked state

异常后 scheduler main loop 不退出，而是进入 parked state：

```python
def enter_fault_parked_state(self, event):
    cleanup_current_iteration_state()
    while True:
        cmd = recovery_queue.get()
        if cmd.command == RETRY_REINIT:
            result = run_retry_reinit_on_main_thread(cmd)
            fault_sentinel.publish_command_result(result)
        elif cmd.command == RESUME and recovery_ok:
            self._engine_paused = False
            return
        elif cmd.command == TERMINATE:
            raise SchedulerTerminateRequested()
```

parked state 的目标是“挂住但可控”，不是 busy spin。

## 10. Out-of-band command channel

推荐使用 ZMQ ROUTER/DEALER 或现有 IPC 扩展：

```text
SentinelManager ROUTER socket
  -> FaultSentinel DEALER socket per scheduler
```

要求：

- command 有 `command_id` 和 `epoch`。
- result 必须带相同 `command_id`。
- 超时后 SentinelManager 标记 sentinel `UNRESPONSIVE`。
- 心跳独立于 command result。
- control channel 不复用 scheduler main loop 的普通 request queue。

## 11. Hard pause 工作流

### 11.1 上层下发 pause

```text
POST /fault_tolerance/apply pause hard=true
  -> SentinelManager.pause(hard=True)
  -> freeze TokenizerManager admission
  -> best-effort send PauseGenerationReqInput(mode=retract)
  -> broadcast HARD_ABORT_COMM to all FaultSentinel
  -> each FaultSentinel disables communicators
  -> each FaultSentinel aborts/destroys ProcessGroups
  -> state COMM_ABORTED or WAITING_OPERATOR
```

`PauseGenerationReqInput` 是正常路径优化；真正打断 collective 的能力来自 `HARD_ABORT_COMM`。

### 11.2 scheduler 异常触发 pause

```text
scheduler exception
  -> wrapper captures event
  -> local FaultSentinel reports event
  -> SentinelManager freezes admission
  -> SentinelManager broadcasts hard pause
  -> FaultSentinel aborts communication domain
  -> scheduler main loop parks
```

### 11.3 heartbeat stall 触发 pause

```text
FaultSentinel detects main loop heartbeat timeout
  -> reports FaultEvent(type=heartbeat_stall)
  -> SentinelManager broadcasts HARD_ABORT_COMM
  -> FaultSentinel aborts communication domain
```

## 12. Communication abort

`distributed_recovery.py`：

```python
class DistributedRecoveryManager:
    def disable_communicators(self): ...
    def abort_communicators(self, timeout_sec: int): ...
    def cleanup_distributed(self): ...
    def reinit_distributed_on_main_thread(self, params: dict[str, Any]): ...
    def health_check(self, timeout_sec: int): ...
```

### 12.1 PyNccl abort

`pynccl_wrapper.py` 增加：

```python
Function("ncclCommAbort", ncclResult_t, [ncclComm_t])
```

`pynccl.py` 增加：

```python
def abort(self):
    if getattr(self, "comm", None) is not None:
        self.nccl.ncclCommAbort(self.comm)
    self.available = False
    self.disabled = True
    self.aborted = True
    self.comm = None
```

### 12.2 ProcessGroup abort/destroy

对每个 `GroupCoordinator`：

1. disable pynccl/custom communicators。
2. abort PyNccl communicator。
3. 尝试 backend abort。
4. fallback 到 `torch.distributed.destroy_process_group(group)`。
5. 清空 `GroupCoordinator` 内部引用。

需要新增：

```python
def get_all_model_groups() -> list[GroupCoordinator]: ...
def safe_abort_or_destroy_group(group: GroupCoordinator, timeout_sec: int) -> GroupAbortResult: ...
```

abort 可使用短生命周期 bounded `ThreadPoolExecutor`，但不能创建常驻 worker thread pool。

## 13. Retry 工作流

### 13.1 SentinelManager.retry

```text
POST /fault_tolerance/apply retry
  -> SentinelManager validates state
  -> state RECOVERING
  -> broadcast PREPARE_RETRY
  -> ensure comm aborted or request HARD_ABORT_COMM
  -> broadcast RETRY_REINIT
  -> collect result from all FaultSentinel
  -> broadcast HEALTH_CHECK
  -> if success: broadcast RESUME, open admission, state RUNNING
  -> if failure: state WAITING_OPERATOR / COMM_ABORTED
```

### 13.2 FaultSentinel retry

FaultSentinel 不直接在 control thread 修改 ModelRunner。它做：

1. 接收 `RETRY_REINIT`。
2. 把 command 投递到 scheduler main loop 的 `recovery_queue`。
3. 等待 main loop 执行 reinit 并返回结果。
4. 超时则返回 failure。

### 13.3 Scheduler main thread reinit

```python
def run_retry_reinit_on_main_thread(cmd):
    cleanup_scheduler_runtime_state()
    model_runner.fault_tolerance_prepare_reinit()
    distributed_recovery.cleanup_distributed()
    distributed_recovery.reinit_distributed_on_main_thread(cmd.params)
    model_runner.fault_tolerance_rebind_distributed_groups()
    model_runner.fault_tolerance_invalidate_cuda_graphs()
    if cmd.params.get("recapture_cuda_graph", True):
        model_runner.fault_tolerance_recapture_cuda_graphs()
    distributed_recovery.health_check(timeout)
    return success
```

## 14. ModelRunner recovery callbacks

在 `ModelRunner` 增加：

```python
class ModelRunner:
    def fault_tolerance_capture_topology(self) -> TopologySnapshot: ...
    def fault_tolerance_prepare_reinit(self) -> None: ...
    def fault_tolerance_reinit_distributed(self, params: dict[str, Any]) -> None: ...
    def fault_tolerance_rebind_distributed_groups(self) -> None: ...
    def fault_tolerance_invalidate_cuda_graphs(self) -> None: ...
    def fault_tolerance_recapture_cuda_graphs(self) -> None: ...
    def fault_tolerance_health_check(self) -> None: ...
```

`TopologySnapshot`：

```python
@dataclass
class TopologySnapshot:
    world_size: int
    rank: int
    local_rank: int
    backend: str
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    attn_cp_size: int
    moe_dp_size: int
    moe_ep_size: int
    dist_init_method: str
    nccl_port: int
```

本轮 retry 使用 same-topology snapshot；后续扩展可在这里接入 topology-changing recovery。

## 15. Scheduler runtime cleanup

fault 后 retry 前必须处理：

- `running_batch`。
- `batch_queue`。
- overlap schedule `last_batch`。
- `chunked_req`。
- 当前 forward batch 临时状态。
- 已分配但不可信的 KV cache block。
- streaming 请求的部分输出状态。

默认策略：

- fault 后使用 `retract` 优先保存可重算请求。
- 已向客户端输出部分 token 的请求标记 interrupted，不能静默续接。
- 无法安全 retract 的请求返回错误或等待上层重试策略。

## 16. HTTP API

### 16.1 Status

```python
@app.get("/fault_tolerance/status")
async def fault_tolerance_status():
    manager = _global_state.sentinel_manager
    if manager is None:
        return {"enabled": False, "state": "DISABLED"}
    return manager.get_status()
```

### 16.2 Apply

```python
class FaultToleranceApplyRequest(BaseModel):
    fault_tolerance_instruction: str
    fault_tolerance_timeout: int | None = None
    fault_tolerance_params: dict[str, Any] = Field(default_factory=dict)
```

本轮处理：

- `pause`
- `retry`
- `terminate`

其他 instruction 作为扩展点返回明确的 not implemented response，不在本文档展开。

## 17. Middleware

非 `RUNNING` 状态拦截普通推理：

```json
{
  "error": {
    "type": "fault_tolerance_unavailable",
    "message": "SGLang engine is controlled by fault tolerance state COMM_ABORTED",
    "state": "COMM_ABORTED"
  }
}
```

放行：

- `/fault_tolerance/status`
- `/fault_tolerance/apply`
- `/health`
- `/metrics`
- 必要 admin endpoint

## 18. 子进程异常策略

现有 `run_scheduler_process` 的外层异常处理会发送 `SIGQUIT`。FT enabled 时需要改为：

1. scheduler main loop wrapper 尽量捕获异常并 park。
2. 如果异常仍冒泡到 process top-level：
   - 发送 `ProcessFaultEvent` 给 SentinelManager。
   - 不默认 kill process tree。
   - 当前 scheduler 标记 `EXITED` 或 `UNRESPONSIVE`。
3. 只有收到 terminate command 或配置要求自动 terminate 时，才调用现有 kill process tree。

FT disabled 时保持现有 fail-stop。

## 19. 测试计划

### 19.1 单元测试

新增：

```text
test/srt/fault_tolerance/test_state.py
test/srt/fault_tolerance/test_sentinel_manager.py
test/srt/fault_tolerance/test_fault_sentinel.py
test/srt/fault_tolerance/test_middleware.py
test/srt/fault_tolerance/test_distributed_recovery.py
```

覆盖：

- 状态迁移。
- fault event 幂等。
- manager command 聚合。
- sentinel heartbeat timeout。
- scheduler exception wrapper 进入 parked state。
- retry command 必须投递到 main loop 执行 reinit。

### 19.2 Mock recovery 测试

- fake communicator abort。
- ProcessGroup destroy fallback。
- reinit 成功/失败。
- health check 成功/失败。
- main loop reinit 超时。

### 19.3 分布式集成测试

- TP=2：scheduler exception -> hard pause -> retry -> all_reduce health check -> 推理恢复。
- TP=2：manual hard pause while decode running -> abort -> retry。
- DP>1：same-topology retry。
- CUDA graph enabled：retry 后旧 graph invalidated，重新 capture 或安全禁用。
- scheduler main loop 模拟 hang：FaultSentinel 仍能接收 hard pause 并 abort comm。

### 19.4 回归测试

- FT disabled 下原 fail-stop 行为。
- 现有 pause_generation tests。
- 权重更新 pause/retract tests。

## 20. PR 拆分

### PR 1：M1 控制面

- `state.py`
- `manager.py`
- HTTP status/apply
- middleware
- ServerArgs
- basic tests

### PR 2：FaultSentinel 线程与 command channel

- `sentinel.py`
- command schema
- scheduler sentinel init
- heartbeat
- command/result path

### PR 3：scheduler exception wrapper 与 parked state

- scheduler loop wrapper
- FaultEvent report
- parked recovery queue
- top-level fail-stop behavior adjustment under FT enabled

### PR 4：hard pause / communication abort

- PyNccl `ncclCommAbort`
- ProcessGroup abort/destroy helper
- DistributedRecoveryManager abort
- manual hard pause E2E

### PR 5：retry reinit

- ModelRunner topology snapshot
- same-topology reinit
- group rebind
- health check

### PR 6：runtime cleanup / CUDA graph / integration

- scheduler batch cleanup
- CUDA graph invalidation/recapture
- streaming interruption policy
- TP/DP integration tests

## 21. 完成定义

完成时必须满足：

1. FT disabled 行为不变。
2. FT enabled 时 scheduler exception 不默认 kill process tree。
3. FaultSentinel 是 scheduler 进程内 out-of-band 常驻控制线程。
4. 上层 hard pause 能在 scheduler main loop hang 时到达 FaultSentinel。
5. hard pause 能 abort/destroy communicator/process group。
6. retry reinit 在 scheduler main thread safe point 执行。
7. retry 后 health collective 成功。
8. retry 后普通推理恢复。
9. retry 失败后管理面仍可查询。
10. 本轮只展开 retry path，后续 topology-changing recovery 可复用 SentinelManager/FaultSentinel 架构。