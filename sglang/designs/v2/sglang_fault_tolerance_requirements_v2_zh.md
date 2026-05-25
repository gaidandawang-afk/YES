# SGLang Fault-Tolerance 需求文档 v2

## 0. 文档信息

- 文档目标：定义 SGLang fault-tolerance 的外部语义、接口边界和验收标准。
- 核心原则：SGLang 提供引擎内的故障接管、rank 状态查询、retry 和 scale_down；实例级调度、重试策略和扩缩容决策由上层服务框架完成。
- 对外接口：`GET /fault_tolerance/status` 和 `POST /fault_tolerance/apply`。
- 对外动作：`retry`、`scale_down`。
- 内部能力：pause-on-error。发生故障后 SGLang 可以自动暂停并等待上层服务决策，但不提供 FT 专用的外部 pause instruction。

## 1. 背景和角色分工

SGLang 在容错场景下默认运行在上层服务框架内。上层服务框架负责实例生命周期、请求重放、流量切换、实例替换和扩缩容策略。SGLang 只负责把本实例内部的 rank 故障转化为可查询、可恢复的运行时状态。

角色分工如下：

| 组件 | 职责 |
| --- | --- |
| 上层服务框架 | 监控 status，决定调用 `retry` 或 `scale_down`，处理客户端请求重放和实例级流量切换。 |
| SGLang FT 控制面 | 捕获故障，更新 rank 状态，执行 pause-on-error，分发 retry/scale_down 命令。 |
| Scheduler/FaultSentinel | 在 scheduler 进程内接收控制命令，清理运行时状态，触发通信域重建。 |
| 通信后端 | 提供通信域 abort、reinit、active-rank 隔离或 fault-tolerant continue 能力。 |

## 2. 需求范围

### 2.1 必须支持

1. FT 默认关闭；开启后故障进入 SGLang FT 控制面。
2. 故障发生后，SGLang 根据配置执行 pause-on-error 或 backend-driven continue。
3. 上层服务可以查询每个 rank 的状态，状态只包含 `healthy`、`dead`、`paused`。
4. 上层服务可以调用 `retry`，在不隔离 rank 的情况下重建通信域并恢复服务。
5. 上层服务可以调用 `scale_down`，隔离指定 rank，其他 rank 重建通信域并恢复服务。
6. `retry` 和 `scale_down` 都必须丢弃本实例所有 in-flight 请求。
7. `scale_down` 接受指定 rank 的当前状态为 `dead`、`paused` 或 `healthy`；API 层不做强校验。
8. 对 mooncake、nixl 等支持故障续推的后端，必须有配置项区分故障后默认暂停还是默认续推。

### 2.2 非目标

以下能力不作为 SGLang FT 对外语义的一部分：

- FT 专用外部 pause instruction。
- 上层请求重放。
- 跨实例请求迁移。
- 替换 dead rank 并恢复到原 world。
- 持久化 in-flight 请求状态。
- 在 status 中暴露复杂阶段、traceback、topology、epoch 或内部 command 结果。

现有非 FT 的 `/pause_generation` 和 `/continue_generation` 属于已有管理接口，不作为本文档定义的 FT 控制面。

## 3. Rank 状态模型

对外只暴露 rank 级状态。

| 状态 | 含义 |
| --- | --- |
| `healthy` | rank 当前参与可服务通信域，可以接收普通推理工作。 |
| `paused` | rank 仍在进程内可控，但本实例已暂停或该 rank 等待恢复动作。 |
| `dead` | rank 已退出、不可达、被判定失效，或已被 `scale_down` 隔离。 |

状态规则：

1. FT 启动成功后，所有 rank 初始为 `healthy`。
2. pause-on-error 策略下，故障后仍可控的 rank 进入 `paused`。
3. 进程退出、heartbeat 不可达、控制面不可达或被 `scale_down` 指定隔离的 rank 进入 `dead`。
4. `retry` 成功后，参与恢复的 rank 进入 `healthy`。
5. `scale_down` 成功后，未隔离 rank 进入 `healthy`，被隔离 rank 保持 `dead`。

## 4. 故障后策略

新增配置项：

```text
--fault-tolerance-on-error-strategy {pause,continue}
```

建议内部字段：

```python
fault_tolerance_on_error_strategy: Literal["pause", "continue"] = "pause"
```

### 4.1 `pause`

`pause` 是通用策略。任一 rank 报告不可继续执行的错误时：

1. SGLang 停止接收普通推理请求。
2. 可控 rank 进入 `paused`。
3. 不可信的 in-flight 请求被中断，不保证继续输出。
4. 上层服务通过 status 看到 rank 状态后，选择调用 `retry` 或 `scale_down`。

该策略适用于普通 NCCL/ProcessGroup 路径，也适用于上层希望统一接管恢复决策的 mooncake/nixl 部署。

### 4.2 `continue`

`continue` 适用于支持故障续推的 backend，例如 mooncake 和未来 nixl 的 active-rank 隔离能力。任一 rank 发生故障时：

1. 后端尝试自动隔离故障 rank。
2. SGLang 将故障 rank 记录为 `dead`。
3. 仍可服务的 rank 保持 `healthy`。
4. 普通推理入口保持打开。
5. 如果后端无法完成隔离或健康检查失败，SGLang 回退到 pause-on-error。

`continue` 只在后端明确支持 fault-tolerant continue 时生效；否则按 `pause` 处理并记录日志。

## 5. Status API

### 5.1 请求

```http
GET /fault_tolerance/status
```

### 5.2 响应

status 只返回每个 rank 的状态。

```json
{
  "ranks": [
    {"rank": 0, "state": "healthy"},
    {"rank": 1, "state": "dead"},
    {"rank": 2, "state": "paused"},
    {"rank": 3, "state": "paused"}
  ]
}
```

约束：

- `rank` 使用 SGLang 启动时的原始 global rank。
- `state` 只能是 `healthy`、`dead`、`paused`。
- 响应不包含全局状态、epoch、last fault、traceback、topology 或 command details。

## 6. Apply API

### 6.1 请求格式

复用现有 `fault_tolerance_instruction` 请求格式：

```http
POST /fault_tolerance/apply
Content-Type: application/json
```

```json
{
  "fault_tolerance_instruction": "retry",
  "fault_tolerance_timeout": 60,
  "fault_tolerance_params": {}
}
```

支持的 instruction：

| instruction | 说明 |
| --- | --- |
| `retry` | 不隔离 rank，重建通信域并恢复到 `healthy`。 |
| `scale_down` | 隔离指定 rank，其他 rank 重建通信域并恢复到 `healthy`。 |

不提供 `pause` instruction。故障暂停由 pause-on-error 自动触发。

### 6.2 响应格式

```json
{
  "success": true,
  "message": "retry succeeded",
  "ranks": [
    {"rank": 0, "state": "healthy"},
    {"rank": 1, "state": "healthy"}
  ]
}
```

失败响应应包含 `success=false` 和简短 `message`。详细 traceback 不通过 FT API 暴露。

## 7. Retry 需求

### 7.1 请求

```json
{
  "fault_tolerance_instruction": "retry",
  "fault_tolerance_timeout": 60,
  "fault_tolerance_params": {
    "torch_empty_cache": true
  }
}
```

### 7.2 行为

`retry` 用于所有需要相同 rank 集合重新建连的恢复场景。

执行要求：

1. 只对未被隔离的 rank 下发恢复命令。
2. 丢弃所有 in-flight 请求，包括 running batch、waiting queue、overlap result queue 和不可信的中间状态。
3. abort/destroy 当前通信域。
4. 使用当前 active rank 集合重建通信域。
5. 执行轻量 health check。
6. 成功后 active rank 状态设为 `healthy`。
7. 失败后 rank 保持 `paused` 或 `dead`，普通请求入口保持关闭。

`retry` 不改变 active rank 集合。若某些必要 rank 已经 `dead`，上层服务应选择 `scale_down` 或实例替换；SGLang 可以让 `retry` 执行失败并保持可查询。

## 8. Scale Down 需求

### 8.1 请求

```json
{
  "fault_tolerance_instruction": "scale_down",
  "fault_tolerance_timeout": 60,
  "fault_tolerance_params": {
    "ranks": [1, 3]
  }
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `ranks` | 需要隔离的原始 global rank 列表。 |

实现可以兼容 `rank`、`ranks`、`isolate_rank`、`isolate_ranks`，但文档推荐使用 `ranks`。

### 8.2 前置条件

`scale_down` 面向 paused 场景。满足以下任一情况即可执行：

- 已触发 pause-on-error，至少一个 rank 为 `paused` 或 `dead`。
- 上层服务确认实例不再接收普通推理请求，并希望隔离指定 rank。

`scale_down` 不强制要求被隔离 rank 当前一定是 `dead`。指定 `healthy` 或 `paused` rank 时，SGLang 也应按隔离处理。

当所有 rank 都是 `healthy` 且普通请求入口打开时，`scale_down` 应返回失败，避免在正常服务中直接改变 active rank 集合。

### 8.3 行为

执行要求：

1. 将请求中的 rank 加入 isolated set，并对外显示为 `dead`。
2. 若被隔离 rank 仍可达，best-effort 通知其停止参与服务；该通知失败不阻塞 scale_down。
3. 对未隔离 rank 清理所有 in-flight 请求。
4. 对未隔离 rank abort/destroy 当前通信域。
5. 使用 active rank 集合重建通信域。
6. 执行 active rank health check。
7. 成功后 active rank 状态设为 `healthy`，普通请求入口恢复。
8. 失败后 active rank 保持 `paused`，isolated rank 保持 `dead`。

### 8.4 In-flight 请求

`scale_down` 成功或失败都不保留本实例 in-flight 请求。上层服务需要按自己的请求语义决定是否重放。

## 9. 普通请求入口

SGLang FT 控制普通推理入口：

1. 所有 active rank 为 `healthy` 时，普通推理请求放行。
2. 任一 active rank 为 `paused` 时，普通推理请求返回 503。
3. `dead` rank 不属于 active rank 集合；如果其他 active rank 均为 `healthy`，scale_down 后可以继续服务。
4. FT 管理接口、health 和 metrics 不受普通请求入口拦截。

503 响应只需要说明引擎暂不可用，不暴露复杂内部状态。

## 10. Mooncake/NIXL 兼容需求

对支持 active-rank 隔离和故障续推的 backend，SGLang 需要提供统一策略开关：

```text
--fault-tolerance-on-error-strategy continue
```

需求：

1. `pause` 策略下，即使 backend 支持续推，SGLang 也按 pause-on-error 处理。
2. `continue` 策略下，SGLang 允许 backend 自动隔离故障 rank 并继续服务。
3. backend 续推成功后，status 中故障 rank 为 `dead`，其他 rank 为 `healthy`。
4. backend 续推失败或能力不可用时，SGLang 回退为 pause-on-error。
5. scale_down 与 backend active-rank 隔离机制使用同一份 isolated/active rank 信息。

## 11. 错误处理

### 11.1 `retry` 失败

`retry` 失败时：

- 普通请求入口保持关闭。
- 可控 rank 状态保持 `paused`。
- 已退出或不可达 rank 状态保持 `dead`。
- 上层服务可以继续查询 status，并决定再次 retry、scale_down 或替换实例。

### 11.2 `scale_down` 失败

`scale_down` 失败时：

- 请求中指定隔离的 rank 仍显示为 `dead`。
- 未成功恢复的 active rank 显示为 `paused`。
- 普通请求入口保持关闭。
- 上层服务可以继续 scale_down 更多 rank、retry active rank，或替换实例。

### 11.3 重复请求

`retry` 和 `scale_down` 应具备幂等倾向：

- 重复隔离同一 rank 不应报错。
- 对已经 `healthy` 的 active rank 执行 retry 可以成功或返回无操作成功。
- 对不可达 rank 的隔离通知失败不应导致整个 scale_down 直接失败。

## 12. 验收标准

### 12.1 Status

- FT 启用后 status 返回所有 rank 的 `healthy` 状态。
- 故障后 status 只包含 `healthy`、`dead`、`paused`。
- status 不返回内部状态机字段、fault 详情或 topology。

### 12.2 Pause-on-error

- `fault_tolerance_on_error_strategy=pause` 时，rank 故障会自动暂停服务。
- 暂停后普通推理请求返回 503。
- 暂停后上层服务可以调用 `retry` 或 `scale_down`。
- FT API 不提供 `pause` instruction。

### 12.3 Continue strategy

- `fault_tolerance_on_error_strategy=continue` 且 backend 支持续推时，故障 rank 被记录为 `dead`，其他 rank 保持 `healthy`。
- backend 不支持或续推失败时，系统回退到 pause-on-error。

### 12.4 Retry

- `retry` 会丢弃所有 in-flight 请求。
- `retry` 会重建通信域并执行 health check。
- `retry` 成功后 active rank 全部为 `healthy`，普通推理恢复。
- `retry` 失败后 status 仍可查询。

### 12.5 Scale Down

- `scale_down` 可以隔离 `dead`、`paused` 或 `healthy` rank。
- 被隔离 rank 对外显示为 `dead`。
- 未隔离 rank 重建通信域并执行 health check。
- `scale_down` 成功后未隔离 rank 为 `healthy`，普通推理恢复。
- `scale_down` 丢弃所有 in-flight 请求。

## 13. 结论

SGLang FT v2 对外提供简洁的 rank 状态查询和两个恢复动作：`retry` 与 `scale_down`。故障后的暂停是内部 pause-on-error 行为，由上层服务框架根据 status 决定后续恢复路径。对 mooncake/nixl 等可续推 backend，通过 `fault_tolerance_on_error_strategy` 明确选择故障后暂停或继续服务。
