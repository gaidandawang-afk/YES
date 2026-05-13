# SGLang 代码走读札记

这份笔记用于持续记录 SGLang runtime 走读中的关键问题。它不是完整架构文档，而是围绕一次请求在 Scheduler 内部如何变成 `Req`、如何进入 batch、如何从 prefill/extend 转入 decode、以及输出为什么有时走 detokenizer、有时直接回 tokenizer manager 的阅读路线。

关联主路线可参考：

- [map.md](M:/Codes/sglang-ft/YES/sglang/notes/map.md)
- [architecture.md](M:/Codes/sglang-ft/YES/sglang/notes/architecture.md)

## 1. Scheduler 输入阶段：`process_input_requests`

Scheduler 每轮 event loop 的入口大致是：

```text
recv_requests()
  -> process_input_requests(recv_reqs)
  -> get_next_batch_to_run()
  -> run_batch(batch)
  -> process_batch_result(batch, result)
```

`process_input_requests()` 做的是把刚从 tokenizer manager 或 RPC 收到的消息分发给对应 handler：

```python
output = self._request_dispatcher(recv_req)
if output is not None:
    self.send_to_tokenizer.send_output(output, recv_req)
```

这里容易误解：对正常 `generate` 请求，`_request_dispatcher` 通常不会返回最终输出。

正常 generate 路径是：

```text
TokenizedGenerateReqInput
  -> handle_generate_request()
  -> 创建 Scheduler 内部 Req
  -> _add_request_to_queue(req)
  -> 返回 None
```

也就是说，正常 generate 在 `process_input_requests()` 阶段还没有跑模型，也没有经过 detokenizer，只是入队。

会直接 `send_to_tokenizer` 的一般是控制类或即时响应，例如：

```text
FlushCacheReqOutput
AbortReq
UpdateWeights 相关输出
health check 输出
session/control 响应
部分错误响应
```

这些对象本身已经是最终业务响应，不需要 token id 转文本，所以不经过 detokenizer。

代码入口：

- `process_input_requests`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1545)
- `handle_generate_request`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1682)

## 2. 为什么有些输出直接回 tokenizer manager，而不是 detokenizer

DetokenizerManager 的职责不是通用 response router，而是处理需要 decode token id 的模型输出。

典型生成输出路径：

```text
Scheduler process_batch_result
  -> BatchTokenIDOutput
  -> DetokenizerManager
  -> BatchStrOutput / decoded text
  -> TokenizerManager
  -> HTTP response / SSE
```

对应代码：

- Scheduler 发送 `BatchTokenIDOutput`：[scheduler_output_processor_mixin.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:1125)
- Detokenizer handler 注册：[detokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/detokenizer_manager.py:128)

Detokenizer 主要处理：

```text
BatchTokenIDOutput
BatchEmbeddingOutput
BatchMultimodalDecodeReq
FreezeGCReq
```

控制类响应没有 token 序列要 decode，直接回 tokenizer manager 更短，也更符合职责边界。

因此可以记成：

```text
控制请求：
TokenizerManager -> Scheduler -> TokenizerManager

生成请求：
TokenizerManager -> Scheduler -> Model -> Scheduler -> DetokenizerManager -> TokenizerManager
```

## 3. `session_id` 如何影响 `Req`

`handle_generate_request()` 里先判断是否带 session：

```text
1. 没有 session_id
   -> 普通 generate 请求
   -> 直接用本次 input_ids 创建 Req

2. 有 session_id 且 session 存在
   -> session.create_req(...)
   -> 基于 session 历史创建 Req

3. 有 session_id 但 session 不存在
   -> 创建 abort Req
   -> 后续通过统一输出路径返回错误
```

不带 session 时：

```text
Req 的上下文 = 当前请求 prompt
```

带有效 session 时：

```text
Req 的上下文 = session 历史 + 当前请求增量
```

直观理解：

```text
不带 session_id：每次从新 prompt 起步。
带 session_id：沿着上一次保存的上下文继续走。
无效 session_id：生成一个失败回执。
```

session 会影响 `Req` 的来源、输入 token 组成、上下文复用方式，以及请求结束后如何回写 session 状态。

## 4. `grammar_manager` 在做什么

`grammar_manager` 负责结构化输出约束，例如 JSON schema、regex、EBNF、choice 等。它不负责模型 forward，也不负责 detokenize。

它插在请求入队之前：

```text
TokenizedGenerateReqInput
  -> Req
  -> validate input
  -> grammar_manager.process_req_with_grammar(req)
  -> ready: 进入 waiting_queue
  -> not ready: 暂存在 grammar queue
```

`handle_generate_request()` 末尾的逻辑是：

```python
added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
if not added_to_grammar_queue:
    self._add_request_to_queue(req)
```

含义是：

```text
没有 grammar，或 grammar 已经准备好：
  -> 返回 False
  -> 直接进入 waiting_queue

带 grammar 且还要编译：
  -> 返回 True
  -> 暂时不进 waiting_queue
  -> 编译 ready 后再入队
```

后续采样时，grammar 会参与构造合法 token mask：

```text
model logits
  -> grammar mask
  -> 屏蔽非法 token
  -> sampler 只能采样合法 token
```

它的核心价值是避免 scheduler 把 grammar 还没准备好的请求拿去跑模型。

## 5. `extend` 是什么

SGLang 里的 `extend` 基本等价于 prefill，但名字更精确：它表示把一段新 token 接到已有上下文后面，并为这段 token 计算 KV cache。

例子：

```text
已有上下文: A B C
新输入:     D E F

extend 做的事：
  处理 D E F
  写入 D/E/F 的 KV cache
  用最后位置 logits 采样第一个 next token
```

它不只发生在新请求开头，也可能发生在：

```text
新请求 prompt prefill
chunked prefill 的某一段
session 追加的新 user input
radix cache 命中 prefix 后的未命中 suffix
mixed chunk 里的新 prefill token
```

和 decode 的区别：

```text
extend:
  一次处理多个新增上下文 token
  主要目的是填 KV cache
  token 来自 prompt / 新增输入

decode:
  每个请求通常一次处理 1 个 token
  token 来自上一轮采样结果
  产出下一个 token
```

一个简单生成流程：

```text
prompt = [10, 20, 30, 40]

extend:
  输入 [10, 20, 30, 40]
  写 prompt KV
  采样 token 50

decode step 1:
  输入 [50]
  写 token 50 KV
  采样 token 60

decode step 2:
  输入 [60]
  写 token 60 KV
  采样 token 70
```

## 6. `last_batch` 在 `get_next_batch_to_run()` 里的作用

`last_batch` 是上一轮 event loop 刚运行过的 batch。在 `get_next_batch_to_run()` 里，最关键的用途是把上一轮完成的 `EXTEND` batch 转入 `running_batch`。

代码逻辑简化：

```python
if self.last_batch and self.last_batch.forward_mode.is_extend():
    self.last_batch.filter_batch(...)
    if not self.last_batch.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch
        else:
            self.running_batch.merge_batch(self.last_batch)
```

语义是：

```text
上一轮 prefill/extend 完成的请求
  -> 如果还没 finished
  -> 如果不是 chunked 未完成等特殊情况
  -> 合并进 running_batch
  -> 后续继续 decode
```

为什么要 filter：

```text
1. finished 请求不该进入 decode
2. prefill-only 请求不需要 decode
3. chunked prefill 可能只是完成了一段 prompt，还不能 decode
4. DLLM / staging 请求可能要暂时排除
```

典型时间线：

```text
loop 1:
  get_next_batch_to_run()
    -> 从 waiting_queue 取 A
    -> 返回 A 的 EXTEND batch
  run_batch(A extend)
  process_batch_result(A extend)
  last_batch = A_extend

loop 2:
  get_next_batch_to_run()
    -> 发现 last_batch 是 EXTEND
    -> filter A_extend
    -> merge 到 running_batch
    -> 如果没有新 prefill
    -> update_running_batch(running_batch)
    -> 返回 A 的 DECODE batch
```

所以 `last_batch` 在这里是 prefill 到 decode 的桥。

## 7. `running_batch` 不是生产消费队列

`running_batch` 容易被误认为是“每轮取出一批，跑完就删除”的消费队列。更准确地说，它是活跃 decode 请求集合。

普通队列消费是：

```text
queue = [A, B, C]
consumer 取出 A
queue = [B, C]
```

`running_batch` 是：

```text
running_batch = [A, B, C]

本轮 decode:
  用 A/B/C 各自上一轮最后 token 做输入
  每个请求生成一个新 token

跑完后:
  running_batch 仍然是 [A, B, C]
  只是每个 req.output_ids 多了一个 token
```

只有请求结束时才会被过滤掉：

```text
A finished
running_batch = [B, C]
```

因此：

```text
waiting_queue: 更像新请求队列
last_batch: prefill 到 decode 的中转暂存
running_batch: 未完成生成请求的工作集
```

## 8. `running_batch` 谁赋值，为什么 `recv_requests()` 后它不一定为空

`recv_requests()` 只负责收新请求或控制消息。`running_batch` 是 Scheduler 对象上的持久状态，跨 event loop 保存，不会因为本轮没有新请求就清空。

第一次请求进入时：

```text
loop 1:
  recv_requests() 收到请求 A
  process_input_requests() 把 A 放入 waiting_queue
  get_next_batch_to_run() 返回 A 的 EXTEND batch
  run_batch(A extend)
  process_batch_result(A extend)
  last_batch = A_extend
  running_batch 可能还是空
```

下一轮即使没有新请求：

```text
loop 2:
  recv_requests() = []
  process_input_requests([]) 什么都不做
  get_next_batch_to_run()
    -> last_batch 是 EXTEND
    -> running_batch = last_batch
    -> 返回 decode batch
```

之后 decode 多轮：

```text
loop 3:
  recv_requests() = []
  running_batch 仍有 A
  update_running_batch(A)
  run decode

loop 4:
  如果 A finished
  update_running_batch/filter_batch 把 A 移除
  running_batch 变空
```

所以即使 `recv_requests()` 每轮都为空，`running_batch` 仍可能有正在生成的旧请求。

## 9. `get_next_batch_to_run()` 拆解

`get_next_batch_to_run()` 是每轮 scheduler 的总调度入口，决定本轮跑什么：

```text
1. 新请求的 prefill/extend
2. 老请求的 decode
3. idle
```

简化流程：

```text
get_next_batch_to_run()
  1. 处理 waiting/running 超时
  2. 处理 DLLM finished 状态
  3. 如果 last_batch 是 EXTEND，把它过滤后合并进 running_batch
  4. 清理 prefill-only running_batch
  5. 尝试从 waiting_queue 构造新的 prefill batch
  6. 如果有 new_batch，优先返回 new_batch
  7. 如果没有 new_batch，从 running_batch 构造 decode batch
  8. 做 DP attention / MLP sync / ngram 等补充处理
  9. 返回本轮要 run 的 batch
```

核心决策可以压缩成：

```python
new_batch = self.get_new_batch_prefill()

if new_batch is not None:
    ret = new_batch
else:
    if not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:
        self.running_batch = self.update_running_batch(self.running_batch)
        ret = self.running_batch if not self.running_batch.is_empty() else None
    else:
        ret = None
```

阅读重点：

- `get_next_batch_to_run`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2130)
- `get_new_batch_prefill`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2238)
- `update_running_batch`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2488)

## 10. `get_new_batch_prefill()` 在做什么

`get_new_batch_prefill()` 从 `waiting_queue` 里挑新请求，构造 `EXTEND` batch。

它会考虑：

```text
grammar 是否 ready
hierarchical cache / HiCache 状态
priority scheduling / preemption
running_batch 是否已满
KV cache/token pool 空间
chunked_prefill_size
prefix/radix cache 命中
LoRA batch 兼容性
prefill_max_requests / max_prefill_tokens
```

成功时：

```text
waiting_queue 中可运行请求
  -> ScheduleBatch.init_new(...)
  -> prepare_for_extend()
  -> 返回 EXTEND batch
```

如果开启 mixed chunk，并且条件允许：

```text
new prefill batch + running_batch decode
  -> new_batch.mix_with_running(self.running_batch)
  -> 本轮一个 batch 同时包含 prefill 和 decode token
```

这也是为什么不能简单理解成“prefill 和 decode 永远分开”。

## 11. `update_running_batch()` 拆解

`update_running_batch()` 是 decode 前的整理函数。它处理已有活跃请求集合：

```text
running_batch 里哪些请求还活着？
KV cache 还够不够？
如果不够，哪些请求要 retract？
如果够，如何准备下一轮 decode input？
```

简化流程：

```text
update_running_batch(batch)
  1. filter_batch(): 移除 finished/abort/无效请求
  2. 如果空了，返回空 batch
  3. check_decode_mem(): 检查下一轮 decode KV 空间是否够
  4. 如果不够，retract_decode(): 回退一部分请求
  5. 更新 new_token_ratio / batch_is_full
  6. 如果仍非空，prepare_for_decode()
  7. 返回可执行 decode 的 batch
```

`prepare_for_decode()` 的直观含义：

```text
对每个 req:
  取上一轮生成的最后一个 token
  作为本轮 input_id

reqA.output_ids[-1] = 103
reqB.output_ids[-1] = 202

batch.input_ids = [103, 202]
```

然后模型 forward：

```text
input [103, 202]
  -> next [104, 203]
```

处理结果时再把新 token append 回各自请求。

`retract_decode()` 的含义不是丢请求，而是在 KV 空间不足时把部分活跃请求暂时撤回，给剩下请求腾空间：

```text
running_batch = [A, B, C, D]
KV 不够
retract C/D
running_batch = [A, B]
C/D 回到等待/重调度路径
```

## 12. 一轮完整状态迁移示例

单请求 A，生成 3 个 token：

```text
初始:
  waiting_queue = []
  running_batch = []
  last_batch = None

loop 1:
  recv A
  process_input_requests -> waiting_queue = [A]
  get_next_batch_to_run -> A_extend
  run_batch(A_extend) -> 生成第 1 个 token
  process_batch_result(A_extend)
  last_batch = A_extend
  running_batch = []

loop 2:
  recv_requests = []
  get_next_batch_to_run:
    last_batch 是 EXTEND
    running_batch = A
    update_running_batch(A)
    return A_decode
  run_batch(A_decode) -> 生成第 2 个 token
  last_batch = A_decode
  running_batch = [A]

loop 3:
  recv_requests = []
  get_next_batch_to_run:
    last_batch 不是 EXTEND，不合并
    running_batch 仍是 [A]
    update_running_batch(A)
    return A_decode
  run_batch(A_decode) -> 生成第 3 个 token
  process_batch_result 后 A finished

loop 4:
  update_running_batch/filter_batch 移除 A
  running_batch = []
  ret = None
```

一句话总结：

```text
get_next_batch_to_run 是调度决策器。
get_new_batch_prefill 是新请求 prefill 构造器。
last_batch 是 prefill 到 decode 的桥。
running_batch 是活跃 decode 请求集合。
update_running_batch 是 decode 工作集整理器。
```
## 13. `chunked_req` 的意义和更新时机

`chunked_req` 表示当前有一个请求的 prefill/extend 太长，不能在本轮一次性处理完，所以被切成多段继续处理。

它不是一个队列，而是 Scheduler 上保存的“当前未完成 chunked prefill 请求”：

```text
self.chunked_req = None 或 某一个 Req
```

核心目的：

```text
1. 限制单轮 prefill token 数，避免长 prompt 独占 GPU
2. 控制 KV/cache 预算，避免一次 prefill 申请过多 token
3. 让长 prompt 可以分多轮 extend，并和其他请求/old decode 交错
4. 在 chunk 未完成前，阻止该 req 被误合并进 running_batch 做 decode
```

普通长 prompt 示例：

```text
prompt 长度 = 10000
chunked_prefill_size = 2048

第 1 轮 EXTEND: 处理 token 0..2047
第 2 轮 EXTEND: 处理 token 2048..4095
...
最后一轮 EXTEND: 处理剩余 token
最后一轮完成后，才允许进入 running_batch decode
```

### 13.1 为什么 `last_batch` 合并到 `running_batch` 时要排除 `chunked_req`

每轮 `get_next_batch_to_run()` 开头会处理上一轮 `last_batch`：

```text
last_batch 是 EXTEND
  -> filter_batch(...)
  -> merge 到 running_batch
```

但如果 `last_batch` 里包含 chunked prefill 尚未完成的请求，这个请求不能进入 decode。原因是它的完整 prompt KV 还没建完。

所以代码会先把当前 `self.chunked_req` 放进排除集合：

```python
if self.chunked_req is not None:
    chunked_req_to_exclude.add(self.chunked_req)
    self.stash_chunked_request(self.chunked_req)
```

之后：

```python
self.last_batch.filter_batch(
    chunked_req_to_exclude=list(chunked_req_to_exclude)
)
```

语义是：

```text
上一轮跑完一个 chunk
  -> 这只是 prompt 的一段
  -> 不要把它当成已完成 prefill 的请求
  -> 先从 last_batch -> running_batch 的合并路径里排除
```

### 13.2 已有 `chunked_req` 如何继续跑下一段

在 `_get_new_batch_prefill_raw()` 里，如果当前已有 `self.chunked_req`：

```python
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

`init_next_round_input()` 会重新计算：

```text
fill_ids = origin_input_ids + output_ids
prefix_indices = 已经命中的/已经缓存的 KV 前缀
extend_input_len = len(fill_ids) - len(prefix_indices)
```

`adder.add_chunked_req()` 会根据本轮剩余 chunk budget 决定这次能处理多少 token：

```python
truncated = req.extend_input_len > _rem_tokens
req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))
req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
self.can_run_list.append(req)
return req if truncated else None
```

所以更新规则是：

```text
如果这一轮仍没处理完整个 prompt:
  self.chunked_req 继续等于这个 req

如果这一轮已经处理完剩余 prompt:
  self.chunked_req = None
```

### 13.3 新的 `chunked_req` 什么时候产生

遍历 `waiting_queue` 时，普通新请求会先 `req.init_next_round_input(tree_cache)`，然后进入：

```python
res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None), ...)
```

如果这个请求的 `extend_input_len` 超过本轮 `rem_chunk_tokens`，`add_one_req()` 会把它截断成一个 chunk：

```python
trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
req.set_extend_input_len(trunc_len)
req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
self.can_run_list.append(req)
self.new_chunked_req = req
```

回到 Scheduler 后：

```python
if adder.new_chunked_req is not None:
    assert self.chunked_req is None
    self.chunked_req = adder.new_chunked_req
```

语义是：

```text
这个 waiting_queue 请求太长
  -> 本轮只处理前一段
  -> 把它记录成 self.chunked_req
  -> 下一轮继续处理剩余部分
```

### 13.4 `is_chunked` 的作用

每次当前存在 `self.chunked_req`，Scheduler 会：

```python
if self.chunked_req is not None:
    self.chunked_req.is_chunked += 1
```

`Req.is_chunked` 是一个计数/标记，表示这个请求正在经历 chunked prefill。`Req` 注释里说明它在被 chunk 时递增，在 chunked request 被处理时递减。阅读时可以把它理解为：这个请求不是普通一次性 prefill，而是被拆段处理过。

### 13.5 一句话总结

```text
chunked_req = 当前那个“prompt 还没 prefill 完、下一轮还要继续 extend”的请求。
```

它会在两种情况下更新：

```text
1. 新请求太长，被 add_one_req 截断：
   adder.new_chunked_req -> self.chunked_req

2. 旧 chunked_req 继续处理下一段：
   self.chunked_req = adder.add_chunked_req(self.chunked_req)
   返回 req 表示还没完，返回 None 表示最后一段处理完
```