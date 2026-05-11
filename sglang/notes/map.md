下面是一条按“徒步路线图”组织的 SGLang 推理主线，基于 [YES/sglang/notes/architecture.md](M:/Codes/sglang-ft/YES/sglang/notes/architecture.md) 和本地源码核对。

**主路线**
```text
Client
  -> FastAPI HTTP Server
  -> TokenizerManager
  -> Scheduler
  -> ScheduleBatch / Req
  -> TpModelWorker
  -> ModelRunner
  -> Scheduler output processor
  -> DetokenizerManager
  -> TokenizerManager
  -> HTTP response / SSE
```

**0. 起点：服务启动**
入口先看 [launch_server.py](M:/Codes/sglang-ft/sglang/python/sglang/launch_server.py)，它解析 `ServerArgs` 后进入 HTTP server。HTTP server 再通过 [http_server.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/http_server.py:2150) 的 `launch_server` 拉起运行时。

真正的多进程 runtime 在 [engine.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/engine.py:602) 的 `_launch_subprocesses`：主进程保留 `HTTP Server + TokenizerManager`，子进程启动 `Scheduler` 和 `DetokenizerManager`。Scheduler 进程启动逻辑在 [engine.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/engine.py:503)。

**1. 山门：HTTP 请求进入**
原生 `/generate` 路由在 [http_server.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/http_server.py:678)。

```text
POST /generate JSON
  -> GenerateReqInput
  -> tokenizer_manager.generate_request(...)
```

如果 `stream=True`，HTTP 层把 `TokenizerManager` 产出的每个 chunk 包成 SSE：

```text
data: {...}\n\n
data: [DONE]\n\n
```

对应代码在 [http_server.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/http_server.py:680)。非流式则等待 `generate_request(...).__anext__()` 的最终结果，见 [http_server.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/http_server.py:699)。

OpenAI 兼容 `/v1/chat/completions` 是一条入口支线，路由在 [http_server.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/http_server.py:1410)，最终也会适配成 `GenerateReqInput`，并回到同一条 runtime 主线。

**2. 第一段山路：TokenizerManager**
核心入口是 [tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:481) 的 `generate_request`。

这里做几件事：

```text
GenerateReqInput
  -> normalize_batch_and_arguments()
  -> 校验 rid / priority / LoRA / pause 状态
  -> _tokenize_one_request()
  -> _create_tokenized_object()
  -> _send_one_request()
  -> _wait_one_response()
```

关键点：

- `obj.normalize_batch_and_arguments()` 统一单请求和 batch 请求。
- 如果用户直接传 `input_ids`，基本跳过文本 tokenize，直接形成 token 输入。
- 如果用户传 `text`，这里调用 tokenizer 得到 `input_ids`。
- 如果有图像、音频、视频，多模态处理也在这里接入。
- `TokenizerManager` 创建并持有 `ReqState`，后续响应回来时靠 `rid` 找回请求状态。

可重点读：

- `ReqState`：[tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:128)
- `TokenizerManager`：[tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:176)
- `_tokenize_one_request`：[tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:668)
- `_create_tokenized_object`：[tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:931)
- `_wait_one_response`：[tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:1122)

跨进程发给 Scheduler 的对象主要是 `TokenizedGenerateReqInput`，定义在 [io_struct.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/io_struct.py:659)。

**3. 第二段：Scheduler 收请求、入队**
Scheduler 主类在 [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:270)。

主循环在 [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1273)：

```text
while True:
  recv_requests()
  process_input_requests()
  get_next_batch_to_run()
  run_batch()
  process_batch_result()
```

普通循环实现见 [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1286)。

`TokenizedGenerateReqInput` 到达后，`process_input_requests` 分发到 `handle_generate_request`：

- `process_input_requests`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1545)
- `handle_generate_request`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:1680)

这里会把跨进程输入转成 Scheduler 内部的 `Req`，然后进入 `waiting_queue`。

**4. 第三段：动态调度，决定 prefill 还是 decode**
调度决策入口是 [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2130) 的 `get_next_batch_to_run`。

可以按这个 mental model 看：

```text
waiting_queue: 还没 prefill 的请求
running_batch: 已 prefill、正在 decode 的请求

每轮调度：
  1. 把上一轮 prefill 后未完成的请求并入 running_batch
  2. 优先尝试构造新的 prefill batch
  3. 否则从 running_batch 构造 decode batch
  4. 没活则 idle
```

关键函数：

- `get_new_batch_prefill`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2238)
- `update_running_batch`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2488)
- `run_batch`：[scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2577)

这里最重要的内部对象在 [schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py)：

- `Req`：[schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py:493)
- `ScheduleBatch`：[schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py:1225)
- `Req.init_next_round_input`，做 prefix cache 匹配：[schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py:872)
- `prepare_for_extend`，准备 prefill/extend batch：[schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py:1478)
- `prepare_for_decode`，准备 decode batch：[schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py:2002)
- `get_model_worker_batch`，跨到 worker 边界：[schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py:2255)

**5. 第四段：模型执行**
Scheduler 的 `run_batch` 会把 `ScheduleBatch` 转成 `ModelWorkerBatch`，再调用 `TpModelWorker.forward_batch_generation`。

关键入口在 [tp_worker.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tp_worker.py:450)。

```text
ScheduleBatch
  -> ModelWorkerBatch
  -> ForwardBatch.init_new()
  -> ModelRunner.forward()
  -> ModelRunner.sample()
  -> GenerationBatchResult
```

重点读：

- `ForwardBatch`：[forward_batch_info.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/forward_batch_info.py:280)
- `ForwardBatch.init_new`：[forward_batch_info.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/forward_batch_info.py:436)
- `ModelRunner`：[model_runner.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/model_runner.py:285)
- `ModelRunner.forward`：[model_runner.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/model_runner.py:2606)
- `_forward_raw`：[model_runner.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/model_runner.py:2665)
- `forward_decode`：[model_runner.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/model_runner.py:2503)
- `forward_extend`：[model_runner.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/model_runner.py:2526)
- `sample`：[model_runner.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/model_executor/model_runner.py:2765)

prefill 走 `forward_extend`，decode 走 `forward_decode`。最后在 TP/PP 最后 rank 上采样出 `next_token_ids`。

**6. 第五段：Scheduler 处理模型输出**
模型返回 `GenerationBatchResult` 后，Scheduler 调 `process_batch_result`，入口在 [scheduler.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler.py:2752)。

实际输出处理逻辑在 mixin：

- prefill 输出处理：[scheduler_output_processor_mixin.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:121)
- decode 输出处理：[scheduler_output_processor_mixin.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:358)

这里会：

```text
next_token_ids
  -> 追加到 Req.output_ids
  -> check_finished()
  -> 判断 stop token / stop string / max_new_tokens / grammar
  -> 需要输出时构造 BatchTokenIDOutput
  -> 发给 DetokenizerManager
```

`Req.check_finished` 在 [schedule_batch.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/schedule_batch.py:1087)。

**7. 第六段：Detokenizer 增量解码**
Detokenizer 主类在 [detokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/detokenizer_manager.py:74)。

事件循环很短，见 [detokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/detokenizer_manager.py:144)：

```text
recv BatchTokenIDOutput
  -> handle_batch_token_id_out()
  -> _decode_batch_token_id_output()
  -> BatchStrOutput
  -> send back to TokenizerManager
```

关键函数：

- `_decode_batch_token_id_output`：[detokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/detokenizer_manager.py:225)
- `handle_batch_token_id_out`：[detokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/detokenizer_manager.py:360)

这里的核心不是简单 `decode(all tokens)`，而是维护增量 decode 状态，避免 UTF-8 或 tokenizer 边界被截断，并在 stop string/token 命中时做 trim。

**8. 终点：TokenizerManager 聚合并返回 HTTP**
Detokenizer 返回 `BatchStrOutput` 后，主进程的 `TokenizerManager.handle_loop` 接收，入口在 [tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:1506)。

`_handle_batch_output` 在 [tokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/tokenizer_manager.py:1515)，它按 `rid` 找到 `ReqState`，更新累计文本、token ids、meta info，然后唤醒 `_wait_one_response`。

最后：

```text
流式:
  ReqState.out_list 有新 chunk
  -> _wait_one_response yield
  -> HTTP SSE data: {...}

非流式:
  等 finished
  -> 返回最终 JSON
```

**关键支线**
1. **OpenAI Chat 支线**  
   `/v1/chat/completions` 在 [http_server.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/http_server.py:1410)，先做 OpenAI 协议适配和 chat template，之后回到 `GenerateReqInput -> TokenizerManager.generate_request`。

2. **Embedding / classify / rerank 支线**  
   `/encode` 在 [http_server.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/entrypoints/http_server.py:710)，仍走 `TokenizerManager.generate_request`，但输入是 `EmbeddingReqInput`，模型输出是 `BatchEmbeddingOutput`，Detokenizer 基本透传，见 [detokenizer_manager.py](M:/Codes/sglang-ft/sglang/python/sglang/srt/managers/detokenizer_manager.py:181)。

3. **Prefix cache / KV cache 支线**  
   读 `Req.init_next_round_input` 和 `ScheduleBatch.prepare_for_extend`。这里决定 prompt 哪些 token 命中 radix/prefix cache，哪些 token 需要真的 prefill，并分配 KV slot。

4. **流式输出支线**  
   流式不是 HTTP 层主动轮询模型，而是 Scheduler 按 `stream_interval` 或 finished 产生 `BatchTokenIDOutput`，经 Detokenizer 变成文本 delta，再唤醒 HTTP SSE generator。

5. **多 GPU / DP 支线**  
   `dp_size > 1` 时，`TokenizerManager` 和 Scheduler 中间会插入 `DataParallelController`，负责把 tokenized 请求路由到不同 DP worker group。主路线不变，只是 `TokenizerManager -> Scheduler` 中间多一个分发站。

建议实际走读顺序：先看 `http_server.py:678`，再一路追 `tokenizer_manager.py:481`、`scheduler.py:1286`、`scheduler.py:2130`、`tp_worker.py:450`、`model_runner.py:2606`、`scheduler_output_processor_mixin.py:121/358`、`detokenizer_manager.py:144`、`tokenizer_manager.py:1515`。这条线读通后，再回头补 `schedule_batch.py` 和 `mem_cache`，整体会清楚很多。