明白。下面完全从**集合通信/数据分布**角度讲，不看代码。

假设：

```text
8 张卡：G0 G1 G2 G3 G4 G5 G6 G7
attention DP = 2
attention TP = 4

A0 = G0 G1 G2 G3
A1 = G4 G5 G6 G7

MoE expert group = G0...G7
top_k = 2
```

每个 token 在 MoE 前都有一个 hidden state：

```text
h_i: [hidden_size]
```

MoE 层要做：

```text
token hidden state
  -> router 选 top-k expert
  -> 发给 expert 所在 GPU
  -> expert 算
  -> 按 top-k 权重加权合并
  -> 回到原 token 所属位置
```

**1. Token 初始分布**
在 attention DP 模式下，attention 之后 token 分布是按 attention DP 组局部存在的。

例如本轮有 8 个 token：

```text
A0 产生:
  t0 t1 t2 t3

A1 产生:
  t4 t5 t6 t7
```

更具体地，可能分布在 rank 上：

```text
G0: t0
G1: t1
G2: t2
G3: t3
G4: t4
G5: t5
G6: t6
G7: t7
```

每张卡本地有若干 token 的 hidden states。

**2. Router/TopK 本地计算**
每张卡对自己持有的 token 做 router：

```text
t0 -> expert 3, expert 11
t1 -> expert 0, expert 9
t2 -> expert 6, expert 14
...
```

这一步没有集合通信。每张卡本地得到：

```text
topk_expert_ids
topk_weights
```

关键是：top-k expert 可能不在本卡，也可能不在本 attention DP 组。

**3. Expert 到 GPU 的映射**
假设 16 个 expert 均匀放在 8 张卡上：

```text
G0: E0,  E1
G1: E2,  E3
G2: E4,  E5
G3: E6,  E7
G4: E8,  E9
G5: E10, E11
G6: E12, E13
G7: E14, E15
```

那么：

```text
t0 -> E3, E11
E3  在 G1
E11 在 G5
```

所以 `t0` 的 hidden state 要复制成两份：

```text
h_t0 -> G1
h_t0 -> G5
```

因为 top_k=2，**一个 token 通常会产生 top_k 份 dispatch 流量**。

**4. Dispatch All-to-All**
dispatch 的目标是：

```text
每张源 GPU 把自己 token 的 hidden states 发到目标 expert 所在 GPU。
```

逻辑上是一个 all-to-all：

```text
源 rank: token owner
目标 rank: expert owner
payload: hidden state + token/expert 元信息
```

以 G0 上的 token 为例：

```text
G0 持有 t0

t0 -> E3  -> G1
t0 -> E11 -> G5

G0 发送:
  to G1: h_t0, weight_for_E3, original_token_id=t0
  to G5: h_t0, weight_for_E11, original_token_id=t0
```

所有 GPU 同时做类似事情：

```text
G0 -> G0..G7
G1 -> G0..G7
...
G7 -> G0..G7
```

这就是 MoE dispatch A2A。

结果是，每张 GPU 收到“路由到自己本地 experts 的 token”。

例如：

```text
G5 收到:
  给 E10/E11 的 token hidden states
  可能来自 G0, G2, G4, G7 ...
```

**5. Expert 本地计算**
dispatch 后，每张 GPU 只算自己本地 expert。

例如 G5 有：

```text
E10, E11
```

它收到一批 token：

```text
for E10: h_a, h_b, h_c
for E11: h_t0, h_d
```

然后本地计算：

```text
E10(h_a), E10(h_b), E10(h_c)
E11(h_t0), E11(h_d)
```

这一步主要是本地 GEMM，不是集合通信。

输出仍然暂存在 expert owner GPU 上。

**6. Combine All-to-All**
expert 算完后，每个 token 的 expert 输出要回到原 token owner 那里，并做加权求和。

继续看 `t0`：

```text
t0 原来在 G0

E3(h_t0)  在 G1 算完
E11(h_t0) 在 G5 算完
```

combine 阶段要把结果送回 G0：

```text
G1 -> G0: y_t0_E3,  weight_E3
G5 -> G0: y_t0_E11, weight_E11
```

G0 收齐后做：

```text
y_t0 = weight_E3 * y_t0_E3 + weight_E11 * y_t0_E11
```

所有 token 都同时这样返回，所以 combine 也是一个 all-to-all：

```text
源 rank: expert owner
目标 rank: original token owner
payload: expert output hidden state + weight / token id
```

**7. 返回后的分布**
combine 完成后，token hidden states 回到 MoE 前的原始分布：

```text
G0: y_t0
G1: y_t1
G2: y_t2
...
G7: y_t7
```

这样下一层 attention 或 residual 后续逻辑才能继续按原来的 batch/token 排布走。

**普通 DP 和 attention DP 的通信差异**
现在对比同样 8 卡：

普通 DP=2, TP=4：

```text
DP0 = G0 G1 G2 G3
DP1 = G4 G5 G6 G7

每个 DP 组都有完整 expert 集合。
```

如果 t0 在 G0，且属于 DP0，那么它的 expert 一定在 G0-G3 内找：

```text
t0 -> E3  -> G1
t0 -> E11 -> 也在 DP0 的某张卡，例如 G2/G3
```

所以 dispatch/combine 只发生在：

```text
G0-G3 内部
```

不会有：

```text
G0 -> G5
G5 -> G0
```

attention DP=2, TP=8：

```text
A0 = G0-G3
A1 = G4-G7
但 expert 集合铺在 G0-G7
```

如果 t0 在 G0，它的 expert 可能在 G4-G7：

```text
t0 -> E11 -> G5
```

于是有跨 attention-DP 组通信：

```text
dispatch: G0 -> G5
combine:  G5 -> G0
```

**通信成本增加在哪里**
精确说增加在这两类流量：

```text
dispatch 阶段:
  原本只在 4 卡组内发送 hidden states
  现在可能在 8 卡全局组内发送 hidden states

combine 阶段:
  原本只在 4 卡组内返回 expert outputs
  现在可能从另一 attention-DP 组返回 expert outputs
```

每个 routed token 的通信量近似是：

```text
dispatch: top_k * hidden_size * dtype_size
combine:  top_k * hidden_size * dtype_size
```

如果 top_k=2：

```text
每个 token 至少涉及约 4 份 hidden state 级别传输:
  2 份 dispatch
  2 份 combine
```

区别是这些传输是否跨组。

普通 DP：

```text
传输限制在 4 卡副本内部。
```

attention DP：

```text
传输扩展到 8 卡 expert group。
其中一部分会跨 A0/A1。
```

如果 expert 均匀分布，top-k 路由也近似均匀，那么 A0 产生的 token，选到 A1 上 expert 的概率很高。以 `top_k=2` 粗略看，两个 expert 都落在本 attention 组的概率约：

```text
(1/2)^2 = 1/4
```

也就是说大约 3/4 的 token 会至少有一路 expert 跨到另一个 attention DP 组。combine 还要回来一次。

**为什么还要这么做**
因为普通 DP=2 时，每个 DP 组都要有完整 expert 集合：

```text
G0-G3 一份 experts
G4-G7 一份 experts
```

attention DP 则是：

```text
G0-G7 共同放一份 experts
```

所以它用更大的 MoE A2A 通信域，换取 expert 权重不复制。

本质 tradeoff：

```text
普通 DP:
  expert 权重复制
  MoE 通信局部

attention DP:
  expert 权重不按 DP 复制
  MoE 通信全局
```

所以你可以把 attention DP 下的 MoE 看成：

```text
token 在 attention 组内产生
expert 在全局组内分布
token 必须出组找 expert
expert 输出必须回组继续后续计算
```

这就是从集合通信角度的完整链路。

# SGLang大致切分方式

假设：

Global TP = 8
attention_DP = 2
ATTN_CP = 2
ATTN_TP = 2

则：

attn_cp_rank ∈ [0, 1]
attn_tp_rank ∈ [0, 1]
attn_dp_rank ∈ [0, 1]

按 SGLang 的 rank 展开顺序：

- 普通 DP -> PP -> Global TP -> ATTN_CP -> ATTN_TP

- 如果开启 attention DP，则严格是：

  普通 DP -> PP -> Global TP -> attention_DP -> ATTN_CP -> ATTN_TP

ATTN_TP 是最内层、变化最快。所以 8 个 tp_rank 可以映射成：

tp_rank 0: attn_dp_rank=0, attn_cp_rank=0, attn_tp_rank=0
tp_rank 1: attn_dp_rank=0, attn_cp_rank=0, attn_tp_rank=1

tp_rank 2: attn_dp_rank=0, attn_cp_rank=1, attn_tp_rank=0
tp_rank 3: attn_dp_rank=0, attn_cp_rank=1, attn_tp_rank=1

tp_rank 4: attn_dp_rank=1, attn_cp_rank=0, attn_tp_rank=0
tp_rank 5: attn_dp_rank=1, attn_cp_rank=0, attn_tp_rank=1

tp_rank 6: attn_dp_rank=1, attn_cp_rank=1, attn_tp_rank=0
tp_rank 7: attn_dp_rank=1, attn_cp_rank=1, attn_tp_rank=1

公式是：

attn_tp_rank = tp_rank % attn_tp_size

attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size

attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)