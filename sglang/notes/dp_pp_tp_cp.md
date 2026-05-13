# DP/PP/TP/CP

```text
总卡数 = DP * PP * TP
先切 DP 副本
每个 DP 副本里再切 PP stage
每个 PP stage 内再切 TP group
```

CP 比较特殊：**CP 不是额外乘一维卡数，而是从 TP 维度里再切出 attention 的 context 维度**。

**例子 1：DP=2, PP=2, TP=4**
总共 16 张卡：

```text
DP0:
  PP0: G0  G1  G2  G3
  PP1: G4  G5  G6  G7

DP1:
  PP0: G8  G9  G10 G11
  PP1: G12 G13 G14 G15
```

每个 `PP stage` 里有一个 TP=4 group：

```text
DP0-PP0 TP group = [G0, G1, G2, G3]
DP0-PP1 TP group = [G4, G5, G6, G7]

DP1-PP0 TP group = [G8, G9, G10, G11]
DP1-PP1 TP group = [G12, G13, G14, G15]
```

PP group 是“同一个 TP lane 跨 pipeline stage”：

```text
DP0 pipeline groups:
  lane0: [G0, G4]
  lane1: [G1, G5]
  lane2: [G2, G6]
  lane3: [G3, G7]

DP1 pipeline groups:
  lane0: [G8,  G12]
  lane1: [G9,  G13]
  lane2: [G10, G14]
  lane3: [G11, G15]
```

**PP 怎么作用**
PP 是按层切模型，不是按 token 切，也不是按 expert 切。

假设模型 80 层，`PP=2`：

```text
PP0 持有 layer 0-39
PP1 持有 layer 40-79 + final norm/lm_head/sampler
```

一次请求在 DP0 上执行时：

```text
Tokenizer
  -> DP0-PP0
      G0-G3 以 TP=4 跑 layer 0-39
      输出 hidden states
  -> DP0-PP1
      G4-G7 以 TP=4 跑 layer 40-79
      输出 logits / sample next token
  -> Detokenizer / TokenizerManager
```

PP stage 之间传的是 **activation / hidden states**，不是 token id，也不是完整请求重新计算。

可以理解成：

```text
PP0:
  input_ids -> embedding -> lower layers -> hidden_states

PP1:
  hidden_states -> upper layers -> logits -> next_token
```

通信边界：

```text
PP 内部：
  相邻 stage 之间 P2P 发送 hidden_states / proxy tensors / batch metadata

TP 内部：
  每个 PP stage 内部做 tensor parallel all-reduce / all-gather 等

DP 之间：
  普通 DP 副本之间不为同一个请求通信
```

**PP 为什么有用**
PP 主要解决两个问题：

```text
1. 单个 TP group 放不下完整模型权重
2. 想把模型层切到更多卡上
```

代价是：

```text
1. PP stage 之间要传 activation
2. decode 时每个 token 都要过完整 pipeline，存在 pipeline bubble
3. 调度更复杂，需要 microbatch / async depth 尽量填满 pipeline
```

SGLang 里 PP event loop 会按 microbatch 循环：

```text
stage P:
  recv 上一 stage 的请求/activation
  run 当前 batch
  send 请求/activation 到下一 stage
  last stage 处理输出
```

所以 PP 的核心是 **层间流水线**。

**TP 在每个 PP stage 内怎么作用**
在每个 PP stage 中，TP=4 表示该 stage 的每一层权重被切到 4 张卡上。

例如某层 linear：

```text
G0-G3 共同持有 PP0 某层权重切片
G4-G7 共同持有 PP1 某层权重切片
```

同一个 PP stage 内部，TP ranks 对同一批 token 同步执行：

```text
输入 hidden_states
  -> 每个 TP rank 算自己的权重分片
  -> all-reduce / all-gather 得到该层输出
```

TP 是**层内张量切分**，PP 是**层间切分**。

**CP 是什么**
CP 是 attention context parallelism。它不是单独的总卡数维度，而是把 TP group 内部的 attention 再拆成：

```text
attention TP
attention CP
```

公式可以简化为：

```text
attn_tp_size = tp_size / attn_cp_size
```

如果没有 attention DP，假设：

```text
TP=8
CP=2
```

那么：

```text
attention TP size = 8 / 2 = 4
```

rank 布局可以看成：

```text
TP group: G0 G1 G2 G3 G4 G5 G6 G7

CP0 attention-TP group:
  [G0 G1 G2 G3]

CP1 attention-TP group:
  [G4 G5 G6 G7]

CP communication groups:
  [G0 G4]
  [G1 G5]
  [G2 G6]
  [G3 G7]
```

也就是说：

```text
attention TP group:
  横向切 hidden/head

attention CP group:
  纵向切 context/sequence/KV
```

**CP 怎么作用在 attention**
以 long context prefill 为例，一个请求有很长上下文：

```text
tokens = [0 ... 131071]
```

CP=2 时，可以粗略理解成：

```text
CP0 负责一部分 context/KV
CP1 负责另一部分 context/KV
```

attention 计算逻辑变成：

```text
每个 CP rank:
  拿本地 context/KV slice
  计算局部 attention 结果

CP group:
  对局部结果做合并/归约/交换
  得到等价于完整 context attention 的输出
```

decode 时也类似：

```text
当前 query token 要 attend 全部历史 KV

CP0:
  query attends KV shard 0

CP1:
  query attends KV shard 1

然后 CP group 合并 partial attention result
```

所以 CP 的目标是：

```text
把超长上下文 attention 的 KV/cache/计算压力拆到多个 CP rank 上
```

它解决的是 attention context 太长的问题，不是 MoE expert 分布问题。

**CP 和 PP/TP 的关系**
如果同时有 PP、TP、CP：

```text
DP 副本
  -> PP stage
      -> TP group
          -> attention 内部再拆 CP + attention TP
```

例如：

```text
DP=1, PP=2, TP=8, CP=2
总卡数 = 1 * 2 * 8 = 16
```

布局：

```text
PP0: G0-G7
  attention:
    CP0 attn-TP: G0-G3
    CP1 attn-TP: G4-G7

PP1: G8-G15
  attention:
    CP0 attn-TP: G8-G11
    CP1 attn-TP: G12-G15
```

PP 仍然按层切：

```text
PP0 lower layers
PP1 upper layers
```

CP 只在每个 PP stage 的 attention 层内部生效。

**一张总图**
```text
DP
└── replica 0
    ├── PP0: lower layers
    │   └── TP group
    │       ├── attention TP/CP groups
    │       └── MLP/MoE TP/EP groups
    └── PP1: upper layers
        └── TP group
            ├── attention TP/CP groups
            └── MLP/MoE TP/EP groups

└── replica 1
    └── 同样结构，普通 DP 下与 replica 0 独立服务不同请求
```

简短总结：

```text
DP: 多份模型副本，分请求
PP: 一份模型按层切，activation 在 stage 间传
TP: 每个 stage 内按张量/权重切，层内 collective
CP: attention 内按 context/KV 切，长上下文 attention 的 collective
```

PP 解决“模型层太多/权重太大，需要跨更多卡放”；CP 解决“attention context/KV 太长，需要把 context 维度拆开”。

# TP 切分计算

您说得对，之前的解释可能偏重原理对比，没把“横切KV的计算开销”这个点**说透**。下面我用最直接的方式，把横切在KV计算中到底产生了哪些额外开销、为什么比竖切差很多，明确指出来。

---

## 一、横切KV时，每个Token的计算流程

假设：
- 隐藏维度 \( h = 4096 \)
- KV头总维度也是 \( h_{kv} = 4096 \)（简化）
- GPU数量 \( g = 4 \)
- 输入激活 \( X \) 形状：`[b, s, 4096]`

### 横切的具体做法
将权重矩阵 \( W \)（形状 `[4096, 4096]`）**按行切分**到 4 张卡：

\[
W = \begin{bmatrix} W_0 \\ W_1 \\ W_2 \\ W_3 \end{bmatrix}, \quad W_i \in \mathbb{R}^{1024 \times 4096}
\]

同时，输入 \( X \) 也必须**按特征维度切分**：

\[
X = [X_0, X_1, X_2, X_3], \quad X_i \in \mathbb{R}^{[b, s, 1024]}
\]

每张卡 \( i \) 计算：

\[
Y_i = X_i \cdot W_i \quad \in \mathbb{R}^{[b, s, 4096]}
\]

注意：\( Y_i \) 已经是**完整的KV输出维度**（4096），但它只基于输入的一部分特征（1024维）计算出的**部分贡献**。

真正的完整 \( K \) 或 \( V \) 应该是：

\[
Y = Y_0 + Y_1 + Y_2 + Y_3
\]

所以必须进行一次 **AllReduce 求和**，使每张卡都得到完整的 \( Y \)。

---

## 二、横切带来的三大具体开销

### 1. 通信开销（最致命）

每生成 **1 个新 token**（在自回归解码中），都要做一次 AllReduce。

- **单次通信数据量**：每张卡需要发送自己算出的 \( Y_i \)，形状 `[b, 1, 4096]`（假设 batch=1, seq_len=1）。即 **4096 个浮点数**。
- **AllReduce 总通信量**（以带宽计）：对于 4 卡，通常需要 2×(g-1)/g × 数据量 的跨卡传输。粗略计算：每卡发送 4096 个数，接收 3×4096 个数。**每个 token 产生约 16KB 的通信**。
- **延时影响**：通信开销直接加到每个解码步骤的延迟上。在高速互联（如 NVLink 900GB/s）下或许能忍，但在跨节点或带宽较低时，会成为显著瓶颈。

**对比竖切**：竖切在投影阶段**零通信**。

---

### 2. 显存冗余开销（无法接受）

横切要求**每张卡都保存完整的 KV Cache**。

为什么？因为每张卡上的 \( Y_i \) 只是部分贡献，只有经过 AllReduce 后的完整 \( Y \) 才能用于后续的注意力计算。为了下一个 token 能重用历史的 K、V，每张卡必须缓存**完整**的 K、V 矩阵（而不是分片）。

- 完整 KV Cache 大小：\( 2 \times \text{层数} \times \text{batch} \times \text{序列长度} \times h_{kv} \)
- 横切下：这个大小要乘以 \( g \) 倍的冗余。
- 举例：`层数=32, batch=1, seq_len=4096, h_kv=4096` → 一个完整 KV Cache 约 1GB（FP16）。4 卡横切 → **4GB 冗余存储**，而竖切只需要 1GB（分片存储）。

这就意味着显存利用率极低，能跑的 batch 大小或序列长度大幅下降。

---

### 3. 计算负载无优势，反而增加额外规约操作

- **总计算量**：横切与竖切在投影层的总浮点运算次数完全相同（都是 \( b \times s \times h \times h_{kv} \)）。
- **但横切需要额外执行 AllReduce**，这部分操作虽然不算“浮点运算”，但占用 GPU 的通信带宽和同步时间，实际耗时不可忽略。

---

## 三、为什么竖切没有这些开销？

竖切时：
\[
W = [W_0, W_1, W_2, W_3], \quad W_i \in \mathbb{R}^{4096 \times 1024}
\]
每张卡计算：
\[
Y_i = X \cdot W_i \quad \in \mathbb{R}^{[b, s, 1024]}
\]
每张卡的 \( Y_i \) 已经是**最终输出的一部分**。跨卡拼接即可得到完整 KV，**不需要 AllReduce**。

KV Cache 也是分片存储：
- 卡0 存 1024 维的 KV（对应第 0～3 个头）
- 卡1 存下一段 1024 维的 KV（对应第 4～7 个头）
- 没有冗余。

---

## 四、一句话总结横切KV的开销位置

> **每个 token 生成时，横切必须做一次全卡 AllReduce 来合并部分结果，这带来通信延迟；同时每张卡都要存一份完整的 KV Cache，造成 g 倍的显存浪费。而竖切完全避免了这两点。**

因此，在大模型的实际系统中（如 Megatron-LM），QKV 投影层**永远使用竖切**。横切只出现在一些特殊地方，比如 MLP 的第二层（降维层）或输出投影层，因为这些地方输入已经被竖切切碎，横切可以避免再次切分输入，但付出的 AllReduce 代价是值得的——绝不会用在需要缓存大量状态的 KV 投影上。

# 附录 Attention 计算
当然。这里给出注意力机制（Scaled Dot-Product Attention，即缩放点积注意力）的完整计算公式，这是 Transformer 及大模型的核心。

---

### 标准自注意力计算公式

给定一个输入序列，通过三个可学习的权重矩阵 \( \mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \) 分别得到 **查询（Query）**、**键（Key）**、**值（Value）** 矩阵：

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}^V
\]

其中：
- \( \mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}} \)：输入矩阵，\( N \) 为序列长度，\( d_{\text{model}} \) 为模型维度。
- \( \mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d_{\text{model}} \times d_k} \)，\( \mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v} \)。
- 通常 \( d_k = d_v = d_{\text{model}} / h \)（\( h \) 为注意力头数）。

---

### 缩放点积注意力输出

\[
\boxed{\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}}
\]

逐项解释：

1. **计算注意力得分矩阵**  
   \[
   \mathbf{S} = \mathbf{Q} \mathbf{K}^\top \quad \in \mathbb{R}^{N \times N}
   \]
   \( S_{ij} \) 表示第 \( i \) 个查询与第 \( j \) 个键的相似度。

2. **缩放**  
   \[
   \mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}
   \]
   除以 \( \sqrt{d_k} \) 是为了防止点积值过大导致 softmax 梯度过于稀疏。

3. **Softmax 归一化（通常作用于行）**  
   \[
   \mathbf{P} = \text{softmax}(\mathbf{S}_{\text{scaled}}) \quad \in \mathbb{R}^{N \times N}
   \]
   \[
   P_{ij} = \frac{\exp(S_{ij} / \sqrt{d_k})}{\sum_{k=1}^{N} \exp(S_{ik} / \sqrt{d_k})}
   \]
   得到注意力权重矩阵，每一行和为 1。

4. **加权求和**  
   \[
   \text{Output} = \mathbf{P} \, \mathbf{V} \quad \in \mathbb{R}^{N \times d_v}
   \]

---

### 带掩码的自回归注意力（因果注意力）

在自回归生成（如 GPT）中，我们需要确保位置 \( i \) 只能看到它之前的位置（包括自己）。引入**掩码矩阵** \( \mathbf{M} \in \mathbb{R}^{N \times N} \)，其中：

\[
M_{ij} = 
\begin{cases}
0 & \text{if } i \ge j \\
-\infty & \text{if } i < j
\end{cases}
\]

修改 softmax 前的值：

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}
\]

加上 \( -\infty \) 使得未授权位置的 softmax 权重趋近于 0。

---

### 多头注意力（Multi-Head Attention, MHA）

将单头注意力复制 \( h \) 次，每个头有独立的投影矩阵，最后将各头输出拼接并投影：

\[
\begin{aligned}
\text{head}_i &= \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) \\
\mathbf{Q}_i &= \mathbf{X} \mathbf{W}_i^Q,\quad
\mathbf{K}_i = \mathbf{X} \mathbf{W}_i^K,\quad
\mathbf{V}_i = \mathbf{X} \mathbf{W}_i^V \\
\text{MultiHead}(\mathbf{X}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, \mathbf{W}^O
\end{aligned}
\]

其中 \( \mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}} \)。

---

### 与张量并行（TP）的联系

上述公式中的矩阵乘法 \( \mathbf{Q} \mathbf{K}^\top \) 在 TP 切分下会如何分布？
- **竖切**：每张卡持有部分 \( \mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i \)（按头切分），本地计算部分 \( \mathbf{Q}_i \mathbf{K}_i^\top \) 和 \( \text{softmax}(\dots)\mathbf{V}_i \)，结果拼接后与原公式完全等价。
- **横切**：不适合直接用于 \( \mathbf{Q}\mathbf{K}^\top \)，因为需要跨卡 all‑reduce 求和，正是之前讨论的“开销所在”。
