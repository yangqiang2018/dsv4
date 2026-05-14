# `SparseAttnSharedkv` 算子完全详解

> **预设读者**：会写普通 Python（`for`、`if`、列表、函数、字典），但**对 numpy/pytorch 不熟，没系统学过线性代数和深度学习，更没写过 NPU/GPU 算子**。
>
> 看完本文后，你将能理解：什么是注意力（attention），什么是 KV cache，什么是滑窗注意力（Sliding Window Attention）、压缩注意力（Compressed Attention）、稀疏注意力（Sparse Attention），以及这三种东西**合在一起**的算子是怎么在 NPU 上跑起来的。
>
> 本文风格与 [`Compressor 详解.md`](./Compressor%20详解.md) 保持一致：每个新概念都先讲清楚再用，每段 numpy 代码都先用纯 Python 的 `for` 循环写一遍对照。如果你对矩阵乘 `@`、转置 `.T`、softmax、Hadamard 乘、`reshape`、广播等概念已经不熟悉，请先回看 [`Compressor 详解.md`](./Compressor%20详解.md) 的「**第 0 章 你需要的几个数学和编程基础**」，本文不再重复那些前置概念。
>
> 配套源码（拉到本地的 atomgit 仓库）：
> - 业务说明：[`ops-transformer/experimental/attention/sparse_attn_sharedkv/README.md`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/README.md)
> - kernel 入口：[`sparse_attn_sharedkv.cpp`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp)
> - SWA 模板（滑窗，最常用）：[`sparse_attn_sharedkv_swa_kernel.h`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_kernel.h)、[`sparse_attn_sharedkv_swa_block_cube.h`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_cube.h)、[`sparse_attn_sharedkv_swa_block_vector.h`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_vector.h)
> - SCFA 模板（稀疏压缩）：[`sparse_attn_sharedkv_scfa_kernel.h`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h)、[`sparse_attn_sharedkv_scfa_block_cube.h`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_cube.h)、[`sparse_attn_sharedkv_scfa_block_vector.h`](../../../dsv4/ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h)

---

## 第 0 章 名字读懂了，事就懂一半了

算子的名字 `SparseAttnSharedkv`（"稀疏注意力，共享 KV"）拆开来看：

- **Sparse**：稀疏。意思是"不全算，只挑一部分算"。
- **Attn**：attention 的缩写，注意力。
- **Shared kv**：共享 KV。这里的"共享"指多个查询头（query heads）**共用**同一份键值对（key/value）。

合在一起就是："**用稀疏的方式做注意力计算，并且 query 头很多但 KV 头只有一份**"的一个算子。

> 注意，本算子的 **Sparse**(稀疏) 不是说连权重矩阵也是稀疏的，而是说"算注意力时，q 不和所有的 KV 都算一遍，只挑一些算"。

下面要把每个词都讲透，目标是：看完第 4 章，你能拿起 §6 的 numpy 版代码读懂；看完第 7 章，你能对照源码看懂 cube/vector 怎么跑。

---

## 第 1 章 大背景回顾：从 attention 到 KV cache

> 这一节是 `Compressor 详解.md` 第 1 章的快速复习版。如果已经熟悉，可以直接跳到 §2。

### 1.1 token、向量、attention 的最简形式

大模型一个字一个字往外吐的过程叫**推理**。每个被处理的"字"叫一个 **token**。每个 token 进入模型后会变成一个**向量**——也就是一串数字，长度通常几千。

模型生成下一个字时，要"回头看"之前所有的字，给每个历史字打一个**相关性分数**，然后按分数加权把信息收集过来。这个"带权重的回看"就是 **attention**。

每个历史 token 提供两个角色：

- **K（Key）**：用来和当前 query 算"相关性分数"的向量
- **V（Value）**：被加权累加的"内容向量"

最简伪代码（D 是向量维度，N 是历史 token 数）：

```python
# q: 当前 token 的 query 向量, shape (D,)
# K: 历史 N 个 token 的 Key,   shape (N, D)
# V: 历史 N 个 token 的 Value, shape (N, D)

scores  = q @ K.T            # (N,) — 对每个历史 token 算一个分数
weights = softmax(scores)    # (N,) — 归一成"和=1 的权重"
output  = weights @ V        # (D,) — 按权重加权求和成新向量
```

`q @ K.T` 这一步：`q` 是 `(D,)`，`K.T` 是 `(D, N)`，结果是 `(N,)`，也就是 N 个分数。

### 1.2 KV cache：为啥要缓存

历史 token 的 K、V 不会变，所以工程上把每个 token 的 (K, V) **算一次缓存起来**。缓存就叫 **KV cache**。生成第 N+1 个 token 时，只算这个新 token 的 q，再用 q 跟 KV cache 里所有的 K、V 做 attention。

```
对话历史:
  历史 token 1 ──► (K_1, V_1) ┐
  历史 token 2 ──► (K_2, V_2) │
  ...                        ├── KV cache（一个大数组）
  历史 token N ──► (K_N, V_N) ┘
```

### 1.3 长上下文有两个痛点

如果对话有 100 万个 token：

1. **KV cache 放不下**（占内存）
2. **算 attention 太慢**（每生成一个新字都要扫全部历史）

我们这个算子就是来解决"**算 attention 太慢**"这个问题的。

> KV cache 放不下的问题，由上一篇文档讲的 [`Compressor`](./Compressor%20详解.md) 算子解决——它把 KV cache 压缩成 1/cmp_ratio 大小。

---

## 第 2 章 三种"省算力"的注意力策略

模型生成一个新 token 时要扫全部 N 个历史 token，这个 N 可能是几十万、上百万。**与其老老实实扫全部，不如只扫"重要的"那几个**。问题：怎么定义"重要"？业界有三种主流策略，本算子**同时支持其中三种的组合**。

### 2.1 策略一：Sliding Window Attention（滑窗注意力，简称 SWA）

**直觉**：当前 token 跟"最近的 K 个" token 关系最密切（就像聊天时关注最近几句话），跟很远的历史关系不大。

**做法**：只算 q 和"最近 W 个 token"的 attention，超出窗口的 token 一律忽略。

```
   历史 token:    [ 0, 1, 2, ..., 996, 997, 998, 999 ]
                                       ↑              ↑
                                       └ 窗口起点      └ 当前位置
   只算 q 跟 [997, 998, 999] 的 attention（W=3）
```

伪代码：

```python
# W = 窗口大小（本算子里 ori_win_left = 127, 即只看最近 128 个 token）
window_K = K[-W:]
window_V = V[-W:]
scores  = q @ window_K.T   # 只对 W 个 token 算分数
weights = softmax(scores)
output  = weights @ window_V
```

**好处**：原来要算 N 次乘加，现在只要算 W 次。N=100 万、W=128 的话快 7800 倍。

**坏处**：远处的信息全丢了——比如对话里你 50 句话之前说过"今天我叫张三"，模型完全看不到了。

### 2.2 策略二：Compressed Attention（压缩注意力，简称 CFA）

**直觉**：远处信息也想看，但不需要看那么细。把每 `cmp_ratio` 个 token 压成 1 个"摘要 token"，然后让 q 跟"摘要们"做 attention 即可。

**做法**：维护两份缓存：
- `ori_kv`：最近 W 个 token 的**原版** K/V（细看）
- `cmp_kv`：远处 token 的**压缩** K/V（粗看，每 cmp_ratio 个原 token 对应 1 个压缩 token）

> `cmp_kv` 就是上一篇 [`Compressor`](./Compressor%20详解.md) 算子吐出来的东西。

q 和**两份**都做 attention，最后**拼起来一起 softmax**：

```
   q ─────► 跟 ori_kv (近 128 个原 token) 算分数 ────┐
                                                   ├──► 拼接 ──► softmax ──► 加权求和
   q ─────► 跟 cmp_kv (远处的压缩摘要) 算分数 ─────────┘
```

**好处**：远处信息不丢，只是分辨率低了；近处信息保持高精度。

**坏处**：cmp_kv 还是有可能很多。比如总共 100 万 token，cmp_ratio=4，cmp_kv 还有 25 万——还是太多。

### 2.3 策略三：Sparse Compressed Attention（稀疏压缩注意力，简称 SCFA）

**直觉**：cmp_kv 那 25 万摘要里，**对当前 q 真正有用的可能只有几百个**。我们提前用某种方法（比如另一个轻量模型、或者直接看分数）挑出这几百个，扫这几百个就行。

**做法**：除了 SWA + CFA，再加一份输入 `cmp_sparse_indices`——这是一个**索引列表**，告诉算子"对于当前 q，只需要从 cmp_kv 里挑这 K 个位置"。

```
   q ─────► 跟 ori_kv (近 128 个原 token) 算分数 ──────────────────┐
                                                                 │
   q ─────► 看 cmp_sparse_indices = [3, 17, 89, ...] (共 K 个索引)
            把 cmp_kv[3], cmp_kv[17], cmp_kv[89] ... 挑出来       ├──► 拼接 ──► softmax ──► 加权求和
            跟 q 算分数 ──────────────────────────────────────────┘
```

**这就是本算子（`SparseAttnSharedkv`）支持的三个场景**：

| 场景  | 输入                                           | 含义                    |
| --- | -------------------------------------------- | --------------------- |
| 一   | 只传 `ori_kv`                                  | 纯 SWA                 |
| 二   | 传 `ori_kv` + `cmp_kv`                        | SWA + CFA（滑窗 + 压缩）    |
| 三   | 传 `ori_kv` + `cmp_kv` + `cmp_sparse_indices` | SWA + SCFA（滑窗 + 稀疏压缩） |

> 索引怎么挑出来的？不在本算子负责的范围内——上层先用别的方法（比如轻量级的相似度计算）选出 topk，再传给本算子。

### 2.4 共享 KV（Shared KV）

最后还有"shared kv"这个名字。它的来源是 **GQA / MQA 架构**。

普通 attention 里，q 有多个"头"（head），每个 head 都有自己的 K、V——比如 64 个头就有 64 套 K、V。这叫 **MHA（Multi-Head Attention）**。

**问题**：64 套 K、V 太占内存了。

**解决**：让多个 q 头**共用**同一套 K、V。本算子的设定是 **N1=64 个 q 头共享 N2=1 套 KV**——所以叫 "Shared KV"。这种 64:1 的比例叫 **MQA（Multi-Query Attention）**，是 GQA（Grouped Query Attention）的一个极端情况。

```
   q heads (共 64 个):  q_0, q_1, q_2, ..., q_63
                         │    │    │         │
                         └────┴────┴────...──┴──► 共用同一套 (K, V)
```

代码里这个 64:1 的比例由 `n1` (q heads) 和 `n2` (kv heads) 表示，`gSize = n1 / n2` 叫"组大小"（group size）。

---

## 第 3 章 名词速查表

下面这些词后面会反复出现：

| 名词                     | 含义                                                 |
| ---------------------- | -------------------------------------------------- |
| **token**              | 输入序列里的一个字/词的最小单元                                   |
| **B（batch）**           | 一次处理几条独立序列                                         |
| **S1 / Q_S**           | query 的序列长度（一次要算多少个 q）                             |
| **S2**                 | `ori_kv` 的序列长度（不压缩的 K/V token 数）                   |
| **S3**                 | `cmp_kv` 的序列长度（压缩后的 K/V token 数）                   |
| **N1（num_q_heads）**    | q 的头数，本算子固定 64                                     |
| **N2（num_kv_heads）**   | KV 的头数，本算子固定 1                                     |
| **G（group size）**      | gSize = N1 / N2 = 64，每 64 个 q 头共用 1 套 KV           |
| **D（head_dim）**        | 每个头的向量长度，本算子固定 512                                 |
| **cmp_ratio**          | 压缩比，4 或 128                                        |
| **ori_kv**             | 原版（不压缩）的 KV cache，对应近处的滑窗                          |
| **cmp_kv**             | 压缩后的 KV cache，对应远处的摘要（来自 Compressor 算子）            |
| **cmp_sparse_indices** | 稀疏选取的索引，告诉算子从 cmp_kv 里挑哪几条                         |
| **K1**                 | 对 ori_kv 一次离散选取的 token 数（默认 512，本算子目前不启用）          |
| **K2**                 | 对 cmp_kv 一次离散选取的 token 数（本算子默认 512）                |
| **softmax_scale**      | softmax 之前给分数乘的缩放系数，通常是 `1/sqrt(D)`                |
| **sinks**              | 一种"注意力下沉"机制，给 softmax 多加一个虚拟 token，下面 §4.5 详讲      |
| **ori_win_left**       | 滑窗左边界（看过去几个 token），默认 127                          |
| **ori_win_right**      | 滑窗右边界（看未来几个 token），默认 0                            |
| **mask_mode**          | 屏蔽哪些 token 不算的策略编号，默认 ori=4, cmp=3                 |
| **PA（Page Attention）** | KV cache 的一种存法，不连续地存到一堆固定大小的 block 里               |
| **block_table**        | PA 模式下的"目录"，告诉每条 batch 用到了哪些 block                 |
| **layout**             | 数据排布格式：BSND、TND、PA_ND                              |
| **Flash Attention**    | 一种把 attention 的 softmax 拆成"在线累加"的算法，§4.4 详讲        |
| **softmax_lse**        | log-sum-exp，softmax 的中间统计量，主要给 flash decoding 拼分块用 |
| **AIC（cube core）**     | NPU 上"专门算矩阵乘"的核                                    |
| **AIV（vector core）**   | NPU 上"算向量四则运算"的核                                   |
| **GM**                 | Global Memory，NPU 的主内存（大但慢）                        |
| **UB / L1 / L0**       | NPU 上类似 CPU 寄存器/缓存（小但快）                            |
| **workspace**          | 算子向系统申请的临时内存                                       |
| **tiling**             | 把大问题切成小块的策略                                        |
| **vec0 / vec1 / vec2** | 算子里 vector 核要做的三段计算（详见 §7）                         |
| **mm1 / mm2**          | 算子里 cube 核要做的两段矩阵乘（详见 §7）                          |
| **double buffer**      | 双缓冲，让搬运和计算重叠的关键技巧                                  |

---

## 第 4 章 attention 涉及的几个关键概念

光知道 §1.1 那段三行伪代码还不够，本算子涉及几个细节，得讲清楚：

### 4.1 `softmax_scale`：为啥要先乘个小数再 softmax

回顾 §1.1：

```python
scores  = q @ K.T
weights = softmax(scores)
```

`scores` 里每个数是 D 个数相乘再相加得到的。**D 越大、scores 的数值波动越大**——比如 D=512 时，scores 可能是几百几千。softmax 对大数特别敏感（`exp(1000)` 直接爆）。

**解决**：先乘一个小常数把 scores 缩小，再做 softmax：

```python
scores = (q @ K.T) * softmax_scale    # softmax_scale 通常是 1/sqrt(D)
weights = softmax(scores)
```

D=512 时 `softmax_scale ≈ 0.04417`，刚好把分数压到合理范围。

算子里这个缩放发生在 mm1 算完 scores、softmax 之前——把每个 score 乘上一个 `softmax_scale` 常数，就是一次最简单的逐元素乘。

### 4.2 `mask_mode`：哪些 token 不能看

不是所有历史 token 都该参与计算。常见两种屏蔽：

1. **因果屏蔽（causal mask）**：生成时不能看未来的 token（不然就作弊了）。具体做法：把"未来 token"的 score 设成 `-∞`，softmax 后这些位置的权重就是 0。
2. **滑窗屏蔽**：只看最近 W 个，超出 W 的也设 `-∞`。

本算子的 `ori_mask_mode = 4` 是滑窗+因果的组合，`cmp_mask_mode = 3` 是因果屏蔽。算子通过 `ori_win_left = 127`（看过去 128 个）和 `ori_win_right = 0`（不看未来）控制窗口边界。

具体边界怎么定？对当前 q 在原序列里的位置 `pos`，可见的 ori_kv 区间是闭区间：

```
[ max(pos - ori_win_left, 0),   pos + ori_win_right ]
```

`ori_win_left = 127, ori_win_right = 0` 时就是"看自己 + 过去 127 个"，刚好 128 个 token。区间之外的 token 在 mask 阶段全部被设成 `-∞`，softmax 之后权重为 0，等同于"看不见"。

### 4.3 `softmax_scale`、`mask`、`sinks` 三件套都是给 softmax "动手脚"

它们都发生在 `q @ K.T` 之后、softmax 之前：

```python
scores  = q @ K.T              # (N,) 原始分数
scores  = scores * softmax_scale  # 缩放
scores  = apply_mask(scores)   # 把不该看的位置设 -inf
scores  = concat(scores, sinks_value)  # 见 §4.5 attention sinks
weights = softmax(scores)      # 然后才 softmax
```

### 4.4 `Flash Attention` 算法：分块 softmax 的诀窍

这是整篇文档**最不好懂**的一节。慢慢来，我们一步步把它讲透。

#### 4.4.1 先回顾标准 softmax 怎么算

§1.1 里 softmax 的标准做法（带数值稳定性的"减最大值"版）：

```python
def softmax(scores):
    # scores: 长度 N 的列表
    m = max(scores)                                # 1) 找到最大值
    e = [math.exp(s - m) for s in scores]          # 2) 每个数减最大值再 exp
    total = sum(e)                                 # 3) 求和
    weights = [ei / total for ei in e]             # 4) 每个除以 sum
    return weights
```

然后 attention 的输出：

```python
output = sum(weights[i] * V[i] for i in range(N))
```

把这两步合在一起，展开成数学式子：

```
            sum_i  exp(scores_i - m) * V_i
output  =  ─────────────────────────────────
              sum_i  exp(scores_i - m)
```

记两个量：

- `s  =  sum_i exp(scores_i - m)`     ← 分母（一个标量）
- `o  =  sum_i exp(scores_i - m) * V_i`  ← 分子（一个 D 维向量）

那么 `output = o / s`。

**关键观察**：`s` 和 `o` 都是"对 N 个 token 求和"的形式——如果能**边读边累加**，就不需要把所有 N 个 scores 同时放在内存里。

#### 4.4.2 为啥不能直接边读边累加？因为 `m` 拦着

上面公式里有个讨厌的依赖：`m = max(scores)` 要**先扫一遍所有 N 个 scores** 才知道。在不知道 `m` 之前，`exp(scores_i - m)` 算不出来。

> 那不减 `m` 行不行？理论上行，但 scores 数值可能很大，`exp(几百)` 会直接溢出成 `inf`，结果就废了。减 `m` 是数值稳定性的硬需求。

如果非要分块算，第一块来的时候我们不知道全局 `m`——它的真正最大值可能藏在第 5 块里。怎么办？

#### 4.4.3 核心思路：用"目前为止见过的最大值"+ 来新值时校正

**直觉**：先用"目前为止见过的最大值" `m_running` 来近似算，每来一块新数据如果出现更大的值就**回头修正**之前的累加。

**关键技巧**：修正可以**整体乘一个系数**，不需要重新扫历史数据。

为啥？因为指数函数有这个性质：

```
exp(scores_i - m_new)
    = exp(scores_i - m_old + m_old - m_new)
    = exp(scores_i - m_old) * exp(m_old - m_new)
       ↑                       ↑
       这是之前用旧 m 算的       这是个标量修正系数
```

所以如果之前算了 `e_i_old = exp(scores_i - m_old)`，现在想换成 `e_i_new = exp(scores_i - m_new)`，**只需要全体乘上 `exp(m_old - m_new)` 一个标量**。

`s` 和 `o` 也是同样的累加形式，**同样只需乘一个标量**：

```
s_new (用 m_new 算的)  =  s_old * exp(m_old - m_new)  +  sum(新块的 exp)
o_new (用 m_new 算的)  =  o_old * exp(m_old - m_new)  +  sum(新块的 exp * V_新)
```

如果 `m_new > m_old`（新最大值更大），那 `exp(m_old - m_new) < 1`，**之前的累加被缩小了**——这就把"之前用错了 m"的偏差校正回来了。

如果 `m_new == m_old`（新块的最大值还是没超过 m_old），那 `exp(0) = 1`，老数据不用改。

#### 4.4.4 完整算法（Python 伪代码）

```python
def flash_attention_chunked(K_chunks, V_chunks, q, softmax_scale):
    """K_chunks, V_chunks: 已经切成块的 K, V (list of arrays)"""
    m_running = -inf      # 目前为止见过的最大 score
    s_running = 0.0       # 目前为止用 m_running 算的 sum(exp)
    o_running = zeros(D)  # 目前为止用 m_running 算的 sum(exp * V) — 即未归一化的输出

    for K_chunk, V_chunk in zip(K_chunks, V_chunks):
        # ----- 这一块的 scores -----
        scores_chunk = (q @ K_chunk.T) * softmax_scale   # shape (块大小,)

        # ----- 1) 更新最大值 -----
        m_chunk = max(scores_chunk)
        m_new   = max(m_running, m_chunk)

        # ----- 2) 算"修正系数" (用来修正旧的 s/o) -----
        alpha = exp(m_running - m_new)   # ≤ 1, 把旧的缩小到新尺度

        # ----- 3) 算这一块的 exp 值（用新的 m_new 算） -----
        e_chunk = exp(scores_chunk - m_new)   # shape (块大小,)

        # ----- 4) 更新 s 和 o -----
        s_running = s_running * alpha + sum(e_chunk)
        o_running = o_running * alpha + e_chunk @ V_chunk

        # ----- 5) 推进 m -----
        m_running = m_new

    # 全部块处理完, 最后归一化
    return o_running / s_running
```

读到这里你可能还在挠头："对，但这真的对吗？跟一次性 softmax 算出来一样吗？"——下面我们用一个能手算的小例子验证一遍。

#### 4.4.5 手算例子：N=4，切成 2 块

假设 q 已经和 K 算完了，scores 是 `[1.0, 3.0, 2.0, 4.0]`（一共 N=4 个）。V 为了好算我们用 D=1（一维向量），数值是 `[10, 20, 30, 40]`。softmax_scale 暂时不管。

**先用标准 softmax 一次性算**（作为参考答案）：

```
m = max(1, 3, 2, 4) = 4
exp(1-4) ≈ 0.0498
exp(3-4) ≈ 0.3679
exp(2-4) ≈ 0.1353
exp(4-4) = 1.0000

s = 0.0498 + 0.3679 + 0.1353 + 1.0 = 1.5530
weights = [0.0321, 0.2369, 0.0871, 0.6439]
output = 0.0321*10 + 0.2369*20 + 0.0871*30 + 0.6439*40
       ≈ 33.428
```

**现在用 Flash Attention 分两块算**：第 1 块是 `scores=[1.0, 3.0], V=[10, 20]`，第 2 块是 `scores=[2.0, 4.0], V=[30, 40]`。

**初始状态**：

```
m_running = -inf
s_running = 0
o_running = 0
```

---

**第 1 块来了**：`scores_chunk = [1.0, 3.0]`, `V_chunk = [10, 20]`

```
1) m_chunk = max(1.0, 3.0) = 3.0
   m_new   = max(-inf, 3.0) = 3.0

2) alpha = exp(-inf - 3.0) = 0     # 历史是空的, 全归零
                                    # (实际代码里第一块走特殊分支)

3) e_chunk = [exp(1-3), exp(3-3)] = [exp(-2), 1.0]
           ≈ [0.1353, 1.0]

4) s_running = 0*0 + (0.1353 + 1.0)     = 1.1353
   o_running = 0*0 + (0.1353*10 + 1.0*20) = 1.353 + 20 = 21.353

5) m_running = 3.0
```

到这里为止，我们"假装 m=3 就是全局最大值"算了一份 s 和 o。

---

**第 2 块来了**：`scores_chunk = [2.0, 4.0]`, `V_chunk = [30, 40]`

```
1) m_chunk = max(2.0, 4.0) = 4.0
   m_new   = max(3.0, 4.0) = 4.0           ← 出现新最大值! 要修正历史

2) alpha = exp(m_running - m_new) = exp(3-4) = exp(-1) ≈ 0.3679

3) e_chunk = [exp(2-4), exp(4-4)] = [exp(-2), 1.0]
           ≈ [0.1353, 1.0]

4) s_running = 1.1353 * 0.3679 + (0.1353 + 1.0)
             = 0.4177 + 1.1353
             = 1.5530                       ← 跟标准算法的 s 完全相同 ✓

   o_running = 21.353 * 0.3679 + (0.1353*30 + 1.0*40)
             = 7.856 + (4.059 + 40)
             = 7.856 + 44.059
             = 51.915

5) m_running = 4.0
```

---

**所有块处理完，归一化**：

```
output = o_running / s_running = 51.915 / 1.5530 ≈ 33.428    ✓ 和标准算法答案完全相同
```

**关键发现**：

- `s_running` 在最后一块结束时刚好等于真正的 `s`
- `o_running / s_running` 等于真正的 attention 输出
- 我们从头到尾**每次只看 2 个 scores**，从未把全部 4 个一起放进内存

这就是 Flash Attention 的全部魔法。**如果这一节哪里没看懂，请回头自己拿纸笔走一遍这个例子**——能算到 33.428 就是真懂了。

#### 4.4.6 算子把"算法的某一步"分给了不同核做

回看 §7.2 的五段拆分，这个分块算法在本算子里被**切到三段核动作**上：

| 算法步骤                                                            | 谁在做                       | 输出                                   |
| --------------------------------------------------------------- | ------------------------- | ------------------------------------ |
| 算 `scores_chunk = q @ K_chunk.T * scale`                        | **mm1**（cube）+ vec1 开头的缩放 | 当前块的 scores                          |
| 算 `m_new`、`alpha = exp(m_old - m_new)`、`e_chunk`，更新 `s_running` | **vec1**（vector）          | e_chunk（给 mm2 当输入）、新的 m、s、修正系数 alpha |
| 算 `e_chunk @ V_chunk`                                           | **mm2**（cube）             | 当前块的"部分加权和"                          |
| 算 `o_running = o_running * alpha + (e_chunk @ V_chunk)`         | **vec2**（vector）          | 累加结果（暂存或最后输出）                        |
| 最后一块：算 `output = o_running / s_running` 并写出                     | **vec2**（vector）          | 最终 attention 输出                      |

**这就是为啥 vec1 和 vec2 必须分开两段**：vec1 只能算到 `e_chunk` 和更新 m、s、alpha（到这里为止还没碰 V）；中间必须夹着 mm2 算 `e_chunk @ V`；mm2 算完 vec2 才能继续做 `o_running` 的累加。三者必须按 vec1 → mm2 → vec2 的顺序串起来。

#### 4.4.7 vec1 一条指令做了什么

vec1 阶段调用了 Ascend C 库提供的一条复合指令（行话叫 `SoftmaxFlashV2`），它把 §4.4.4 算法里 1)~4) 这四步**一次性算完**——除了 `o_running * alpha` 那部分，留给 vec2 做（因为那时候 `o_running` 还没被 mm2 算出来）。

这条指令的语义可以理解成一个函数：

```
输入:
  - scores_chunk        本块刚算出来的 scores (M × N 矩阵, M 行 q, N 个 KV)
  - m_old, s_old        上一块结束时留下的 m_running, s_running（M 行各一个标量）
输出:
  - e_chunk             覆盖回原内存，本块的 exp 值 (M × N)
  - m_new               新的 m_running（M 行）
  - s_new               新的 s_running（M 行）
  - alpha               修正系数 exp(m_old - m_new)（M 行，给 vec2 用）
```

**为啥这四步要"打包"成一条指令？** 因为它们之间共享中间变量（比如 m_new 一旦算出来，e_chunk、alpha、s_new 都要用），写成一条复合指令能避免重复读写 UB，速度比拆 4 条指令快得多。

**第一块怎么办？** 第一块进来时没有"上一块"——算子用预设的初始值：

| 变量 | 第一块时的值 | 为啥 |
|---|---|---|
| `m_old` | `-2e38`（接近 `-∞`） | 让 `alpha = exp(-2e38 - m_chunk) ≈ 0`，自动把"不存在的旧累加"乘成 0 |
| `s_old` | `1.0` | 占位用，反正下一步会被覆盖 |

**双缓冲怎么存？** vec1 在流水里同时维护 2 份缓冲（本轮一份、上轮一份），轮流读写。这样上一轮写的 m、s 不会被本轮覆盖——下一轮还要把它们当成 `m_old, s_old` 读回来。

> 注意：m、s、alpha 这三个量都是**按行**存的——每行 q 各算各的 softmax，互不干扰。所以 m_running 不是一个全局标量，而是一个长度等于行数（M）的向量。

#### 4.4.8 vec2 怎么用 `alpha` 累加 `o_running`

mm2 算完 `e_chunk @ V_chunk` 后，vec2 完成 §4.4.4 算法里 `o_running = o_running * alpha + (e_chunk @ V_chunk)` 那一行的具体动作。逻辑分三种情况：

```
情况 ①：本块是第一块
   o_running = (e_chunk @ V_chunk)        # 直接当起始值，没历史可累加
   暂存到 workspace 留给下一块

情况 ②：本块不是第一块、也不是最后一块
   prev_o = 上一块暂存的 o_running         # 从 workspace 读出来
   o_running = prev_o * alpha + (e_chunk @ V_chunk)
   暂存到 workspace 留给下一块

情况 ③：本块是最后一块
   prev_o = 上一块暂存的 o_running
   o_running = prev_o * alpha + (e_chunk @ V_chunk)
   output    = o_running / s_running       # 归一化，得到最终 attention 输出
   cast 成 fp16/bf16，写到最终输出张量
```

里面有两个"按行"的运算需要解释：

- **`prev_o * alpha`（按行乘）**：每一行 q 对应一个 alpha 标量，但 prev_o 那一行有 D=512 个值。所以是"每行的 D 个值都乘上这一行的 alpha"。NPU 的 SIMD 一次处理一整行，所以会先把这 1 个 alpha 复制成 D 份铺成一整行（这个动作叫"广播"，broadcast），再做整行的元素乘。
- **`o_running / s_running`（按行除）**：同理，每行的 D 个值都除以这一行的 s_running。

**关键观察**：每一块都依赖上一块的 `o_running`——所以 vec2 必须**按 s2 块的顺序串行执行**（同一行 q 内部），不能跳过、不能并行。这给整个算子的 tiling 增加了约束：每个核负责的 S2 切片必须连续，不能交叉分配给别的核。

#### 4.4.9 整体数据流图

```
   每轮循环 (loop = 第 k 块 S2):

                        ┌────────────────────────────────────┐
                        │ 上一块 (k-1) 留下来的 m_old, s_old   │
                        │ (存在双缓冲的另一槽位)               │
                        └─────────────────┬──────────────────┘
                                          │
   ┌──── mm1 (cube): scores = q @ K_chunk.T ────┐
   │                                            │
   ▼                                            ▼
   scores_chunk                          m_old, s_old
   │                                            │
   ├──────────► vec1 一条复合指令 ◄───────────────┤
   │            (做 4.4.4 算法的 1)~4) 步)。      │
   │                                            │
   │            输出:                            │
   │            ① e_chunk     (覆盖 scores)     │
   │            ② m_new       (本轮槽位)         │
   │            ③ s_new       (本轮槽位)         │
   │            ④ alpha       (给 vec2 用)       │
   │
   ▼
   e_chunk (cast fp16, 给 mm2 当输入)
   │
   ▼
   mm2 (cube): mm2_res = e_chunk @ V_chunk
   │
   ▼
   vec2 (vector):
     如果是第一块:
       o_running = mm2_res
     否则:
       prev_o    = 上一块的 o_running
       o_running = prev_o * alpha + mm2_res
     如果是最后一块:
       output = o_running / s_running
       cast fp16/bf16, 写到最终输出
     否则:
       把 o_running 暂存, 留给下一块
```

**总结一句话**：Flash Attention 把"先全扫一遍再 softmax"改成"边扫边累加 + 每次校正"。算子用三个"按行"的缓存量分别对应公式里的 `m_running`、`s_running`、`alpha`；vec1 用一条复合指令更新 m、s、alpha；vec2 用 alpha 修正之前的 o_running 并累加；最后一块除以 s 得到最终输出。

### 4.5 `sinks`（注意力下沉）：给 softmax 加个"虚拟 token"

**问题**：softmax 强制权重和=1。如果当前 q 跟所有历史 token 都不太相关，本来应该"什么都不看"，但因为强制和=1，softmax 还是会逼模型挑一个"不那么不相关"的当主角，造成噪声。

**解决**：给 softmax 加一个虚拟的"出口"——一个固定值的 sink。如果所有真实 token 都不相关，权重都跑到 sink 上去，等于"什么都没看"。

```python
real_scores = q @ K.T * scale     # shape (N,)
sink_value  = sinks_for_this_head  # 一个标量, 每个 q head 一个值
all_scores  = concat(real_scores, sink_value)   # shape (N+1,)
weights     = softmax(all_scores)               # shape (N+1,)
real_weights = weights[:N]                      # 丢掉 sink 那一位
output       = real_weights @ V                  # 注意 V 不变, 还是 N 个
```

算子里 sinks 是一个长度 N1=64 的 fp32 向量（每个 q head 一个值）。它在 vec1 阶段被广播到每一行 q 对应的位置上，作为 flash softmax 初始的 `m_old` 的下限值，一起参与归一化竞争——效果等同于"每行 softmax 多了一个虚拟 token，分数就是 sink 值"。

> "Attention sinks" 是 OpenAI 在 GPT-OSS 中引入的技术，用来缓解 streaming generation 时"早期 token 占太多权重"的问题。这里把它当成 softmax 的一个内置"逃生出口"理解就行。

---

## 第 5 章 数据是怎么排布的：layout、page attention、cu_seqlens

这一节讲算子拿到的输入数据**长什么样、怎么取出来**。是工程层面的，但不懂这些就理解不了后面 §7 算子内部为啥那么多 if 分支。

### 5.1 `layout`：BSND vs TND

一个 batch 里通常有多条独立序列，每条长度可能不一样。怎么把它们装进一个张量？两种主流做法：

**(1) BSND**：固定长度装，短的补 0（padding）。

```
   假设 b=2, s=4, n=1, d=2:
   batch 0: token0, token1, token2, token3       (4 个真实 token)
   batch 1: token0, token1, 0,      0            (2 个真实 + 2 个 padding)
   shape = (B=2, S=4, N=1, D=2)
```

**优点**：四维张量好算 stride。**缺点**：浪费内存（padding 多的话）。

**(2) TND**：所有 batch 的真实 token 直接拼在一起，配一个 `cu_seqlens` 数组告诉每个 batch 从哪到哪。

```
   batch 0: 4 个 token; batch 1: 2 个 token
   T = 4 + 2 = 6 个 token 拼在一起
   shape = (T=6, N=1, D=2)
   cu_seqlens = [0, 4, 6]   # 前缀和: batch 0 是 [0:4], batch 1 是 [4:6]
```

**优点**：不浪费。**缺点**：要先查 cu_seqlens 才知道 batch 边界。

本算子 `q` 支持 `BSND` 和 `TND`，`KV` 支持 `BSND/TND/PA_ND`。

算子对每个 batch 取真实长度的方式：

- **TND**：从 `cu_seqlens[bIdx]` 和 `cu_seqlens[bIdx+1]` 两个前缀和相减得到这一 batch 真实有多少个 token。
- **BSND**：要么所有 batch 等长（直接用 `qSeqSize`），要么从 `seqused_q` 数组读对应 batch 的真实长度。

不同 layout 走的是编译期分支（不同模板实例），运行时没有额外开销。

### 5.2 `PA_ND`：Page Attention 布局

KV cache 还有个特殊布局 `PA_ND`，灵感来自操作系统的"分页内存"。

**动机**：长对话时 KV cache 总长度变化大，提前分配一大块连续内存浪费。改成"按需分配 + 不连续存"。

**做法**：把内存切成固定大小的 block（比如 128 个 token 一个 block），每个 batch 用到的 block **不一定连续**。维护一个 `block_table` 记录每个 batch 用了哪些 block：

```
   全局 KV memory:
   ┌──────┬──────┬──────┬──────┬──────┬──────┐
   │block0│block1│block2│block3│block4│block5│  (每个 block 存 128 token 的 KV)
   └──────┴──────┴──────┴──────┴──────┴──────┘

   block_table = [
       [3, 1, 5],   # batch 0 用了 block 3、1、5（共 384 个 token）
       [0, 4],      # batch 1 用了 block 0、4（共 256 个 token）
   ]
```

要读 batch 0 第 200 个 token 的 KV：
- 200 / 128 = 1，余 72
- 查 `block_table[0][1] = 1`，去 block 1
- 在 block 1 里偏移 72 个 token

**对算子的影响**：搬数据时不能简单一次性把 "N 个连续 token" 一口气搬出来——因为它们在物理上可能跨了多个不连续的 block。算子要按 block 边界**分多次搬**：搬完一个 block 内的部分，去 block_table 查下一个 block 的物理位置，再搬下一段。这是 Page Attention 模式下数据搬运逻辑里那些 while 循环存在的根本原因。

### 5.3 query 的特殊性：n1 头共用一份 q

虽然 q 有 N1=64 个头，但本算子里 KV 只有 N2=1 头——意味着 64 个 q 头要做"同一组 K, V"的 attention。

算子把 64 个 q 头看作 64 个不同的 q 向量分别走 attention，所以 attention 的 m 维（行数）= **batch_size × q_seq_len × n1**，而不是 batch_size × q_seq_len。

m 维的基本切块大小（`mBaseSize`）正好取 64，意思是：**一个 m 基本块就是"一个 KV head 服务的 64 个 q 头"**。这 64 行 q 共用一份 K、V，做 attention 的时候可以一起算（mm1、mm2 一次处理 64 行）——这是 MQA 架构带来的天然分块单位。

---

## 第 6 章 用 Python 把整个算子实现一遍

跟 Compressor 详解一样，先用 Python 把整个 SparseAttnSharedkv 写一遍，等价于"参考实现"。这就是算子真正要算的全部数学；后面 §7 讲的所有工程细节（多核分工、流水、内存层级）都只是为了"把这段 Python 在 NPU 上跑得飞快"，**算出来的数值必须和这段 Python 一模一样**。

### 6.1 numpy 版（简化版：单 batch、单头、所有 query 共享一份 sparse_indices）

> **简化提醒**：本节为了把代码写短，**故意把 `cmp_sparse_indices` 当成"所有 query 共享的一份长度 K2 列表"**——所有 q 用同一批索引。
> **真实算子里 `cmp_sparse_indices.shape = [Q_S, N2, K2]`，每个 query 位置各自独立的一份 K2 索引**（"每个 q 自己选自己感兴趣的远处 token"，而不是大家挑同一批）。完整正确的版本在 §6.2。
> 这一节先用共享版本帮你看清滑窗 + 稀疏 + 因果约束的整体骨架。

```python
import numpy as np

def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sparse_attn_sharedkv_numpy(q, ori_kv, cmp_kv, cmp_sparse_indices,
                               softmax_scale, ori_win_left, cmp_ratio):
    """
    q:                  (S1, D)        当前 batch 的 q（这里假设 n1=1 简化）
    ori_kv:             (S2, D)        滑窗的原始 KV（key 和 value 数值相同，共享）
    cmp_kv:             (S3, D)        压缩的 KV
    cmp_sparse_indices: (K2,) int 索引 —— !!!简化!!! 假装所有 query 共享一份。
                        真实算子是 (S1, N2, K2)，每个 q 自己一份；§6.2 是完整版。
    softmax_scale:      float          缩放系数，通常 = 1/sqrt(D)
    ori_win_left:       int            滑窗左边界长度
    cmp_ratio:          int            压缩比（4 或 128）
    """
    S1, D = q.shape
    output = np.zeros_like(q)

    for s1 in range(S1):                # 对每个 q token 独立处理
        # === Step 1: 取出该 q 的"可见 KV 集合" ===

        # 1a. 滑窗：取 ori_kv 里 [s1 - ori_win_left, s1] 的范围（因果屏蔽: 不看未来）
        ori_left = max(0, s1 - ori_win_left)
        ori_visible = ori_kv[ori_left : s1 + 1]               # (W, D)

        # 1b. 稀疏 + 因果：从 sparse_indices 里挑出"不指向未来"的那些索引,
        #     再去 cmp_kv 里把对应行取出来。cmp_kv[i] 对应原序列 i*cmp_ratio 之前的内容,
        #     所以 idx 必须 < (s1+1) / cmp_ratio
        cmp_S2IdLimit = (s1 + 1) // cmp_ratio
        valid_cmp_idx = cmp_sparse_indices[cmp_sparse_indices < cmp_S2IdLimit]
        cmp_visible = cmp_kv[valid_cmp_idx]                    # (K2', D)

        # 1c. 拼起来
        K = np.concatenate([ori_visible, cmp_visible], axis=0) # (W+K2', D)
        # 注意: 本算子里 V = K（共享同一个张量）, 所以下面 weights @ K 就是 weights @ V

        # === Step 2~4: 标准 attention ===
        scores  = (q[s1] @ K.T) * softmax_scale                # (W+K2',)
        weights = softmax_np(scores, axis=-1)
        output[s1] = weights @ K                               # (D,)

    return output
```

**这段就是算子的核心数学**：每个 q 各自找自己看得见的 KV 集合，按标准 attention 公式算。

### 6.2 完整版：多头、batch、sinks、每 query 独立的 sparse_indices

上面只考虑单 batch、单头、忽略 sinks，并把 sparse_indices 简化成"全 query 共享"。完整版的接口长这样（**这才是算子实际的输入形状**）：

```python
def full_sparse_attn_sharedkv(
    q,                  # (B, S1, N1, D)  N1=64 q heads
    ori_kv,             # (B, S2, N2, D)  N2=1 KV heads → 64 q 共用
    cmp_kv,             # (B, S3, N2, D)
    cmp_sparse_indices, # (B, S1, N2, K2)  每个 (batch, s1, n2) 一个独立的 topk 列表
    sinks,              # (N1,)            每个 q head 一个 sink 值
    softmax_scale, ori_win_left, cmp_ratio,
):
    B, S1, N1, D = q.shape
    output = np.zeros_like(q)
    for b in range(B):
        for s1 in range(S1):
            for n1 in range(N1):
                n2 = 0  # KV heads 只有 1 个, n1=64 q heads 共用
                # gather KVs
                ori_visible = ori_kv[b, max(0, s1 - ori_win_left) : s1 + 1, n2]
                cmp_S2IdLimit = (s1 + 1) // cmp_ratio
                idxs = cmp_sparse_indices[b, s1, n2]          # ← 每个 q 拿自己那行 K2 索引
                idxs = idxs[idxs < cmp_S2IdLimit]             # ← 因果过滤
                cmp_visible = cmp_kv[b, idxs, n2]
                K = np.concatenate([ori_visible, cmp_visible], axis=0)

                # attention with sink
                scores = (q[b, s1, n1] @ K.T) * softmax_scale       # (W+K',)
                # 把 sink 拼到 scores 末尾, softmax 后丢弃 sink 那位
                scores_with_sink = np.concatenate([scores, [sinks[n1]]])
                weights = softmax_np(scores_with_sink, axis=-1)[:-1]
                output[b, s1, n1] = weights @ K
    return output
```

**这就是算子的全部数学功能**。剩下的所有工程复杂度都是为了"在 NPU 上跑得快"。

> **跟 §6.1 简化版的差别**：§6.1 假装所有 query 共享同一份 `cmp_sparse_indices`，所以循环里直接用 `cmp_sparse_indices` 当成整份索引。完整版里**用 `cmp_sparse_indices[b, s1, n2]` 按 query 位置取出当 query 专属的 K2 行**——这才是算子实际的行为。对 kernel 内部实现关系不大（搬运逻辑都一样），但对**理解上层为啥那么传参**很关键。

> **真实 NPU 算子还多了**：① 当 N1=64 + S1 都很大时，q 用 group 维度切；② Flash Attention 的分块在线 softmax（§4.4）；③ Page Attention 的 block_table 翻译；④ cube/vector 双核流水；⑤ TND 变长 batch 处理。我们先把数学搞透，下面再讲工程。

---

## 第 7 章 在 NPU 上是怎么实际跑的

跟上一篇 Compressor 一样，NPU 算子的关键不是"算对"（Python 版已经能算对），而是"跑得**快**"。这一节讲算子的工程结构。

### 7.1 cube + vector 异构核：还是那一套

回顾 [`Compressor 详解.md`](./Compressor%20详解.md) §4：

- **AIC（cube core）**：专门做矩阵乘
- **AIV（vector core）**：做向量四则运算（softmax、mul、add、cast）

本算子也用 1 个 cube 核配 2 个 vector 核的混合配置，互相并行。

差别在于：Compressor 算子里 cube 干 1 件事（mm1）、vector 干 1+1 件事（vec1+vec2）。**本算子 cube 干 2 件事（mm1+mm2），vector 干 2 件事（vec1+vec2）**——SCFA 模式还多一件事 vec0。

### 7.2 一个 attention 包含 5 段计算

和 §6 numpy 版对照：

```
   ┌─────────────────────────────────────────────┐
   │  原 attention 公式                            │
   │  O = softmax(q @ K^T * scale) @ V             │
   └─────────────────────────────────────────────┘

   把这一行公式拆成 5 段, 分别跑在不同核上:

   ┌────────────────────────┬─────────┬──────────────┐
   │ 段名                    │ 跑在    │ 干啥                                              │
   ├────────────────────────┼─────────┼──────────────────────────────────────────────────┤
   │ vec0 (仅 SCFA 模式)     │ vector  │ 按 sparse_indices 把 cmp_kv 的几行拼到一块连续内存 │
   │ mm1 (Q @ K^T)           │ cube    │ 算 scores, shape (M, S2 or K2)                    │
   │ vec1 (scale + mask + softmax) │ vector │ 缩放 + 屏蔽 + flash softmax + cast 成 fp16   │
   │ mm2 (P @ V)             │ cube    │ 算加权和, shape (M, D)                            │
   │ vec2 (累加 + 输出)        │ vector  │ flash 累加 + 除以 sum + cast + 写 GM            │
   └────────────────────────┴─────────┴──────────────────────────────────────────────────┘
```

### 7.3 vec0：SCFA 模式独有的"按索引收集"

SCFA 模式下，cmp_sparse_indices 告诉算子"从 cmp_kv 里挑 K2=512 行"。这些行**在 cmp_kv 内存里是离散的**——不能直接给 cube 算 mm1（cube 要求输入连续）。

vec0 的工作：**先按 sparse_indices 把要用的 K2 行从 cmp_kv 搬到一段连续的临时内存里**。之后 mm1 直接读这段连续内存，跟正常 attention 一样。

流程上：

```
for 每两个相邻索引 (idx_a, idx_b) in sparse_indices:
    从 cmp_kv 把 cmp_kv[idx_a] 和 cmp_kv[idx_b] 搬到 vector 的本地小内存里
    攒够一定数量后, 一次性写到 workspace 的连续区域
```

这个"先散读，再写到一块连续区域"的操作，本质是 **gather + scatter**。SWA 和 CFA 模式不需要 vec0（因为 ori_kv / cmp_kv 本身就是顺序连续读）。

**为啥不让 cube 直接散读？** cube 的矩阵乘要求右矩阵在 L1 上是连续排布的，从 GM 散读再拼成 cube 要求的格式，cube 本身做不到。让擅长 gather 的 vector 先把数据拼好，再交给 cube 顺序读——这是合理的分工。

### 7.4 mm1：Q @ K^T

这是 attention 第一个矩阵乘。在每个核上：
- 左矩阵 A = q：(M=64, K=512)，从 GM 搬到 L1 → L0A
- 右矩阵 B = K（其实就是 ori_kv 或 cmp_kv 的某一段）：(N=512, K=512)，从 GM 一步一步搬进 cube 的高速内存
- 输出 C = A @ B^T：(M=64, N=512)，写到 workspace 上，类型 fp32

> 注意，B 是 K 矩阵但要算的是 `q @ K^T`，所以 B 在加载时要做一次转置。cube 硬件本身支持加载时直接转置，不用单独额外算一遍。

K 维（512）切成 2 个 256 的子块，每个子块再切成 2 个 128 的更小块，用 ping-pong 双缓冲交替计算——目的是让"搬下一块"和"算当前块"同时进行，掩盖搬运延迟。

### 7.5 vec1：scale + mask + flash softmax + cast

mm1 输出的 scores 是 fp32，shape (M, S2_block)（一次处理一块 S2）。vec1 顺序做 5 步：

```
   1. scores *= softmax_scale         # 缩放
   2. apply mask                       # 把屏蔽位置设 -inf
   3. flash softmax 更新 m / s / alpha  # §4.4 的在线累加
                                       # 同时输出本块的 e_chunk = exp(scores - m_new)
   4. e_chunk = cast(e_chunk, fp16)    # 因为后面 mm2 要 fp16 输入
   5. 把 e_chunk 写到 workspace, 留给 mm2
```

flash softmax 维护的"按行最大值 m"、"按行 sum s"、"修正系数 alpha"三个量在 vec1 这一轮被更新，在 vec2 那一轮被用来"修正之前累加的 mm2 输出"。

### 7.6 mm2：P @ V

P = 上一步的 e_chunk（fp16），shape (M, S2_block)。
V = ori_kv 或 cmp_kv（fp16），shape (S2_block, D)。
输出 mm2_res = P @ V，shape (M, D)，fp32，写到 workspace。

跟 mm1 类似，cube 核做这件事。区别是 K 轴换成了 S2_block，N 轴是 D=512。

### 7.7 vec2：flash 累加 + 除以 sum + 写出

vec2 是 attention 的最后一段。回看 §4.4 的 Flash Attention 公式：

```
o_new = o_prev * alpha + e_chunk @ V_chunk
final_output = o_final / s_final
```

vec2 的逻辑分两步：

```
# 1. 如果不是第一块, 把上一块的 mm2 结果用 alpha 修正后加到本块
if not 是第一块:
    prev_mm2  = 上一块暂存的 o_running
    prev_mm2 *= alpha           # 按行乘修正系数
    cur_mm2  += prev_mm2        # 累加

# 2. 如果是最后一块, 除以 sum, cast, 写到最终输出
if 是最后一块:
    cur_mm2 /= s_running        # 按行除, 完成 softmax 归一化
    cur_mm2 = cast(cur_mm2, fp16/bf16)
    写到 attention 输出张量
else:
    把 cur_mm2 暂存到 workspace, 等下一轮继续累加
```

### 7.8 五段 + 双缓冲 = 三任务流水

如果只是按顺序跑 vec0 → mm1 → vec1 → mm2 → vec2，会有大量空闲（cube 算的时候 vector 在等，反之亦然）。**双缓冲**让两个核可以错开干：

```
   时间 ──────────────────────────────────────────────────►
   cube 核:    │ mm1[块0] │ mm2[块-1] │ mm1[块1] │ mm2[块0] │ ...
                                          │           │
                                          ▼           ▼
   vector 核:           │ vec1[块0] │ vec2[块-1] │ vec1[块1] │ vec2[块0]│ ...
                                                    │           │
                                                    ▼           ▼
                                                  vec0[块2] (SCFA 模式下穿插)
```

这是"**triple-cache 流水**"——同时维护三个块的状态：

| 块编号  | 状态                                      |
| ---- | --------------------------------------- |
| 当前块  | 正在算 mm1（cube）+ 正在算 vec0（vector，SCFA 模式） |
| 上一块  | 正在算 vec1（vector）+ 正在算 mm2（cube）         |
| 上上一块 | 正在算 vec2（vector）                        |

主循环的伪代码逻辑是：每跑一轮，**同时**处理三个相邻块的不同阶段：

```
for k = 0, 1, 2, ...:
    本轮 (块 k):       cube 算 mm1[k] / vector 算 vec0[k]（SCFA）
    上一轮 (块 k-1):   cube 算 mm2[k-1] / vector 算 vec1[k-1]
    上上一轮 (块 k-2): vector 算 vec2[k-2]
```

每次循环往前推进一格，三块同时在不同阶段流水。算子用一个长度为 3 的"任务环形缓存"轮替地存这三块的参数信息。

### 7.9 cube 和 vector 之间的同步信号

两个核之间通过 4 对硬件 flag 同步——本质就是一组共享的小整数标志位，"我做完了就设上"、"我要用就等它被设上"。生产者-消费者关系如下：

| flag | 谁负责"设上" | 谁负责"等到" | 含义 |
|---|---|---|---|
| `V0→C1` | vector (vec0 写完) | cube (mm1 读前) | SCFA 模式：vec0 拼好的 KV 已就绪 |
| `C1→V1` | cube (mm1 写完) | vector (vec1 读前) | mm1 输出可读 |
| `V1→C2` | vector (vec1 写完) | cube (mm2 读前) | e_chunk 已就绪 |
| `C2→V2` | cube (mm2 写完) | vector (vec2 读前) | mm2 输出可读 |

直觉上的顺序：

```
cube 写 mm1 → 通知 vector → vector 算 vec1 → 通知 cube
         → cube 算 mm2 → 通知 vector → vector 算 vec2
```

这种 pair-wise 的小同步比"全核栅栏（SyncAll）"轻得多——只阻塞相关的两个核，其它核不受影响。

### 7.10 多核分工：metadata 提前告诉每个核干啥

NPU 一上来就有几十个 cube 核 + 几十个 vector 核。每个核处理输入的哪一段？这个**分核策略**不是在 kernel 内动态算的，而是**提前由 host 端的一个 aicpu 算子**（`npu_sparse_attn_sharedkv_metadata`）算好，存在 `metadata` 张量里（shape [1024]）。

每个核启动时去 metadata 里读自己的"任务范围"——具体来说是三个嵌套维度的起止：

```
   for bN2 in [bN2Start, bN2End):       # 当前核负责的 batch × kv_head 范围
       for gS1 in [gS1Start, gS1End):   # 当前核负责的 (q head × q seq) 范围
           for s2 in [s2Start, s2End):  # 当前核负责的 KV 序列范围
               do_attention_block(...)
```

**为啥要前置算？** 因为不同 batch 的真实长度可能差别很大，简单按"batch 数 / 核数"分配会负载不均（有的核累死、有的核闲死）。这个 aicpu 算子按"实际算量"来分，让每个核大致干一样多的活。metadata 里还有一位"启用标志"——某些核如果总算量太少就直接被标记成不干活，免得起灶的开销大于干活的开销。

### 7.11 workspace 内存布局

每个核都需要一段 GM workspace 暂存中间结果。布局如下：

```
   GM workspace 布局（每个核独立分一段）:
   ┌────────────────────────────────────────────────┐
   │ mm1 结果         [双缓冲 2 份] fp32              │  ← cube 写, vec1 读
   ├────────────────────────────────────────────────┤
   │ vec1 结果（e_chunk）  [双缓冲 2 份] fp16/bf16    │  ← vec1 写, cube 读 (mm2 的输入 A)
   ├────────────────────────────────────────────────┤
   │ mm2 结果         [双缓冲 2 份] fp32              │  ← cube 写, vec2 读
   ├────────────────────────────────────────────────┤
   │ vec2 累加暂存    [双缓冲 2 份] fp32              │  ← vec2 写, vec2 下一轮读（累加用）
   ├────────────────────────────────────────────────┤
   │ 散读拼接结果（仅 SCFA） [4 份]                   │  ← vec0 写, cube 读 (mm1 的 B)
   ├────────────────────────────────────────────────┤
   │ 散读拼接长度信息（仅 SCFA）                       │  ← vec0 写, cube 读
   └────────────────────────────────────────────────┘
```

每段都按"核编号 × 单核大小"做偏移，让每个核读写自己专属的内存区域，互不干扰。每段都有 ping-pong 两份缓冲（即"双缓冲"），让上一块的数据在被读的同时下一块可以写到另一份。

### 7.12 vector 核内部的小内存（UB）

vector 核内部还有一个比 GM 快得多但只有 ~192 KB 的"小内存"（UB），是真正参与计算的地方。整个 vec0、vec1、vec2 的所有中间数据都要塞进这块小内存。算子靠**复用 + ping-pong** 把内存挤出来。粗略分配如下：

| 用途 | 大小 | 备注 |
|---|---|---|
| 主输入缓冲（vec1 / vec2 复用） | 32K × 2 ping-pong | mm1 / mm2 结果搬进来 |
| 副输入缓冲（vec2 第二输入） | 8K × 2 ping-pong | 累加 prev_o 用 |
| 主输出缓冲 | 32K | cast 后的输出 |
| softmax 临时空间 | 32K | flash softmax 内部用 |
| `m_running`（按行最大值） | 1K × 2 | 双缓冲，让上下两轮不冲突 |
| `s_running`（按行 sum） | 1K × 2 | 同上 |
| `alpha`（修正系数） | 1K × 2 | 同上 |
| sinks 原始值 | 128 × 4B | 64 个 q head 一个标量 |
| sinks 广播展开 | 128 × 4B × 8 × 3 | 广播到每行的形式 |

理解一句话：**双缓冲是流水的本钱**——所有"上轮要读、本轮要写"的数据都得开两份，否则流水就卡住了。

---

## 第 8 章 三种模板：SWA、CFA、SCFA

回顾 §2.3 的三个场景，每个场景在算子内部对应一个模板分支：

| 场景           | 模板编号 | 实现方式                          |
| ------------ | ---- | ----------------------------- |
| 一：纯 SWA      | 0    | 基础模板，只处理 ori_kv               |
| 二：SWA + CFA  | 1    | 基础模板的扩展：外层 s2 循环多走一段处理 cmp_kv |
| 三：SWA + SCFA | 2    | 独立模板：在 mm1 之前多一段 vec0         |

### 8.1 SWA 和 CFA 共用一套实现

SWA 和 CFA 共用同一个 kernel 模板，区别只在外层 s2 循环里多走一段——前半段处理 ori_kv 的几个 s2 块，后半段（仅 CFA）继续处理 cmp_kv 的几个 s2 块。每一块上挂一个布尔标志 `isOri`，告诉 cube 核"这块该去 ori_kv 还是 cmp_kv 拿数据"。

key 在于：**flash softmax 在 ori 和 cmp 两段之间是连续累加的**。也就是说，softmax 的"分母 s"和"分子 o"在跨过 ori→cmp 的边界时不会清零，会照常用 alpha 修正——这等同于把 ori_kv 和 cmp_kv 的 scores 拼接成一长串再做一次大 softmax。这就实现了 §2.2 说的"拼接 + 一起 softmax"。

### 8.2 SCFA 模式多了 vec0 + 用 workspace 中转

SCFA 模式的 cmp 部分 KV 不连续（要按 sparse_indices 挑）。所以多了一段 vec0：vector 核先把要用的 cmp_kv 行从离散位置 gather 到 workspace 上一段连续的临时区域，**等待 vec0 写完的信号到了，cube 再去读 mm1**。

执行时的分工：

- **ori 块**：cube 直接从 ori_kv 读，跟 SWA / CFA 没区别。
- **cmp 块**：cube 等 vec0 的同步信号 → 读 vec0 拼好的连续区域 → 跑 mm1。

vec0 只在 cmp 块的轮次跑，ori 块的轮次 vector 核可以提前做别的事。

### 8.3 怎么选模板

host 端在算子编译期就根据用户传入哪些可选参数决定走哪个模板：

| 传入参数                                       | 走的模板 |
| ------------------------------------------ | ---- |
| 只有 `ori_kv`                                | SWA  |
| `ori_kv` + `cmp_kv`                        | CFA  |
| `ori_kv` + `cmp_kv` + `cmp_sparse_indices` | SCFA |

模板编号会被编进 tilingKey，runtime 启动 kernel 时根据 tilingKey 走对应的代码分支。所以这三个模板**编译期就被特化**，运行时没有 if/else 的开销，三种场景各跑各的。

---

## 第 9 章 用一个迷你例子走完整流程

为了好画图，故意用极小的尺寸（注意这不是合法配置，只为讲解）：

```
B = 1, S1 = 4, S2 = 8, S3 = 4, N1 = 2, N2 = 1, D = 4
ori_win_left = 3, cmp_ratio = 2, cmp_sparse_indices = [0, 2]  (取 cmp_kv 的第 0、2 行)
```

q 是 `(1, 4, 2, 4)`：1 个 batch、4 个 q token、2 个 q head、向量长度 4。
ori_kv 是 `(1, 8, 1, 4)`，cmp_kv 是 `(1, 4, 1, 4)`。
sparse_indices 是 `(1, 4, 1, 2)`，每个 (batch, s1, n2) 给 2 个索引。

### 9.1 计算 q 的第 2 个 token (s1=2) 第 0 个 head 的 attention

**Step 1**: 确定可见 KV 范围

- 滑窗：s1=2，左边界 = max(0, 2-3) = 0，右边界 = 2。所以 ori_visible = ori_kv[0:3]，3 个 token。
- 稀疏：cmpS2IdLimit = (2+1)//2 = 1，所以 cmp_sparse_indices=[0, 2] 里只有 idx=0 有效（idx=2 ≥ 1，过未来了）。cmp_visible = cmp_kv[0]，1 个 token。
- 拼起来 K = ori_visible + cmp_visible，共 4 行。

```
   q[0,2,0,:]:    [q0, q1, q2, q3]   (D=4)
   K (4 行):
                  ┌────────────────┐
                  │ ori_kv[0]      │  ← ori 滑窗第 1 个
                  │ ori_kv[1]      │  ← ori 滑窗第 2 个
                  │ ori_kv[2]      │  ← ori 滑窗第 3 个 (=s1=2 自己)
                  │ cmp_kv[0]      │  ← 稀疏选的 1 个
                  └────────────────┘
```

**Step 2**: q @ K.T

`(1, 4) @ (4, 4) = (1, 4)`，得到 4 个 scores。

**Step 3**: scale + softmax (with sink)

```python
scores       = scores * softmax_scale         # 缩放
scores_full  = concat(scores, [sinks[0]])     # 追加 sink，shape (5,)
weights      = softmax(scores_full)[:4]       # 丢掉 sink, shape (4,)
```

**Step 4**: weights @ K

`(4,) @ (4, 4) = (4,)`，得到 output 向量 (D=4)。这就是 q[0,2,0,:] 的输出。

### 9.2 NPU 上的执行流

跟 §6.2 numpy 版相比，NPU 把 4 个 q token × 2 q heads = 8 行 q 看作 m=8 的一批，**一次性**对所有 8 行做 attention（不像 Python 版那样 for 循环）。

```
   M = 8 行 q (s1∈[0,3], n1∈[0,1])
   每行各自有不同的 K 范围 (因为 mask), 但 ori_kv 部分大致连续

   cube 核:
   - mm1: q[8 行] @ ori_kv[s2_block].T  → scores[8, S2_block]
          (ori 阶段)
   - mm1: q[8 行] @ cmp_visible.T       → scores[8, K2']
          (cmp 阶段，SCFA 模式下 cmp_visible 是 vec0 拼好的)
   - mm2: weights[8, total] @ K_or_V    → out[8, D]

   vector 核:
   - vec0 (仅 SCFA): 按 sparse_indices 把 cmp_kv 几行拼到 workspace 上的连续区域
   - vec1: 缩放 + mask + flash softmax + cast → e_chunk
   - vec2: flash 累加修正 + 除以 sum + cast → 写最终 attention 输出
```

每个 s2 块（256 或 512 个 KV token）是一个 `s2LoopIdx`。整个序列可能切成几十个 s2 块，按块流水。

---

## 第 10 章 一句话总结

> **`SparseAttnSharedkv` = 在长上下文推理里，让一个 q 同时跟"近处的 ori_kv 滑窗"+"远处 cmp_kv 中按 sparse_indices 选出的几行"做带 sinks 的 flash attention。**
>
> 在 NPU 上拆成 5 段（vec0 → mm1 → vec1 → mm2 → vec2）跑 cube / vector 异构流水，3 任务双缓冲覆盖搬运延迟，按 metadata 把 batch × n2_head × m_block × s2_block 分给几十个核同时干活，所有中间数据走 workspace 接力，最终合作完成一次 attention 计算。

---

## 附录 A：和 Compressor 算子的关系

这两个算子是配套的：

```
   原始 KV cache (100 万 token)
            │
            ▼
   ┌───────────────────────────┐
   │  Compressor 算子           │   ◄── 上一篇 [Compressor 详解.md] 讲的
   │  每 cmp_ratio 个 token     │       (压缩, 上游)
   │  压成 1 个                 │
   └───────────────────────────┘
            │
            ▼
   cmp_kv (25 万 token, cmp_ratio=4)
            │
            │  + ori_kv (近 128 个 token, 不压缩)
            │  + cmp_sparse_indices (从 cmp_kv 里选的 topk 索引)
            │  + q (当前生成的 query)
            ▼
   ┌───────────────────────────┐
   │  SparseAttnSharedkv 算子   │   ◄── 本文讲的
   │  滑窗 + 稀疏压缩 attention │       (注意力, 下游)
   └───────────────────────────┘
            │
            ▼
   attention output → 喂给下一层 Transformer
```

`cmp_sparse_indices` 这个 topk 列表是上层用别的算子（不在本算子负责）算出来的——常见做法是先用 q 跟 cmp_kv 做粗略匹配（比如用一个轻量化的相似度计算），选 top-K2 个最相关的位置作为索引。

---

## 附录 B：常见问题

**Q：为啥 K 和 V 用同一个张量（K=V）？**

A：本算子的 "Shared KV" 不止指多 q head 共享，还指 **K 和 V 数值上是同一个张量**。这是 DeepSeek 的 MLA / NSA 路线下的设计：上游 Compressor 算子直接产出一份 KV，没有分开的 K/V 投影（投影合在 cmp_kv 里了）。所以本算子的 mm1 (Q @ K^T) 和 mm2 (P @ V) 用的是**同一个张量**——只是 mm1 把它当 K 用（参与第一次矩阵乘，做了一次转置）、mm2 把它当 V 用（参与第二次矩阵乘），方向不同。

**Q：为啥 N1=64 但 N2=1？这不是浪费表达能力吗？**

A：这是 **MQA**（Multi-Query Attention）的极端配置。MQA 牺牲一点表达能力换 KV cache 体积变 1/64——对长上下文推理（KV cache 是瓶颈）非常关键。DeepSeek-V2/V3、Llama-3 等大模型普遍采用 GQA/MQA。

**Q：vec0、vec1、vec2 为啥要拆开？合一起不行吗？**

A：拆开是为了让 cube 的 mm1/mm2 能和 vector 的 vec0/vec1/vec2 **并行流水**。如果合在一起，cube 算的时候 vector 必须空等。拆成 5 段，不同块的 5 段交错跑，cube 和 vector 都没空闲。

**Q：为啥 SCFA 模式要先 vec0 拼 KV，不让 cube 直接按索引读？**

A：cube 的矩阵乘要求 B 矩阵在 L1 上是连续的（NZ 排布有特殊步长要求）。从 GM 按索引散读再排成连续对 cube 来说是逆操作，cube 没有这个能力。让 vector 干这件事（vector 擅长 gather）后写到 workspace，cube 只需要顺序读连续内存——分工合理。

**Q：sinks 加进 softmax 后，输出还是按"和=1 的权重"加权 V 吗？**

A：注意，sink 那一位**不参与** `weights @ V`——它只参与 softmax 归一化，让真实 token 的权重和**小于** 1（差额跑到 sink 上）。所以最终输出 = `(weights[:N] @ V)`，weights[:N] 之和 < 1。这正是"如果都不相关就什么也不输出"的效果。

**Q：fp32 中间结果 vs fp16/bf16 输入输出，为啥要混精？**

A：fp16 精度有限，做大量累加会损失精度。所以"读输入"和"写输出"用 fp16/bf16（省内存和带宽），但**累加和归一化在 fp32 上做**。具体来说：mm1 输出、mm2 输出、flash softmax 的 m/s/alpha 三个累加量都是 fp32；只有交付给下游 mm 的 e_chunk 和最终写出的 attention 输出做了 cast。

**Q：怎么调试本算子？**

A：常用三种手段：
1. **CPU 端先写 numpy 参考实现**（就是 §6 那种），保证数学是对的。
2. **NPU dump 中间结果**：让 kernel 把 mm1 / vec1 / mm2 的中间结果拷回 host，用 numpy 对比每一段。
3. **小 shape 跑通了再放大**：先 B=1, S1=64, S2=128, N1=64, D=512 这种能手算的 shape，确认每一段对了再上正常 shape（B=4, S1=128, S2=8192）。

---

## 附录 C：本算子 vs Compressor 算子对照

| | **Compressor** | **SparseAttnSharedkv** |
|---|---|---|
| 数学功能 | 把每 `cmp_ratio` 个 token 压成 1 个 | 给 q 计算"滑窗 + 稀疏 cmp"的 attention |
| 输入到输出比 | cmp_ratio → 1（压缩） | M → M（attention 不改长度） |
| 是否融合多个 op | 是，融合 7 个 op | 是，融合 5 段（含 1~2 段 mm + 2~3 段 vec） |
| 是否依赖跨调用状态 | 是（state_cache 累积） | 否（只看本次输入） |
| cube/vector 协作 | 1 cube + 2 vector，1 段 mm | 1 cube + 2 vector，2 段 mm + 2~3 段 vec |
| 流水深度 | 双缓冲 (2) | 三任务流水 (3) |
| 全核同步 | 每 nSize 轮一次 SyncAll | 不需要，只用 pair-wise flag |
| 多核分工 | tiling 内嵌算 | host 端 aicpu 算子算好放 metadata |
| 模板个数 | 2 (FullLoad / 普通) | 3 (SWA / CFA / SCFA) |
| 关键挑战 | 在 vec1/vec2 之间做窗口 reshape | flash attention 的在线 softmax + 多个 KV 来源拼接 |

读完这两篇文档，你应该能看懂大模型长上下文推理里"压缩 KV → 稀疏选 KV → 算 attention"的完整链路，以及它们在 NPU 上是怎么协作的。
