# 三级软件流水设计 (PreloadPipeline TileLang 端口)

> 配套 `PERF_HANDOFF.md` §5。本文是动主 kernel 前的相位结构设计稿,供 review。
> 目标:把当前"每 chunk 串行 + 54 个 barrier_all"重排成 Ascend C 的双缓冲三级流水,
> 吃掉 19ms 的 cube/vector 互等 gap,并让 vector 核内 MTE2/VEC/MTE3 跨 chunk 重叠。

---

## 1. 五个相位 + 依赖链 (从 Ascend C 抄过来的)

一个 chunk 的完整处理拆成 5 个相位,横跨 cube/vector 两个核:

| 相位 | 核 | TileLang scope | 干什么 | 当前 kernel.py 行 |
|---|---|---|---|---|
| **V0** gather | vector | `T.Scope("V")` | 逐行 gather KV → `kv_ub_multi` → 批量写 `ws_kv`;建 mask | 473–613 |
| **MM1** QK | cube | `T.Scope("C")` | `ws_kv`→`kv_lo/hi`(L1)→ gemm QK → `ws_score` | 384–428 |
| **V1** softmax | vector | `T.Scope("V")` | mask 加性、scale、online-softmax(出 P + 出 rescale 因子 α)→ `ws_p` | 615–719 |
| **MM2** PV | cube | `T.Scope("C")` | `ws_p`→`p_lo/hi`(L1)→ gemm PV → `ws_o` | 430–456 |
| **V2** merge | vector | `T.Scope("V")` | `ws_o`→ 输出累加器递推 `acc = O + α·acc` | 721–735 |

**跨核依赖链 (4 个 forward cross-flag,和 Ascend C 的 syncV0C1/C1V1/V1C2/C2V2 一一对应):**

```
V0 ──KV_READY(V→C,MTE3)──▶ MM1 ──SCORE_READY(C→V,FIX)──▶ V1 ──P_READY(V→C,MTE3)──▶ MM2 ──PV_READY(C→V,FIX)──▶ V2
   syncV0C1                  syncC1V1                     syncV1C2                  syncC2V2
```

这 4 个 flag 当前 kernel 已经有(`_FLAG_KV_READY/_SCORE_READY/_P_READY/_PV_READY`);
第 5 个 `_FLAG_ITER_DONE` 在流水版里改作**跨 work-item 边界 flag**(对应 Ascend C 的 flag `3`,见 §4:
不需要任何 per-buffer back-flag)。

> ⚠️ 与 Ascend C 的唯一架构差异:Ascend C 的 **ori** gather 在 cube 侧(DataCopyPA),
> 所以 ori 的 MM1 不等 V0;TileLang 里 ori/cmp gather **都在 vector**,所以 MM1 永远等 KV_READY。
> ori 只占 5 个 chunk 里的 1 个,影响可忽略。

---

## 2. 关键重构:把输出累加器递推从 V1 挪到 V2 (这是能流水起来的前提)

**问题**:online-softmax 的输出累加器 `acc_o` 是跨 chunk 串行递推的:

```
acc_o_i = acc_o_{i-1} · α_i + O_i        (α_i = exp(m_{i-1} - m_i),  O_i = P_i@V)
```

当前 kernel 把这个递推拆在两处:
- V1 里 `T.tile.mul(acc_o, acc_o, m_i_prev)` —— rescale 旧累加器 (kernel.py:698–705)
- V2 里 `T.tile.add(acc_o, acc_o, acc_o_ub)` —— 加本 chunk 的 O (kernel.py:733)

流水后 V1(i-1) 和 V2(i-2) 在同一拍跑,两者都写 `acc_o` → **冲突**,无法重叠。

**Ascend C 的解法**(`scfa_block_vector.h` `DealBmm2ResBaseBlock` line 881–957):
**rescale 和 add 全部塞进 V2,V1 完全不碰累加器**。V1 只负责算出 α_i 并 stash;
V2(i) 做完整递推 `acc = O_i + α_i · acc`。

**TileLang 落地**:
- V1(i):照常算 `m_i`(running max)、`m_i_prev = exp(m_old - m_new) = α_i`、`sumexp = sumexp·α_i + sumexp_i`、出 P_i。**删掉** 698–705 的 `acc_o *= α`。把 α_i 存进双缓冲 `alpha[i%2]`。
- V2(i):`acc_o = ws_o[i] + alpha[i%2] · acc_o`(rescale+add 融合)。

**为什么单缓冲 `acc_o` 就够**(不需要给累加器开双份,省 64KB UB):
- V1 不再碰 `acc_o`,所以 V1(i-1) 和 V2(i-2) 不在 `acc_o` 上冲突。
- V2 是 `acc_o` 的唯一读写者,且 V2(i)、V2(i+1) 按程序序串行(同一核同一 pipe),递推天然正确。
- V2 的 VEC 部分(rescale+add)和 V1 的 VEC(softmax)本来就同在 VEC pipe、必然串行,
  单缓冲累加器**不损失**重叠;真正能和 V1-VEC 重叠的是 V2 的 `ws_o` MTE2 load(进临时 buffer)。

> `sumexp`(running sum)、`m_i`(running max)也是 V1 单缓冲、V1-串行,不动。
> 只有传给 V2 的 **α** 需要双缓冲(V1(i) 产、V2(i) 在 +1 拍后消费,距离 1 拍,×2 够)。

---

## 3. 错位调度 (steady state + prologue + epilogue)

令 `NI = NI_total`(8K 场景 = 5)。统一用一个 step 循环 `t ∈ [0, NI+2)` 驱动两个 scope,
各相位用 `if` 守卫错位:

**Vector scope** (`for t in range(NI+2)`):
```
V0(t)    if t < NI            # gather chunk t      → ws_kv[t%2],   set KV_READY
V1(t-1)  if 1 <= t <= NI      # softmax chunk t-1   wait SCORE_READY → ws_p[(t-1)%2], set P_READY
V2(t-2)  if 2 <= t <= NI+1    # merge chunk t-2     wait PV_READY,   acc_o = ws_o + α·acc_o
```

**Cube scope** (`for t in range(NI+2)`):
```
MM1(t)   if t < NI            # QK chunk t          wait KV_READY → ws_score[t%2], set SCORE_READY
MM2(t-1) if 1 <= t <= NI      # PV chunk t-1        wait P_READY  → ws_o[(t-1)%2], set PV_READY
```

**时间线**(NI=5,▣=该拍该核在跑的相位):

```
step t :   0      1      2      3      4      5      6
cube   : MM1·0  MM1·1  MM1·2  MM1·3  MM1·4    -      -
         -      MM2·0  MM2·1  MM2·2  MM2·3  MM2·4    -
vector : V0·0   V0·1   V0·2   V0·3   V0·4     -      -
         -      V1·0   V1·1   V1·2   V1·3   V1·4     -
         -      -      V2·0   V2·1   V2·2   V2·3   V2·4
         └prologue┘   └────── steady ──────┘   └─epilogue─┘
```

稳态(step 2–4)里:**同一拍 vector 同时推进 gather(chunk t)、softmax(chunk t-1)、merge(chunk t-2)**,
分别压 MTE2 / VEC / MTE3 三条 pipe → vector 总时间从 `Σ(V0+V1+V2)` 降到 `max(V0,V1,V2)`。
**同一拍 cube 推进 QK(t) 和 PV(t-1)** → cube 不再空等 vector,gap 收缩。

**Forward flag 计数自动平衡**(每个 flag set/wait 各 NI 次):

| flag | set | wait | 次数 |
|---|---|---|---|
| KV_READY | V0, t∈[0,NI) | MM1, t∈[0,NI) | NI |
| SCORE_READY | MM1, t∈[0,NI) | V1, t∈[1,NI] | NI |
| P_READY | V1, t∈[1,NI] | MM2, t∈[1,NI] | NI |
| PV_READY | MM2, t∈[1,NI] | V2, t∈[2,NI+1] | NI |

---

## 4. 多缓冲 + 为什么 forward flag + 深度就够(无需 back-flag,复刻 Ascend C)

**多缓冲**(对齐 Ascend C `loop % preLoadNum`,`preLoadNum=2`):4 个 GM workspace 都按 chunk 奇偶 ×2 轮转。

| workspace | 现 shape | 流水后 | 索引 |
|---|---|---|---|
| `ws_kv` | `[core,BI,D]` | `[core,2,BI,D]` | `[cid, t%2]` |
| `ws_score` | `[core,H,BI]` | `[core,2,H,BI]` | `[cid, t%2]` |
| `ws_p` | `[core,H,BI]` | `[core,2,H,BI]` | `[cid, t%2]` |
| `ws_o` | `[core,H,D]` | `[core,2,H,D]` | `[cid, t%2]` |
| `alpha`(UB) | (复用 m_i_prev) | `[2, ub_len]` | `[t%2]` |

GM 涨一倍:`ws_kv` 256KB/core、`ws_o` 256KB/core 等,总 workspace 翻倍,可接受(GM 充裕)。

**Ascend C 没有任何 per-buffer back cross-flag**(精读确认:只有 4 forward `syncV0C1/C1V1/V1C2/C2V2`
+ 1 个跨 work-item 的边界 flag `3`)。WAR 安全靠 **buffer 深度 ≥ 生产者最大领先量**,而领先量被 forward
互等格卡死。追一遍最容易出问题的 `ws_kv`:

> WAR 冒险:`V0(t+2)` 复用 `ws_kv[t%2]`,需保证 `MM1(t)` 已读完。
> - `V0(t+2)` 在 vector-step t+2 跑;vector 到 step t+2 **必先做完 step t+1**;
> - step t+1 里 `V1(t)` 等 `syncC1V1`(`MM1(t)` 发)→ **vector 做完 step t+1 ⟺ MM1(t) 完成 ⟺ ws_kv[t%2] 已读走**;
> - ∴ `V0(t+2)` 覆盖时 `MM1(t)` 必已读完。**×2 深度 provably 安全,零 back-flag。**

其余 3 个同理(每个都被对侧 forward-wait 卡在 ≤2 拍领先):
- `ws_score[t]`:`MM1(t+2)` 覆盖前,cube 须做完 step t+1 → `MM2(t)` 等 `syncV1C2` → `V1(t)` 已读 ✓
- `ws_p[t]`:`V1(t+2)` 覆盖前,vector 须做完 step t+2 → `V2(t)` 等 `syncC2V2` → `MM2(t)` 已读 ✓
- `ws_o[t]`:`MM2(t+2)` 等 `syncV1C2`(`V1(t+2)`,vec-step t+3 发)→ 排在 `V2(t)`(vec-step t+2)之后 → 已读 ✓

**结论:总 cross-flag = 4 forward + 1 边界 flag,和 Ascend C 一致,比现在的 5 flag 还简单。**
(Ascend C 的 `kvMerge` 用 `%4` 是它 gather 预载更超前——`cmpLoop` 只在 cmp chunk 计数、且 gather 与 MM1
不锁步;我们 `V0(t)`/`MM1(t)` 同拍锁步,×2 即足。若 profile 显示 gather 想跑更远,`ws_kv` 单独加深到 ×4 即可,GM 便宜。)

---

## 5. barrier_all → pipe 级 set_flag/wait_flag (重叠真正生效的关键)

当前 54 个 `barrier_all()` 把**所有 pipe 抽干**,即使错位了也无法重叠。要把相位内部的
`barrier_all` 换成 `T.set_flag(src_pipe, dst_pipe, eid)` / `T.wait_flag(...)`(核内 pipe 间细粒度同步),
只在**真依赖**的 pipe 对之间挡:

- V0 gather:`MTE2`(逐行 DMA)→ `MTE3`(批量写 ws_kv),用 `set_flag("mte2","mte3",e)`。
- V1 softmax:`MTE2`(load ws_score)→ `V`(算)→ `MTE3`(写 ws_p),`V` 内部连续算用 `pipe_barrier("v")`。
- V2 merge:`MTE2`(load ws_o)→ `V`(rescale+add)。
- 跨核仍用 `set_cross_flag`/`wait_cross_flag`(§1/§4 的 4 forward + 1 边界 flag)。

这样稳态下 V0 的 MTE2、V1 的 V、V2 的 MTE2 才能在硬件上并行。
`_pipe` 合法值:`fix / mte1 / mte2 / mte3 / m / v`。

---

## 6. 顺带:向量化 per-head scalar 循环 (§5.4,削 aiv_scalar 6.9ms)

多处 `for h_i in range(v_block)`(32 次/处 + 每次 barrier)是 6.9ms scalar 的主因:
- mask select 673–680、acc_s 减 m_i 615–626、acc_o rescale(将搬到 V2)、最终 div 738–745。

改成 tile 广播一次处理 `[v_block, BI]`/`[v_block, D]`。需 NPU 验证 `T.tile.sub/mul/div` 是否支持
`[v_block,N] op [v_block,1]` 广播(若不支持,用 `Brcb` 式先把标量列广播成 `[v_block,N]` 再逐元素)。
这条和流水正交,可独立成刀。

---

## 7. 分阶段落地 (每阶段独立 NPU 验证:先正确性后 profile)

| 阶段 | 内容 | 风险 | 预期 |
|---|---|---|---|
| **S1 重构** | §2:rescale 从 V1 挪到 V2 + α stash。**不改调度/缓冲**,仍单缓冲串行。 | 低(行为保持,易验对) | 性能持平,隔离 online-softmax 改动 |
| **S2 流水** | §3+§4+§5:×2 缓冲 + 错位调度 + 4 forward+1 边界 flag(无 back-flag)+ barrier→pipe flag。 | **高**(flag 平衡、死锁、prologue/epilogue 守卫) | 吃 gap,乐观 ~30% |
| **S3 向量化** | §6:per-head 循环广播化。 | 中 | 削 scalar,叠加到 ~40% |

S1 先行的价值:把"online-softmax 数学重排"(行为保持)和"调度重排"(易出死锁/竞态)解耦,
S2 出问题时能确定锅在调度而非数学。对齐 BI=128 那刀"闯 4 层坑"的渐进打法。

---

## 8. 与 Ascend C 的唯一剩余差距 + NPU 待验证点

**唯一"完全复刻"还差的一层:跨 work-item 连续流水。** Ascend C 的 `gloop` 横跨所有 work-item
(batch×s1)连续推进,prologue/epilogue 全局只各一次;当前 TileLang 是**每 token 一对 `Scope("C")/("V")`、
各自排空**,所以每 token 都付一次填充/排空气泡(NI=5 时约 2/7 拍)。完全复刻需把流水抬到外层
`for slot` token 循环之上(per-token 的 sink seed / acc_o 清零 / 输出写回也要并进流水,跨 token 边界用边界 flag
携带依赖)。这是更深的重构 → **建议 S2(inner pipeline)验完看 per-token 气泡实测占比,再决定是否做这层**;
inner pipeline 已经拿到 V0∥V1∥V2 + MM1∥MM2 的主要重叠收益。

**NPU 待验证点(不是设计选择,是实现后要盯的):**
1. **cube L0C 串行已确认复刻**:Ascend C 的 MM1/MM2 也是各自 L0C 排空到 GM 再做下一个,本就无需双 L0C。
2. **alpha 双缓冲是否够**:§2 论证 ×2 足够(产/消距离 1 拍),S2 验证时重点盯 **chunk 2(ori→cmp 切换处**,
   历史上 online-softmax 跨 chunk 出过 bug)。
3. **prologue/epilogue 的相位守卫**:前 1–2 拍 V1/V2/MM2 要 skip(`if t>=1/2`),forward flag 计数要全程平衡,
   否则死锁,易错,需仔细对 §3 的计数表。

---

## 9. S2b 具体落地方案（读 AscendC PreloadPipeline / scfa_block_vector 后定稿）

> 前置:S2a forward 双缓冲已验证(tag `s2a-forward-verified`,41.5→36.3ms/18.9%)。profile 显示
> **gap ~15ms 是核内 `barrier_all` 抽干 pipe 造成**:每核串行 pipe-sum ~21ms,pipe-max 只 ~6-7ms。
> S2b 目标:让核内 MTE2∥VEC∥MTE3 真并行 → 每核朝 ~7ms、两核 Duration 朝 ~10ms(~60%+)。

### 9.1 within-core 重叠 = 显式 pipe flag + ping-pong（AscendC 不是自动的,是手搓的）

AscendC 的核内重叠靠 `SetFlag/WaitFlag<HardEvent::V_MTE2>(eid)` + ping-pong buffer,不是 TQue 魔法:
- buffer 开 2 份:scores `inputBuff1`=32K×2、kv gather `inputBuff2`=16K×2(`scfa_block_vector.h:207-208`)。
- 模式(`:461-484`):`ub = buf[ping*OFFSET]; WaitFlag<V_MTE2>(FLAG+ping); <MTE2 load> <VEC compute>; SetFlag<V_MTE2>(FLAG+ping); ping^=1`。→ MTE2 把 half B 载入的同时 VEC 在算 half A。
- **TileLang 对应**:`T.set_flag("v","mte2",eid)`(V 产完)/ `T.wait_flag("v","mte2",eid)`(MTE2 重载前等),buffer `[2,…]` 用 `t%2` 选 half。`_pipe∈{fix,mte1,mte2,mte3,m,v}`。跨核仍 `set_cross_flag/wait_cross_flag`(保持 §1 的 4 forward+1 边界)。
- ⚠️ `wait_flag(src,dst,e)` 跑在 **dst** pipe 上(见 tilelang-pitfalls)。

### 9.2 UB 192KB 墙 → 16-row/16-head sub-tile（复刻 AscendC，附数字）

去 barrier 后同时活:`acc_o`(64K) + `acc_s_ub/ub_/half`(40K) + gather + `acc_o_ub`。全 64K 粒度 = 234K 爆墙。
- **gather**:`kv_ub_multi [64,512]bf16=64K` → `[16,512]bf16=16K ×2 ping-pong`(32K)。每趟 gather 16 行 + 写 ws_kv,4 趟。
- **V2 merge**:`acc_o_ub [32,512]f32=64K` → `[16,512]f32=32K`。每趟 merge 16 头进 `acc_o`(`acc_o` 整块 64K 不动),2 趟。
- 新峰值 ≈ 64+40+32+32+2 = **170K < 192K**(22K 余量)。`acc_o` 是全 32 头输出累加器,不能拆。

### 9.3 增量步骤（每步独立 NPU 验证:先正确性后 profile;回退点 tag `s2a-forward-verified`）

- **S2b.0 — UB sub-tile,barrier 全保留**:gather 改 4×16 行(`cff7e59`)、merge 改 2×16 头 + un-alias(`0fe56b7`)。**✅ decode+prefill 已验证**,布局对、行为不变。把"布局正确性"和"去 barrier 竞态"解耦(同 S1/S2a 渐进打法)。
- **S2b.1 — vector 去核内 barrier**(下一刀,最难):V0/V1/V2 之间 + pipe-stage 之间换 `set_flag/wait_flag` + ping-pong,V 内连续算用 `pipe_barrier("v")`。验对 + 看 `aiv_time` 是否朝 pipe-max(~7ms)掉。
- **S2b.2 — cube 去核内 barrier**:MM1/MM2 同理。
- 风险:竞态/死锁(flag 配对、ping-pong 深度);按经验大概率要逐步闯。每步只改一小撮、必过 NPU 再推下一步。

### 9.4 S2b.1 怎么拆（设计要点,实施前定稿）

**核心原则**:去掉 vector scope 的 `barrier_all`(全 pipe drain)。**同一 pipe 上的操作硬件自动 in-order,不需任何同步**;只在**跨 pipe 的真依赖**之间补 `set_flag(src,dst,eid)`/`wait_flag(src,dst,eid)`。没数据依赖的跨 pipe 操作(如 V0(t) 的 gather-MTE2 与 V1(t-1) 的 softmax-VEC,不同 chunk/不同 buffer)就让它们并行——这正是 gap 的来源。

**去 barrier 后要补的跨 pipe flag**(每相位内):
- V0 gather:每趟 16 行 gather(MTE2) → 写 ws_kv(MTE3):`set_flag("mte2","mte3")` / wait。
- V1 softmax:load ws_score(MTE2) → 算(VEC):`set_flag("mte2","v")`;算完 → 写 ws_p(MTE3):`set_flag("v","mte3")`。VEC 串内部用 `pipe_barrier("v")`。
- V2 merge:load ws_o(MTE2) → rescale+add(VEC):`set_flag("mte2","v")`。
- 跨核 4 forward cross-flag 不变。⚠️ `wait_flag(src,dst,e)` 跑在 **dst** pipe 上。

**还需的 ping-pong**(让下一 chunk 的 load overlap 当前 chunk 的 compute,复刻 AscendC `inputBuff1` 32K×2):
- gather `kv_ub_multi` 已是 `[2*16]` ping-pong ✓。
- score load `acc_s_ub_`、ws_o load `acc_o_ub` 要开 ×2(`acc_s_ub_` 16K→32K、`acc_o_ub` 32K→64K),否则 V1(t)/V2(t) 的 MTE2 load 会和 V1(t-1)/V2(t-2) 的 VEC 抢 buffer。⚠️ 这会顶 UB 墙(acc_o_ub ×2 = 64K),需要重新核 170K 预算——可能 score 或 ws_o load 只 ×2 一个、或 acc_o_ub 维持单缓冲(merge 的 load∥compute 不重叠,只保跨相位重叠)。**实施前先把 ping-pong 深度和 UB 预算重算清楚**。

**增量子步**(每步 NPU 验证):
- **S2b.1a**:只让 **V0(gather-MTE2) ∥ V1(softmax-VEC)** 重叠——去掉 V0↔V1 之间 + 各自内部 barrier、补上面的 flag,V2 仍 `barrier_all` 隔离。验对 + 看 `aiv_time` 是否开始掉(gather 7.1ms 藏到 softmax 后)。这是最小的"有收益"步。
- **S2b.1b**:把 V2 纳入重叠(去 V1↔V2 barrier + V2 内 flag)。
- **S2b.1c**:相位间 + 跨 chunk 全打通 + ping-pong 调深。
- 回退点:tag `s2a-forward-verified` 或 S2b.0 的 `0fe56b7`。
