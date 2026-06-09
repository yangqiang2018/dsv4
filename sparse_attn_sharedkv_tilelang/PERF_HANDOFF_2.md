# sparse_attn_sharedkv TileLang 性能优化 — 工作交接 #2（S2 软件流水进行中）

> 续 `PERF_HANDOFF.md`。读完即可接着干。配合 MEMORY.md + tilelang plugin 的 skill 一起看。
> 当前在做 **§5 的三级软件流水（Ascend C PreloadPipeline）**,正卡在 TVMScript 双缓冲的 parse 闯关。

---

## 0. 一句话现状

- **S1 已验证**（commit `cc6ee1c`）:把 online-softmax 的输出 rescale 从 V1 挪进 V2。decode+prefill 都过,数值对。
- **S2a 软件流水**:把每 chunk 串行重排成错位流水(cube `MM1(t)∥MM2(t-1)`,vector `V0(t)∥V1(t-1)∥V2(t-2)`)。
  - **reverse 版已验证 + profile**（commit `1c52835`）:单缓冲、相位倒序,**Duration 48.8ms→41.5ms(14.1%→16.6%)**,但 gap 还剩 14.4ms。这是**已知能跑的退路**。
  - **forward 版进行中**（HEAD `800808c`）:正序(= Ascend C 的 PreloadPipeline,重叠更紧)+ parity 双缓冲。**正在逐个 op 闯 TVMScript parse 关**:gemm✓、select✓ 已过,待 NPU 验 decode。

**性能口径**:`性能% = AscendC(6.87ms) ÷ TileLang Duration × 100%`。

---

## 1. ⭐ 立刻要做的（接手第一件事）

在 NPU 服务器(`/sdb/yq/dsv4`)上:

```bash
git pull   # 确认到 800808c 或更新
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode"
```

**三种结果分支:**

**(A) decode 过了** → 跑 prefill 正确性 + profile,对比 reverse 版的 41.5ms / gap 14.4ms:
```bash
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "prefill" --runslow
msprof --output=./prof_fwd --aic-metrics=PipeUtilization \
  --application="python sparse_attn_sharedkv_perf_compare.py --scenarios scfa_prefill --only tilelang --warmup 5 --iters 3"
```
forward 顺序应把 gap 往 0 压、Duration 朝两核的 ~27ms 靠(~25%)。过了就进 **§4 的 S2b**(更大头)。

**(B) decode 还报 parse 错(大概率在 `alpha`:V1 的 `T.copy(m_i_prev, alpha[pv1, :])` 或 V2 的 `tile.mul(..., alpha[pv2, h_i])`)** → 按 §3 的"operand 类型规则"对症修那一个 op,再推。剩的 parity 访问只有这俩,改完应该就过。

**(C) 如果 op-by-op 还在反复挂** → 直接切 §5 的**兜底方案(分开整 buffer + 运行时 if)**,一劳永逸,别再逐个试。

---

## 2. 当前 kernel 结构（HEAD 800808c,kernel.py）

正序流水,保留 barrier(S2a 只吃跨核 gap;核内 pipe 重叠是 S2b):

- **cube** `with T.Scope("C")`,`for t in range(NI_total+1)`:`if t<NI: MM1(t)` 然后 `if t>=1: MM2(t-1)`。
- **vector** `with T.Scope("V")`,`for t in range(NI_total+2)`:`if t<NI: V0(t)` / `if 1<=t<=NI: V1(t-1)` / `if t>=2: V2(t-2)`。
- **4 个 forward cross-flag**:`_FLAG_KV_READY/SCORE_READY/P_READY/PV_READY`,每个每 token set/wait 各 `NI_total` 次(平衡)。`_FLAG_ITER_DONE` 已弃用(forward 互等格 + 深度2 保证 WAR 安全,无需 back-flag)。
- **双缓冲(parity = chunk%2)**:
  - GM:`ws_kv/ws_score/ws_p/ws_o` 形状 `[core,2,...]`,索引 `ws_kv[cid, pv, ...]`(4 维全索引,OK)。
  - L1:`kv_lo/kv_hi` 形状 `[2, BI_half, D]`,gemm 操作数 `kv_lo[pa, :, :]`(**裸冒号**,见 §3)。
  - UB:`mask_ub`/`mask_ub_2` `[2, BI//8]`、`alpha` `[2, ub_len]`,加了 `mask_sel`(`[BI//8]` 整 buffer 给 select 用)。
- **online-softmax 跨 chunk 状态**:`m_i`(running max)、`sumexp`(running sum)单缓冲、V1 串行;`acc_o`(输出累加器)单缓冲、V2 独占(S1 把 rescale 挪进 V2 后 V1 不碰它);只有传给 V2 的 rescale 因子 `alpha` 需 ×2。

---

## 3. ⭐⭐ 硬核知识:TVMScript 双缓冲 parse 的 operand 类型规则（这次踩透的坑）

这是 S2 卡最久的地方。**`[2,…]` parity buffer 怎么取一份切片喂给不同 op,每个 op 要求不同**:

| 取法 | 产出类型 |
|---|---|
| `buf[i]`(少索引) | 报错 "N-dim buffer indexed with 1-dim" |
| `buf[i, 0:M, 0:K]`(标量 + 显式范围) | **BufferLoad** |
| `buf[i:i+1, ...]`(宽度 1 切片) | **塌缩成标量 → BufferLoad** |
| `buf[i, :, :]`(标量 + **裸冒号**) | **BufferRegion** ✅ |
| `buf`(不带下标,整个) | **Buffer**(有 `.access_ptr`) |

各 op 收什么(看 `tilelang/language/ascend.py` 和 `ascend_tile.py` 源码):
- **`gemm_v0`**:要 `Buffer` 或 `BufferRegion`(`_retrieve_shape`)。→ 用裸冒号 `kv_lo[pa, :, :]`。**拒 BufferLoad**。
- **`tile.select` 的 selMask**:直接调 `.access_ptr`(`ascend_tile.py:583`),只有**整个 Buffer**有。dst/src0 走 `retrieve_ptr` 能收 BufferRegion,**唯独 selMask 不行**。→ 拷进整 buffer `mask_sel` 再传。
- **`tile.compare` / `tile.bitwise_and`**:收 BufferRegion(parity 切片 `mask_ub[pv0, :]` 直接用,已验证过 parse)。
- **`tile.mul` 标量第三参 / `T.copy` 两端**:收 BufferLoad / BufferRegion(`m_i_prev[h_i]`、`acc_s_ub[h_i,:]` 都是先例)。

> 参考:能跑的例子 `tilelang-ascend/examples/chunk_gated_delta_rule/expert_chunk_gated_delta_rule.py:132`
> `T.gemm_v0(w_chunk_l1[pid, :, :], ...)` —— 裸冒号喂 gemm 的权威写法。
>
> **验证通过后,把这张表记进 tilelang-pitfalls skill**(源仓库 `/Users/yzmac/.claude-plugins/tilelang/skills/` + 运行时缓存 `~/.claude/plugins/cache/yzmac-personal/tilelang/0.1.0/skills/`,两处都改 + diff,源仓库 git commit 风格 `tilelang: ...`)。

---

## 4. 下一刀:S2b（profile 显示这才是最大头）

reverse 版 profile（41.5ms,8K scfa_prefill,稳态）:
- Duration 41.5ms;cube `aicore_time` 27.06ms;vector `aiv_time` 26.98ms;**gap 14.4ms**。
- vector pipe 分解:`mte2(gather) 9.0 + scalar 6.6 + vec(softmax) 5.3 + mte3 4.1` → **串行 sum≈27ms,但 max 只有 ~9ms**。
- cube pipe 分解:`fixpipe 6.4 + mte2 5.8 + mte1 2.7 + mac 2.36` → pipe-max ~6.4ms。

**两个杠杆:**
1. **forward 顺序(S2a 正在做)** → 吃 14.4ms gap → Duration 朝 27ms。
2. **S2b:去掉核内 barrier_all,换 pipe 级 `T.set_flag/T.wait_flag`** → 让 vector 的 MTE2(gather i)∥VEC(softmax i-1)∥MTE3(merge i-2) 真并行 → **两核 27ms→~9ms**。这是数量级的大头。
   - `set_flag(src_pipe, dst_pipe, eid)` / `wait_flag(...)`,pipe ∈ {fix,mte1,mte2,mte3,m,v}(见 `ascend.py`)。
   - **前置依赖**:S2b 去 barrier 后 V0/V1/V2 同时活,**必须双缓冲**(就是 S2a 正在搞的)→ 所以 forward 双缓冲非过不可,reverse 单缓冲版做不了 S2b。
   - **UB 192KB 墙**:去 barrier 后 `acc_o(64KB)∥acc_o_ub(64KB)∥kv_ub_multi(64KB)` 同时活 = 192KB,挤掉 acc_s。需**把 gather 和 V2 merge sub-tile 成半块**(= 复刻 Ascend C 的 32K/16K ping-pong UB),详见 PIPELINE_DESIGN.md §8。

3. **S3(可独立)**:向量化 `for h_i in range(v_block)` 那几处(mask select、acc_s 减 m_i、acc_o rescale/div)→ 削 `aiv_scalar` 6.6ms。需验证 `T.tile.sub/mul/div` 支不支持 `[v_block,N] op [v_block,1]` 广播。

---

## 5. 兜底方案（若 §3 的 op-by-op 继续磨,直接上这个）

**所有双缓冲 buffer 改成分开的整 buffer + 运行时 `if (t%2)==0` 选**。因为**整个 Buffer 所有 op 都收**(gemm/select/compare/copy 全都收,原 kernel 就这么用),彻底绕开 BufferRegion/BufferLoad/access_ptr 的类型分歧。

- `kv_lo0/kv_lo1/kv_hi0/kv_hi1`(各 `[BI_half,D]` 整 buffer)、`mask0/mask1`、`alpha0/alpha1`。
- cube MM1:`if (t%2)==0: 载入+QK 用 kv_lo0/kv_hi0、ws_*[cid,0] ; else: ...1`。MM2 同理用 `(t-1)%2`。
- vector 各相位同理。`if (t%2)==0:` 是运行时 TIR if(两支都编译,运行时只走一支,像 `if is_ori:`),flag 的 wait/set 放在 if 外保证计数平衡。
- 代价:cube/vector 相位体翻倍(代码量),但运行时只执行一支,性能不变。**确定能过**。

---

## 6. commit 地图（main 上,可回退）

- `cc6ee1c` **S1**(rescale V1→V2)。✅ 已验证(decode+prefill)。**最稳的回退点**。
- `1c52835` **S2a reverse**(单缓冲倒序)。✅ 已验证 + profile **41.5ms/16.6%**。**能跑的次稳回退点**(但做不了 S2b)。
- `800808c` **S2a forward**(HEAD,双缓冲正序)。⏳ parse 闯到 select 已过,待 NPU 验 decode。
- 中间那串 `e18711d/643cdfa/079bacd/9070daa/ba0af97/369c026` 都是 forward 双缓冲的失败/中间态(每个修一层 parse 坑),不用回去看,知识已汇总到 §3。

---

## 7. 工作流程（务必遵守,同 PERF_HANDOFF.md §8）

- **本地无 NPU**:编译/正确性/profile 全靠用户在 NPU 手动跑,贴结果。不能假装跑过。
- **本地能做**:`export PATH=$HOME/.local/bin:$PATH` 后 `ruff format` + `ruff check` + `python -m py_compile`(只能查语法/格式,查不出 TVMScript 语义)。
- **改完主动 commit+push 到 dsv4**(用户偏好,不等他说)。push 前 ruff format+check。commit message 英文,结尾 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。push 后 `git pull` 同步本地 main。
- **正确性别用 assert_close**(bf16 噪声误报),用仓库 `_check_result` 口径。先 decode(快)后 prefill --runslow(慢)。
- 回复正文用中文,代码/注释/commit 用英文。
- 每刀验证有效后,通用手段记进 tilelang skill(perf→tilelang-perf,坑→tilelang-pitfalls;两处都改 + diff)。

---

## 8. 关键文件 / 命令

- `sparse_attn_sharedkv_tilelang/kernel.py` — 核心 kernel(改这)。
- `sparse_attn_sharedkv_tilelang/PIPELINE_DESIGN.md` — 流水相位设计(相位/flag/缓冲/UB墙,§8 有 S2b 的 sub-tile 计划)。
- Ascend C 标杆:`ops-transformer/.../sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h`(`PreloadPipeline` line 784-827,正序 MM1∥MM2 / V0∥V1∥V2);`..._block_vector.h`(`DealBmm2ResBaseBlock` = V2 输出递推在 V2 做)。
- TVMScript 原语:`tilelang-ascend/tilelang/language/ascend.py`(set_flag/wait_flag/cross_flag/gemm_v0)、`ascend_tile.py`(tile.select 等)。
- 测试命令见 §1。kernel 名 msprof 里是 `sparse_attn_sharedkv_kernel`。
