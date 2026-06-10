# sparse_attn_sharedkv TileLang 性能优化 — 工作交接 #3（S2b 软件流水去 barrier 中）

> 续 `PERF_HANDOFF_2.md`。读完即可接着干。配合 `PIPELINE_DESIGN.md §9`(S2b 完整设计) + MEMORY.md + tilelang plugin 的 skill 一起看。
> 当前在做 **§5 三级软件流水的 S2b**(去核内 barrier、换 pipe 级 flag,让 vector 的 MTE2∥VEC∥MTE3 真并行)。地基已全部验通,卡在最复杂的一刀:拆 V1 内部 barrier。

---

## 0. 一句话现状

- **基线**:forward 双缓冲软件流水,decode+prefill 数值都对,**8K scfa_prefill 稳态 36.3ms / 18.9%**(`性能% = AscendC 6.87ms ÷ TileLang Duration × 100%`)。HEAD `bd56634`,main 干净。
- **profile 结论**:Duration 36.3ms,cube `aicore_time`≈21.2ms,vector `aiv_time`≈21.2ms,**跨核 gap≈15ms**。gap 的根因是**核内 `barrier_all` 把 pipe 抽干** → 每核串行 pipe-sum≈21ms,而 pipe-max 只≈7ms(vector mte2 7.1 / scalar 6.7 / vec 5.3 / mte3 2.8)。
- **S2b 目标**:去核内 barrier、补 pipe 级 `set_flag/wait_flag`,让每核 MTE2∥VEC∥MTE3 并行 → 每核朝 ~7ms、Duration 朝 ~10ms(~60%+)。这是数量级的大头。
- **进度**:S2b.0(UB sub-tile + un-alias)✅、S2b.1a(pipe-flag 原语 smoke)✅、S2b.1b(V0 gather 内部 ping-pong)✅ —— **三层机制全验通**。下一刀 **S2b.1c**(拆 V1 内部 barrier)是最复杂、最 race-prone 的一刀,设计已定稿,待实施。

---

## 1. ⭐ 立刻要做的（接手第一件事）：S2b.1c — 拆 V1 内部 barrier

**完整设计在 `PIPELINE_DESIGN.md §9.4`（连同 §9.1/§9.2 的机制和 UB 预算）。** 下面是可直接动手的实施清单。

**目标**:去掉 V0-end barrier + V1 内部全部 `barrier_all`,让 V0 的 gather-MTE2 和 V1 的 VEC 在硬件上并行。V2 仍 `barrier_all`(它当 step 边界、drain 上一 chunk 的 VEC,这样**免掉跨 chunk WAR 的 ping-pong**)。

**关键事实(本 session 已坐实,放心用)**:
- `T.copy(GM→UB)` = **MTE2**,`T.copy(UB→GM)` = **MTE3**,`T.copy(UB→UB)` = **VEC**(源码:`tilelang-ascend/src/transform/common/operation_config.h` 标 `copy_ub_to_ub` → `"PIPE_V"`)。
- 所有 `T.tile.*`(compare/add/mul/sub/exp/max/select/reduce_*/fill/cast)= **VEC**。
- ⇒ **同一 pipe 上的 op 硬件自动 in-order,不用任何同步**;只在**跨 pipe 真依赖**补 flag。
- `T.set_flag(src,dst,eid)` / `T.wait_flag(src,dst,eid)`:`src` 产、`dst` 等,`wait_flag` 跑在 **dst** pipe 上。`_pipe∈{fix,mte1,mte2,mte3,m,v}`。不同 `(src,dst)` 是不同 event,eid 空间独立。

**V1(t-1) 当前的 op 序(都被 `barrier_all` 隔开,kernel.py 里 `if t>=1: if t<=NI_total:` 那段)**:
```
fill acc_s_ub_=0 (VEC)
copy mask_ub[pv1,:]→mask_sel (VEC, UB→UB)
for h_i: select → acc_s_ub[h_i]  (VEC, 读 acc_s_ub_=0 和 mask_sel)
copy m_i→m_i_prev (VEC)
wait_cross_flag(SCORE_READY)                          # 跨核
copy ws_score[cid,pv1,...]→acc_s_ub_ (MTE2 load)      # 覆写 acc_s_ub_
add  acc_s_ub += acc_s_ub_ (VEC)
mul scale / reduce_max→m_i / tile.max(m_i,m_i_prev,m_i) / sub m_i_prev / exp m_i_prev (VEC)
copy m_i_prev→alpha[pv1*ub_len:+ub_len] (VEC)
for h_i: sub acc_s_ub[h_i]-=m_i[h_i] (VEC)
exp acc_s_ub / reduce_sum→sumexp_i / mul sumexp / add sumexp (VEC)
copy acc_s_ub→acc_s_half (VEC, fp32→bf16 cast)
copy acc_s_half→ws_p[cid,pv1,...] (MTE3 write)
set_cross_flag("MTE3", P_READY)
```

**改法**:把 V1 里的 `barrier_all` 全删,**只补这 3 处跨 pipe flag**(其余 VEC↔VEC 自动 in-order):
1. **WAR**:`select`(VEC 读 `acc_s_ub_`)→ `load ws_score`(MTE2 覆写 `acc_s_ub_`)。
   `T.set_flag("v","mte2",0)` 放在 **select loop 之后**;`T.wait_flag("v","mte2",0)` 放在 **load 之前**。
   (否则 load 抢在 select 前覆写 `acc_s_ub_`,select 读到 score 而不是 0 → 错。这是最易漏的一处。)
2. **RAW**:`load`(MTE2)→ `add`(VEC 读 `acc_s_ub_`)。
   `T.set_flag("mte2","v",0)` 放在 **load 之后**;`T.wait_flag("mte2","v",0)` 放在 **add 之前**。
3. **RAW**:`cast→acc_s_half`(VEC)→ `写 ws_p`(MTE3)。
   `T.set_flag("v","mte3",0)` 放在 **cast 之后**;`T.wait_flag("v","mte3",0)` 放在 **write 之前**。
- `wait_cross_flag(SCORE_READY)` 保留(在 load 前;它自身 serialize 指令流)。
- **V0-end barrier 删掉**:在 `set_cross_flag("MTE3", _FLAG_KV_READY)` 之前那个 `T.barrier_all()`(S2b.1b 的 drain 之后那个)删掉,让 gather 的 MTE2 不被 drain、能流进 V1 的 VEC 期间。⚠️ 但 V0 的 back-flag drain(`wait_flag("mte3","mte2",0/1)`)要保留(平衡)。
- **V1-end barrier 保留**:`set_cross_flag(MTE3,P_READY)` 之后到 V2 之间留一个 `barrier_all`,隔离 V2(V2 仍全 barrier)。

**eventId 不撞确认**:V0 gather 用 `mte2↔mte3`(eid 0,1)、`mte3↔mte2`(eid 0,1);V1 用 `v↔mte2`(eid 0)、`mte2↔v`(eid 0)、`v↔mte3`(eid 0)。全是不同 `(src,dst)` event,即便同 step 无 barrier 也不撞。

**预期**:正确性必须保住;**perf 收益有限**(gather 和 score-load 同在 MTE2 串行 → gather 只 ∥ V1 的 fill+mask+select 几个 VEC op,softmax 大头还没 ∥ MTE2)。**这步主要是验 V1 debarrier 的 flag 设计对不对、不死锁**,真大头在下一步 S2b.1d。

**验证**(NPU `/sdb/yq/dsv4`,本地无 NPU):
```bash
git pull
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode"
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "prefill" --runslow
```
- 都过 → V1 debarrier flag 设计验通,进 **S2b.1d**(下面)。
- 挂/死锁/数值错 → 锅在那 3 个 flag 的摆位/方向/WAR(V1 单独改的,V0/V2 没动),按现场对症。

---

## 2. 紧接着：S2b.1d — 跨 chunk score 预取 ping-pong（真大头）

**这才把 softmax-VEC ∥ MTE2、vector 21ms→~7ms。** 复刻 AscendC `inputBuff1` 32K×2 ping-pong(`ops-transformer/.../sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h:461-484`)。

- `acc_s_ub_` 开 ×2([2,v_block,BI] 或扁平),MTE2 预取 chunk c+1 的 ws_score 进 half (c+1)%2,**while** VEC 算 chunk c 的 softmax(half c%2)。配 `V_MTE2` 风格 flag(`set_flag("v","mte2",half)` / `wait_flag` 按 ping-pong index)。
- ⚠️ **UB 预算**(§9.2,当前峰值 176K / 墙 192K):`acc_s_ub_` ×2 = +16K → 186K(还行,6K 余量)。`acc_o_ub` ×2 = +32K **会爆** → V2 的 ws_o 预取这步先不做,或挤别的。**实施前重算 §9.2 的 170K 预算表**。
- 这步要重排 V1 的循环结构(把 score-load 提前一拍),比 S2b.1c 更动调度,**务必小步 + 每步 NPU 验**。

之后还有 **S2b.2**(cube 侧 MM1/MM2 去 barrier,同理)。

---

## 3. commit 地图（main 上,可回退）

- `91d42f7` **alpha 扁平 1D 修复**。✅ decode+prefill 验证。(根因:V2 逐头 rescale 的 2D 标量 `alpha[pv2,h_i]` 被 tile `binary_op` 只取 `indices[0]`、丢了 `h_i`。)
- `tag s2a-forward-verified`(= `04aa6be`) **S2a forward 双缓冲已验证 + profiled 36.3ms/18.9%**。**最稳回退点**。
- `cff7e59` **S2b.0a** gather sub-tile(4×16 行 ping-pong shape)。✅
- `0fe56b7` **S2b.0b** merge sub-tile(2×16 头)+ un-alias 到 170K 布局。✅ **S2b.0 整刀完成**。
- `ec9b282` **S2b.1a** pipe-flag smoke(gather MTE2→MTE3 换 flag)。✅
- `1d1bcfc` **S2b.1b** V0 gather 内部 ping-pong(forward+back flag + 平衡 drain)+ 记录 `copy_ub_to_ub=VEC`。✅
- `bd56634` **HEAD**,docs(S2b.1c/1d 设计)。

---

## 4. 当前 kernel 结构（HEAD bd56634, kernel.py 要点）

- 闭包常量:`GATHER_ROWS=16`/`N_GATHER_PASS=4`(gather 4×16 行)、`MERGE_HEADS=16`/`N_MERGE_PASS=2`(merge 2×16 头)、`ub_len`、`mask_w=32`。
- UB 布局(§9.2,un-aliased,峰值 176K):`acc_o`@0(64K)、`acc_s_ub`@64K、`acc_s_ub_`@80K、`acc_s_half`@96K、scalars/masks/alpha@104K、`kv_ub_multi`@112K(32K,gather ping-pong [2*16,D])、`acc_o_ub`@144K(32K,merge [16,D])、`acc_o_half`@112K(别名 kv_ub_multi,epilogue-only)。
- vector loop `for t in range(NI_total+2)`:V0(t) gather / V1(t-1) softmax / V2(t-2) merge,错位流水。cube loop `for t in range(NI_total+1)`:MM1(t) / MM2(t-1)。**注意:`for...in range()` 在 prim_func 里是 TIR loop,loop var 是 Var(不是 Python int)。**
- V0 gather(ori+cmp)已 ping-pong:per pass `if gp>=2: wait_flag("mte3","mte2",pp)` → 16 行 gather(MTE2) → `set_flag/wait_flag("mte2","mte3",pp)` → write(MTE3) → `set_flag("mte3","mte2",pp)`;loop 后 drain `wait_flag("mte3","mte2",0/1)` + `barrier_all`(V0-end,S2b.1c 要删) + `set_cross_flag(MTE3,KV_READY)`。
- V1 / V2 仍全 `barrier_all`(待 S2b.1c / 之后拆)。

---

## 5. 关键文件 / 命令

- `sparse_attn_sharedkv_tilelang/kernel.py` — 核心 kernel(改这)。
- `sparse_attn_sharedkv_tilelang/PIPELINE_DESIGN.md` — **§9 是 S2b 完整设计**(§9.1 pipe-flag+ping-pong 机制、§9.2 UB 预算数字、§9.3 增量步骤、§9.4 S2b.1c/1d 详细 flag 清单)。
- `sparse_attn_sharedkv_tilelang/PERF_HANDOFF_2.md` / 本文件 — 交接历史。
- Ascend C 标杆:`ops-transformer/.../sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h`(`PreloadPipeline` 785-827)、`..._scfa_block_vector.h`(`:461-484` 是 inputBuff1 32K×2 ping-pong + `WaitFlag<V_MTE2>` 模板,S2b.1d 照抄)。
- TileLang 原语:`tilelang-ascend/tilelang/language/ascend.py`(set_flag/wait_flag/pipe_barrier/set_cross_flag/gemm_v0)、`ascend_tile.py`(tile.* + binary_op)。
- 测试:`pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode"`(快)/ `-k "prefill" --runslow`(慢)。kernel 名 msprof 里是 `sparse_attn_sharedkv_kernel`。
- profile:`msprof --output=./prof_fwd --aic-metrics=PipeUtilization --application="python sparse_attn_sharedkv_perf_compare.py --scenarios scfa_prefill --only tilelang --warmup 5 --iters 3"`。

---

## 6. 工作流程（务必遵守,同 PERF_HANDOFF_2.md §7）

- **本地无 NPU**:编译/正确性/profile 全靠用户在 NPU(`/sdb/yq/dsv4`)手动跑、贴结果。**不能假装跑过**。
- **本地能做**:`export PATH=$HOME/.local/bin:$PATH` 后 `ruff format` + `ruff check` + `python -m py_compile`(只查语法/格式,查不出 TVMScript 语义/竞态)。
- **改完主动 commit+push 到 dsv4**(用户偏好,不等他说)。push 前 ruff format+check。commit message 英文、结尾 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。push 后 `git pull` 同步本地 main。
- 正确性别用 `assert_close`(bf16 噪声误报),用仓库 `_check_result` 口径(test 里已用)。先 decode(快)后 prefill --runslow(慢)。
- 回复正文用中文,代码/注释/commit 用英文。
- **S2b 是 race-prone**:每步只改一小撮、必过 NPU 再推下一步(同 S1/S2a/S2b.0 渐进打法)。漏 flag = 竞态/死锁,比之前的 bug 难调。
- 每刀验证有效后,通用手段记进 tilelang skill(perf→tilelang-perf,坑→tilelang-pitfalls;源仓库 `~/.claude-plugins/tilelang/skills/` + 运行时缓存 `~/.claude/plugins/cache/yzmac-personal/tilelang/0.1.0/skills/` **两处都改 + diff**,源仓库 git commit 风格 `tilelang: ...`)。

---

## 7. 本 session 已沉淀进 tilelang-pitfalls skill 的坑（背景,别重复踩）

1. **`for...in range()` 在 `@T.prim_func` 里是 TIR loop**,loop var 是 `Var`、不是 Python int ⇒ 不能拿它索引 Python 容器(tuple/list),要索引 TIR Buffer。
2. **parity UB buffer 行 stride 必须 ≥32B**:`[2, BI//8]` uint8 奇数行落 +16B 未对齐 → AIV `UB address ... not aligned` device fault。pad 到 `[2,32]` 整行裸冒号。
3. **tile `binary_op` 标量第三参(BufferLoad)只转发 `indices[0]`**:2D 标量 `buf[a,b]` 会丢 `b`、读成 `buf.flat[a]`。per-element 标量来源用**扁平 1D** buffer + 单个计算索引(alpha 就是这么修的)。
4. **`[2,…]` parity buffer 喂不同 op 的 operand 类型规则**(gemm_v0 收裸冒号 BufferRegion、select 的 selMask 要整 Buffer、compare/and 收 BufferRegion)。
5. (本文件 §1 的事实)`copy` 的 pipe:gm→ub=MTE2 / ub→gm=MTE3 / ub→ub=VEC;tile.*=VEC;同 pipe 自动 in-order。

> 这些 skill 是给"写/review TileLang Ascend kernel"用的,新 session 写 flag 时 skill 会自动触发,照着来。
