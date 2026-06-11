# sparse_attn_sharedkv TileLang 性能优化 — 工作交接 #3（S2b 软件流水去 barrier 中）

> 续 `PERF_HANDOFF_2.md`。读完即可接着干。配合 `PIPELINE_DESIGN.md §9`(S2b 完整设计) + MEMORY.md + tilelang plugin 的 skill 一起看。
> 当前在做 **§5 三级软件流水的 S2b**(去核内 barrier、换 pipe 级 flag,让 vector 的 MTE2∥VEC∥MTE3 真并行)。地基已全部验通,卡在最复杂的一刀:拆 V1 内部 barrier。

---

## 0. 一句话现状

- **基线**:forward 双缓冲软件流水,decode+prefill 数值都对,**8K scfa_prefill 稳态 36.3ms / 18.9%**(`性能% = AscendC 6.87ms ÷ TileLang Duration × 100%`)。HEAD `0487312`,main 干净。
- **profile 结论**:Duration 36.3ms,cube `aicore_time`≈21.2ms,vector `aiv_time`≈21.2ms,**跨核 gap≈15ms**。gap 的根因是**核内 `barrier_all` 把 pipe 抽干** → 每核串行 pipe-sum≈21ms,而 pipe-max 只≈7ms(vector mte2 7.1 / scalar 6.7 / vec 5.3 / mte3 2.8)。
- **S2b 目标**:去核内 barrier、补 pipe 级 `set_flag/wait_flag`,让每核 MTE2∥VEC∥MTE3 并行 → 每核朝 ~7ms、Duration 朝 ~10ms(~60%+)。这是数量级的大头。
- **进度**:S2b.0(UB sub-tile + un-alias)✅、S2b.1a(pipe-flag 原语 smoke)✅、S2b.1b(V0 gather 内部 ping-pong)✅、**S2b.1c(拆 V1 内部 barrier)✅ decode+prefill 全 pass(NPU 2026-06-10,commit `0487312`)** —— **核内 debarrier 机制全验通**。下一刀 **S2b.1d**(跨 chunk score 预取 ping-pong)是 vector 21→~7ms 的真大头,设计见 §2/§9.4,**实施前先重算 §9.2 UB 预算**。

---

## 1. ✅ 已完成并验证：S2b.1c — 拆 V1 内部 barrier（commit `0487312`）

> **状态:decode + prefill 全 pass(NPU 2026-06-10)。核内 debarrier 的 flag 设计验通、不死锁。下一刀 S2b.1d 见 §2。** 下面这份清单保留作 flag 设计的事后参照(万一 1d 重排循环时要回看 V1 的 hazard 边)。

**完整设计在 `PIPELINE_DESIGN.md §9.4`（连同 §9.1/§9.2 的机制和 UB 预算）。** 下面是已落地的实施清单。

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

## 2. ⭐ 立刻要做（接手第一件事）：S2b.1d — 跨 chunk score 预取 ping-pong（真大头）

**这才把 softmax-VEC ∥ MTE2、vector 21ms→~7ms。** 复刻 AscendC `inputBuff1` 32K×2 ping-pong(`ops-transformer/.../sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h:461-484`)。

**前置(1c 已铺好的地基)**:V1 已 debarrier、3 对 pipe flag(`v↔mte2` WAR / `mte2↔v` RAW / `v↔mte3` RAW)就位、V1-end barrier 还在当 step 边界。1d 要做的是把 `acc_s_ub_` 开 ×2、把 score-load 提前一拍(预取 c+1)让它 ∥ 当前 chunk 的 softmax-VEC,flag 的 eid 改成跟 ping-pong index 走(`set_flag("v","mte2",half)` / `wait_flag` 按 half)。⚠️ 提前 score-load 后,V1-end barrier 可能要从"全 drain"退化成只 drain VEC/MTE3、放 MTE2 预取穿过 —— 这步会动 step 边界语义,**最易引入跨 chunk 竞态,务必小步 + 每步 NPU 验**。

- `acc_s_ub_` 开 ×2([2,v_block,BI]),MTE2 预取 chunk c+1 的 ws_score 进 half (c+1)%2,**while** VEC 算 chunk c 的 softmax(half c%2)。配 `V_MTE2` 风格 flag(`set_flag("v","mte2",half)` / `wait_flag` 按 ping-pong index)。

**✅ UB 预算已重算(本 session,kernel.py `ub_addr` 实测 offset)**:当前峰值 **176K**(acc_o 64K + acc_s_ub 16K + acc_s_ub_ 16K + acc_s_half 8K + scalars ~2.3K + kv_ub_multi 32K + acc_o_ub 32K)。`acc_s_ub_` ×2 = +16K → **峰值 186.3K < 192K(~5.7K 余量)✓**;`acc_o_ub` ×2 = +32K → 218K **爆墙 ✗** ⇒ **1d 只预取 score,`acc_o_ub` 维持单缓冲**(V2 ws_o 预取不做)。但 `acc_s_ub_` 现在 80–96K、夹在 acc_s_half@96K 与 kv_ub_multi@112K 之间,扩 32K **不能原地长,必须重排 `ub_addr`**。建议新布局(大 buffer 先放):
  `acc_o`@0(64K)→`kv_ub_multi`@64(32K)→`acc_o_ub`@96(32K)→`acc_s_ub_`@128(**32K,[2,32,128]**)→`acc_s_ub`@160(16K)→`acc_s_half`@176(8K)→scalars@184(~2.3K);`acc_o_half`(epilogue,32K bf16)别名 `kv_ub_multi`@64(与 gather 时段不重叠)。峰值 186.3K。

**小步拆分(各自独立 NPU 验,复刻 S2b.0 的"先布局后 overlap")**:
- **1d-α ✅ 完成并 NPU 验证(2026-06-10,commits `3812768`+`e56f05b`+`69ba88c`)**:`acc_s_ub_` 扁平 `[2*v_block,BI]`(half=行 `pv*v_block..+v_block`),V1 全程只用 half pv1,行为不变。途中坐实三个事实(都已进 pitfalls skill):① Var 起点 2D slice 是 BufferLoad → fill 编译炸/add 走标量路静默错 → fill 整 buffer、add 逐行;② annotate 布局顶部不是自由空间,planner 把 reduce/sort 隐藏 tmp 排在 named 峰值之后,新布局须留 ≥13K 尾巴 → `acc_s_half` 别名进 `acc_o_ub` 头 8K(named 峰值 178.3K,尾 13.7K);③ 该别名靠 V1-end+V2-end barrier 串行,**β 保留 V1-end barrier 时安全,若动 barrier 必须先拆别名**。
- **1d-β ✅ 完成并 NPU 验证(2026-06-10,commit `d9d6552`)**:预取 chunk t score 进 half t%2 ∥ V1(t-1) softmax;fill+select+逐行 add 塌缩成单 select 直接套 score;稳态预取在 softmax 链后/cast 前;WAR `v↔mte2` eid=half(pre-set ×2+drain ×2)、RAW `mte2↔v` eid=half;V1-end barrier 保留。decode+prefill pass。
- **1d 后 profile(8K scfa_prefill 稳态)**:Duration ~36.9ms(基线 36.3 持平),cube≈22.6 / aiv≈22.5,gap≈14;vector pipe:vec 6.49 / scalar 9.4 / mte2 8.9 / mte3 4.4(pipe-sum 29 > aiv 22.5 ⇒ **V0/V1 重叠生效**,但 Duration 没动)。
- **S2b.1e debarrier V2 merge:已试、变差、已回退(`be2f1a2` → revert `cd2e958`)**。正确性 pass 但 Duration 36.9→**40.9ms**:aiv_vec 6.49→5.57(空泡确实少了),可 scalar 9.4→11.3、mte3 4.4→5.9、cube 22.6→27.0 全涨——V2 的 MTE2/标量流失去 barrier 节流后与 V0 gather/V1 抢 MTE2/总线,KV_READY 变晚把 cube 链拖长。**教训:核内 barrier 不全是纯开销,部分起"隔离资源竞争"作用;V2 debarrier 不赚,别再试同款**。
- **S2b 阶段结论**:核内 pipe 重叠机制(flag/ping-pong/预取)全部验通,但 Duration 36.3→36.9 没赚——瓶颈不在核内 pipe 串行。
- **S2c 也试了、也回退了(2026-06-10)**:cube MM2→t-2 的 PreloadPipeline skew(S2c.0,`4044e56`+parity 修复 `b6d306a`,decode+prefill 全 pass)Duration 36.9→**44.4ms**;full S2c(双侧)有 ws_kv 越界 bug 修后未单独验。`c9f9948` 回退到 1d-β。
- **⭐ 全程 profile 数据链与硬结论**:基线 36.3 → 1c/1d 36.9(机制生效:pipe-sum 29>aiv 22.5) → 1e 40.9(回退) → S2c.0 44.4(回退)。**每个"调度结构"刀都不赚反亏;gap≈14ms 恒定;vector scalar≈10ms + mte2≈8-9ms 是基底**。与 AscendC 6.87ms 的差距≈100×/chunk,来源是 **per-op 粒度**(32×select、32×add、64×逐行 gather DMA、每 op 标量发射)而非调度——AscendC 单条 vector 指令处理整 tile,TileLang tile.* 逐头逐行。**调度层已挖完,继续在 skew/barrier 上动刀没有肉**。下一阶段唯一有肉的方向:**砍 per-op 数**(select/add tile 化、gather 合并 DMA、mask 改加性盖在 score 上),做之前先建小型微基准验单 op 成本,别再上整 kernel 试。

**✅ 微基准已建并跑通(`bench_microop.py`,NPU 2026-06-10)**,vector 核成本模型:
- 整块 [32,128] mul = 44.6ns;**拆 32 行 = 501ns(11×罚);scalar-mul/select per-row ≈31ns/行**。
- DMA:16×2KB = 1×32KB ≈ 166ns,**纯带宽限制,合并 DMA 无肉**(gather 碎片不亏)。
- flag/barrier 空流水下 ≈0。
- 每 chunk per-row 链 ≈3µs,640 chunk/核 ≈2ms 可削(V1 select 32 + V2 mul/add 64)。
- `select_fused` 19.8ns / `add_fused` 40.4ns 已实测(vs 拆行 1016/501ns)——但**两条 fused 路全死**:① select_fused 的 mask 是连续位流(4096 元素要 512B mask、非 128bit 周期),要 32 份行 mask 复制、净赚减半;② **binary_op/fill 的 dst 不收子 tile slice**——`acc_o[hbase:hbase+16,:]` 即便常量起点也塌成 BufferLoad(只有整 buffer 有 access_ptr),V2 fused add 编译挂、已回退(`4606d4a`→`b695965`)。**子 tile fused 的语言锁已解**:`binary_op` 收 tvm `BufferRegion`,prim_func 内手工构造(`tvm_tir.BufferRegion`+`Range.from_min_extent`,见 `052423a` 的 `_sub_tile` helper)即可绕过 slice→BufferLoad 塌缩——编译/数值在 NPU 验证通过。**但 V2 fused add 这刀 perf 又反向(36.9→43.3ms,`15c176c` 回退)**:aiv_vec 6.49→5.10 局部赚,V2 提前完成让下一 step 的 gather/写更早抢 MTE2/MTE3,cube mte2 5.9→6.8、Duration +6.4。这是 1e/S2c.0 的同款共振模式,坐死结论:**36.9 是当前结构下的脆弱平衡点,任何局部加速都触发资源抢占共振,前向优化线收官**(改进剩两条:tilelang 提供整 tile 跨行带步原语;或换全局调度=重写 chunk 分配)。BufferRegion 技法已记入 pitfalls(它真正的价值留给反向 kernel)。
- **AscendC 对账(2026-06-10,scfa_block_vector.h)坐实两个未复刻大头,皆为 TileLang 语言能力缺口**:① `SoftmaxFlashV2`(:442)——整 tile softmax 单库调用,我们是 ~10 op 长链+32 行 per-row sub;② `Brcb`+`RowMuls`(:330,924-928)——alpha 广播+整块行乘,我们是 32 次标量 mul。sync 密度(58 SetFlag/WaitFlag)与我们持平,非差距点。**修正(2026-06-10 终验)**:`T.tile.broadcast` 在部署版已存在(容器 hasattr 验证),V1 32 行 sub→broadcast+整块 sub 已试(`70b4a31`):**局部全赚(vec −1.4ms、scalar −0.55、mte2 −0.6)而 Duration 36.9→41.8ms,共振第四例,已回退(`aba7c85`)**。四次独立实验(1e/S2c.0/V2-fused/Brcb-sub)同模式:**任何缩短 V1/V2 的局部加速都恶化 cross-flag 接力相位 → 36.9ms 平衡点确认是该全局结构的极小值;最后差距=per-chunk lockstep + GM 接力的结构本身(AscendC 单 chunk ≈5µs vs 我们 ≈30µs)。前向收官;再前进唯一路径是重写工作分配(per-token 多 chunk 并行/cube-vector 紧耦合),非局部刀**。
- 坑(已 push 修):JIT cache 按 AST,闭包注入 body 9 case 全跑第一个 kernel → 每 case 写显式 prim_func + 解析期常量裁剪;fill(Duplicate)不收 uint8 → mask 用 compare 生成。

---

## 3. commit 地图（main 上,可回退）

- `91d42f7` **alpha 扁平 1D 修复**。✅ decode+prefill 验证。(根因:V2 逐头 rescale 的 2D 标量 `alpha[pv2,h_i]` 被 tile `binary_op` 只取 `indices[0]`、丢了 `h_i`。)
- `tag s2a-forward-verified`(= `04aa6be`) **S2a forward 双缓冲已验证 + profiled 36.3ms/18.9%**。**最稳回退点**。
- `cff7e59` **S2b.0a** gather sub-tile(4×16 行 ping-pong shape)。✅
- `0fe56b7` **S2b.0b** merge sub-tile(2×16 头)+ un-alias 到 170K 布局。✅ **S2b.0 整刀完成**。
- `ec9b282` **S2b.1a** pipe-flag smoke(gather MTE2→MTE3 换 flag)。✅
- `1d1bcfc` **S2b.1b** V0 gather 内部 ping-pong(forward+back flag + 平衡 drain)+ 记录 `copy_ub_to_ub=VEC`。✅
- `bd56634` docs(S2b.1c/1d 设计)。
- `0487312` **S2b.1c** debarrier V1 softmax(删 V0-end barrier + V1 全部 20 个 barrier,补 3 对 pipe flag:`v↔mte2` WAR / `mte2↔v` RAW / `v↔mte3` RAW;保留 V1-end barrier 当 step 边界)。✅ **decode+prefill 验证**。**最新稳回退点**。
- **HEAD** = `0487312`(+ 本次 docs 更新)。

---

## 4. 当前 kernel 结构（HEAD `0487312`, kernel.py 要点）

- 闭包常量:`GATHER_ROWS=16`/`N_GATHER_PASS=4`(gather 4×16 行)、`MERGE_HEADS=16`/`N_MERGE_PASS=2`(merge 2×16 头)、`ub_len`、`mask_w=32`。
- UB 布局(§9.2,un-aliased,峰值 176K):`acc_o`@0(64K)、`acc_s_ub`@64K、`acc_s_ub_`@80K、`acc_s_half`@96K、scalars/masks/alpha@104K、`kv_ub_multi`@112K(32K,gather ping-pong [2*16,D])、`acc_o_ub`@144K(32K,merge [16,D])、`acc_o_half`@112K(别名 kv_ub_multi,epilogue-only)。
- vector loop `for t in range(NI_total+2)`:V0(t) gather / V1(t-1) softmax / V2(t-2) merge,错位流水。cube loop `for t in range(NI_total+1)`:MM1(t) / MM2(t-1)。**注意:`for...in range()` 在 prim_func 里是 TIR loop,loop var 是 Var(不是 Python int)。**
- V0 gather(ori+cmp)已 ping-pong:per pass `if gp>=2: wait_flag("mte3","mte2",pp)` → 16 行 gather(MTE2) → `set_flag/wait_flag("mte2","mte3",pp)` → write(MTE3) → `set_flag("mte3","mte2",pp)`;loop 后 drain `wait_flag("mte3","mte2",0/1)` + `set_cross_flag(MTE3,KV_READY)`。**V0-end barrier 已删(S2b.1c)** → gather MTE2/MTE3 能流进 V1 的 VEC 窗口。
- **V1 已 debarrier(S2b.1c)**:内部零 `barrier_all`,3 对 pipe flag(`set_flag/wait_flag`):`v→mte2`(WAR,select 读 `acc_s_ub_` → load 覆写)、`mte2→v`(RAW,load → add)、`v→mte3`(RAW,cast → write ws_p),全 eid 0;`wait_cross_flag(SCORE_READY)` 自身 serialize 保留;**末尾保留一个 `barrier_all` 当 step 边界**(drain 上一 chunk,使单缓冲 scratch 免跨 chunk ping-pong)。
- V2 仍全 `barrier_all`(待 S2b.2 或被 1d 触及时再拆)。

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

**S2b.1c 验通后(2026-06-10)新沉淀进 `tilelang-perf` skill「手段 3:核内 debarrier」**:`barrier_all` 是全 pipe drain → 单核 pipe-sum;删它、只在跨 pipe 真依赖补 `set_flag/wait_flag`(同 pipe 自动 in-order)→ pipe-max。含 hazard 机械判定法、`wait_cross_flag` 自身 serialize(后面不用补 barrier)、渐进式 debarrier(先保留 step-boundary barrier 免跨 chunk ping-pong)。perf 数字栏待 S2b.1d 回填。

> 这些 skill 是给"写/review TileLang Ascend kernel"用的,新 session 写 flag 时 skill 会自动触发,照着来。

---

## 11. ⚠️ fork 切换引入 prefill 数值回归（2026-06-11，环境问题，非内核）

- **现状**:cube-direct(SWA,KV 不过 vector)+ 编译器 GM→L1 子块写补丁(fork `b5efa1b`/`52ad83a`,noClear 跳过子块清块竞争)decode 三场景两 dtype 全绿;**prefill 全场景 <99.5%**(scfa 99.22%)。
- **Ground truth(`87df937` 探针)**:把 kernel 换成 session 前"prefill 验过"的 1d-β(`d9d6552`),在同一 fork 上**只有 98.95%,比当前 99.22% 还差**。⇒ **prefill 失败是 fork 本身(领先旧 tilelang-ascend 37 commit,数值变了),不是 cube-direct/编译器/FUSE 的锅;我们的工作反而改善了 prefill**。
- **已坐实的 bug 与修复(都在当前 HEAD)**:① cube-direct 集成把 back-flag drain 误门控 `t>=NI_ori` → prefill 多 slot 事件计数饱和死锁(`cc06dfa` 改回 `not cube_direct` 逐 chunk);② FUSE-V0 ori 块 gather 的页边界分支 prefill 错(`1ee7873` 退回逐行,95.69→99.22);③ FUSE-V1/V2 整块 broadcast/别名全退回逐行(`85ed7cb`/`88b9151`)。
- **HEAD `356912c`** = 当前最佳:cube-direct(SWA decode 验证)+ SCFA/CFA 逐行老路。`probe-current-9922` 分支留存。
- **下一步(环境三选一,留给用户定向)**:① bisect fork 的 37 commit 找数值回归点;② 把编译器子块补丁打到原 0.1.9 源(若 0.1.9 上 prefill 过)而非 fork;③ 暂以 decode 验证为准、prefill 借线另行 triage(metadata 算子用户已说先不管)。
- 编译器补丁是干净独立产出:codegen 检出子块(行 extent<dstM)传 noClear=1 跳过整块 InitConstValue;whole-block 路径字节等价(SCFA/CFA 不受影响)。
