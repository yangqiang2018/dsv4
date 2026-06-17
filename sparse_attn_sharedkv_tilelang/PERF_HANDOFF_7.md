# sparse_attn_sharedkv 性能优化 — 工作交接 #7（C 跨-slot 流水进行中：S1+S2 验绿，S3a 待实现）

> 续 `PERF_HANDOFF_6.md`（C 可行性 §3.8、Ascend C 蓝本、S3 full 设计 §3.9 都在那；深细节回查它）。本文件自包含，读完即可接着干。配 `MEMORY.md`。
> **当前在做 C（跨-slot 软件流水）。S1+S2 已验绿，S3a 是下一步。**

## 0. 一句话现状 + 北极星

- **目标（用户北极星）**：复刻 Ascend C 这个算子的计算逻辑 + 性能方案，让 swa 前向做到 AscendC 的 **80–100%**。Ascend C 源码是蓝本，别发明 kernel 侧小技巧；过不去改 fork 编译器（`yangqiang2018/tilelang-ascend`，NPU 验过的成功改动打 `cfeat-*` annotated tag，攒着提上游 issue）。
- **诚实天花板**：vector 侧便宜刀**已用尽**（brcb/row_muls rescale、SoftmaxFlashV2 fused softmax 都 NPU 验过 = perf 中性，见 _6 §3.6/3.7）。**~50% 是 kernel 侧天花板**。破它只剩 **C（跨-slot 软件流水）**。**注意：即使 C 全成也就 ~55，够不着北极星 80**——80 还得叠 cube 侧（_6 §3.5② L1-ring 预取，又一大改）。**用户已知情拍板「硬上 C」**（知道 ~16-25 容器轮、最好 ~55、砸了全回退）。
- **perf 现状**：swa **35.6** / cfa **42.3** / scfa 15.0（S1+S2 后）。**比 C 前 baseline swa 42.4 / cfa 49.9 降了 ~7**——这是 S1 把 acc_o 挪 GM 的纯 DMA 成本，**所有恢复+增益押在 S3**。
- **进度**：S1✅ S2✅ Q4✅ S3b✅(swa35→41.3) C1 cube debarrier✅(41.3→45.3,+4,过baseline) C2 bulk KV gather✅但**perf平**(45.3→45.1) **C3=L0C ping-pong(M-tiling)已实现+双审过,待容器验**。**关键转折**:C2 平→re-profile 实锤 cube scalar 51% 不是 gather(是 3× decode + C1 加的 flag,C2 只把 mte2 820→754);真瓶颈=C1 让 cube pipe 真重叠(和126%)后**暴露的单块 128KB L0C 串行 MAD/FIX**。C3 复刻 Ascend C 的 2×64KB L0C ping-pong 重叠 MAD/FIX(gemm_v0 不能切操作数→用 **M-tiling 切 heads**(连续切片)而非 D-tiling,免编译器改+保住 C2;详见 §0.6.3)。到 80 仍需叠 decode-carry(削 scalar)+ vector。

## 0.6 S3b 设计已定案（2026-06-16，Plan agent 读真源码 + 我审核）—— 纯 kernel 重写、无需改编译器、分阶段把 all-or-nothing 收敛到一步

> Q4 实证「intra-slot 不够、必须 cross-slot」后,用户拍板「硬上 80,需要就改编译器」。S3b = 复刻 Ascend C 的 `PreloadPipeline` flat-gloop 跨-slot 流水。**只盯 SWA(NI_total=1,task==slot)。** 蓝本源码在本地:`ops-transformer/.../arch32/sparse_attn_sharedkv_swa_kernel.h`(PreloadPipeline `:741-769`)+ `_swa_block_{cube,vector}.h`。

**关键定案(都有源码/pass 实锤):**
1. **无需改编译器 —— 纯 kernel.py 重写。** `CombineCV`(`tilelang-ascend/src/transform/ascend_combinecv.cc:843-862`)把整个 loop nest **逐字复制**进 cube 程序和 vector 程序两份(只按 scope 把叶子 op 置空),所以在 kernel.py 写一个 flat `for gloop`,cube/vector 各跑自己那份、靠 cross-flag 会合 —— 正是 Ascend C `PreloadPipeline`(一个函数体里 `if AIC{..} if AIV{..}` 并列)。所需原语(TIR for/if、set/wait_cross_flag、set/wait_flag、GM-region Var-parity)全已 NPU 验过,**不用打新 cfeat**。
2. **flat-gloop 结构(SWA)**:`for g in range(total_tasks+2)`,3-深 skew —— cube 做 `MM1(g) ‖ MM2(g-1)`,vector 做 `V0(g) ‖ V1(g-1) ‖ V2/tail(g-2)`(tail=normalize+writeback+LSE **折进 V2(g-2)**,这样 slot g-2 的 GM round-trip 埋在 slot g-1/g 的 VEC 下 = Q4 做不到的 cross-slot 交织)。每相位 `if 0<=off<total_tasks ∧ valid(off):` 门控(=Ascend C `isValid`),`+2` 是末尾 drain(`extraLoop=PRELOAD_NUM`)。
3. **死锁铁律(#1 风险,`:1259` 血泪)**:每个 cross-flag 的 set 和它的 wait **用同一个 valid(同一 slot index) 谓词门控** → set 次数 ≡ wait 次数 = `|{i:valid(i)}|`,与 batch×slot 数无关 → 结构上不可能 leak。SWA cube_direct **只 3 个 live flag**(KV_READY dead:cube 自拉 KV)。prologue(g=0,1)/epilogue(g=S,S+1)靠 `0<=off<S` 守,不发空 set/wait。intra-core ping-pong(score eid 0/1、Q4 eid 2/3)的 pre-set/drain 移到**包住 gloop**(不是 per-slot)。
4. **parity 表**:`ws_acc_o`(S2 已 slot-parity)→ V2/tail 读 `(g-2)%2`、seed 写 `g%2`;**`sumexp`/`m_i` 必须加 gloop-parity**(`_7 §1` 的开放问题:V1(g-1) 写新 max/sum 时 V2/tail(g-2) 还在读旧的 → 单缓冲会被覆写;差一个 slot 故 `[2]` 够。**避 `:1499`**:别在 tile-op 操作数上切 Var-range,用 flat `[2*ub_len]` + 标量 base-offset `sumexp[parity*ub_len+h]`,或整行 region `sumexp[parity]`,哪个能编过在 S3b.1 验);`alpha`/`ws_kv/score/p/o`/`mask_ub`/`acc_s_ub_` 的 `t%2` chunk-parity **重解释成 `g%2`**(GM 不扩、只换索引);**`q_l1` 加 `[2,H,D]` gloop-parity**(MM1(g) 与 MM2(g-1) 用不同 slot 的 Q;L1 64→128KB,峰值 400/512KB 够);`p_lo/p_hi` 可留单缓冲(MM2 内即用即弃);**UB tiles(acc_o_work/work2/ub/half、acc_s_ub/half)留单缓冲**(Var-parity 踩 `:1499`,跨 step 复用靠 step 边界 barrier drain)。
5. **分阶段(把 all-or-nothing 收到一步)**:**S3b.0** gloop 重构成 no-op(还串行、0-skew,验 byte-identical via `get_kernel_source` dump diff + fast + decode)→ **S3b.1** sumexp/m_i 加 parity(还串行=inert,**最险 IR 编辑,隔离验**)→ **S3b.2** 重解释 ws_*/alpha/... parity + q_l1 双缓冲(还串行=inert)→ **S3b.3 引入 skew**(唯一改时序的一步=硬 gate:counter-audit flag 平衡 → fast(leak) → decode(race) → perf;过不了/hang/regress 就**整体回退 acc_o→GM 栈**,`_7 §5`)→ **S3b.4** 收紧剩余 barrier_all 为精确 flag(可选 perf 抛光)。前三步都 inert/可单验,风险全压在 S3b.3。
6. **复用 S1+S2+Q4**:S1 GM accumulator、Q4 debarrier flag + acc_o_work 双缓冲(与 cross-slot 正交,组合用)全留;S2 的 ws_acc_o parity 从「inert 总被 drain」变「live 跨-slot 双缓冲」。SCFA 全程不动(只 `slot→g` 改名,不上 skew)。
7. **诚实预期**:S3b 填 ~25% bubble + 埋 vector GM DMA → **swa ~50-55**(VEC 总忙时不减、cube 在 ~2.72ms 成共瓶颈,故封顶 ~55);**到 80 还要叠 cube 侧 L1-ring KV 预取(`_6 §3.5②`,S3b 之外另一大改)。** S3b 估 **~13-23 容器轮**(S3b.0~2 各 1-4 inert,S3b.3 ~6-10 高方差)。失败序:死锁 > no-overlap > cross-slot WAR > `:1499`。

**详细伪代码 + 行号在对话的 Plan agent 报告里(cube/vector 两 scope 的 gloop 体、decode(off) 的 OOB-safe 读、flag 表)。**

### ✅ S3b 已实现(2026-06-16,construct agent + 2 轮 adversarial review + 我审,待容器验)

**形态:新增独立 lean prim_func `sparse_attn_sharedkv_swa`(kernel.py ~1893-2576),`return ... if NI_total==1 else ...` 选择;旧 prim_func 字节同一(CFA/SCFA 不动)。** flat `for g in range(total_work+2)`,3-深 skew:cube `MM1(g)|MM2(g-1)`、vector `V0(g)|V1(g-1)|V2+tail(g-2)`。死锁铁律(set/wait 同 valid 谓词)+ score-machine pre-set/drain 包 gloop,review 把 1022 种 validity pattern 全模拟过 = 平衡。

**关键发现(review 阶段我抓到、第一轮 review 漏判的真 bug + 同时是大简化)**:**单-chunk SWA 的 online-softmax 累加器 trivially `acc = 0*alpha + O = O = ws_o`(P@V 输出),没有跨-chunk 累加** —— Ascend C 蓝本直接 normalize 它的 O buffer(vec2ResGm),根本没有独立累加器。所以 V2+tail **直接 normalize ws_o**(load ws_o → div sumexp → Output),**砍掉整个 ws_acc_o GM round-trip(S1 的 −7 来源,对 SWA 本就多余)+ no-op 的 rescale**。这同时**修了一个真 bug**:ws_acc_o 的 2-深 parity 撑不过 3-gloop-step skew —— V0(g) 对 half g%2 的清零被 V2(g-2) 的 store 覆写(g%2==(g-2)%2),V2(g-2) 真正依赖的 re-zero 是 V0(g+2) 的,而它对**最后几个有效 slot 被 skip**(g+2 越界/padded)→ 读到 stale O(g-4)。改读 ws_o(MM2(g-2) 刚写、PV_READY 守)无此问题。**`ws_acc_o` 现在是 SWA 路径的 dead arg(auto-alloc,如 SCFA 容忍)。** sumexp/m_i 用 :1499-safe 的 save/restore(region copy + 单缓冲 *_rt 喂 tile-op)跨 slot 带到 tail;q_l1 双缓冲(L1 重排到 400KB)。

**⚠️ 诚实:这是大重写、本地编不了 tilelang。container 三验是真 gate**:① fast(抓 prefill deadlock —— flag review 说低风险但 hang 最坏)② **`decode and dtype0`(抓 data hazard/wrong-output —— 我抓的那类 bug 的真 gate)** ③ perf。**预期**:Fix B 砍了 ws_acc_o round-trip(−7 源)+ cross-slot overlap 填 bubble → swa 有望从 35 冲过 baseline 42 向 ~50-55。砸了(hang/红/没动)→ 查 flag 平衡 / data hazard / overlap,或回退。

### ✅ S3b 容器验绿 + profile(2026-06-17)—— green,swa 35→41.3,但有大 headroom(不是天花板)

- **green**:fast + `decode and dtype0` 全过(两次 build error 先后修:`(tuple)[mp%2]` FloorMod、`q_l1[g%2]` 3D 缺切片 —— 都是 tilelang parse 错、py_compile 抓不到、容器逐个暴露)。
- **perf**:swa 35.0→**41.3**(cross-slot overlap + 砍 ws_acc_o 有效,逼近 baseline 42.4)。cfa 44.2 / scfa 15.9 不变(S3b 只动 SWA,CFA/SCFA 走旧 prim_func)。
- **profile(swa,稳态 ~3640us/kernel)—— 实锤「不是天花板」**:
  - **vector 是长板(aiv 3542us)但 34% 在 idle**(busy 仅 66%:vec 29%+scalar 24%+mte2 7%+mte3 7%)。idle 是**等 cube 的 SCORE/PV**。
  - **cube data-movement-bound**:`mte2`(KV/Q GM→L1)**25% 是最大 pipe**;数据搬运(mte1+mte2+fixpipe)56% ≫ mac 14%;cube pipe 基本**串行**(和 82%=18% bubble)。cube_util 94%。
  - **结论**:cube 的 KV 载入暴露+串行 → SCORE/PV 产得慢 → vector 干等 34%。**离 80 有明确路,不是顶。**
- **下一刀 = 组合拳(见 [[feedback-combined-levers]],单上一个可能没涨甚至降)**:**① cube debarrier/L1-ring**(去掉 cube gloop 的 barrier_all,让 KV 载入 mte2 叠到 gemm mac 下 → SCORE/PV 产得快 → 减 vector idle)+ **② vector debarrier**(去掉 vector gloop 的 barrier_all,让 V0/V1/V2 phase + VEC/MTE 重叠 → 减 vector busy)。**为什么必须组合**:只② → vector 变快后**更早撞 SCORE/PV 等待、idle 更多**(cube-bound),不涨;要①把 cube 喂快了②才兑现。cube 是 root,先做/一起做。
  - **⚠️ cube L1-ring 的真障碍 + 解法(实现前必读)**:cube 要 overlap 就得去 step-boundary barrier,但 **kv 只有 2-深 parity** → load(g) 写 kv[g%2] 撞 MM2(g-2) 读 kv[(g-2)%2](同 parity)= **跨-gloop WAR**,要 back-flag;而**3-深 kv 超 L1 预算**(3×kv 384 + 2×q 128 + p 16 = 528>512;q 单缓冲→464 fit 但 q 又生 WAR)。**解法(避死锁)**:这条 WAR back-flag 的 set 和 wait **都用 `valid(g-2)` 谓词门控**(被复用的那块 kv 的 owner slot)—— set 在 MM2(g-2)(step g-1,valid(g-2))读完 kv 后,wait 在 MM1(g) load(step g)写 kv 前;**两端同 valid(g-2) → set==wait,不死锁**(同 S3b 的 forward-flag 铁律)。**代价:cube 要 decode 3 个 offset(g/g-1/g-2,像 vector 那样),多一个 decode 拿 valid(g-2) 当 wait 门。** 这是这刀最难、最该谨慎 construct+review 的点(死锁雷区,但门控对了就安全)。
  - 次要:vector scalar 24%(832us)偏高 = gloop 每步 3× decode,可 carry decode 标量跨步不重算(降 vector busy)。

### 0.6.1 C1 = cube debarrier/L1-ring ✅ 容器验绿(2026-06-17,swa 41.3→45.3 +4,首次过 baseline)

> **结果**:fast + decode/dtype0 全过(L0C 别名修复经住了真数值 gate);swa **41.3→45.3**(+4,**首次越过 pre-C baseline 42.4**)。cfa 43.3/scfa 16.9 是噪声(走 main prim_func,本刀字节不动)。**cube 单刀就 +4** → 印证 profile 判断(cube data-movement 是 root,prefetch mte2 叠到 MAD/FIX 下确实把 cube 喂快了),也印证 [[feedback-combined-levers]](但这次单刀就兑现了一部分)。**下一步第②刀 vector debarrier 叠上。** 设计/坑见下(已落地代码,审过)。

**做法**:`sparse_attn_sharedkv_swa` 的 cube scope(kernel.py ~2042-2345)**6 个 `barrier_all` 全删**,换成精确 pipe flag。新增 cube `decode(g-2)→valid2`。flag 表(**eid 空间是 per-HardEvent-DIRECTION**,fork 自己的 gemm 模板 `common.h:586-592/932-947` 把 eid 0 跨 M_FIX/FIX_M/MTE2_M/M_MTE2 同时复用就证明了这点 → within-block 的 (m,fix,0/1/2)/(fix,m,1)/(mte2,m,0) 跟 back-flag 不撞,即便 MM1/MM2 现在重叠):
- **BACK1 = (m→mte2, eid 0)**:kv/q_l1/p_lo 的跨-gloop WAR。wait 在每步顶(gated `valid2`,排空 MTE2 管),set 在 MM2 两个 gemm 读完 kv 后(gated `valid1`)。set/wait 同 pin slot g-2 → set==wait,无 leak。**一个 MTE2-drain 的 wait 覆盖 kv+q_l1+p_lo 三个**(q_l1/p_lo 靠 in-order MAD/MTE2 传递)。无 pre-set(头两次 kv 写无前读者,valid2=F 自动跳)。
- **BACK_LC = (fix→m, eid 0)**:**关键坑——`acc_s_l0c` 和 `acc_o_l0c` 别名同一 L0C 地址**(`l0c_addr = {"acc_s_l0c":0, "acc_o_l0c":0}` 行 263 注释「disjoint phases ⇒ alias」;acc_s [64,64]=16KB ⊂ acc_o [64,512]=128KB)。原来纯靠 barrier 把 MM1 的 score-L0C(FIX 读)和 MM2 的 out-L0C(MAD 写)隔成「disjoint phases」。去 barrier 后这俩在同一条链 A1→A2→A3→A4(MM1)→B1→B2→B3(MM2)→A1'... **必须一个统一 WAR flag**:set 在每个 block 最后一次 FIX 读后(MM1 copy_hi / MM2 copy_o),wait 在每个 block 第一次 MAD 写前(MM1 gemm_lo / MM2 gemm)。block-by-block **交替 w,s,w,s** 就把两条缝都缝上(缝1:copy_hi→同步 MM2 gemm;缝2:copy_o→下步 gemm_lo)。pre-set 武装第一个 wait,loop 后 drain 吃最后一个 set。**两个独立 flag(分别管 acc_s/acc_o)会各自配错对、漏掉跨-alias 缝**——这是第一版的真 bug,第 2 个 reviewer 抓到、我合并修了。
- **within-MM1**:load→gemm RAW 新增 (mte2→m,3);原 (m,fix,0)/(fix,m,1)/(m,fix,2) 链不动。**within-MM2**:原 (mte2,m,0)/(m,fix,1) 不动;两个 P@V gemm 之间的 barrier 删(MAD in-order)。`set_cross_flag("FIX", SCORE/PV_READY)` 本就由 FIX 管序,删它前的 barrier 安全。

**对抗审结论**:① flag 平衡/死锁 reviewer:全 (src,dst,eid) set==wait,prefill(多 slot)/decode(单 slot)/gap/空核/+2 epilogue/冷启全过,无环(每个 blocked wait 的 producer 都是已入队的过去 op)。② 数据冒险 reviewer:**抓到 L0C 别名漏洞**(已修)。③ 合并修复 reviewer:6 项全 PASS,无残留。**两个 load-bearing 前提(都成立)**:N0==N1(靠 +2 epilogue 保证每个 slot 的 MM2 都落在 range 内);`if valid` 必须 lower 成设备分支、wait+set 成对进出同一 `if`(现状 GREEN 行为)。

**预期**:cube 把 mte2(25%,最大 pipe)prefetch 叠到 MAD/FIX 串行链下 → cube 每步更快 → 喂快 SCORE/PV → 砍 vector 34% idle。**按 [[feedback-combined-levers]]:这是组合拳第①刀(root),先单独验 correctness**(死锁风险集中在 cube 跨-gloop WAR)。**perf 可能只部分兑现甚至持平**(vector 自己的 barrier 还在,消费速度受限)——别据此判死,第②刀 vector debarrier 下一轮叠上才看组合 perf。砸了(hang→查 flag 平衡;红→查 data hazard;尤其再查别名):回退本 commit 即可(只动 swa cube scope)。

**容器三验**(改 kernel.py,`cd /sdb/yq/dsv4 && git pull` 即生效):① `pytest -q sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv_fast.py`(prefill deadlock/leak)② `pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode and dtype0"`(真数值,抓 race/WAR)③ `python sparse_attn_sharedkv_perf_compare.py`(perf%)。

### 0.6.2 C2 = cube SCALAR 削减(bulk KV gather)已实现(2026-06-17,待容器验)

> **C1 后重新 profile(msprof PipeUtilization,swa)实锤瓶颈搬家了**:C1 让 cube pipe 真重叠(mac+scalar+mte1+mte2+fix **和 126%** vs C1 前串行 82%),**新瓶颈 = cube SCALAR 51%(1584us,长板核的最大 pipe)**。cube aicore_time 3140us(util 96.7%,长板);vector aiv_time 3167us 但 **idle 24%**→busy 仅 2407us(被 cube 卡)。所以**先打 cube scalar,不是 vector**(vector debarrier 会打错目标——这就是 profile-first 的价值)。
>
> **cube scalar 来源**:每步整数 `//`/`%`——3× decode(g/g-1/g-2)的 `//max_seq`+`%max_seq`(3+3)+ **KV gather 的 8 个 16-row chunk 各算 `g0//ori_block_size`+`g0%ori_block_size`+边界 if(8+8,大头)**。`//`/`%` 在 Ascend scalar 单元很贵。

**做法(复刻 Ascend C 的 bulk DataCopy,不是发明)**:Ascend C `CopyGmToL1`(swa_block_cube.h:244)用 `DataCopy(nd2nz, nValue=srcN)` **一次 bulk 拷连续段**,不是 16-row chunk。SWA 的 ori window 是连续 run,故把每个 64-row half 用 **1-2 个 runtime-extent `T.copy`(per 分页块段)** 拷,替掉 8-chunk loop。`ori_block_size=128`(测试 `block_size1=128`,来自 KV tensor shape)≥ BI_half=64 → 一个 half 跨 ≤2 块 → 每 half 1 div + 1 mod(**全步 2+2,从 8+8 砍下来**)。编译期 `if ori_block_size >= BI_half:` 守(只 trace 一条路,无 IR bloat),`else` 留原 16-row chunk loop 作小块 fallback。**正确性**:新拷贝产出与原 8-chunk **字节同一** kv_lo/kv_hi(全局行 [ori_left0, ori_left0+128),已逐 case 验:对齐/跨界/lo-vs-hi)。`is-subtile-runtime-extent` cfeat 撑 runtime-extent 拷贝。

**预期**:cube scalar 8+8→2+2 div/mod(gather 大头)→ cube 时间往 vector floor(2407us)掉 → swa 45→~55+?(若 cube 掉破 vector floor,vector 变瓶颈,再轮到 vector debarrier / decode-carry)。**这刀单独应见效**(gather 占 scalar `//`/`%` 的大头,~73%,非组合拳——不像 cube/vector debarrier 互相依赖)。**风险**:correctness(错地址→错 KV→错输出),decode/dtype0 gate 抓得到;无死锁风险(纯地址算术,无新同步)。砸了回退本 commit。

**下一步若 cube 不再是瓶颈**:decode-carry(把 valid(g) 存 3-深 rotating UB,g-1/g-2 不重算,省 2+2 div/mod,**cube+vector scalar 都受益**)+ vector debarrier(seed/restore/LSE 的 glue barrier)。先 profile 再定。

### 0.6.3 C3 = L0C ping-pong(M-tiling)已实现(2026-06-17,construct + 2 轮对抗审 + 我审,待容器验)

> **C2 平 → 第二次 re-profile(swa,post-C2)实锤真瓶颈**:cube aicore_time 3047us(util 96.7%,长板),scalar 1577us(51%,C2 后**几乎没变** 1584→1577,证实 gather 不是 scalar 大头);mte2 820→754(C2 唯一效果)。cube pipe **和 126%**(C1 让它们真重叠了),但 cube 时间没怎么降——因为**单块 128KB L0C 逼着 MAD/FIX 串行**(acc_o[64,512]=128KB 占满 L0C,acc_s 只能别名它,故 MM1/MM2 整条 gemm→copy 链串行)。Ascend C 用 **2×64KB L0C ping-pong**(`cL0BufIter%2`,swa_block_cube.h:560)让 copy(FIX)叠到下个 gemm(MAD)下——这是我们最大的结构性差异。

**关键障碍 + 解法**:Ascend C 靠 **D-tiling**(D_SPLIT=256)切出 64KB L0C tile。但 **TileLang `gemm_v0` 不切操作数**(kernel.py:246,N 从 output C 的 shape 推、要 B 的 N=output N)→ D-tiling 需要 D-切的 strided kv 操作数,gemm_v0 不吃(要么 KV D-split 大改+撤销 C2,要么改编译器)。**解 = 改用 M-tiling(切 heads)**:P@V 的 output acc_o[64,512] 按 head 切成 acc_o_a[0:32]、acc_o_b[32:64](各 [32,512]=64KB,正好一个 slot),A 操作数 `p_lo[0:32,:]`/`p_lo[32:64,:]` 是**连续行切片**(gemm_v0 吃,reviewer 读 codegen 确认 offset/M-from-C 都对)。**免编译器改、保住 C2、不碰 KV load**。MM1 用 acc_s_lo/acc_s_hi 当两个 ping-pong output(本就两个 [64,64])。

**⚠️ build-fix(第一推 container 红,reviewer 看错文件)**:`T.gemm_v0` 实际用的是 `tilelang/language/ascend.py`(不是 reviewer 读的 `pto.py`)。它的 `_retrieve_shape` 只吃 `Buffer`/`BufferRegion`;**2D range-slice `p_lo[0:HALF_HEADS,:]` lower 成 `BufferLoad` 被拒**(报 `Unsupported argument type: BufferLoad for buffer p_lo[0:32, 0:64]`)。现有能用的操作数(`kv_lo[pb,:,:]`)是**前导标量索引**→ BufferRegion。**解**:p_lo/p_hi 改 3D `[2, HALF_HEADS, BI_half]`,用 `p_lo[0,:,:]`/`p_lo[1,:,:]`(前导标量 0/1 → BufferRegion);同字节/同 L1 地址(head-half 本就连续),p-load 拆成 4 个 copy(按 head)。M-tiling 的 head 映射不变(p_lo[0]=heads 0:32),reviewer 的 M-tiling/flag 分析仍成立。**教训**:静态确认 gemm_v0 要锁定 `ascend.py` 那份。

**实现**:L0C 两 slot acc_s_a/acc_o_a@0、acc_s_b/acc_o_b@64KB(slot 内 acc_s⊂acc_o 别名,disjoint MM1/MM2 phase)。p_lo/p_hi 3D head-split。flag:per-slot RAW(m→fix eid{0=A,1=B}:copy 等 gemm)+ WAR(fix→m eid{0=A,1=B}:下个 gemm 等本 slot 上次 copy),pre-set 两 WAR、loop 后 drain 两个。**替掉旧 BACK_LC**(acc_s/acc_o 不再别名一块串行 buffer,改 2-slot ping-pong)。BACK1/load-RAW/p-load-RAW 不动。groups 顺序 MM1_lo(A)/MM1_hi(B)/MM2_m0(A)/MM2_m1(B)→相邻 group 不同 slot→copy 叠下个 gemm。

**对抗审(2 reviewer + 我)**:① flag 平衡/死锁:全 (src,dst,eid) set==wait,gap/冷启/空核全过,无环;唯一 load-bearing 前提 N0==N1==N2(+2 epilogue 保证)。② 正确性 5 项全 PASS——M-tiling 重构出与旧 acc_o[0:64] 字节同一;slot 内 acc_s⊂acc_o 别名被 per-slot WAR 完整覆盖(gate 整 slot 非字节范围);**gemm_v0 M-slice(p_lo[32:64,:] 非零起点连续行切片)reviewer 读 codegen 确认对**(_retrieve_ptr offset=2048→GetSliceInfo 出 [32,64]→CreateCubeVariable base+offset,M 从 C 取);残留只是「静态确认非编译确认」,容器一跑即知。

**预期**:copy(FIX,大头是 acc_o[*,512] 的 O 写)叠到 MAD 下 → cube 往 vector floor(~2407us)掉 → swa 45→~52-54。**scalar 51% 仍在**(C3 不碰 decode),到 80 还要 decode-carry + vector。砸了(hang→flag;红→M-tiling 地址/gemm_v0 M-slice;尤其若 build error 提示 gemm operand)回退本 commit(只动 swa cube L0C/MM1/MM2)。

## 0.5 UPDATE 2026-06-16：S3a 前提证伪 → 改做 Q4（debarrier+双缓冲），已 push 待验

**两条独立 dataflow 分析（本人 + Plan agent，读全代码）一致证伪 S3a 对 SWA 的前提**：
1. **已验硬事实**：`T.barrier_all()` → `pipe_barrier(PIPE_ALL)` = **核内全 pipe 屏障**（非跨核，查 fork `codegen_ascend_pto.cc`）。SWA cube_direct **cube 自己拉 KV**（`kernel.py:~627`），`wait_cross_flag(KV_READY)` 在 `:~874` 的 dead else —— cube 启 MM1 不等 vector。
2. **瓶颈是 vector，cube 空转**：S1 给 vector 加了 ~10 个 GM round-trip（seed 存 / V2 load+store / tail load），每个被 `barrier_all` 墙隔开 → **DMA 与 VEC 强制串行**。**周期 ≈ vector 忙时；−7 = vector 侧被串行化的 GM 延迟**。
3. **S3a（跨-slot tail 重排）动不了 vector 忙时**：重排不减 vector 总忙时；它想要的「cube MM1(slot) ‖ vector tail(slot−1)」**本来就免费存在**（cube 在 tail 期间本就空转）；且把 tail 排在 V1(slot+1) 前 → P_READY 产得更晚，cube 跑不了更前。预测 ~34-36，**过不了自己硬 gate**。

**Q4 = 真因杠杆（已实现，commit `78d78fe`，cube_direct only，SCFA byte-identical）**：把 S1 那些 `barrier_all` 墙换成精确 pipe flag + **双缓冲 `acc_o_work`**（新增 `acc_o_work2`，两块 32KB 填满腾出的 64KB acc_o 槽），让 pass mp+1 的 GM load(MTE2) 叠 pass mp 的 rescale/div(VEC)+store(MTE3)。复刻 Ascend C 的 overlapped `vec2ResGm`（GM 驻留不是 −7，串行化才是）。
- **V2 merge**：per-pass barrier_all → `mte2→v`(loads RAW) / `v→mte3`(store RAW) / `v→mte2 eid2`(acc_o_ub 单缓冲 WAR，set@mp0/wait@mp1 平衡)。
- **tail**：tail-entry barrier_all（V2→tail GM RAW + buf WAR）+ per-pass flag（`mte2→v` RAW、`pipe_barrier v` div→cast、`v→mte3` casts→Output）。
- **Q4 eid {mte2→v 2,3; v→mte3 1,2; v→mte2 2} 与 score 机器 {v→mte2 0,1; mte2→v 0,1; v→mte3 0} 不交**。flag 平衡 + WAR/RAW 覆盖**已独立审计**（SWA+CFA 脚本验平衡、SCFA 字节同一、双缓冲地址不重叠）。
- **预期（已证伪）**：原以为 swa 35.6→~40-42 / cfa 42.3→~49。

### ✅ Q4 容器验结果（2026-06-16）：GREEN 但 perf 基本没动 —— 证实 intra-slot debarrier 不够

- **green**：fast(swa/cfa/scfa prefill) + `decode and dtype0` 全过 —— **flag 方案数值正确**（无 race、无 deadlock，acc_o_ub WAR + 双缓冲 + 跨 phase 同步都对）。**首推有个 build error**（`(tuple)[mp%2]`:range loop 的 mp 是 TIR Var、不能索引 Python 元组;`if cube_direct` 是编译期分支故 scfa build 跳过没踩到、swa/cfa 踩到 —— 第二推 `1a11be7` 手动展开 2-pass 修好）。
- **perf**：swa 35.6→**35.0**（持平/噪声内）/ cfa 42.3→**43.7**（+1.4，passes 多故略有收益）/ scfa 15.2（不变）。**−7 没收回。**
- **结论（重要）**：**−7 的真因不是 intra-slot 的 barrier_all 串行化**（debarrier 了也没回来）。S1 给 vector 每 slot 加的 ~8 个 GM round-trip(seed 2 store / V2 2 load+2 store / tail 2 load)是 **GM 延迟**;单 slot 的 VEC 计算量**不够遮**这些 DMA(2-pass 流水只藏得下一两个 load)。要遮全得靠**跨-slot 把 DMA 铺到邻居 slot 的 VEC 下**(Ascend C PreloadPipeline 的 softmax(n-1)‖merge(n-2) 跨-task 交织) = **S3b**。这反而**印证了原 handoff 的 cross-slot 论点**(我之前对 intra-slot 能不能救持怀疑、现已实证不能)。cfa +1.4 是因为 chunk 多、intra-slot overlap 有点用,但离 −6 差得远。
- **当前态:swa 35 / cfa 43.7,仍 BELOW baseline(42.4/49.9)。保留 Q4 而不上 S3b = 严格劣于 baseline → 不上 S3b 就该整体回退。**
- **决策点(待用户拍板)**:**(A) 回退 S1+S2+Q4 到 baseline 42.4/49.9**(诚实终点:80 这条路够不着 —— vector 便宜刀尽、intra-slot 已证不够、S3b 最好也就 ~55、cube L1-ring 又一大改);**(B) 上 S3b**(flat gloop 跨-slot 重写,~8-12 高方差轮、all-or-nothing、最好 ~55 仍非 80、可能 regress/hang);(C) 先 profile 确认瓶颈再定(便宜)。
- **容器三验命令**:① fast(leak)② `-k "decode and dtype0"`(race)③ perf_compare。

**§1 的 S3a 设计 ⚠️ SUPERSEDED（保留作分析记录，别再照做）。**

## 1. ~~立刻干：S3a（vector deferred-tail skew，硬 gate）~~ ⚠️ SUPERSEDED（见 §0.5，前提证伪 → 改做 Q4）

**目标**：overlap **slot k 的 normalize+writeback**（vector 尾，正是 S1 加重的那段 cube 空泡）和 **slot k+1 的 cube MM1**。**只盯 SWA（NI_total=1）**；CFA 的 chunk-skew × slot-skew 交互更复杂，留 S3b。

**思路**（重排 **vector scope** 的 for-slot 体；**cube scope 不动**）：
```
每 slot 迭代:  V0(slot) gather   [提前 → 早设 _FLAG_KV_READY(slot) → cube MM1(slot) 能启动]
              if slot>=1: tail(slot-1) = normalize+writeback+LSE   [‖ cube MM1(slot) = THE OVERLAP]
              slot 的 seed/fill（m_i←Sinks, sumexp←1.0, ws_acc_o[slot%2]←0）
              V1(slot); V2(slot)
+ epilogue（for-slot 后）: tail(末 slot)
```
- overlap = **cube MM1(slot) ‖ vector tail(slot-1)**。carried 累加器 = `ws_acc_o[(slot-1)%2]`（S2 已 parity，GM 持久跨迭代）。

**⚠️ 开放设计问题（fresh session 必须仔细解，别照搬 _6 §3.9 里「sumexp/m_i 单缓冲够」那个过快结论——它没考虑 seed/fill 覆写）**：
- `tail(slot-1)` 要读 slot-1 的 `sumexp`/`m_i`，但 slot 的 **seed/fill 会覆写它们**。要 `tail` 读得到旧值 → `tail` 必须在 slot 的 seed/fill **之前**；但 `tail` 又要在 `V0(slot)` **之后**（才有 overlap，KV_READY 先设）。而 V0/seed/fill/V1/V2 现在全在 `for t` 循环里。**两条出路**：(a) 把 SWA 的 `for t`（NI_total=1：V0@t=0 / V1@t=1 / V2@t=2）**摊平成直线**再重排（V0 提前、seed/fill+V1+V2 放 tail 后）——最干净；(b) 给 `sumexp`/`m_i` 加 slot-parity——但它们在 `div`/`ln`/`add`/`softmax_flashv2` 里用，Var-parity 会踩 `kernel.py:~1499` 的「UB tile-op no Var-offset range slice」坑，更险。**先试 (a)。**
- **flag 平衡（死锁雷区，`kernel.py:~1259` 注释记载「+2/slot leak 死锁 prefill」）**：deferred tail 本身不发 cross-flag（只写 `LSE_out` 到 GM），理论不改 flag set/wait 计数。但 V0/V1/V2 跨迭代重排后，4 个 forward flag（`_FLAG_KV/SCORE/P/PV_READY`，行 ~112-116）和 intra-core ping-pong（`v↔mte2` 等）的 per-slot 平衡可能被打破。**必须数清每个 flag 的 set==wait，且每步 prefill（多 slot 抓 leak）+ decode（单 slot 抓 race）都验。**

**硬 gate**：S3a 落地后若 swa **连 S1 的 −7 都收不回**（没回到 ~42），**abort + 回退**（见 §5），**不进 S3b**。S3a 是这条路上第一个有真实风险/回报信号的点。

## 2. C 全景 + staging（深设计回查 _6 §3.8 / §3.9）

- **S1 ✅**：acc_o UB→GM 单缓冲（commit `8595d49` + 跨-pass WAR 修 `cf7320c`）。cube_direct only，SCFA 逐字不变。
- **S2 ✅**：`ws_acc_o` GM slot-parity（`d88dca9`，inert、byte-identical）。
- **S3a**（本文 §1）= 1-深 vector deferred-tail skew，硬 gate。
- **S3b** = full flat-gloop 重写（_6 §3.9）：`T.Scope("C")/T.Scope("V")` 是**整函数**分区（CombineCV 塌成两个顶层程序），真正的跨-slot 流水要把主循环整个改成 Ascend C 的 `PreloadPipeline`（flat `gloop` + 3-深 ring + isValid 门控 + 末尾 `extraLoop` drain）；per-buffer parity table、cross-flag scheme 都在 _6 §3.9。~8-12 轮高方差。
- 全 C 估 ~16-25 容器轮，最好 ~55，砸了**整套 acc_o→GM 全回退**。

## 3. 关键代码状态

- **`ws_acc_o`**：GM workspace（prim_func arg，`@tilelang.jit(... workspace_idx=[13,14,15,16,17])`，**api.py 不用改 = auto-alloc**），shape `[core_num, 2, H_per_block, D]`，index `[cid, slot%2, vid*v_block+heads, :]`。4 个 copy 点：fill store / merge load+store / writeback load。
- **`acc_o_work`**：32KB UB 工作 tile（`[MERGE_HEADS, D]`，annotate 到 acc_o 空出的区，**cube_direct only，单缓冲**——parity 放 GM 不放 UB tile，避 `:1499` 坑）。
- **cube_direct vs SCFA**：所有 acc_o→GM 改动 **cube_direct only**；SCFA 留 UB `acc_o` 累加器**逐字不变**（每处 `if cube_direct: <GM> else: <原样>`；`cube_direct` 是编译期 bool，两路编进不同 binary，互不串）。改任何 cube_direct 路径前先确认没碰到 SCFA。
- **S1 同步形态**：merge/normalize 的 GM round-trip 用**全 barrier_all**，含两条跨-pass WAR 守（merge store 后、writeback cast 后，`cf7320c`——`acc_o_work` 跨 pass 复用，上一 pass 的 store/cast 读没排空、下一 pass load 就覆写）。S3b 再换精确 flag。
- 常量：`v_block=32, ub_len=32, MERGE_HEADS=16, N_MERGE_PASS=2, BI=128, D=512, H_per_block=64, accum_dtype="float"`。SWA `NI_total=1`，CFA/SCFA `>1`。
- grid：`with T.Kernel(core_num, is_npu=True) as (cid, vid)`，`vid∈{0,1}`（每 AIV 半边 heads）。slot loop：`for slot in T.serial(total_work)`（行 ~564），`pid = linear_start + slot`。

## 4. 验证 workflow（本地无 NPU + 无 tilelang，全在容器）

- 本地只能 `python -m py_compile sparse_attn_sharedkv_tilelang/kernel.py` + `ruff`（`ruff` 在 `~/.local/bin/ruff`，先 `export PATH=$HOME/.local/bin:$PATH`；**别 pip install ruff**）。
- **改 kernel.py（dsv4）→ 容器 `cd /sdb/yq/dsv4 && git pull` 即生效（JIT，不用重装 .so）。** 只有改 fork 的 .cc/.h 才 `USE_ASCEND=True pip install -e . --no-build-isolation`（fork 在容器 `/app/data/tilelang-ascend`，branch `ascendc_pto`）。
- **容器三验**：① `pytest -q sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv_fast.py`（build + prefill swa/cfa/scfa，**抓 deadlock leak**）② `pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode and dtype0"`（**真数值 gate，抓 race/WAR**）③ `python sparse_attn_sharedkv_perf_compare.py`（swa/cfa/scfa，perf%=AscendC/TileLang）。**S3 每步 ①②都必须跑——缺一漏一种 bug（leak 只 prefill 抓得到、race 只 decode 抓得到）。**
- **`git push` 必须单独成一条命令**（别和 `git add`/`commit` 串成复合命令，否则整条以 `git add` 开头匹配不上 `Bash(git push *)` allow 规则、落到 auto-mode classifier 被软拒 —— 与仓库归属无关，见 `feedback_push_workflow` 记忆）。dsv4 push `main`；fork push `ascendc_pto`。push 后 `git checkout main && git pull` 同步（全局 CLAUDE.md）。
- 改完 `ruff format` + `ruff check --fix` + `py_compile`；commit 英文 message + 结尾 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`；回复正文中文。

## 5. 回退选项（S3a 砸了 / 决定收手 → 体面终点）

`git revert` S1+S2 的 commit（`8595d49` `cf7320c` `d88dca9` + 其 doc commit），或把 `kernel.py` revert 到 SoftmaxFlashV2-ship 态（`78838ea` 的 kernel）→ 回 **swa 42.4 / cfa 49.9**（已验过的态，**不用再烧容器**）。vector 刀收官 + 6 个 cfeat tag + 完整 C 设计分析（_6 §3.8/3.9 + 本文）全留作记录。这是诚实、体面的收官。

## 6. 这次 session 已 ship（别重做）

- **SoftmaxFlashV2 fused softmax**：fork builtin `tl.ascend_softmax_flashv2`（4 处注册；`common.h` wrapper 从 ASC stdlib `asc/include/adv_api/activation/softmaxflashv2.h`+`softmax_tiling.h` include，`__has_include` 守 + fallback `kernel_operator.h`）+ kernel wiring（cube_direct V1，替手写 ~8 VEC pass）。NPU green，但 **perf 中性**。tag `cfeat-softmax-flashv2`。
- 补打 `cfeat-tile-op-bufferload`（d789b93，binary/unary 吃 BufferLoad 切片，green user = brcb/row_muls 复用 `_handle_buffer_load`）。
- **fork 共 6 个 cfeat tag**：`gm-l1-subblock-write` / `reduce-tmp-half` / `is-subtile-runtime-extent` / `brcb-row-muls` / `tile-op-bufferload` / `softmax-flashv2`。
- **C 进度**：S1（acc_o→GM）、S2（ws_acc_o parity），都验绿；C 可行性 + S3 full 设计 + S3a 设计（_6 §3.8/3.9 + 本文 §1）。

## 7. 关键坑（本/前 session 血泪，回查 _6 §7 更全）

- **acc_o→GM 的 `acc_o_work` 跨-pass WAR**：共享工作 tile，上一 pass 的最后一次读（merge GM store=MTE3 / writeback cast=VEC）没排空、下一 pass 的 load（MTE2）就覆写 → ~30% 错、max rel 2.0。修=每个跨-pass 复用点后补 barrier_all（`cf7320c`）。**S3 会大量复用 GM round-trip，这类 WAR 是头号雷。**
- **cross-slot flag 失衡 = 死锁**：`:1259` 记载 +2/slot leak 死锁 prefill（事件计数器跨 slot 累积、prefill 多 slot 撑爆；decode 单 slot 躲过）。**flag set==wait 必须数平，prefill+decode 都验。**
- **失败形态判读**：1-3% 错/两 dtype 都崩/max rel 2.0 整齐 = 结构 bug；fast 绿不代表覆盖（merge/decode 才压满多 pass）。
- **UB tile-op 不吃 Var-offset range slice**（`:1499`）：parity 放 GM copy（region context 容 Var），别放 tile-op 操作数。
- **别名 buffer 进 SCFA 的 IR 会扰动它**（+4ms）：cube_direct-only 的 buffer 只在 `if cube_direct:` 块 annotate，SCFA 别看见。
