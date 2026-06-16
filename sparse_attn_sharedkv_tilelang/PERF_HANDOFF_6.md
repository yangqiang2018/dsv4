# sparse_attn_sharedkv 性能优化 — 工作交接 #6（编译器层 + lever stack）

> 续 `PERF_HANDOFF_5.md`（cube-direct 收官于 SUCC9，profiling 定根因）。本文件自包含，读完即可接着干。
> 配 MEMORY.md（`feedback-compiler-feature-tags` / `project-fork-prefill-regression` / `project-tilelang-fork` / tilelang-perf skill）。

---

## 0. 一句话现状

- **目标**（用户定，北极星）：**完全复刻 Ascend C 这个算子的计算逻辑 + 性能优化方案**，让 TileLang 前向 perf 做到 AscendC 的 **80–100%**。**Ascend C 源码是蓝本** —— perf 缺口靠「它怎么优化就怎么复刻」，**别自己发明 kernel 侧小技巧**（本 session 发明的 V2-merge wide-add / broadcast-mul 两刀都撞编译器/硬件墙回退了；§3-B KV 复用 / §3-C 内存布局才是复刻 Ascend C 的省内存 + 流水方案）。**过不去就改编译器**（fork `yangqiang2018/tilelang-ascend` 是我们的；修 bug / 加特性，每个 NPU 验过的成功改动打 `cfeat-*` annotated tag，**攒着一起给官方 `tile-ai/tilelang-ascend` 提 issue / 需求** —— 见 §4）。
- **perf**（`perf_compare`，sharedkv 列，perf%=AscendC/TileLang，越高越接近；忽略 metadata 算子）：**swa 42.7% / cfa 49.1% / scfa ~15-16%**（HEAD `27d274c`，**brcb/row_muls V2 rescale 向量化已 green + ship**，见 §3.6；scfa 14.9 vs 旧 16.3 大概率噪声、新 session 复测）。旧 baseline swa 41.4 / cfa 48.7（`e6f2b65`）；V2-merge 向量化前两刀 `fa63798`(wide-add)/`b30b447`(broadcast-mul) 回退，第三刀 brcb/row_muls 成了。
- 自 SUCC9（swa 37.0 / cfa 42.8 / scfa 15.9）以来，靠 cube debarrier + V1/V2/normalize 向量化/debarrier 把 **swa +4.4、cfa +5.9** 推上来；scfa（lockstep + 离散 gather）基本平。

## 1. 最后一刀已验：V2 full-tile add 回归 decode → 已回退（dsv4 `b68dffa`）

**结论**：dsv4 `fa63798`（V2 merge 逐-head add 折成一个 range-slice full-tile add）NPU **decode 回归**——`swa 97.3% / cfa 98.5%`（要 99.5%，max rel err 2.0）；fast prefill 套件 + scfa decode 仍绿。已回退：`b68dffa` 的 kernel 逻辑与验过的 `e6f2b65` **逐字节一致**（只多了记录死路的注释）。

- **根因（已坐实，非编译器 bug）**：`binary_op` 给 full-tile add 发的 intrinsic 与逐-head 循环**完全相同**（`ascend_add`，ptr=`acc_o+hbase*D`，size=`MERGE_HEADS*D`，src=`acc_o_ub`；`_handle_buffer_load` 的 Ramp 假设成立，`:`→`Ramp(0,1,D)`，否则 size=16 会错 ~94% 而非 2.7%）。真正的病：**合并成一条宽 VEC op 后**，对 `acc_o_ub` 的单条宽读比 16 条窄读**排空慢**，与下一 pass 的 `T.copy(ws_o, acc_o_ub)`(MTE2) **抢**（debarrier 路径这条跨-pass WAR 没 flag 守）。decode 跑很多次 V2-merge pass 才暴露；fast prefill 跑得少没踩到。**注意：§1 原先预判的 build / `size must be same` 失败模式没发生——是更隐蔽的跨-pass 同步竞争。**
- **fork `d789b93`（range-slice operand 支持）保留**：additive、当前无 in-kernel user 故 inert；§3-A 的 rescale broadcast 需要它。**没打 `cfeat-*` tag**（§4 纪律：要有 NPU 验过的 user 才打；现在没有）。
- **容器复验回退绿（应是走形式，逻辑 == e6f2b65）**：
  ```
  cd /sdb/yq/dsv4 && git pull
  pytest -q sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv_fast.py
  pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode and dtype0"
  python sparse_attn_sharedkv_perf_compare.py   # 应回到 swa 41.4 / cfa 48.7 / scfa 16.3
  ```
- **教训（带进 §3-A）**：把 debarrier 的**逐行 op 合并成一条宽 op** 会把原本被窄 op 时序掩盖的**跨-pass WAR 暴露**出来。合并时要么宽 op 后补一条 drain/flag，要么用**不跨 pass 复用**的独立 buffer。
- **下一步**：直接做 §3-A（rescale mul broadcast 向量化，更大的刀），按上面教训配同步。

## 2. 已完成的 lever stack（SUCC9 → 现在，都已 NPU 验，除 §1）

| 刀 | gate | 效果 |
|---|---|---|
| cube debarrier MM1+MM2（barrier_all→m/fix/mte2 pipe flag）| 全 | scfa lockstep 收紧；swa/cfa cube 变快但被 vector 卡(平衡) |
| V1 softmax max-subtract 向量化（broadcast `m_i`→`m_i_brd`，**别名空闲 `kv_ub_multi`**）| cube_direct | swa 37→40，cfa 44→46 |
| V2 merge debarrier（删逐-head 的 ~96 barrier/chunk）| cube_direct | swa 40→41，cfa 46→47 |
| normalize debarrier（删 64 barrier/slot）| cube_direct | cfa 47→48.7 |
| ~~V2 full-tile add via 编译器 range-slice~~（**已回退** `b68dffa`）| cube_direct | 回归 decode（跨-pass WAR，见 §1/§7）|

**关键模式**：① broadcast scratch 别名 cube_direct 下空闲的 `kv_ub_multi`（无 vector gather → 闲）；② 所有 debarrier/向量化都 **gate 到 `cube_direct`**（swa/cfa），SCFA 保留带 barrier 的逐行形式（lockstep 里 debarrier/broadcast **会共振**，§5 + skill「broadcast 行 sub」反例）。

## 3. 下一步 lever（按优先级）

- **A. V2 rescale mul broadcast 向量化 —— ⚠️ 已试，被 AscendC Broadcast 宽-dst 卡住（暂搁）**。逐-head `acc_o[h]*=alpha[h]` → broadcast `alpha[MERGE_HEADS]`→`[MERGE_HEADS,D]` + full-tile mul。编译器已就绪（fork `1a70fed` 让 `broadcast` 吃 BufferLoad 切片，**dump 实锤 lowering 正确**：axis=1 / extent `[16,512]` / src-ptr=`alpha+(pv2*32+mp*16)` 全对，buffer 地址也不重叠）。**但 dsv4 `b30b447` 上 NPU prefill 回归 swa/cfa ~87%**（max rel 2.0）。根因（dump + 失败算术坐实）：**AscendC `Broadcast` 不填满 512-宽 dst，每行尾部 ~一个 64-block 留旧值** → `4257616/65536 行 ≈ 65/512 列错 = 12.7%`。V1 softmax broadcast 只 128 宽（`m_i_brd[32,128]`）→ 阈值下、填满、所以一直没事。**要成立得先解 Broadcast 宽-dst（compiler/库层调查，不是 kernel 侧）** —— 或换 `brcb` / 把 D 切 ≤128 列块广播（变复杂、收益打折）。已回退 `9b073f5`，per-head 形式留着。
- **B. KV 滑窗复用 —— ⚠️ 作废（前提错，见 §3.5）**。Ascend C **没有** KV 滑窗复用：grouped kernel，KV 每 (group×window) 只流式读一次。cube mte2 22% 是 KV 流式（被流水/L1-ring 隐藏），不是冗余重载。别去做 L1 滑窗。
- **C. 编译器：内存规划器**（腾 UB）。kernel 顶到 178.3K/192K 是**这版**布局撑的（不是物理上限，AscendC 同硬件留得出空间）。更紧的 hidden-tmp / liveness 复用 → 腾出空间给 broadcast buffer + 跨 slot 双缓冲 → **解锁最大的杠杆**（V2 mul broadcast 无需 time-share、跨 slot 流水填气泡）。大 codegen 改（.cc，要重装 .so）。

**瓶颈画像（swa profile，post §2）**：Duration 3.93ms，两核各 ~95% util、~25% 内部气泡；vector 略长板（pipe-sum 2.83 vs cube 2.72）。**要 80%（≤2.375ms）得：vector 削（A）+ cube 削（B，KV 复用）+ 消气泡（跨 slot 流水）三者叠加，且都卡 UB/L1 资源墙 → 所以走编译器破墙（C）**。诚实说：纯 kernel 侧便宜刀快用尽，~50% 一带是 kernel 侧天花板，破 80% 要 C。

## 3.5 Ascend C SWA 性能方案蓝本（源码研究 2026-06-15，4 agent 交叉验证）

读了 `ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/swa_*` 全套（kernel/cube/vector + op_host/tiling）。**两条纠正了之前的误解：**

1. **没有 KV 滑窗复用 —— §3-B 前提错，作废**。这是 grouped/prefill kernel：一个 M-block = 整个 query group，SWA 窗口每个 (group×window) **只流式读一次、读完即弃**，相邻 query token 不复用 KV 窗口。真正「载一次复用多次」的是 **Q**（每 group 载一次 L1，所有 KV chunk 复用，最后一个 n-loop 才放槽；`swa_block_cube.h:545-589`）。所以 cube mte2 22% 不是冗余重载，是 KV 流式 + 被流水 / L1-ring 隐藏。
2. **rescale 用 `Brcb`+`RowMuls`，不是 `Broadcast` —— lever-3A 的正解**。`alpha=exp(m_prev-m_new)` 由 `SoftmaxFlashV2`（config `SOFTMAX_OUTPUT_WITHOUT_BRC`）直接输出成**每行标量 `[rows,1]`**（`swa_block_vector.h:353`）；rescale = `Brcb`（每行标量铺成 1 个 32B / 8-fp32 block，`:687`）+ 自定义 `RowMuls`（`Mul` src1 stride-0，按 64-fp32 repeat 走完 512 列 **+ 显式标量尾段 remainder**，`:691,:769-848`）。**那个 remainder 正是我们 `Broadcast([16,512])` 缺的尾巴 64 列** → lever-3A 正确复刻 = `Brcb` + 带 remainder 的 strided mul，**全程不用 AscendC Broadcast**。

**Ascend C SWA 核心优化（按 perf 权重）：**

- **① 全局 3-深跨核软件流水（headline，我们最大的 gap）**：一条 `PreloadPipeline`（`swa_kernel.h:741-769`），3-slot task ring（`extraInfo[3]`，`gloop%3`），`gloop` 单调计数 **跨 query slot / batch 永不 reset**。一 step 内：cube 做 MM1(n)‖MM2(n-1)，vector 做 softmax(n-1)‖merge(n-2)，**3 个 (slot,chunk) task 同时在飞**。**只在 core 的最后一个 task 才 drain**（`extraLoop=isEnd?PRELOAD_NUM:0`，`:726`），**slot 边界不 flush** → slot k 尾 chunk 和 slot k+1 头 chunk 在同一 ring 重叠。GM workspace 按 `loop%PRELOAD_NUM`(=2) 双缓冲。c:v=**1:2 MIX**（1 cube 配 2 vector，每个 256 行 M-tile 两 AIV 对半分；`.cpp:48`）。**我们的港：V0(t)/V1(t-1)/V2(t-2) 跨 chunk 流水已有，但每 slot flush（「slot 间不流水」），c:v 是 1:1。**
- **② cube/vector 内存分治 —— §3-C 的真相**：Ascend C 是 MIX kernel，**GEMM operand + 其双缓冲全在 cube 独占的 L1（QP×4=256K + KV×3=192K = 448/512K）+ L0（A/B/C 各 ×2 ping-pong）**；UB（192K，用 176.5K，~8% 余量，**和我们一样满**）只装 vector 的 softmax/merge working set。**「余量」不在 UB，在 cube 的 L1/L0**。所以 §3-C 不是「省 UB」，是「GEMM 双缓冲别占 UB + cube 侧 KV-L1 做 3-深 ring 预取」。
- **③ 轻量 vector 工作（降 pipe-sum）**：`SoftmaxFlashV2` 融合 max-sub/exp/sum；Brcb-rescale；`actualColumnCount==0` 快路径跳全-mask chunk（`:356,:584`）；**unitFlag 融合 L0C drain**（最后 K-slice 翻 `unitFlag 0b10→0b11`，fixpipe 重叠下一 mmad，无显式 PipeBarrier，`swa_block_cube.h:577-581`）；持久 event id（一次 alloc）。

**Tile/常量对照**：Ascend C s2BaseSize=**512**（cube 内 N_SPLIT 128×4）、M_SPLIT=128、K_L1_SPLIT=256/K_L0=128、D_SPLIT=256、mBaseSize=gSize=64、headDim=512 固定、c:v=1:2。我们 BI=128。

**修正后破 80% 路线（取代上面 §3-A/B/C 旧框架）**：(a) chunk 流水**延伸到跨 slot**（去 per-slot flush，slot loop 喂同一 skewed pipeline）→ 填 ~25% bubble（约到 ~55%）；(b) **vector work 复刻 Ascend C 轻量 idiom**（Brcb rescale、融合 softmax、全-mask 快路径、unitFlag drain）→ 降 pipe-sum 才上 80%。**两者都要；本 session 那种发明式单-op 微优化不是路。** 便宜先做 = Brcb rescale（顺带正解 lever-3A、降 vector scalar），大头 = 跨-slot 流水。

## 3.6 ✅ Brcb rescale 已 ship（green，swa 41.4→42.7 / cfa 48.7→49.1，HEAD `27d274c`）

复刻 Ascend C 的 V2 rescale（§3.5 第 2 条）成了：16 个逐-head 标量 mul → 1 `brcb` + 1 `row_muls`/pass（§3.5 ③「轻量 vector idiom」第一刀）。fork tag `cfeat-brcb-row-muls`(`d7add81`)。

**最终 kernel 形态**（cube_direct V2 merge）：`brcb(alpha_brd8[16,8], alpha[abase:abase+16], 2,1,8)` → `pipe_barrier("v")` → `row_muls(acc_o[hbase:hbase+16,:], 同, alpha_brd8, 16,512,512)` → `pipe_barrier("v")` → 逐-head add → `barrier_all()`。`alpha_brd8`(512B) annotate 到 `kv_ub_multi+16KB`（cube_direct only）。

**编译器侧（fork，已 tag）**：① `common.h` 加 `tl::ascend::row_muls<T>`（镜像 Ascend C RowMuls）。② 注册 `tl.ascend_brcb`/`tl.ascend_row_muls` builtin（`ascend.h` 声明 + `ascend.cc` `TIR_DEFINE_TL_BUILTIN` kOpaque + `codegen_ascend.cc` `PrintOpCall`，模板名走 args[0]）。③ brcb/row_muls 吃 BufferLoad 切片。

**踩坑链（5 个坑，全进 §7）**：Broadcast 宽-dst = 厂商 bug → 换 brcb/row_muls → BufferLoad handling → **`call_extern` 被 codegen 丢弃**（必须注册 builtin，这就是 brcb「NOT implemented」的真意）→ 缺 `PIPE_V` 同步（brcb→row_muls RAW）→ **adds→下一 pass `T.copy(ws_o,acc_o_ub)` 的 VEC→MTE2 WAR**（row_muls 时序新暴露的 fa63798 同款，要 full barrier 或 V→MTE2 flag，`pipe_barrier("v")` 守不住）。诊断法：3× barrier_all 绿 → 是同步不是 row_muls bug → 再收回轻屏障。

---

## 3.7 新 session 接着干什么（按优先级）

**A. 收尾/小修（快）**：
1. **复测 scfa — ✅ 结清**：重跑得 scfa **16.5%**（= 旧 16.3；之前那个 14.9 是噪声），无回归。证实 brcb/row_muls 不碰 scfa（per-head barriered、非 cube_direct），`alpha_brd8` 无条件 alloc（512B）也没扰动 scfa 布局。
2. **3rd 同步轻量化 — ✅ VERIFIED GREEN（`d877ee4`，origin/main）**：V2 merge 那条 `barrier_all()`（adds→下个 mp `T.copy(ws_o→acc_o_ub)` 的 VEC→MTE2 WAR）换成 `T.set_flag("v","mte2",2)` + `T.wait_flag`（eid 2 free，0/1 是 prefetch ping-pong；back-to-back 平衡无 leak；MTE2 in-order 故下个 mp 的 copy 自动后置）。**这是复刻 Ascend C、不是偏离**：`swa_block_vector.h:689-696` 的 rescale 同步 = `PipeBarrier<PIPE_V>`（VEC→VEC RAW，= 我们 2 条**没动**的 `pipe_barrier("v")`：brcb→row_muls、row_muls→add）+ **`SetFlag<V_MTE2>`**（VEC→MTE2 WAR，= 这条新 flag）。**A2 前那条 `barrier_all()` 才是偏离**（Ascend C 此处不做全核 barrier）。WAR 不能用 `pipe_barrier("v")`：跨 pipe（VEC 写完 acc_o_ub 前 MTE2 不能覆盖），PIPE_V 只排空 VEC、漏 MTE2 → 之前纯 PIPE_V = 97%（§7）。**NPU 已验**：fast ✅ + `decode and dtype0` ✅；perf swa **42.8** / cfa **49.3**（原 42.7/49.1，**噪声内、perf 未真动**）—— 价值 = 同步忠实 Ascend C（去掉一个不该有的全核 barrier），非 perf 刀；回归兜底才回退成 `barrier_all`。**push 备忘**：直推 main 的 `git push` 要**单独成一条** Bash 命令（别与 `git add`/`commit` 串成复合，否则整条以 `git add` 开头匹配不上 `Bash(git push *)` allow 规则、落到 auto-mode classifier 被软拒；与仓库归属无关）。
3. **补 tag — ✅ 已做**：`d789b93`（tile-op BufferLoad operand）已打 `cfeat-tile-op-bufferload` 并 push 到 fork。green user = brcb/row_muls 复用 `_handle_buffer_load`（§3.6 ship 绿，brcb src / row_muls dst-src0 都是 BufferLoad 切片）。tag message 已注明：binary/unary 的 BufferLoad **分支本身**仍无直接 green user（其 fa63798 user 因跨-pass WAR 回退、非 BufferLoad lowering 之过），但 helper 经 brcb/row_muls 已 NPU 验。`1a70fed`（broadcast BufferLoad）仍无 green user（lever-3A 回退）、不打。

**B. §3.5 ③ 轻量 vector idiom — 调研完毕（4-agent + 源码比对），只剩 SoftmaxFlashV2 一条有 headroom，已选定**：
- `actualColumnCount==0` 全-mask 快路径 → **已处理/no-op**：SWA `NI_total=1`（单 chunk 永不空）、CFA cmp 是 dense `[0,topk_cmp)`，没有空 tile 可跳。
- cube `unitFlag` 融合 fixpipe drain → **低优**：cube 非瓶颈（vector pipe-sum 2.83 > cube 2.72），且 cube barrier 已被 vector 流水遮蔽。
- **`SoftmaxFlashV2` 融合 → 真 headroom，但是编译器 builtin**：我们 V1 softmax 是 ~8 个独立 VEC pass（`kernel.py` 1294-1351:reduce_max/max/exp(alpha)/broadcast/sub/exp/reduce_sum/rescale），Ascend C 一个 `SoftmaxFlashV2`（`swa_block_vector.h:349-355`，config `SOFTMAX_OUTPUT_WITHOUT_BRC`）全融合。攻当前长板（vector），收益 ~几个点（被 cube 变长板封顶）但忠实复刻 + 攒 tag。

  **① 编译器 plumbing 已 land（fork `0473fb3`，ascendc_pto，inert）**：`tl.ascend_softmax_flashv2` builtin，brcb/row_muls 同款 4 处注册 —— `common.h` 加 `tl::ascend::softmax_flashv2<T>` wrapper（→ `SoftMaxFlashV2TilingFunc` + `SoftmaxFlashV2`，namespace-scope `constexpr SoftmaxConfig WITHOUT_BRC`，`kernel_operator.h` 用 `__has_include` 守不破别的 kernel）+ `ascend.h`/`ascend.cc`（decl + `TIR_DEFINE`，`-1`/kOpaque）+ `codegen_ascend.cc`（emit 7 buffer + 3 scalar，`{1,8}{8,N}`）+ `ascend_tile.py`（`T.tile.softmax_flashv2`）。无 green user 故未打 tag。
  **② kernel wiring ✅ 已做（dsv4 `78838ea`，cube_direct only，SCFA 留 manual chain，py_compile+ruff 绿、本地无法 build/NPU 验）**：1294-1351 换成一个 `T.tile.softmax_flashv2(acc_s_ub, sumexp, m_i, alpha_exp, sumexp_i_ub, m_i_prev, softmax_tmp, v_block, BI, BI)`；删了 `m_i_brd`（fused op 内部做 max-subtract），其 16KB slot 给 `softmax_tmp`，加 `alpha_exp[ub_len]`，都 cube_direct annotate 到 idle kv_ub_multi（disjoint）。SCFA else 分支逐字保留（只去掉死的 m_i_brd broadcast）。**③ 是当前唯一待办**。**buffer 映射（SoftmaxFlashV2 要 in≠out 双缓冲 max/sum）**：in_max=`m_i_prev` / out_max=`m_i`（留 1287 的 `copy(m_i,m_i_prev)` 喂 prev）；in_sum=`sumexp_i_ub`（复用空出的 chunk-sum scratch）/ out_sum=`sumexp`（加 `copy(sumexp,sumexp_i_ub)` 喂 prev）；exp_out=新 `alpha_exp[ub_len]`→ `copy` 到 `alpha[pv1*ub_len:]`（替 1314）；tmp=复用 cube_direct 下空出的 `m_i_brd` 16KB slot reinterpret uint8（fused op 内部做 max-subtract，m_i_brd 不再用）；scale `mul(acc_s_ub,softmax_scale)` 1288 保留（Ascend C ElewiseCompute 单独、在 SoftmaxFlashV2 前）。把 1294-1351 整段 gate 成 `if cube_direct: softmax_flashv2 else: <manual chain>`。
  **③ 容器验（build-gated 未知，本地无 Ascend toolkit 全验不了）**：先 `git pull`(fork+dsv4) → 重装 .so（`USE_ASCEND=True pip install -e . --no-build-isolation`）→ build smoke-test（坑:`kernel_operator.h` 路径、`constexpr SoftmaxConfig` NTTP、`AscendC::` 限定、tmp 尺寸）→ dump 确认 `tl::ascend::softmax_flashv2` 出现 → fast + `decode and dtype0`（**数值核对 WITHOUT_BRC 语义 == manual chain**，尤其 alpha=exp(prev-new) 符号、sum 是否内部已乘 alpha）+ perf。绿 → 打 `cfeat-softmax-flashv2`。

**C. 大头 = §3.5 ① 跨-slot 软件流水（项目，破 80% 唯一路）**：前提 `acc_o`（64KB UB，单缓冲）→ GM workspace（slot-parity 双缓冲，像 Ascend C `vec2ResGm`），腾 64KB UB + 解锁 slot k 尾叠 slot k+1 头（SWA `NI_total=1`，跨-slot 才有并行）。slot loop = `kernel.py:530`，blocker = acc_o-in-UB（本 session 用 agent map 过，blocker/carry-state 清单见对话）。all-or-nothing 多轮高风险、本地不能验，得用户拍板投入。

**北极星不变**（§0）：复刻 Ascend C 方案到 80–100%，别发明 kernel 侧小技巧；过不去改 fork 编译器、打 `cfeat` tag。

**预期**：16 标量 mul/pass → 1 brcb + 1 row_muls/pass，降 V2 vector scalar（§3.5 ③「轻量 vector idiom」第一刀）；perf 增益中等，但正解 lever-3A + 攒 2 个编译器特性。

### 3.6 实测进展（第一轮失败，根因 = `call_extern` 被 codegen 丢弃）

第一轮上 NPU：rescale **整个没了** —— dump `grep -c "brcb|row_muls" = 0`，V2 merge 只剩 `AscendC::Add(acc_o, acc_o, acc_o_ub)`，没有 rescale mul（SWA fast 这里其实 >1 chunk，所以缺 rescale 就红 ~96-97%）。踩坑链：
1. row_muls C++ 模板编译装上了（`.so` 不报错 = C++ 写对了）。
2. BufferLoad handling（brcb src / row_muls dst-src0 是 range slice）—— 已加。
3. alpha_brd8 placement —— **红鲱鱼**：annotate 到 kv_ub_multi+16KB，失败率只微动（96.27→97.06）。因为 brcb 被丢 → alpha_brd8 是死 buffer，失败率动只是布局扰动。
4. **真根因**：`brcb`/`row_muls` 走 `T.call_extern`，**这套 Ascend codegen 直接丢弃 call_extern**。能用的 tile op 全是**注册的 `tl.ascend_*` builtin**。**brcb 当初的「NOT implemented」警告就是这意思**（C++ 模板在、TIR codegen 没接）。

**剩余修法（.cc，要重装 .so）—— 镜像 `ascend_broadcast`（已定位 4 处）**：
1. `src/op/ascend.h`（~169 行旁）：`TVM_DLL const Op &ascend_brcb();` + `ascend_row_muls();`
2. `src/op/ascend.cc`（~1116 行旁）：`TIR_DEFINE_TL_BUILTIN(ascend_brcb).set_num_inputs(-1).set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));` + row_muls 同。
3. `src/target/codegen_ascend.cc`（~563 行旁的 if-else 链）：`else if (op->op.same_as(tl::ascend_brcb())) { BrcbCodegen(op); }` + 写 `BrcbCodegen`/`RowMulsCodegen` 发 `tl::ascend::brcb<T>(...)` / `tl::ascend::row_muls<T>(...)`（镜像 `BroadcastOpCodegen`）。
4. `tilelang/language/ascend_tile.py`：brcb/row_muls 改用 `tir.call_intrin("handle", tir.op.Op.get("tl.ascend_brcb"), ...)`（dtype 由 codegen 的 `<T>` 出，去掉字符串拼）。
5. 重装 .so → dump 确认 `tl::ascend::brcb`/`row_muls` 出现在生成码 → fast+decode+perf → 打 tag。

**现状**：kernel 已回退 per-head 绿（`9333fda` == `b68dffa`）；fork 留着 row_muls C++ 模板 + Python wrapper + BufferLoad（inert，等 builtin 注册）。kernel-side rescale（alpha_brd8 + brcb/row_muls 调用）等 builtin 接上再重贴（git `bb643fc` 有原版）。

## 4. 编译器修改纪律（用户定，必守）

- **兼容性铁律**：改编译器**必须 additive / 向后兼容**，别破坏现有路径（Buffer/BufferRegion 等已有分支原样保留），**别影响其他用 tilelang 写算子/用编译器的人**。`d789b93` 范例：只**新加** BufferLoad 分支。
- **每个特性 NPU 验过 → 打 `cfeat-<slug>` annotated tag**（why/what 写成可提 issue 的程度），`git push origin <tag>`。**供日后逐个提 issue 到上游 `tile-ai/tilelang-ascend`**。
- 已打 5 个：`cfeat-gm-l1-subblock-write`(52ad83a)、`cfeat-reduce-tmp-half`(9a0d62d)、`cfeat-is-subtile-runtime-extent`(025ef5c)、`cfeat-brcb-row-muls`(d7add81，§3.6)、**`cfeat-tile-op-bufferload`(d789b93，本 session §3.7 A3)**。`d789b93`（tile-op 吃 BufferLoad 切片）的 green user 经 brcb/row_muls 复用 `_handle_buffer_load` 坐实（binary/unary 分支的**直接** user `fa63798` 因跨-pass WAR 回退，但 helper 本身已 NPU 验，tag message 已注明）。**仍没打：`1a70fed`(broadcast BufferLoad)** —— 唯一 user（`b30b447` broadcast-mul）因 AscendC 宽-dst bug 回退、无 green user；additive/inert，留着（§3-A 一旦解了 Broadcast 宽-dst 就能用）。
- **fork = `/Users/yzmac/Documents/WorkContent/tilelang-ascend`**（唯一一份；`dsv4/tilelang-ascend` 重复 clone 已删；坏过的留作 `tilelang-ascend.broken` 可删）。改 .py 容器 `git pull` 即生效；改 .cc/.h 才 `USE_ASCEND=True pip install -e . --no-build-isolation` 重装。

## 5. 环境 / 命令

- 内核：`dsv4/sparse_attn_sharedkv_tilelang/kernel.py`（JIT，改即生效）。编译器：fork（见 §4）。本地无 NPU + 无 tilelang，**只能 py_compile+ruff，真验在容器**。
- 快测（dev loop，秒级，默认跑）：`pytest -q sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv_fast.py`（S1=1024、bf16、3 prefill 场景，覆盖多块分页/边界拆分/scfa 多块离散 gather/多核）。全量门禁：`-k prefill --runslow`（8K）。decode：`-k "decode and dtype0"`。
- perf：`python sparse_attn_sharedkv_perf_compare.py`（dsv4 根目录跑；只计时，正确性以 pytest 为准）。
- profile：`msprof --output=./prof --aic-metrics=PipeUtilization --application="python sparse_attn_sharedkv_perf_compare.py --scenarios swa_prefill --only tilelang --warmup 5 --iters 3"`。看 cube/vector 的 aicore_time、pipe 分解、气泡（aicore_time − pipe_sum）、cube_utilization。
- 工作流：改完 `ruff format`+`ruff check --fix`+`py_compile`，commit+push（英文 commit，结尾 Co-Authored-By；正文回复中文）。dsv4 push main；fork push ascendc_pto。

## 6. commit 地图

**dsv4 main**（HEAD `9b073f5`，== `b68dffa` 逻辑）：`9b073f5` 回退 `b30b447`（V2 rescale broadcast；AscendC Broadcast 不填满 512-宽 dst，prefill 回归 12.7%）← `b30b447`（**已回退**，lever 3-A）← `30b92de` docs ← `b68dffa` 回退 `fa63798`（V2 full-tile add 跨-pass WAR 回归 decode；逻辑回到 `e6f2b65`）← `fa63798`（**已回退**）← `e6f2b65` normalize debarrier ← `532a2d9`/`7e55000` V2 merge debarrier ← `c4a75fc`/`3640c2b`/`70f784e` V1 向量化的 scfa-不回归修复（m_i_brd 别名/scoping/parse 几轮）← `f26da2f` V1 max-subtract 向量化 ← `30341dd` cube MM2 debarrier ← `0043a37` cube MM1 debarrier ← `83544be` handoff_5 §5 profiling ← `20be4ea`/`cc6c02e`(SUCC9) CFA cube-direct。

**fork ascendc_pto**（HEAD `1a70fed`）：`1a70fed` `broadcast` 吃 BufferLoad 切片（lowering dump 验证正确；用它的 lever 3-A 因 AscendC Broadcast 宽-dst 回退，无 green user、未 tag）← `d789b93` tile op（binary/unary）吃 BufferLoad 切片（`_handle_buffer_load`；用它的 fa63798 回退，无 green user）← `025ef5c` is_subtile runtime-extent ← `9a0d62d` reduce-tmp /2 ← `52ad83a` GM→L1 子块写。

## 7. 关键坑（本 session 血泪，别重蹈）

- **tvmscript 把 `if cube_direct:` 块作用域化**：块内 alloc 的 buffer 在块外（annotate/用处）「未定义」→ 顶层无条件 alloc + **条件 annotate**（dict-unpack `**(...)` 不支持 → 用第二个 `T.annotate_address` 调用，annotate 累加语义）。
- **别名 buffer 进无关场景的 IR 会扰动它**：`m_i_brd` 别名 `kv_ub_multi`、即使 scfa 不写它，也让编译器在 scfa 的 gather 周围加保守同步（+4ms）→ **别名按场景条件化**（scfa 根本不声明该 buffer）。
- **范围切片 `acc_o[a:b,:]` 在 tile-op arg 求值成 `BufferLoad`**（T.copy region 上下文里才是 BufferRegion）→ 编译器加 BufferLoad 分支（`d789b93`）。单行 `acc_o[i,:]` 不受影响（本来就过）。
- **debarrier/broadcast 只在 cube_direct gate**，scfa（lockstep）保留——否则共振（局部赚而 Duration 涨，§5 四连证）。
- **编译器源码 working tree 坏了也能从对象库读**：`git show <ref>:path`；坏 checkout 直接重 clone（`--no-recurse-submodules`，改 .py 用不上 submodule）。
- **把 debarrier 的逐行 VEC op 合并成一条宽 op，会暴露被窄 op 时序掩盖的跨-pass WAR**（fa63798 血泪）：16 条 `acc_o[h]+=acc_o_ub[h]` 合成一条 `acc_o[hbase:hbase+16]+=acc_o_ub`，intrinsic 完全等价（已核 `binary_op`），但宽读 `acc_o_ub` 排空慢，与下一 pass 的 `T.copy(...,acc_o_ub)`(MTE2) 抢 → **decode 回归 97-98%**（prefill 跑得少没踩到）。合宽 op 时补 drain/flag，或别跨 pass 复用同一 buffer。已回退 `b68dffa`。**判读**：失败比例 1-3%、两个 dtype 都崩、max rel err 2.0 整齐 → 结构性 bug 不是尾噪声；fast 套件绿不代表覆盖（V2 merge 要 `t>=2` 多 chunk 才跑，decode 才压满）。
- **AscendC `Broadcast` 不填满宽 dst（512）—— 每行尾部留旧值**（lever 3-A 血泪）：broadcast `[N]`→`[N,512]`(axis=1) 只填了 ~前 448 列，最后 ~64-block 留上一次的值 → rescale 乘了错的 alpha → prefill swa/cfa ~87%（`4257616/65536 行 ≈ 65/512 列错 = 12.7%`，max rel 2.0）。**128 宽（V1 softmax `m_i_brd[32,128]`）填得满、没事**；512 宽崩。lowering 正确（`get_kernel_source` dump 实锤 axis/extent/ptr 全对、buffer 地址不重叠）—— 是 AscendC Broadcast op 本身的宽-dst 上限。要做宽广播得先查它（或切 ≤128 列块）。已回退 `9b073f5`。
- **`get_kernel_source()` dump 大法 + 失败算术 这次立功**（[[reference-get-kernel-source]]）：先 dump 排除 lowering（broadcast 的 axis/extent/ptr、buffer 地址全对）→ 锁定不是 codegen；再用「失败元素数 / 行数 = 每行错几列」（65/512）直接指向「按列尾部没填」。**本地无 NPU 时，dump + 失败形态比盲改快得多**——写完改动先 dump 验同步/extent/地址，别盲烧 NPU。
- **TileLang tile op 不发射？→ 它是不是走 `call_extern`**（brcb/row_muls 血泪）：`T.call_extern` 在这套 Ascend codegen 会被**丢弃**（dump `grep -c fn_name`=0、生成码里没有就是它）；能用的 tile op 全是**注册的 `tl.ascend_*` builtin** —— 加一个要 4 处：`ascend.h` 声明 `const Op& ascend_X()` + `ascend.cc` `TIR_DEFINE_TL_BUILTIN(ascend_X).set_attr<TCallEffectKind>(kOpaque)` + `codegen_ascend.cc` else-if 调 `PrintOpCall`（模板名 `f"X<{dtype}>"` 当 args[0] 传、codegen 前缀 `tl::ascend::`）+ Python 用 `tir.call_intrin(Op.get("tl.ascend_X"), ...)`。**brcb 的「NOT implemented」假警告就是这意思**（C++ 模板在、TIR 没接）。
- **同步原语分层（别动不动 `barrier_all`）**：`T.pipe_barrier("v")` = AscendC `PipeBarrier<PIPE_V>`（同-pipe VEC drain，轻）；`T.set_flag(src,dst,eid)`/`T.wait_flag` = 跨-pipe flag（VEC↔MTE2 等，轻）；`T.barrier_all()` = 全 pipe + 跨核（重）。**VEC→VEC RAW**（brcb→row_muls）用 `pipe_barrier("v")`；**VEC→MTE2 WAR**（row_muls/adds 写完 → 下一 pass `T.copy` 抢 acc_o_ub）`pipe_barrier("v")` **守不住**，要 `barrier_all` 或 `set_flag("v","mte2")/wait_flag`。Ascend C 的分层（`PipeBarrier<PIPE_V>` ×3 + `SetFlag<V_MTE2>`，`swa_block_vector.h:689-696`）是模板。
- **诊断「同步 vs 真 bug」**：可疑同步问题 → 临时把相关同步全换 `barrier_all` 看绿不绿。绿 = 同步覆盖不足（再逐个收回轻屏障）；还红 = 真算法/codegen bug（回退）。
- 每刀通用手段记进 tilelang-perf/pitfalls skill（源仓库 + 缓存两处，MEMORY 有约定）。
