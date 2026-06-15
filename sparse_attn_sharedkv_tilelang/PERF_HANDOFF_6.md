# sparse_attn_sharedkv 性能优化 — 工作交接 #6（编译器层 + lever stack）

> 续 `PERF_HANDOFF_5.md`（cube-direct 收官于 SUCC9，profiling 定根因）。本文件自包含，读完即可接着干。
> 配 MEMORY.md（`feedback-compiler-feature-tags` / `project-fork-prefill-regression` / `project-tilelang-fork` / tilelang-perf skill）。

---

## 0. 一句话现状

- **目标**（用户定，北极星）：**完全复刻 Ascend C 这个算子的计算逻辑 + 性能优化方案**，让 TileLang 前向 perf 做到 AscendC 的 **80–100%**。**Ascend C 源码是蓝本** —— perf 缺口靠「它怎么优化就怎么复刻」，**别自己发明 kernel 侧小技巧**（本 session 发明的 V2-merge wide-add / broadcast-mul 两刀都撞编译器/硬件墙回退了；§3-B KV 复用 / §3-C 内存布局才是复刻 Ascend C 的省内存 + 流水方案）。**过不去就改编译器**（fork `yangqiang2018/tilelang-ascend` 是我们的；修 bug / 加特性，每个 NPU 验过的成功改动打 `cfeat-*` annotated tag，**攒着一起给官方 `tile-ai/tilelang-ascend` 提 issue / 需求** —— 见 §4）。
- **perf**（`perf_compare`，sharedkv 列，perf%=AscendC/TileLang，越高越接近；忽略 metadata 算子）：**swa 41.4% / cfa 48.7% / scfa 16.3%**（最后完整验证 = dsv4 `e6f2b65`；现 HEAD `9b073f5` kernel 逻辑与之逐字节一致 —— V2 merge 向量化两刀 `fa63798`(wide-add) + `b30b447`(broadcast-mul) 都回退了，见 §3-A/§6/§7）。
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

## 3.6 下一步 build：Brcb rescale（用户选定 2026-06-15；跨-slot 大手术暂缓）

复刻 Ascend C 的 rescale（§3.5 第 2 条），正解 lever-3A 的 Broadcast 宽-dst 墙、降 V2 vector scalar。**用户在「跨-slot 大手术 vs Brcb」里选了 Brcb（受控、可验、攒 tag）**；跨-slot 流水（前提 `acc_o`→GM，all-or-nothing 多轮高风险、本地不能验）记为大项目待启。

**关键事实（已查）**：
- fork `brcb` 的 **C++ runtime 已存在**（`src/tl_templates/ascend/common.h:789` → `AscendC::Brcb`）；只是 Python wrapper（`ascend_tile.py:716`）挂着**过时的「NOT implemented」假警告**。un-stub 即可。
- `Broadcast`（`common.h:834`）是**纯转发 `AscendC::Broadcast`**，宽-dst bug 在**厂商库里、我们改不了** → 必须走 brcb+RowMuls 绕开（Ascend C 就是这么做的，全程不碰 Broadcast）。
- Ascend C `RowMuls` = `swa_block_vector.h:769-848`：逐行 `AscendC::Mul`，`BinaryRepeatParams` 的 `src1BlkStride=0/src1RepStride=0`（把 brcb 出的 8-lane block 沿列广播）+ 列分 `dLoop=actualCol/64` 个 repeat **+ `dRemain` 尾段**。我们 col=512 → `dLoop=8, dRemain=0`（512 是 64 整数倍，**根本不需要尾段**；Broadcast 之所以崩是它内部 mis-tile，RowMuls 老实发 8 个 repeat 就填满了）。

**实现清单（改 .h → `USE_ASCEND=True pip install -e . --no-build-isolation` 重装 .so）**：
1. **un-stub** `ascend_tile.py` `brcb`（去假警告；C++ 已在）。
2. **加 `row_muls` runtime 模板**到 `common.h`，照搬 `RowMuls`（fp32：`repeatElementNum=64, blockElementNum=8`；`columnCount<256*8` 分支；`dLoop<=dealRowCount` 走列-repeat；带 `dRemain` 尾段以防非-512 配置）。
3. 加 `row_muls` Python wrapper（`call_extern "tl::ascend::row_muls<T>"`，传 dst/src0/src1 ptr + dealRowCount/columnCount/actualColumnCount）。
4. **kernel**（V2 merge cube_direct）：逐-head rescale mul → `brcb(alpha_brd8[MERGE_HEADS,8], alpha[abase:abase+MERGE_HEADS], rep=MERGE_HEADS, blk=1, rep_stride=8)` + `row_muls(acc_o[hbase:hbase+MERGE_HEADS,:], 同, alpha_brd8, MERGE_HEADS, D, D)`；**add 保持逐-head**（acc_o_ub 的跨-pass WAR，§7）。`alpha_brd8`(=MERGE_HEADS*8*4=512B) 别名 idle `kv_ub_multi`。
5. **验**：重装 .so → `get_kernel_source` dump 核 row_muls 发了 8 个 repeat 填满 512 → fast + decode + perf。
6. 绿后打 tag：`cfeat-brcb-enable`、`cfeat-row-muls`（攒着提 issue）。

**预期**：16 标量 mul/pass → 1 brcb + 1 row_muls/pass，降 V2 vector scalar（§3.5 ③「轻量 vector idiom」第一刀）；perf 增益中等，但正解 lever-3A + 攒 2 个编译器特性。

## 4. 编译器修改纪律（用户定，必守）

- **兼容性铁律**：改编译器**必须 additive / 向后兼容**，别破坏现有路径（Buffer/BufferRegion 等已有分支原样保留），**别影响其他用 tilelang 写算子/用编译器的人**。`d789b93` 范例：只**新加** BufferLoad 分支。
- **每个特性 NPU 验过 → 打 `cfeat-<slug>` annotated tag**（why/what 写成可提 issue 的程度），`git push origin <tag>`。**供日后逐个提 issue 到上游 `tile-ai/tilelang-ascend`**。
- 已打 4 个（前 3 追溯）：`cfeat-gm-l1-subblock-write`(52ad83a)、`cfeat-reduce-tmp-half`(9a0d62d)、`cfeat-is-subtile-runtime-extent`(025ef5c)。**两个 BufferLoad-切片特性 `d789b93`(binary/unary) + `1a70fed`(broadcast) 都 lowering 正确但无 green kernel user**——用它们的 levers（`fa63798` wide-add 跨-pass WAR、`b30b447` broadcast-mul AscendC 宽-dst）**都因别的原因回退了** → **都没打 tag**；等有 NPU 验过的 user 再打。两者 additive/inert，留着（§3-A 一旦解了 Broadcast 宽-dst 就能用）。
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
- 每刀通用手段记进 tilelang-perf/pitfalls skill（源仓库 + 缓存两处，MEMORY 有约定）。
