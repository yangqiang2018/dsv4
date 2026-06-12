# sparse_attn_sharedkv TileLang 性能优化 — 工作交接 #4（cube-direct + 编译器补丁 + fork prefill 回归）

> 续 `PERF_HANDOFF_3.md`（§1–§11 是 S2b/S2c/FUSE/微基准的完整历史）。本文件自包含，读完即可接着干。
> 配合 `PERF_HANDOFF_3.md §11`（fork prefill 回归详记）+ MEMORY.md（`project-tilelang-fork` / `project-fork-prefill-regression`）+ tilelang plugin 的 skill。

---

## 0. 一句话现状

- **目标**（用户定）：把 TileLang 版 sparse_attn_sharedkv 前向 perf 做到 AscendC 的 **80–100%**（当前基线 36.9ms ≈ 18.6%；AscendC 6.87ms）。
- **HEAD**：dsv4 main = `acb9026`（kernel = cube-direct SWA **已启用**（paged 边界拆分）+ 逐行 SCFA/CFA 老路）。配 fork `025ef5c`（is_subtile runtime-extent 修复）。
- **编译器 fork**：`yangqiang2018/tilelang-ascend` 分支 `ascendc_pto`，HEAD = `9a0d62d`（GM→L1 子块写补丁 `52ad83a` + #978 reduce-tmp 修复）。
- **验证状态**：**decode + prefill 三场景全绿**（swa 用 cube-direct，paged 边界 bug 已修）。**perf（sharedkv，%AscendC）**：swa 37.0% / cfa 30.7% / scfa 16.1%（目标 80%）。
- **✅ prefill 回归已解决**：真凶 = upstream **#978**(`65a22c5`)把我们 ascendc 路径的 reduce tmp 翻倍，撞极紧的手工 UB → 污染 prefill；修复 = fork 保持 `/2`(`9a0d62d`)。⚠️ 曾误判为 #1002(AscendWorkspaceReduction)——那是**旧-.so 假信号**(只改 Python 没 `pip install`)，已洗清。**教训：NPU 任何省事信号必须在干净重装 .so 上复现。**

---

## 1. ✅ 已解决：fork 的 prefill 数值回归（真凶 #978）

**曾是 ship blocker，现已结案。** 真凶 = upstream **`65a22c5`(#978 "change ascendc reduce tmp buffer size")**：把 `allocate_tmp_buffer.cc` 的 `GetAscendCTmpBufferSize_`（我们 ascendc/auto 路径）`ascend_reduce` 分支 tmp 尺寸从 `args[3]*bytes()/2` 改成 `*bytes()`（翻倍）。我们 kernel 的 online-softmax reduce（`T.reduce_max`/`T.reduce_sum` on fp32 `acc_s_ub`）隐藏 tmp 翻倍，撞极紧的手工 UB 布局（`kernel.py:255-260`，只剩 ~13.7K tail）→ 布局移位/别名污染 prefill 的 -inf 路径；decode mask 全 1 不敏感。

- **修复（已 ship）**：fork 保持 `/2` —— `9a0d62d`（`src/transform/allocate_tmp_buffer.cc`）。验证：decode 全绿 + scfa/cfa prefill 全绿。fork-local 取舍：上游那个 fp32 gemv 例子会回退，但不是我们的算子。
- **⚠️ 定位教训（必读）**：曾误判为 #1002(`AscendWorkspaceReduction`)，因为"注释 `phase.py:72` → prefill pass"。那是**旧-.so 假信号**——只改 Python 没 `pip install`，跑的是重装前的 .so；干净重装后 pass-off 根本不修复（且该 pass 反而帮忙）。**NPU 任何省事信号必须在干净重装的 .so 上复现；真凶是靠"当前 fork 上逐个 revert 嫌疑 + 干净重装"定位的。** #1002 的 opt-out 全部撤回（dsv4 `d9dec74`、fork `2b2a3c3`）。
- 已排除红鲱鱼：`1e763f4`(#1027 TROWSUM)是 `GetPTOTmpBufferSize_` pto 专用；`577d34c`(#1000 cast)只动 bf16→fp32 无损上转。完整 triage 见 `PREFILL_TRIAGE_NOTES.md`。

### ✅ cube-direct SWA prefill 已修复并提速（2026-06-12）

`swa_prefill` 走 cube-direct 全 pass，且 decode + scfa/cfa prefill 全绿。**根因 = paged block 边界读错**：cube-direct 16 行 GM→L1 拷贝每 pass 只查一次 block table，prefill 窗口起点 `ori_left` 非 block 对齐时，`rowc=g0%block>block-16` 的跨界 pass 跨 2 个分页 block 却只读一个 → 读错物理块（对照 AscendC `DataCopyPA` 的边界分段实锤；早先 kv_lo WAR-race 假设是错的——cube 顺序执行无 race）。
- **修复=两处**：① fork **`025ef5c`**（`ascend.cc`：`is_subtile` 对 **runtime extent** 也判 sub-tile→跳过整块 clear，**关键**）；② dsv4 **`acb9026`**（`kernel.py`：cube-direct 加载在 block 边界拆两段，非跨界 pass 仍编译期 16 行拷）。
- **(a) 失败教训**：最初 kernel 级拆分（`f33240e`）拆分逻辑+zN 地址其实都对（zN col-0 偏移线性 `16*r`），唯一错是 runtime extent → `as<IntImmNode>()` null → is_subtile=false → 整块 clear 从 runtime 偏移**越界清** → 73.8%。is_subtile 一修就成立 ⇒ **没用大改 (b)（移植 DataCopyPA）**。
- **perf（cube-direct 生效后，sharedkv 列，perf%=AscendC/TileLang）**：swa_prefill **37.0%**（~18.6% 基线翻倍，超预测的 27.6%）；scfa 16.1% / cfa 30.7%（仍老路）。距 80% 仍有空间，下一杠杆见 §9。

---

## 2. ✅ 已完成并 decode 验证：cube-direct KV（SWA，KV 不过 vector）

**这是对账 AscendC 查出的最后一块未复刻结构**：AscendC 的 `DataCopyPA`（`ops-transformer/.../sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_common.h:137`）是 **cube 侧**手写块拷，GM→L1，**KV 全程不过 vector 核**——没有 V0 gather、没有 ws_kv 往返、没有 KV_READY 握手。这是 SWA 它 1.87ms / 我们 6.8ms 的结构性差距来源。

- **落地**（kernel.py，`cube_direct = (NI_cmp == 0)` 门控，仅 SWA）：cube 的 MM1 ori chunk 直接 `T.copy(ori_KV[blk, row0:row0+16, 0, :], kv_lo[pa, gp*16:..., :])` GM→L1，4 趟 16 行块拷覆盖 64 行；vector 的 ori gather/ws_kv/KV_READY 全删（SWA 链上 0 个 KV 同步）。
- **SCFA/CFA**：`cube_direct=False`，走**逐行老路**（vector gather 写 ws_kv + KV_READY），与 1d-β 一致。
- **依赖编译器补丁**：cube-direct 写 L1 是**子块行写**（16 of 64 行），需要 fork `52ad83a` 的 GM→L1 子块支持（见 §3）。
- **下一步扩展**：CFA 的 cmp 也是 dense 连续索引，理论上也能 cube-direct（cube 标量读 cmp_block_table 直拷）。SCFA 的 topK 离散索引只能留 vector 路。**做之前等 prefill 回归解决**，否则验不了。

---

## 3. ✅ 编译器补丁（fork `52ad83a`）：GM→L1 子块行写

**问题**：`copy_gm_to_l1` 模板按整块 NZ(zN) 布局拷，且对 tail 调 `InitConstValue` **清整块**。cube-direct 的子块写（gp=0 清全 64 行、gp=1~3 在独立调用写 16~63 行）→ 清零与兄弟趟的写 WAW 竞争，几乎全清成 0。

**修法（最小补丁，3 文件）**：
- `src/op/ascend.cc`（gm2l1 分支）：检出子块（dst 行 extent < dstM，常量比较），传 `noClear=1`；**dst 指针完全用自然 OffsetOf**（NZ 偏移本就对，gp*256 = zN GetOffset，别再算）。whole-block 路径 `noClear=0`。
- `src/tl_templates/ascend/common.h`（`copy_gm_to_l1`）：加 `uint32_t noClear=0` 参；`if (noClear==0 && tail mismatch) InitConstValue(...)`——子块跳过清零（4 趟合起来覆盖全块，本就不需清）。
- `src/target/codegen_ascend.cc`：`{"copy_gm_to_l1", 4}`（extra_args 4：strideN/validRow/validCol/noClear）。

**踩过的坑（别重蹈，全在 fork commit log）**：
- `c1fbca9`：c0 用 `dst->dtype.bytes()` = lower 后 storage 化的 1 字节 → 偏移翻倍 gp*512。**dst dtype 在 lower 后是 1B**。
- `82a84bb`→`e370037`→`52ad83a`：一度把 nz_off 既塞 dst 指针又传 rowOffset 让模板再进一次 → **双偏移 gp*512**。**生成码 `get_kernel_source()` 一看就抓到**（见 §6 调试法）。最终：自然 OffsetOf 一直对，只需 noClear。
- zN 子块地址几何正确性：用 `3rdparty/catlass/include/catlass/layout/matrix.hpp` 的 `zN::GetOffset` 实算证实，行 r0(16 倍数) col 0 → `(r0/16)*256 + (r0%16)*16` = r0*16（bf16 C0=16）。

**重编**：C++ 改动要 `cd /app/data/tilelang-ascend && git pull && USE_ASCEND=True pip install -e . --no-build-isolation`（codegen 编进 .so，**改 .cc 必重装**；改 header `common.h` 因 kernel JIT 重编可能 pull 即可，但稳妥起见一起重装）。验证 codegen 生效：`get_kernel_source()` 看 `copy_gm_to_l1` 的实参。

---

## 4. ❌ 本 session 试过并回退的（别再走）

- **FUSE-V1/V2/V0（broadcast 整块 select/sub/mul/add + ori 块 gather）**：decode 验证过、vector 6.49→3.35ms，但 **Duration 不赚反亏（共振，见 §5）**，且 **prefill 有 bug**（FUSE-V0 块 gather 页边界错 95.69→99.22;FUSE-V1 整块 select 部分窗口错;brc_tmp/mask_full 别名 acc_o_ub 头跨 chunk 竞争）。**全部退回逐行**（`85ed7cb`/`88b9151`/`1ee7873`）。`T.tile.broadcast` 在 fork 里**存在可用**，但整块化的 perf 被共振吃掉、正确性坑多，**不值**。
- **S2b.1e（V2 debarrier）/ S2c（PreloadPipeline skew）**：见 #3 §2，局部加速触发 cross-flag 接力相位失配，Duration 全部反涨。**四连证：lockstep 流水里任何局部刀都反伤**（已记 tilelang-perf skill）。

---

## 5. 关键结论：36.9ms 是当前调度结构的脆弱平衡点

S2b/S2c/FUSE/V2 四次独立实验全证：**核内 pipe 重叠机制全验通(pipe-sum 29 > aiv 22.5)，但 Duration 不动**——瓶颈在**跨核 lockstep**（每 chunk 5 cross-flag 接力，两核各 ~61% busy，gap≈14ms 恒定）。**cube-direct 是唯一动了结构的刀**（SWA 把 KV 同步整条删掉），是奔 80% 去的正解。继续在核内 barrier/skew/fuse 上动刀没肉。

**微基准成本模型**（`bench_microop.py`，已验）：整块 op 44ns vs 拆 32 行 501ns（11×）；DMA 纯带宽限制（16×2K=1×32K≈166ns，合并无肉）；flag/barrier≈0。

---

## 6. 调试方法（本 session 验证有效，写进流程）

- **`get_kernel_source()` 对账生成码**：`func = build_...(...); print(func.get_kernel_source())`。一发 shell（别进交互式，粘贴易掉字符）：
  ```bash
  python -c "from sparse_attn_sharedkv_tilelang.kernel import build_sparse_attn_sharedkv
  f=build_sparse_attn_sharedkv(batch=1,max_seq=8192,total_tokens=8192,ori_block_num=64,ori_block_size=128,ori_table_len=64,cmp_block_num=16,cmp_block_size=128,cmp_table_len=16,scenario=1,topk_cmp=0)
  s=f.get_kernel_source(); i=s.find('ASCEND_IS_AIC'); print(s[i:i+2500])"
  ```
  双偏移、stale build、参数顺序错——全靠这个一眼定位。**比盲改+NPU 跑快一个量级。**
- **Ground-truth 探针**：怀疑"我们引入回归"时，把 kernel 换成已知good的旧 commit（`git show <sha>:path > kernel.py`），同环境跑——证伪/证实，不再猜。
- **decode pass ≠ 正确**：decode 是末位 query、窗口全满、mask 全 1，**-inf 屏蔽路径根本不走**。**mask/窗口相关的改动必须 prefill 验**。

---

## 7. commit 地图

**dsv4 main**（可回退）：
- `acb9026` **HEAD** cube-direct SWA 启用（paged 边界拆分；配 fork `025ef5c` is_subtile 修复）→ swa 37%
- `e26d2fb`/`50328f1` 中途 fallback + revert（`f33240e` 拆分尝试，缺 is_subtile 修复故失败）
- `9012315`/`0a9d130` docs（prefill triage 校正到 #978）
- `a27c565` docs(fork prefill 回归记录)
- `356912c` cube-direct + 逐行 kernel（`probe-current-9922` 同此）
- `1ee7873` 退回逐行 ori gather（95.69→99.22，FUSE-V0 是 prefill bug）
- `88b9151`/`85ed7cb` 退回 FUSE-V1/V2 逐行 + 删 brc_tmp/mask_full
- `cc06dfa` 修死锁（back-flag drain 改 `not cube_direct`）
- `tag s2-forward-balance-36.9`(=`230a551`) **纯调度最优回退点 36.9ms**（cube-direct 之前，FUSE 之前）

**fork ascendc_pto**（编译器，可回退到 `5d3fcc9` = 无补丁）：
- `9a0d62d` **HEAD** #978 reduce-tmp 修复（`allocate_tmp_buffer.cc` 保持 `/2`）
- `52ad83a` GM→L1 子块写补丁
- `5d3fcc9` fork 基线（无我们的补丁）

---

## 8. 环境 / 文件 / 命令

- 内核：`sparse_attn_sharedkv_tilelang/kernel.py`（改这）。编译器：`tilelang-ascend/`（fork，改 codegen/模板）。
- 容器 NPU：`/sdb/yq/dsv4`（pull 跑测试）+ `/app/data/tilelang-ascend`（编译器，git pull + pip 重装）。本地无 NPU。
- 装 fork：`USE_ASCEND=True pip install -e . --no-build-isolation`（USE_ASCEND 必设，否则探 nvcc 炸 metadata）。
- 测试：`pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode"` / `-k "prefill" --runslow`（慢）。阈值 99.5% within tol。
- perf：`python sparse_attn_sharedkv_perf_compare.py --scenarios swa_prefill --only both`（⚠️ 它只计时不一定校验正确性，正确性以 pytest 为准）。
- 工作流（MEMORY）：改完 `ruff format`+`ruff check --fix`+`py_compile`，主动 commit+push dsv4；commit 英文、结尾 Co-Authored-By；正文回复中文。**改完代码主动 push 不用等**。

---

## 9. 通往 80% 的路线（prefill 回归已解决）

1. ✅ **解 prefill 回归** —— 已完成（§1，真凶 #978，fork `9a0d62d`）。
2. ✅ **cube-direct SWA prefill 已修 + 量到 37%**（§1）：边界拆分（dsv4 `acb9026`）+ is_subtile runtime-extent 修复（fork `025ef5c`）。swa sharedkv 18.6%→37%。
3. **⭐下一棒：扩 cube-direct 到 CFA cmp**（dense 连续，cfa 30.7%→应像 swa 跳）。cmp cube-direct 桩已在 `kernel.py` `if False and is_cfa`（启用 + 加同样的边界拆分 + gating 放开 CFA）。**注意跨核重构**：cmp 的 vector 端（createvecindex/gather/KV_READY）当前不按 cube_direct gate，CFA 启用 cube-direct 后会和 cube 的 cmp 加载冲突，需 gate 掉 + 防 deadlock（性质类似大改，多轮 NPU 迭代）。SCFA topK 离散留 vector。
4. **若仍不够 80%**：跨核握手深度（ws_* 多缓冲 + cube 提前一拍，复刻 PreloadPipeline）——但注意 §5 的 lockstep 教训，单刀会共振，要整体重排工作分配。
5. 每刀通用手段记进 tilelang-perf/pitfalls skill（源仓库 + 缓存两处，MEMORY 有约定）。
