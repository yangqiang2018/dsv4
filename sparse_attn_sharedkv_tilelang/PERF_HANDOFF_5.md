# sparse_attn_sharedkv TileLang 性能优化 — 工作交接 #5（CFA cube-direct）

> 续 `PERF_HANDOFF_4.md`（prefill #978 数值回归 + cube-direct SWA 已收官；深历史看 #4）。本文件自包含，读完即可接着干。
> 配 MEMORY.md（`project-fork-prefill-regression` / `project-tilelang-fork` / tilelang skill）。

---

## 0. 一句话现状

- **目标**（用户定）：TileLang 版前向 perf 做到 AscendC 的 **80–100%**。
- **HEAD**：dsv4 main = `c9f0eb8`（kernel = cube-direct SWA 已启用 + 逐行 SCFA/CFA 老路）。fork `yangqiang2018/tilelang-ascend` 分支 `ascendc_pto` = `025ef5c`（#978 reduce-tmp `/2` 修复 + is_subtile runtime-extent 修复）。
- **验证**：**decode + prefill 三场景全绿**（swa 用 cube-direct）。
- **perf**（`perf_compare`，sharedkv 列，perf%=AscendC/TileLang，越高越接近 AscendC；**忽略 metadata 算子**）：**swa 37.0% / cfa 30.7% / scfa 16.1%**（TOTAL sharedkv 20.5%）。swa 已被 cube-direct 从 ~18.6% 基线翻倍。
- **本棒任务**：**扩 cube-direct 到 CFA 的 cmp**（dense 连续索引），cfa 30.7% → 目标 ~swa 量级。SCFA 的 topK 离散索引留 vector 路（不动）。

---

## 1. ⭐ 立刻要做：CFA cube-direct

**思路**：cube-direct 提速的本质 = 砍掉 KV 的跨核同步（cube 自己 GM→L1 直拷，不过 vector gather / ws_kv / KV_READY）。swa 已这么做。CFA 的 cmp 也是 **dense 连续**（`gc0=(t-NI_ori)*BI+...`，cube 标量读 `cmp_block_table` 直拷），同样能 cube-direct。

**配方**（`kernel.py`）：
1. **gating 放开 CFA**：`cube_direct = (NI_cmp == 0) or is_cfa`（当前是 `NI_cmp == 0`，只 swa）。SCFA（`is_cfa=False` 且 `NI_cmp!=0`）仍 `cube_direct=False`，留 vector。
2. **启用 cmp 桩 + 加边界拆分**：`if False and is_cfa:`（约 `kernel.py:677`，cmp 的 lo/hi 16 行块拷桩）→ 改成启用，并**套用 §2 的 paged 边界拆分**（用 `cmp_block_size`/`cmp_block_table`，跨界 pass 拆"本块尾+下块头"）。
3. **gate 掉 cmp 的 vector 端**（⚠️ 跨核重构，deadlock 高发）：CFA 的 cmp vector 端（`createvecindex` ~`kernel.py:930`、cmp gather、`set_cross_flag(KV_READY)` ~`:1045`）当前**不按 `cube_direct` gate**——CFA 启用 cube-direct 后这些仍会跑、和 cube 的 cmp 直拷冲突（双载 / KV_READY 计数错 / 死锁）。需把它们包进 `if not cube_direct`，并保证 back-flag drain / KV_READY 计数一致（参考 swa 的死锁修复 `cc06dfa`/`44f09ac` 的做法）。
4. **ori chunk**：CFA 的 ori chunk（`t<NI_ori`）走已成的 cube-direct ori 路（`if cube_direct and t<NI_ori`，已带边界拆分），不用改。

**验证**：`pytest ...-k "cfa_prefill" --runslow -v` 过线 → 全回归（prefill+decode 三场景）→ `perf_compare` 量 cfa 收益。**注意**：fork 已含所需修复（is_subtile），改 `kernel.py` 是 JIT、不用重装 .so；只有动 fork `.cc` 才重装。

---

## 2. cube-direct paged 边界拆分机制（swa 已成，复用到 cfa）

- **bug 画像**：16 行 GM→L1 块拷每 pass 只查一次 block table（`KV[blk, rowc:rowc+16]`），窗口起点非 block 对齐时，`rowc=g0%block > block-16` 的跨界 pass 跨 2 个**分页** block 却只读一个 → 读错物理块。对照 AscendC `DataCopyPA`（`ops-transformer/.../sparse_attn_sharedkv_common.h`，while 循环"一次只处理一个 Block"、边界分段）。
- **修法**（已在 swa ori 路落地，`kernel.py` cube-direct lo/hi 循环）：非跨界 pass 走编译期 16 行拷；跨界 pass（`ori_block_size - rowc < GATHER_ROWS`）拆两段：本块 `rowc:rowc+n0`（`n0=block-rowc`）+ 下块 `0:16-n0`，dst 用 runtime 偏移 `gp*16+n0`（zN col-0 偏移线性 `16*r`，OffsetOf 算得对）。
- **关键依赖**（fork `025ef5c`，`src/op/ascend.cc`）：`is_subtile` 对 **runtime extent** 也判 sub-tile（`is_subtile = full && (!ext || ext->value < full->value)`）→ noClear=1 跳过整块 clear。**否则** runtime extent → `as<IntImmNode>()` null → is_subtile=false → 整块 clear 从 runtime 偏移**越界清** → 污染（这是 swa 拆分第一次失败 73.8% 的真因）。CFA 的 cmp 拆分同样依赖它（fork 已带，无需再改 .cc）。

---

## 3. 环境 / 命令

- 内核：`sparse_attn_sharedkv_tilelang/kernel.py`（改这，JIT）。编译器：`/app/data/tilelang-ascend`（fork，改 codegen/模板才重装）。本地无 NPU。
- 容器：`/sdb/yq/dsv4`（pull 跑测试）+ `/app/data/tilelang-ascend`（`git pull` + `USE_ASCEND=True pip install -e . --no-build-isolation`，仅 .cc/.h 改动需要）。
- 测试：`pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "cfa_prefill" --runslow -v`；全套 `-k "prefill" --runslow` / `-k "decode"`。阈值 99.5% within tol。
- perf：`python sparse_attn_sharedkv_perf_compare.py --scenarios swa_prefill --only both`（跑三场景；只计时，正确性以 pytest 为准；忽略 metadata 算子）。
- 工作流（MEMORY）：改完 `ruff format`+`ruff check --fix`+`py_compile`，主动 commit+push；commit 英文、结尾 Co-Authored-By；正文回复中文。
- ⚠️ **NPU 调试铁律**（血泪教训）：任何"省事信号"必须在**干净重装的 .so** 上复现——`pip install -e .` 不重跑就是旧 .so（曾因旧 .so 假信号误判 #1002 一整轮）。怀疑某 upstream commit 时，在**当前 fork** 上逐个 revert 嫌疑 + 干净重装，别 bisect 旧代码（旧代码 + 当前 CANN 编不动/挂死）。

---

## 4. commit 地图

**dsv4 main**：
- `c9f0eb8` **HEAD** docs(handoff)
- `acb9026` cube-direct SWA 启用（paged 边界拆分）→ swa 37%
- `0a9d130`..`e26d2fb` prefill #978 triage + cube-direct fallback 中途态（已被 acb9026 取代）

**fork ascendc_pto**（可回退 `5d3fcc9`=无我们补丁）：
- `025ef5c` **HEAD** is_subtile runtime-extent → sub-tile（cube-direct 边界拆分依赖）
- `9a0d62d` #978 reduce-tmp 保持 `/2`（prefill 数值修复）
- `52ad83a` GM→L1 子块写补丁（cube-direct 基础）

---

## 5. 其它杠杆（CFA 之后）

- **scfa 16.1%（最慢）**：topK 离散 gather，没法 cube-direct，是硬骨头。可能方向：优化离散 gather 的批量化（见 tilelang-perf skill 的批量 gather 手段），或减少其跨核 lockstep。
- **§4#5 跨核 lockstep**（PERF_HANDOFF_3/4 §5）：核内 pipe 重叠已验通但 Duration 卡在跨核 lockstep（每 chunk 多个 cross-flag 接力）。cube-direct 砍 KV 同步就是动这个的有效刀；继续往"减跨核握手深度 / cube 提前一拍（PreloadPipeline）"走，但 §5 教训：lockstep 里单刀会共振，要整体重排工作分配。
- 每刀通用手段记进 tilelang-perf/pitfalls skill（源仓库 + 缓存两处，MEMORY 有约定）。
