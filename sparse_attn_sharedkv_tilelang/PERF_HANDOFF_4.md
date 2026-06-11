# sparse_attn_sharedkv TileLang 性能优化 — 工作交接 #4（cube-direct + 编译器补丁 + fork prefill 回归）

> 续 `PERF_HANDOFF_3.md`（§1–§11 是 S2b/S2c/FUSE/微基准的完整历史）。本文件自包含，读完即可接着干。
> 配合 `PERF_HANDOFF_3.md §11`（fork prefill 回归详记）+ MEMORY.md（`project-tilelang-fork` / `project-fork-prefill-regression`）+ tilelang plugin 的 skill。

---

## 0. 一句话现状

- **目标**（用户定）：把 TileLang 版 sparse_attn_sharedkv 前向 perf 做到 AscendC 的 **80–100%**（当前基线 36.9ms ≈ 18.6%；AscendC 6.87ms）。
- **HEAD**：dsv4 main = `a27c565`（kernel = cube-direct SWA + 逐行 SCFA/CFA 老路）。当前最佳态另存 branch `probe-current-9922`。
- **编译器 fork**：`yangqiang2018/tilelang-ascend` 分支 `ascendc_pto`，HEAD = `52ad83a`（我们的 GM→L1 子块写补丁，净效果就这一个 commit）。
- **验证状态**：**decode 三场景(scfa/swa/cfa) × 两 dtype(bf16/fp16) 全绿**。**prefill 全场景 <99.5% 借线**（scfa 99.22%）。
- **⚠️ 最重要的结论**：**prefill 借线是 fork 环境回归，不是我们的内核/编译器工作**。Ground-truth 探针（`87df937`，把 kernel 换成 session 前"prefill 验过"的 1d-β `d9d6552`，同 fork 上跑）只有 **98.95%，比当前 99.22% 还差**。⇒ fork（领先旧 tilelang-ascend 37 commit）的上游改动改了数值，连旧基线都跌破阈值；decode 不敏感（全窗口 mask 全 1）所以全绿，prefill（部分窗口走 -inf 屏蔽）才暴露。

---

## 1. ⭐ 立刻要做（接手第一件事）：triage fork 的 prefill 数值回归

**这是 ship 的 blocker，且与 perf 优化正交。** prefill 要重新过 99.5%，三条路（按推荐序）：

1. **bisect fork 的 37 个上游 commit**（`5d3fcc9` 是我们 fork 时的基线；旧 tilelang-ascend 上 1d-β prefill 是过的）。
   - 容器里：`cd /app/data/tilelang-ascend && git log --oneline <旧基线>..5d3fcc9`（注意是 fork 基线之前的上游历史；fork 本身基于上游某点）。
   - 高嫌疑：动 codegen 数值/round/cast/reduce tmp 的 commit。`git log --oneline --grep -iE "round|cast|precision|reduce|softmax|fp32|accum"`。
   - 二分时用**最快的单 case**：`pytest ...-k "prefill and dtype0 and scfa" --runslow`（一次 ~几分钟）。
2. **把 GM→L1 子块补丁打到原 0.1.9 源**（容器原装 tilelang 0.1.9，若它 prefill 过）。代价：要在 0.1.9 上重做 §3 的补丁（codegen 结构可能不同）。
3. **暂以 decode 验证为准**，prefill 借线挂账，先推进 perf（cube-direct 的 SWA 收益用 decode + perf compare 量）。

**判据**：随便挑一个旧 fork commit（比如 `5d3fcc9`）checkout 编译装上、跑 1d-β prefill。若过 → 回归在 `5d3fcc9` 之后（但那是 fork HEAD，矛盾，说明回归更早）；若不过 → 回归在我们 fork 基线之前的上游。**最干净是直接 bisect 到具体 commit。**

> 注意：我们的子块补丁（`52ad83a`）对 whole-block 路径**字节级等价**（SCFA/CFA 走的就是 whole-block），所以补丁本身不是 prefill 回归源——已用「探针 1d-β + 补丁 fork = 98.95%」证实。

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
- `a27c565` **HEAD** docs(fork prefill 回归记录)
- `356912c` 当前最佳 kernel（= cube-direct + 逐行；`probe-current-9922` 同此）
- `1ee7873` 退回逐行 ori gather（95.69→99.22，FUSE-V0 是 prefill bug）
- `88b9151`/`85ed7cb` 退回 FUSE-V1/V2 逐行 + 删 brc_tmp/mask_full
- `cc06dfa` 修死锁（back-flag drain 改 `not cube_direct`）
- `tag s2-forward-balance-36.9`(=`230a551`) **纯调度最优回退点 36.9ms**（cube-direct 之前，FUSE 之前）

**fork ascendc_pto**（编译器，可回退到 `5d3fcc9` = 无补丁）：
- `52ad83a` **HEAD** GM→L1 子块写最终补丁（净效果就这个）
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

## 9. 通往 80% 的路线（prefill 回归解决后）

1. **解 prefill 回归**（§1）——前置，否则 cube-direct 验不了 prefill。
2. **量 cube-direct SWA 收益**：prefill 三场景过 + `perf_compare swa_prefill`。SWA 把 KV 同步整删，应大跳（27.6% 起）。
3. **扩 cube-direct 到 CFA cmp**（dense 连续，cube 标量读 cmp_block_table 直拷 GM→L1）。SCFA topK 离散留 vector。
4. **若仍不够 80%**：跨核握手深度（ws_* 多缓冲 + cube 提前一拍，复刻 PreloadPipeline）——但注意 §5 的 lockstep 教训，单刀会共振，要整体重排工作分配。
5. 每刀通用手段记进 tilelang-perf/pitfalls skill（源仓库 + 缓存两处，MEMORY 有约定）。
