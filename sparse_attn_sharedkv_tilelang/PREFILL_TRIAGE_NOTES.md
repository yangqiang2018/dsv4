# prefill 数值回归 triage —— 本地静态分析（接 PERF_HANDOFF_4.md §1）

> 本文件是 §1 的本地推进结果（无 NPU 也能做完的部分）。实跑验证在容器（`/sdb/yq/dsv4` + `/app/data/tilelang-ascend`）按 §4 做。
> **v2（深挖后重定向）**：判断从最初的"cast/reduce 舍入"转到 **跨核 CV 结构 pass**。最初的头号 cast 嫌疑 #1000 已被代码级排除，详见 §3。

---

## ✅ RESOLVED（2026-06-11，隔离探针坐实 + 修复已推）

**根因 = `AscendWorkspaceReduction` pass（#1002 `4477f9a`, `phase.py:72`）**。用户实测：注释掉 `phase.py:72` 重 JIT → prefill scfa 恢复 ≥99.5%。机制见 §3。

**修复 = 整函数 opt-out attr `disable_workspace_reduction`**（已推，待容器验）：
- fork ascendc_pto **`03f8858`**：`AscendWorkspaceReductionPass::Substitute` 顶部守卫，带该 attr 即原样返回。
- dsv4 main **`8343436`**：`kernel.py` body 首行 `T.func_attr({"disable_workspace_reduction": True})`。
- **容器验证**：两仓 pull → fork `.cc` 改了**必须重装 .so**（`USE_ASCEND=True pip install -e . --no-build-isolation`）→ 跑 prefill 全场景两 dtype + decode 回归检查。

下面 §1–§5 是定位过程的完整记录（含已排除的红鲱鱼），保留备查。

---

## 0. 一句话（定位时）

把 §1 的 66-commit 盲 bisect 收窄到 **2 个动我们跨核 CV 机制的结构性 pass**（#1002 workspace-reduction、#1102 cross-core if-fix）+ 1 个 reduce 尺寸（#978）。其余（含名字最像的 #1027 TROWSUM、cast #1000、pad #1118）已用代码级分析排除/降权。**最便宜的第一刀**：注释掉 `phase.py:72` 的 `AscendWorkspaceReduction()`，kernel 重 JIT 跑 prefill（不用重装 .so）。→ **此刀已被用户执行并命中，见顶部 RESOLVED。**

---

## 1. 范围已钉精确

- **旧基线（prefill 验过的点）= upstream tile-ai `2e27af7`（#968）**。证据：sibling 旧克隆 `WorkContent/tilelang-ascend` HEAD `0de76b6` 的 upstream parent (`0de76b6^2`) = `2e27af7`。
- **fork 基线 = `5d3fcc9`（#1128）**，其上叠我们 7 个 GM→L1 补丁到 `52ad83a`。
- **范围 = `2e27af7..5d3fcc9` = 66 commits**（PERF_HANDOFF_4 写的 ~37 是约数）。

---

## 2. 关键甄别：我们走 non-pto 路径 → 大半 commit 是红鲱鱼

- kernel 用 **default non-pto Ascend C lowering**（`kernel.py:15`；`T.Kernel(is_npu=True)`）。
- `allocate_tmp_buffer.cc` 的 `createTmpBuffer_` 按 target 分流 `GetPTOTmpBufferSize_`(pto) vs `GetAscendCTmpBufferSize_`(我们)⇒ **所有纯 [PTO] codegen 改动字节级无关**。
- **❌ 排除头号红鲱鱼 `1e763f4`(#1027 "TROWSUM row-reduce tmp buffer fix")**：名字最像 softmax 分母,但改的是 `GetPTOTmpBufferSize_`(pto 专用)。别再查它。

---

## 3. 嫌疑排序（v2，深挖代码后）

bug 画像：借线小幅退化（scfa 99.22%，差阈值 0.28pt），decode 全绿、prefill 才挂。decode↔prefill 差异 = prefill 走更多 mask/window 的**条件控制流** + 更满的 workspace 占用。**结构性 sync/workspace 改动比舍入更贴这个画像**。

### ⭐ 头号（动我们跨核 CV 机制，且可廉价测）
| commit | 改动 | 为什么贴 | 怎么测 |
|---|---|---|---|
| **`4477f9a` (#1002)** | 新增 898 行 `AscendWorkspaceReduction` pass，`phase.py:72` **无条件**跑；注释 "Erase manual workspace allocations for **virtual CV copy** in Ascend"；pass 内按 `cid * dst_full_extent` 重导 GM workspace（`ascend_workspace_reduction.cc:242/260`） | 正对我们手工分配、按 core-id 分片的 CV 跨核 workspace `ws_kv/ws_p/ws_o`（`ws_p[cid,...]`，`workspace_idx=[13,14,15,16]`）。重导布局若与 kernel 手工访问不符→污染。prefill 吃满 workspace、更易暴露 | **注释 `phase.py:72`**（pass 在 #968 基线根本不存在，禁掉 = 回到基线该阶段行为）→ kernel 重 JIT 跑 prefill。**纯 Python 改、不用重装 .so**。若禁掉后编不过/跑不了 = kernel 依赖该 pass，改走 checkout #1002 边界对比 |
| **`1bc1002` (#1102)** | `cross_core_pipeline.cc` +370，给该 pass **新增 `IfThenElseNode` 处理**（之前收集跨核 buffer/scope 不进 if 分支） | 我们 kernel 跨核路径全是 if（`if cube_direct and t<NI_ori`、`if not cube_direct`、prefill mask 分支）。自动 sync 现在进 if 分支插/改 flag，可能与我们**手工 set_flag/wait_flag** 打架。decode 不怎么走这些分支、prefill 走得多→prefill 专属 | checkout `1bc1002` vs `1bc1002^`，各重装 .so 跑 prefill fast case |

### 〇 中等
| `65a22c5` (#978) | reduce tmp `bytes()/2`→`bytes()`（翻倍）。**git blame 坐实在我们路径 `allocate_tmp_buffer.cc:773`**（`GetAscendCTmpBufferSize_` 的 `ascend_reduce()` 分支，对应 `T.reduce_max`/`T.reduce_sum`） | 隐藏 reduce scratch 翻倍→UB 布局移位。⚠️ 但 `kernel.py:255-260` 注解显示作者已用 178.3K 布局给 scratch 让位，溢出或已补偿→存疑 | revert：`:773` 改回 `... / 2`，重装 .so 跑 |

### ❌ 已排除 / 降权（代码级理由）
- **`577d34c` (#1000) copy_ub_to_ub cast** —— **排除**。#1000 只把 `(T1=float,T2=bf16)` 加进 `CAST_NONE` 列表，即 **bf16→fp32 上转**（无损，舍入模式无关）。我们 epilogue 的是 **fp32→bf16 下转**（`T.copy(acc_s_ub, acc_s_half)`，src=fp32→dst=bf16 ⇒ `copy_ub_to_ub<T1=bf16,T2=float>`），落 `else` 分支用 `CAST_RINT`，**#1000 没碰它**。fp16 下转同理一直 `CAST_RINT`。⇒ 对两 dtype 都惰性。
- **`4f4a060` (#1118) copy_gm_to_ub pad 门控** —— **降权**。只在 `maskShapeN` 已 32B 对齐时把 `rightPadding` 关掉（对齐宽度本不需 pad），`Duplicate` pad-fill 块未动 ⇒ 数值上≈no-op。
- **`3573c21` (#969) T.copy dynamic shape** —— **降权**。codegen_ascend.cc 仅 9 行加动态 shape 路径，我们 kernel 全静态 shape。

---

## 4. 容器行动计划（v2，按性价比）

### 步骤 1 —— 禁 #1002 pass（最便宜，纯 Python，不重装 .so）
```bash
cd /app/data/tilelang-ascend
# 注释掉 tilelang/engine/phase.py:72 的 AscendWorkspaceReduction()(mod)
cd /sdb/yq/dsv4
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "prefill and scfa" --runslow -v
```
- prefill 回 ≥99.5% → **#1002 坐实**（同时拿到 workaround：对本 kernel 禁该 pass，或让 pass 跳过我们手工 workspace）。
- 编不过/跑不了 → kernel 依赖该 pass，改走"checkout `4477f9a` vs `4477f9a^` 对比"。
- 顺带零成本分流：这步跑了**两 dtype**。结构 bug 应 dtype 无关；若只 bf16 挂、fp16 过，说明另有 bf16-specific codegen（回头单查）。

### 步骤 2 —— checkout 测 #1102
`git checkout 1bc1002 && pip 重装 && 跑 prefill`；再 `git checkout 1bc1002^ && 重装 && 跑`。翻转即坐实。

### 步骤 3 —— revert #978 一行
`allocate_tmp_buffer.cc:773` `bytes()`→`bytes() / 2`，重装跑。⚠️ #978 本是 bugfix，revert 仅诊断。

### 兜底 —— 只在 our-path commit 上 bisect（不是全 66）
our-path commit（动 `codegen_ascend.cc` / `tl_templates/ascend/` / 非 pto-gated transform）只有 ~7 个：`#969 #978 #980 #1002 #1034 #1102 #1118`(+端点 #1128)。在这几个边界手动二分即可，别盲跑全 66（1d-β 内核未必每个中间 commit 都能 JIT，全 bisect 易误判）。

---

## 5. 关键 SHA / 位置速查
- 旧基线 `2e27af7`(#968) ｜ fork 基线 `5d3fcc9`(#1128) ｜ fork HEAD `52ad83a`
- 头号嫌疑：`4477f9a`(#1002, `phase.py:72` + `src/transform/ascend_workspace_reduction.cc`)、`1bc1002`(#1102, `src/transform/cross_core_pipeline.cc` IfThenElse 处理)
- 中等：`65a22c5`(#978, `allocate_tmp_buffer.cc:773`)
- 已排除：`1e763f4`(#1027 pto)、`577d34c`(#1000 cast 上转无损)、`4f4a060`(#1118 对齐 pad)、`3573c21`(#969 动态 shape)
- 本地仓：fork = `dsv4/tilelang-ascend`(`ascendc_pto`)；旧基线快照 = `WorkContent/tilelang-ascend`(HEAD `0de76b6`)
