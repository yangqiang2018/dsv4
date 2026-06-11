# prefill 数值回归 triage —— 本地静态分析（接 PERF_HANDOFF_4.md §1）

> 本文件是 §1 的本地推进结果：**无 NPU 也能做的部分已做完**（把 66 个上游 commit 收窄到 3 个真正经过我们路径的数值嫌疑，并排除头号红鲱鱼）。
> 实跑验证要在容器（`/sdb/yq/dsv4` + `/app/data/tilelang-ascend`）按下面"行动计划"做。

---

## 0. 一句话

把 §1 的"bisect fork 上游 commit"从 **66 个盲 bisect** 收窄到 **3 个真正经过我们 (ascendc/auto) 路径的数值嫌疑**，并**排除了头号红鲱鱼 `#1027`（TROWSUM row-reduce）—— 它改的是 pto 专用函数，根本不碰我们的路径**。先做一步零重编诊断即可分流。

---

## 1. 范围已钉精确

- **旧基线（prefill 验过的点）= upstream tile-ai `2e27af7`（PR #968）**。
  证据：sibling 旧克隆 `WorkContent/tilelang-ascend` 的 HEAD `0de76b6` 是本地 merge upstream，其 `0de76b6^2`（upstream parent）= `2e27af7 (#968)`。
- **fork 基线 = `5d3fcc9`（#1128 TopK fix）**，其上叠我们 7 个 GM→L1 补丁到 `52ad83a`。
- **bisect 范围 = `2e27af7..5d3fcc9` = 66 commits**（PERF_HANDOFF_4 写的 ~37 是约数，以 66 为准）。

---

## 2. 关键甄别：我们走 non-pto 路径 → 大半 commit 是红鲱鱼

- kernel 用 **default non-pto Ascend C lowering**（`kernel.py:15` 明示；`T.Kernel(is_npu=True)`）。
- 编译器 `src/transform/allocate_tmp_buffer.cc` 的 `createTmpBuffer_` 按 target 分流：
  - `"pto"` → `GetPTOTmpBufferSize_`
  - `"ascendc"/"auto"` → `GetAscendCTmpBufferSize_`（**我们走这条**）
- ⇒ **所有纯 `[PTO]` codegen / `GetPTOTmpBufferSize_` 改动对我们字节级无关**，66 个里大半出局。

### ❌ 已排除的头号红鲱鱼
- **`1e763f4` (#1027) "Fix PTO row-reduce temporary buffer size and type mismatch for TROWSUM"** —— 名字最像（TROWSUM = softmax 分母），但它改的 `GetPtoRowReduceTmpCols` / TROWSUM 分支在 **`GetPTOTmpBufferSize_`（pto 专用）**，我们的 ascendc 路径走 `GetAscendCTmpBufferSize_`，**不受影响**。别在它上面浪费 NPU 时间。

---

## 3. 真正经过我们路径的数值嫌疑（按机制排序）

bug 画像：**借线小漂移（scfa 99.22%，差阈值 0.28pt），decode 全绿、prefill 才挂**。decode↔prefill 唯一功能差 = prefill 走 -inf 屏蔽（部分窗口、行内含 exp(-inf)=0、tail 处理）。小漂移更像 **cast/round 精度** 或少量元素污染，而非布局越界（越界会是灾难性 fault 不是 99.22%）。

| 序 | commit | 改动 | 文件:行 | 机制 | dtype |
|---|---|---|---|---|---|
| A | `577d34c` (#1000) | `copy_ub_to_ub` 新增 float→bf16 cast 分支 | `src/tl_templates/ascend/common.h` (`copy_ub_to_ub`) | 小幅精度漂移，正好 99.22% 量级 | **仅 bf16** |
| B | `65a22c5` (#978) | reduce tmp buffer `dtype.bytes()/2` → `dtype.bytes()`（**翻倍**） | `allocate_tmp_buffer.cc:773`（`GetAscendCTmpBufferSize_` 的 `ascend_reduce()` 分支，**git blame 坐实是 #978**） | 隐藏 reduce scratch 翻倍 → UB 布局移位/越界踩相邻别名 buffer。⚠️ 但 `kernel.py:255-260` 注解显示作者**已用 178.3K 布局给 scratch 让位**，溢出可能已补偿 → 存疑 | dtype 无关 |
| C | `4f4a060` (#1118) | `copy_gm_to_ub` pad 门控加 `\|\| (maskShapeN*sizeof(T))%32==0` | `src/tl_templates/ascend/common.h:~196` | 32B 对齐宽度改走非 pad 路径 → partial-window/tail（prefill 专属）数据可能带进 pad 区 | dtype 无关 |

次要（结构非数值，低优先）：`3573c21`(#969 T.copy dynamic shape, `codegen_ascend.cc`)、`4477f9a`(#1002 workspace reduction pass)、`1bc1002`(#1102 cross_core_pipeline if-fix)。

我们 kernel 确实用 `T.reduce_max`(`kernel.py:1029`) / `T.reduce_sum`(`kernel.py:1063`) 做 online-softmax 行规约 → lower 成 `ascend_reduce()`，正是 #978 改尺寸的那个 op。

---

## 4. 容器行动计划（按性价比，从省到贵）

### 步骤 0 —— 零重编诊断（先做，免费分流嫌疑）
当前 fork HEAD 上，prefill **两 dtype 都跑**：
```bash
cd /sdb/yq/dsv4
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "prefill and scfa" --runslow -v
```
- **只 bf16(dtype0) 挂、fp16 过** → 直指 **A (`#1000` cast)**。
- **两 dtype 都挂** → 指向 **B (`#978`) / C (`#1118`)**（dtype 无关）。

### 步骤 1 —— 假设优先单点测（每个 = 1 行 revert + 重编 + 跑 fast case）
重编：`cd /app/data/tilelang-ascend && USE_ASCEND=True pip install -e . --no-build-isolation`
fast case：`pytest ...-k "prefill and dtype0 and scfa" --runslow -q`

- 若步骤0指向 **bf16-only** → 先查 **A**：临时去掉 `copy_ub_to_ub` 里 `(std::is_same_v<T1,float> && std::is_same_v<T2,bfloat16_t>)` 那个 case，看 bf16 cast 路径是否变化（⚠️ 删了可能 fall-through 编译失败，这种就改成对照不同 cast 实现来测）。
- 若步骤0是 **两 dtype 都挂** → 先 revert **B**：`allocate_tmp_buffer.cc:773` 把 `src_buffer_node->dtype.bytes();` 改回 `src_buffer_node->dtype.bytes() / 2;`，重编跑。⚠️ #978 本是 bugfix（某 op 需要更大 buffer），revert 仅作诊断，若 prefill 回 >99.5% 即坐实，再设计不踩别名的正解。
  - B 不中 → revert **C**：`common.h:~196` 把 `if (maskShapeN == dstN || (maskShapeN * sizeof(T)) % 32 == 0)` 改回 `if (maskShapeN == dstN)`，重编跑。

### 步骤 2 —— 兜底 `git bisect`（仅当上面都不中）
```bash
cd /app/data/tilelang-ascend
git bisect start 5d3fcc9 2e27af7
git bisect run bash -c '
  cd /app/data/tilelang-ascend &&
  USE_ASCEND=True pip install -e . --no-build-isolation -q || exit 125 &&
  cd /sdb/yq/dsv4 &&
  pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "prefill and dtype0 and scfa" --runslow -q
'
```
⚠️ **caveat**：66 commit 跨度大，1d-β 内核未必每个中间 commit 都能 JIT 编译（gather/T.copy/atomic 等 API 漂移）。kernel JIT 失败会让 pytest 报 ERROR(非 FAIL)→ bisect 误判为 bad。`exit 125` 只挡 pip 装失败，挡不了 JIT 失败。所以**优先步骤 0/1**，bisect 当最后手段，且遇可疑结果手动 `git bisect skip`。

---

## 5. 关键 SHA 速查
- 旧基线：`2e27af7`(upstream #968) ｜ fork 基线：`5d3fcc9`(#1128) ｜ fork HEAD(含补丁)：`52ad83a`
- 红鲱鱼（别碰）：`1e763f4`(#1027 pto-only TROWSUM)
- 嫌疑 A/B/C：`577d34c`(#1000) / `65a22c5`(#978) / `4f4a060`(#1118)
- 本地仓：fork = `dsv4/tilelang-ascend`（branch `ascendc_pto`）；旧基线快照 = `WorkContent/tilelang-ascend`（HEAD `0de76b6`，有 `upstream` remote）
