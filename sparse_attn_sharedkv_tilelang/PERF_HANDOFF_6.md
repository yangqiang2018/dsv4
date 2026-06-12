# sparse_attn_sharedkv 性能优化 — 工作交接 #6（编译器层 + lever stack）

> 续 `PERF_HANDOFF_5.md`（cube-direct 收官于 SUCC9，profiling 定根因）。本文件自包含，读完即可接着干。
> 配 MEMORY.md（`feedback-compiler-feature-tags` / `project-fork-prefill-regression` / `project-tilelang-fork` / tilelang-perf skill）。

---

## 0. 一句话现状

- **目标**（用户定）：TileLang 前向 perf 做到 AscendC 的 **80–100%**。**需要就改编译器**（fork `yangqiang2018/tilelang-ascend` 是我们的）。
- **perf**（`perf_compare`，sharedkv 列，perf%=AscendC/TileLang，越高越接近；忽略 metadata 算子）：**swa 41.4% / cfa 48.7% / scfa 16.3%**（最后完整验证 = dsv4 `e6f2b65`）。
- 自 SUCC9（swa 37.0 / cfa 42.8 / scfa 15.9）以来，靠 cube debarrier + V1/V2/normalize 向量化/debarrier 把 **swa +4.4、cfa +5.9** 推上来；scfa（lockstep + 离散 gather）基本平。

## 1. ⚠️ 立刻要做：验证最后一刀（还没上 NPU）

**dsv4 `fa63798` + fork `d789b93` 这对改动用户还没在容器跑过**——新 session 第一件事就是拿到它的结果。

- **改了啥**：编译器让 tile op（`T.tile.add/mul`）吃**范围切片** `buf[a:b,:]`（之前求值成 `BufferLoad` 报 `no access_ptr`）；kernel 把 V2 merge 的逐-head add 折成**一个 full-tile add** `acc_o[hbase:hbase+MERGE_HEADS,:] += acc_o_ub`。
- **验证（容器要 pull 两个 repo）**：
  ```
  cd /app/data/tilelang-ascend && git pull   # fork d789b93,.py 改 editable 装即生效,不重装 .so
  cd /sdb/yq/dsv4 && git pull
  pytest -q sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv_fast.py
  pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode and dtype0"
  python sparse_attn_sharedkv_perf_compare.py
  ```
- **预期**：增益小（add 16→1 op/pass）；真价值 = 打通「改编译器 → kernel 用」通路 + 解锁 range-slice 向量化。scfa 不变。
- **风险**：`d789b93` 的 `_handle_buffer_load`（`tilelang/language/ascend_tile.py`）**假设切片 BufferLoad 的索引是 `Ramp`**（range→base+lanes）。本地没 tilelang 没法 trace 确认结构。**若 build 报错或 `size must be same` 断言挂**，把报错贴来——多半是 `:` 那维不是 Ramp，调 `_handle_buffer_load` 一轮就好。
- **过了之后**：① 给 fork `d789b93` 打 tag `cfeat-tile-op-region-slice`（约定：NPU 验过才 tag；见 §4）。② 接着做下面 §3-A。

## 2. 已完成的 lever stack（SUCC9 → 现在，都已 NPU 验，除 §1）

| 刀 | gate | 效果 |
|---|---|---|
| cube debarrier MM1+MM2（barrier_all→m/fix/mte2 pipe flag）| 全 | scfa lockstep 收紧；swa/cfa cube 变快但被 vector 卡(平衡) |
| V1 softmax max-subtract 向量化（broadcast `m_i`→`m_i_brd`，**别名空闲 `kv_ub_multi`**）| cube_direct | swa 37→40，cfa 44→46 |
| V2 merge debarrier（删逐-head 的 ~96 barrier/chunk）| cube_direct | swa 40→41，cfa 46→47 |
| normalize debarrier（删 64 barrier/slot）| cube_direct | cfa 47→48.7 |
| **（待验 §1）** V2 full-tile add via 编译器 range-slice | cube_direct | — |

**关键模式**：① broadcast scratch 别名 cube_direct 下空闲的 `kv_ub_multi`（无 vector gather → 闲）；② 所有 debarrier/向量化都 **gate 到 `cube_direct`**（swa/cfa），SCFA 保留带 barrier 的逐行形式（lockstep 里 debarrier/broadcast **会共振**，§5 + skill「broadcast 行 sub」反例）。

## 3. 下一步 lever（按优先级）

- **A. V2 rescale mul 也 broadcast 向量化**（接 §1）。逐-head `acc_o[h]*=alpha[h]` → broadcast `alpha[pass 的 MERGE_HEADS 个]`→`[MERGE_HEADS,D]` + full-tile mul（range-slice 现在能用了）。broadcast buffer `[MERGE_HEADS,D]`=32KB，**time-share `kv_ub_multi`**：V1 用 `m_i_brd`(16KB) 用完，V2 接着用同一块（V1(t-1)/V2(t-2) 是顺序相位，VEC pipe in-order 安全）。这刀砍 vector scalar 的大头。
- **B. KV 滑窗复用**（cube `mte2` 22%，830us）。swa 相邻 query token 窗口重叠 127/128，我们每 slot 重载整窗，AscendC 复用。复杂（L1 滑窗管理）；且被 vector 平衡卡（要配 A 一起两核同降才动 Duration）。
- **C. 编译器：内存规划器**（腾 UB）。kernel 顶到 178.3K/192K 是**这版**布局撑的（不是物理上限，AscendC 同硬件留得出空间）。更紧的 hidden-tmp / liveness 复用 → 腾出空间给 broadcast buffer + 跨 slot 双缓冲 → **解锁最大的杠杆**（V2 mul broadcast 无需 time-share、跨 slot 流水填气泡）。大 codegen 改（.cc，要重装 .so）。

**瓶颈画像（swa profile，post §2）**：Duration 3.93ms，两核各 ~95% util、~25% 内部气泡；vector 略长板（pipe-sum 2.83 vs cube 2.72）。**要 80%（≤2.375ms）得：vector 削（A）+ cube 削（B，KV 复用）+ 消气泡（跨 slot 流水）三者叠加，且都卡 UB/L1 资源墙 → 所以走编译器破墙（C）**。诚实说：纯 kernel 侧便宜刀快用尽，~50% 一带是 kernel 侧天花板，破 80% 要 C。

## 4. 编译器修改纪律（用户定，必守）

- **兼容性铁律**：改编译器**必须 additive / 向后兼容**，别破坏现有路径（Buffer/BufferRegion 等已有分支原样保留），**别影响其他用 tilelang 写算子/用编译器的人**。`d789b93` 范例：只**新加** BufferLoad 分支。
- **每个特性 NPU 验过 → 打 `cfeat-<slug>` annotated tag**（why/what 写成可提 issue 的程度），`git push origin <tag>`。**供日后逐个提 issue 到上游 `tile-ai/tilelang-ascend`**。
- 已打 4 个（前 3 追溯）：`cfeat-gm-l1-subblock-write`(52ad83a)、`cfeat-reduce-tmp-half`(9a0d62d)、`cfeat-is-subtile-runtime-extent`(025ef5c)；**待打** `cfeat-tile-op-region-slice`(d789b93，§1 验过再打)。
- **fork = `/Users/yzmac/Documents/WorkContent/tilelang-ascend`**（唯一一份；`dsv4/tilelang-ascend` 重复 clone 已删；坏过的留作 `tilelang-ascend.broken` 可删）。改 .py 容器 `git pull` 即生效；改 .cc/.h 才 `USE_ASCEND=True pip install -e . --no-build-isolation` 重装。

## 5. 环境 / 命令

- 内核：`dsv4/sparse_attn_sharedkv_tilelang/kernel.py`（JIT，改即生效）。编译器：fork（见 §4）。本地无 NPU + 无 tilelang，**只能 py_compile+ruff，真验在容器**。
- 快测（dev loop，秒级，默认跑）：`pytest -q sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv_fast.py`（S1=1024、bf16、3 prefill 场景，覆盖多块分页/边界拆分/scfa 多块离散 gather/多核）。全量门禁：`-k prefill --runslow`（8K）。decode：`-k "decode and dtype0"`。
- perf：`python sparse_attn_sharedkv_perf_compare.py`（dsv4 根目录跑；只计时，正确性以 pytest 为准）。
- profile：`msprof --output=./prof --aic-metrics=PipeUtilization --application="python sparse_attn_sharedkv_perf_compare.py --scenarios swa_prefill --only tilelang --warmup 5 --iters 3"`。看 cube/vector 的 aicore_time、pipe 分解、气泡（aicore_time − pipe_sum）、cube_utilization。
- 工作流：改完 `ruff format`+`ruff check --fix`+`py_compile`，commit+push（英文 commit，结尾 Co-Authored-By；正文回复中文）。dsv4 push main；fork push ascendc_pto。

## 6. commit 地图

**dsv4 main**（HEAD `fa63798`，待验）：`fa63798` V2 full-tile add（用编译器 range-slice）← `e6f2b65` normalize debarrier ← `532a2d9`/`7e55000` V2 merge debarrier ← `c4a75fc`/`3640c2b`/`70f784e` V1 向量化的 scfa-不回归修复（m_i_brd 别名/scoping/parse 几轮）← `f26da2f` V1 max-subtract 向量化 ← `30341dd` cube MM2 debarrier ← `0043a37` cube MM1 debarrier ← `83544be` handoff_5 §5 profiling ← `20be4ea`/`cc6c02e`(SUCC9) CFA cube-direct。

**fork ascendc_pto**（HEAD `d789b93`，待验）：`d789b93` tile op 吃 BufferLoad 切片（`_handle_buffer_load`）← `025ef5c` is_subtile runtime-extent ← `9a0d62d` reduce-tmp /2 ← `52ad83a` GM→L1 子块写。

## 7. 关键坑（本 session 血泪，别重蹈）

- **tvmscript 把 `if cube_direct:` 块作用域化**：块内 alloc 的 buffer 在块外（annotate/用处）「未定义」→ 顶层无条件 alloc + **条件 annotate**（dict-unpack `**(...)` 不支持 → 用第二个 `T.annotate_address` 调用，annotate 累加语义）。
- **别名 buffer 进无关场景的 IR 会扰动它**：`m_i_brd` 别名 `kv_ub_multi`、即使 scfa 不写它，也让编译器在 scfa 的 gather 周围加保守同步（+4ms）→ **别名按场景条件化**（scfa 根本不声明该 buffer）。
- **范围切片 `acc_o[a:b,:]` 在 tile-op arg 求值成 `BufferLoad`**（T.copy region 上下文里才是 BufferRegion）→ 编译器加 BufferLoad 分支（`d789b93`）。单行 `acc_o[i,:]` 不受影响（本来就过）。
- **debarrier/broadcast 只在 cube_direct gate**，scfa（lockstep）保留——否则共振（局部赚而 Duration 涨，§5 四连证）。
- **编译器源码 working tree 坏了也能从对象库读**：`git show <ref>:path`；坏 checkout 直接重 clone（`--no-recurse-submodules`，改 .py 用不上 submodule）。
- 每刀通用手段记进 tilelang-perf/pitfalls skill（源仓库 + 缓存两处，MEMORY 有约定）。
