# sparse_attn_sharedkv 性能优化 — 工作交接 #6（编译器层 + lever stack）

> 续 `PERF_HANDOFF_5.md`（cube-direct 收官于 SUCC9，profiling 定根因）。本文件自包含，读完即可接着干。
> 配 MEMORY.md（`feedback-compiler-feature-tags` / `project-fork-prefill-regression` / `project-tilelang-fork` / tilelang-perf skill）。

---

## 0. 一句话现状

- **目标**（用户定）：TileLang 前向 perf 做到 AscendC 的 **80–100%**。**需要就改编译器**（fork `yangqiang2018/tilelang-ascend` 是我们的）。
- **perf**（`perf_compare`，sharedkv 列，perf%=AscendC/TileLang，越高越接近；忽略 metadata 算子）：**swa 41.4% / cfa 48.7% / scfa 16.3%**（最后完整验证 = dsv4 `e6f2b65`；现 HEAD `b68dffa` kernel 逻辑与之逐字节一致 —— §1 那刀回退了）。
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

- **A. V2 rescale mul 也 broadcast 向量化**（接 §1）。逐-head `acc_o[h]*=alpha[h]` → broadcast `alpha[pass 的 MERGE_HEADS 个]`→`[MERGE_HEADS,D]` + full-tile mul（range-slice 现在能用了）。broadcast buffer `[MERGE_HEADS,D]`=32KB，**time-share `kv_ub_multi`**：V1 用 `m_i_brd`(16KB) 用完，V2 接着用同一块（V1(t-1)/V2(t-2) 是顺序相位，VEC pipe in-order 安全）。这刀砍 vector scalar 的大头。**注意（fa63798 教训，见 §1/§7）**：broadcast + full-tile mul 是 range-slice 宽 op，要么 broadcast buffer 不跨 pass 复用、要么宽 op 后补一条 drain/flag——否则跟 §7 的跨-pass WAR 同病（窄 op 时序碰巧掩盖、合宽就暴露）。
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

**dsv4 main**（HEAD `b68dffa`）：`b68dffa` 回退 `fa63798`（V2 full-tile add 跨-pass WAR 回归 decode；逻辑回到 `e6f2b65`）← `fa63798`（**已回退**）← `e6f2b65` normalize debarrier ← `532a2d9`/`7e55000` V2 merge debarrier ← `c4a75fc`/`3640c2b`/`70f784e` V1 向量化的 scfa-不回归修复（m_i_brd 别名/scoping/parse 几轮）← `f26da2f` V1 max-subtract 向量化 ← `30341dd` cube MM2 debarrier ← `0043a37` cube MM1 debarrier ← `83544be` handoff_5 §5 profiling ← `20be4ea`/`cc6c02e`(SUCC9) CFA cube-direct。

**fork ascendc_pto**（HEAD `d789b93`，待验）：`d789b93` tile op 吃 BufferLoad 切片（`_handle_buffer_load`）← `025ef5c` is_subtile runtime-extent ← `9a0d62d` reduce-tmp /2 ← `52ad83a` GM→L1 子块写。

## 7. 关键坑（本 session 血泪，别重蹈）

- **tvmscript 把 `if cube_direct:` 块作用域化**：块内 alloc 的 buffer 在块外（annotate/用处）「未定义」→ 顶层无条件 alloc + **条件 annotate**（dict-unpack `**(...)` 不支持 → 用第二个 `T.annotate_address` 调用，annotate 累加语义）。
- **别名 buffer 进无关场景的 IR 会扰动它**：`m_i_brd` 别名 `kv_ub_multi`、即使 scfa 不写它，也让编译器在 scfa 的 gather 周围加保守同步（+4ms）→ **别名按场景条件化**（scfa 根本不声明该 buffer）。
- **范围切片 `acc_o[a:b,:]` 在 tile-op arg 求值成 `BufferLoad`**（T.copy region 上下文里才是 BufferRegion）→ 编译器加 BufferLoad 分支（`d789b93`）。单行 `acc_o[i,:]` 不受影响（本来就过）。
- **debarrier/broadcast 只在 cube_direct gate**，scfa（lockstep）保留——否则共振（局部赚而 Duration 涨，§5 四连证）。
- **编译器源码 working tree 坏了也能从对象库读**：`git show <ref>:path`；坏 checkout 直接重 clone（`--no-recurse-submodules`，改 .py 用不上 submodule）。
- **把 debarrier 的逐行 VEC op 合并成一条宽 op，会暴露被窄 op 时序掩盖的跨-pass WAR**（fa63798 血泪）：16 条 `acc_o[h]+=acc_o_ub[h]` 合成一条 `acc_o[hbase:hbase+16]+=acc_o_ub`，intrinsic 完全等价（已核 `binary_op`），但宽读 `acc_o_ub` 排空慢，与下一 pass 的 `T.copy(...,acc_o_ub)`(MTE2) 抢 → **decode 回归 97-98%**（prefill 跑得少没踩到）。合宽 op 时补 drain/flag，或别跨 pass 复用同一 buffer。已回退 `b68dffa`。**判读**：失败比例 1-3%、两个 dtype 都崩、max rel err 2.0 整齐 → 结构性 bug 不是尾噪声；fast 套件绿不代表覆盖（V2 merge 要 `t>=2` 多 chunk 才跑，decode 才压满）。
- 每刀通用手段记进 tilelang-perf/pitfalls skill（源仓库 + 缓存两处，MEMORY 有约定）。
