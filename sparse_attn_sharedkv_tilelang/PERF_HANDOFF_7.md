# sparse_attn_sharedkv 性能优化 — 工作交接 #7（C 跨-slot 流水进行中：S1+S2 验绿，S3a 待实现）

> 续 `PERF_HANDOFF_6.md`（C 可行性 §3.8、Ascend C 蓝本、S3 full 设计 §3.9 都在那；深细节回查它）。本文件自包含，读完即可接着干。配 `MEMORY.md`。
> **当前在做 C（跨-slot 软件流水）。S1+S2 已验绿，S3a 是下一步。**

## 0. 一句话现状 + 北极星

- **目标（用户北极星）**：复刻 Ascend C 这个算子的计算逻辑 + 性能方案，让 swa 前向做到 AscendC 的 **80–100%**。Ascend C 源码是蓝本，别发明 kernel 侧小技巧；过不去改 fork 编译器（`yangqiang2018/tilelang-ascend`，NPU 验过的成功改动打 `cfeat-*` annotated tag，攒着提上游 issue）。
- **诚实天花板**：vector 侧便宜刀**已用尽**（brcb/row_muls rescale、SoftmaxFlashV2 fused softmax 都 NPU 验过 = perf 中性，见 _6 §3.6/3.7）。**~50% 是 kernel 侧天花板**。破它只剩 **C（跨-slot 软件流水）**。**注意：即使 C 全成也就 ~55，够不着北极星 80**——80 还得叠 cube 侧（_6 §3.5② L1-ring 预取，又一大改）。**用户已知情拍板「硬上 C」**（知道 ~16-25 容器轮、最好 ~55、砸了全回退）。
- **perf 现状**：swa **35.6** / cfa **42.3** / scfa 15.0（S1+S2 后）。**比 C 前 baseline swa 42.4 / cfa 49.9 降了 ~7**——这是 S1 把 acc_o 挪 GM 的纯 DMA 成本，**所有恢复+增益押在 S3**。
- **进度**：S1 ✅、S2 ✅ 都容器验绿。**S3a 是下一步实现**（第一个真 overlap + 硬 gate）。

## 1. 立刻干：S3a（vector deferred-tail skew，硬 gate）

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
