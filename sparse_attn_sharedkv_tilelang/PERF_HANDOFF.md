# sparse_attn_sharedkv TileLang 性能优化 — 工作交接

> 给接手的新 session 看。目标:把 TileLang 版 `sparse_attn_sharedkv`(sharedkv 那个算子)
> 的 NPU 性能从当前 **~14%** 提到 Ascend C 的 **80%+**。本文件自包含,读完即可接着干。
> 配合用户 memory(MEMORY.md)和 tilelang plugin 的 skill(tilelang-perf / -pitfalls / -debugging)一起看。

---

## 0. 一句话现状

8K(S1=8192)SCFA prefill 下,TileLang sharedkv kernel 单次 `Duration` 已从 **88.2ms → 48.8ms**
(性能 7.7% → 14.1%,= AscendC 6.87ms ÷ TileLang Duration)。下一步是**最大的一刀**:
实现 Ascend C 的**三级软件流水(双缓冲)**,这是通往 80% 的正确路。

**性能口径**:用户要的是 `性能% = AscendC延迟 ÷ TileLang延迟 × 100%`(性能=延迟倒数),不是延迟直接相除。

---

## 1. 关键文件位置

**TileLang 实现**(要优化的):`sparse_attn_sharedkv_tilelang/`
- `kernel.py` — **核心 kernel**(`@T.prim_func def sparse_attn_sharedkv`),所有优化都在这。
- `api.py` — 高层入口(layout 分发、scenario 路由、调 metadata + kernel)。
- `metadata.py` — 调度器,**纯 host Python 端口**(不是 NPU kernel,msprof 抓不到)。
- `test_sparse_attn_sharedkv.py` — pytest 测试(正确性,用 `_check_result` 口径)。
- `probe_cube_gather.py` — cube-gather 可行性探针(结论见 §4,已确认此路不通,但不影响主线)。
- `golden.py` — CPU golden + paged KV 数据生成器。

**Ascend C 参考实现**(性能标杆,要照着抄思路):
`ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/`
- `arch32/sparse_attn_sharedkv_scfa_kernel.h` — **顶层调度 + `PreloadPipeline`(双缓冲核心,line 784-827)**。
- `arch32/sparse_attn_sharedkv_scfa_block_cube.h` — cube 侧(MM1/MM2 gemm + ori 的 DataCopyPA gather)。
- `arch32/sparse_attn_sharedkv_scfa_block_vector.h` — vector 侧(V0 gather / V1 softmax / V2 merge)。
- `sparse_attn_sharedkv_common.h` — `DataCopyPA`(line 136-177)、`DataCopyGmNDToL1`(112-128)、各 struct。

**性能对比脚本**:`sparse_attn_sharedkv_perf_compare.py`(仓库根目录,非 pytest,`python ...py` 直接跑)。

**TileLang 官方文档 / 例子**(写 kernel 必查):
- `tilelang-ascend/docs/TileLang-Ascend Programming Guide.md`(§4.1 内存/搬运/计算/调度原语)。
- `tilelang-ascend/examples/sparse_flash_attention/example_sparse_flash_attn_mask_pa.py`
  — **主 kernel 的原型**,paged attention 的权威写法(gather 在 vector→workspace→cube)。

---

## 2. 当前 kernel 架构(kernel.py)

- `T.Kernel(core_num=24)` persistent dispatch;metadata 驱动每个 AIC 核的 `(bn2,m)` work range。
- 每个 work item = 一个 query token,走 `NI_total` 个 chunk。**BI=128**(KV tile),所以
  `NI_ori=1`(滑窗 128 连续) + `NI_cmp=4`(topk 512 ÷ 128) = **5 chunk**。
- **每 chunk 严格串行**(这是要打掉的):
  1. **vector** `T.Scope("V")`:逐行 gather KV → `kv_ub_multi`(UB) → 批量写 `ws_kv`(GM workspace)。
  2. **cube** `T.Scope("C")`:`ws_kv`→`kv_lo`/`kv_hi`(L1) → `gemm QK` → `ws_score`(GM)。
  3. vector:`ws_score`→softmax→`ws_p`(GM)。
  4. cube:`ws_p`→`gemm PV`→`ws_o`(GM)。
  5. vector:`ws_o`→merge 进 online-softmax 状态。
  - cube/vector 用 5 个 `cross_flag` 严格握手,**每 chunk 一个完整往返,零跨-chunk 重叠**。
- BI=128 后 kv 操作数 128KB 超 L0B(64KB),所以 kv 拆成 **`kv_lo`/`kv_hi` 两个 [64,512] 物理 buffer**,
  p 拆 `p_lo`/`p_hi`,QK/PV 各做 2 次 gemm(`init=False` 累加)。详见 §6 坑。

---

## 3. 已完成的优化(3 刀)+ profile

| 刀 | 做了什么 | Duration | 性能 | commit |
|---|---|---|---|---|
| cut0 基线 | BI=64,逐行 gather,串行 | 88.2ms | 7.7% | tag `before_performance` |
| cut1 | **批量 gather**:逐行单 buffer+逐行 barrier → 多行 buffer+一个 barrier+批量写出 | 65.8ms | 10.4% | `eb28372` |
| cut2 | **BI 64→128**(对齐 Ascend C N_SPLIT=128),chunk 10→5;配套手动 gemm L0 切分 | 48.8ms | 14.1% | `3347032`+`8976426`+`455375f`+`4a778c8` |

**cut2 稳态 profile(48.8ms 的构成,8K scfa_prefill)**:
- `gap`(= Duration − aicore_time,cube/vector 互等)= **19.1ms (39%)** ← 最大块
- `aiv_scalar`(逐 head 循环 + barrier)= 6.9ms
- `aiv_mte2`(cmp 散列 gather)= 6.07ms
- `aiv_vec`(softmax)= 5.34ms
- `aiv_mte3` = 2.57ms;`aic_mac`(真矩阵算)= 2.36ms;cube 搬运 ≈ 10ms
- vector busy(aiv_time)≈ 29.6ms;cube busy(aicore_time)≈ 29.7ms;Duration 48.8ms。

---

## 4. ⚠️ 关键方向修正(务必先读,别重蹈覆辙)

前一个 session 一度以为"把 KV gather 移到 cube 侧"是关键,**这是错的**,已查证:

- **cube 不能逐行写 L1**:`probe_cube_gather.py` 证明 `T.copy(KV[phys,row], kv_l1[i,:])` 逐行(1D 或 2D 切片)
  都失败,只有整块/连续多行 GM→L1(`T.copy(KV[0,0:N,0,:], kv_l1)`)能用。**所以 cube 做不了散列 gather。**
- **但这不重要**:精读 Ascend C 发现,**cmp(SCFA 主要开销)的散列 gather 两边都不在 cube** ——
  Ascend C 是 `vector(V0) gather → kvMergeGm_(GM workspace) → cube 整块读`
  (`scfa_block_cube.h` line 448-478 从 `kvMergeGm_` 读,不是间接 gather);
  这和当前 TileLang 的 `vector gather → ws_kv(GM) → cube 读` **架构本质一样**。
  cube 的 `DataCopyPA` 只用于 **ori(连续)**,是次要开销。
- **真正的差距是软件流水(双缓冲),不是 gather 在哪个核。** 见 §5。

**结论:不要再去碰 cube-side gather。** `probe_cube_gather.py` 的唯一价值是确认了
"TileLang cube 无逐行 L1 写"这个 pitfall(可记 skill),它探的方向不在关键路径上。

---

## 5. 下一步(主线):实现三级软件流水 = Ascend C 的 `PreloadPipeline`

**Ascend C 怎么做到 6.87ms**(`scfa_kernel.h` line 784-827):每轮 loop 同时推进 3 个相位的任务
(3-task cache),让 cube 和 vector 各自的不同阶段**跨 loop 重叠**:

```
extraInfo0 = cache[loop % 3]      // 本轮 i
extraInfo2 = cache[(loop+2) % 3]  // 上一轮 i-1
extraInfo1 = cache[(loop+1) % 3]  // 上两轮 i-2
同一次调用里:
  cube(AIC):    MM1(i)              + MM2(i-1)
  vector(AIV):  V0 gather(i)  ∥  V1 softmax(i-1)  ∥  V2 merge(i-2)
```

**为什么这让 vector 从 20.9ms 降到接近 Duration**:AIV 核内有 **MTE2/VEC/MTE3 多条 pipe**,
gather(MTE2)、softmax(VEC)、merge(MTE3) 放在**不同 loop** 上就能并行 → vector 总时间从
`sum(各阶段)` 变 `max(各阶段)`。cube 同理(MM1(i) ∥ MM2(i-1))。TileLang 现在每 chunk 串行
+ 海量 barrier,完全没利用 pipe 并行 → 这就是 gap 19ms + vector 慢的根因。

**TileLang 实现要点**(纯调度优化,不需要 cube gather,语言层支持):
1. 把"每 chunk 串行 gather→QK→softmax→PV→merge"重排成**软件流水**:
   prologue 填满流水 → 主循环每轮错位推进(gather chunk i+? 时,softmax/PV 在 chunk i,merge 在 chunk i-?)
   → epilogue 排空。
2. `ws_kv`/`ws_score`/`ws_p`/`ws_o` 等 workspace **多缓冲**(N 份,轮转),让不同 loop 的数据不打架。
3. 用 `T.set_flag`/`T.wait_flag` 在 pipe 间(而非全核 `barrier_all`)做细粒度同步;
   把现在的 `barrier_all`(还有 ~49 个)尽量换成 pipe 级同步,让 MTE2/VEC/MTE3 真正并行。
4. **顺带**:向量化 softmax 里的逐 head 循环(`for h_i in range(v_block)` 32 次,多处)→ 用 tile 广播
   一次处理所有 head,削 `aiv_scalar` 6.9ms。需验证 `T.tile.sub/mul` 是否支持 [v_block,BI] op [v_block,1] 广播。

参考写法:`example_sparse_flash_attn_mask_pa.py` 的 cube/vector scope + cross_flag 握手是**单缓冲**版本
(和当前 kernel 一样),Ascend C 的 `PreloadPipeline` 是**多缓冲流水**版本 —— 目标是把前者改成后者。

**这是大重构,预计要分几步、闯几层坑**(参考 BI=128 那刀闯了 4 层)。建议:先设计 TileLang 版的
流水相位结构(V0/V1/V2 + MM1/MM2 对应到 TileLang 的 scope/flag),给用户过一眼,再动主 kernel。
每步都要 NPU 验证(先正确性后 profile)。

**预期**:gap 19ms 大部分被吃掉 + scalar 削平 → 乐观 ~30%(重叠)→ ~40%(加向量化 softmax);
要冲 80% 可能还需更高效的 gather/softmax 原语,届时再评估。但方向是对的。

---

## 6. 已知坑(踩过的,别再踩)— 也都在 tilelang-pitfalls skill 里

- **`from __future__ import annotations` 会废掉 `@T.prim_func`**:PEP 563 把注解字符串化,
  TVMScript 要求值注解 → `expected Object but got str`。定义 prim_func 的文件**绝不能**加这行。
- **`gemm_v0` 的 operand 必须是完整 buffer 且整体 ≤ L0(64KB)**:不接受切片(`q_l1[:,0:256]` →
  `ValueError: BufferLoad`),也不自动 tiling(传 >L0 的完整 operand 能编过但**运行时 MTE error**,
  plog 特征 `mte error!=0 / cube error==0`)。大 matmul 要拆成独立的 ≤64KB 小 buffer(如 kv_lo/kv_hi),
  K 方向 split 用 `init=False` 在 L0C 累加。
- **body 里的赋值/if/for 都是 TIR,不是 Python**:`x = D//2` 写在 prim_func body 内,`x` 变运行时 Var,
  `buf[:,0:x]` 的 slice 建不出常量 Ramp。编译期常量(tile 宽、split 半宽)一律在 prim_func **外**算成闭包常量。
- **dtype 在 `T.Tensor(...)` 里用变量,别写字面量**:`T.Tensor([...], "int32")` 可能出问题,用 `indices_dtype="int32"` 变量。
- **GM/UB/L1 的 `T.copy` 要 2D**;cube 的 GM→L1 只支持整块/连续多行,不支持逐行单行写。
- **`T.tile.max(dst, src1, src2)` dst 必须放最后**,否则静默丢一个 src(online-softmax 跨 chunk 会崩)。
- **GM→UB/L1 的 `T.copy` 是异步 DMA**,读目标 buffer 前要同步。

---

## 7. 硬件约束(用户的 NPU,Atlas A3 / 910_93)

- **L0C 128KB / L0A 64KB / L0B 64KB / UB 192KB / L1 512KB**。
- cube/vector = 1 AIC : 2 AIV(vid ∈ {0,1},每个 AIV 处理一半 head)。
- API 约束:N1(q head)=64,N2(kv head)=1,D=512,BI 必须整除 topk_cmp 和 ori_block_size。

---

## 8. 工作流程(重要约束,务必遵守)

- **本地无 NPU**:所有编译/正确性/性能验证都靠**用户在 NPU 上手动跑**,贴结果回来。不能假装跑过。
- **验证顺序**:先**正确性**(`pytest ... -k decode` 快,再 `-k prefill --runslow` 慢),过了再 **profile**。
  正确性**别用 `assert_close`**(bf16 尾噪声会误报),用仓库的 `_check_result` 口径
  (≥99.5% 元素过 + 最大归一化相对误差 <10)。
- **每次改完代码主动 `commit`+`push` 到 dsv4 仓**(用户偏好,不用等他说)。push 前先
  `ruff format .` + `ruff check . --fix`(ruff 在 `~/.local/bin/ruff`,不在默认 PATH;
  `export PATH=$HOME/.local/bin:$PATH`)。commit message 结尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- **每刀优化验证有效后,审视手段是否通用,通用的记入 tilelang plugin 的 skill**:
  性能手段→`tilelang-perf`,踩坑→`tilelang-pitfalls`,诊断→`tilelang-debugging`。
  **改两处并 diff 验证**:源仓库 `/Users/yzmac/.claude-plugins/tilelang/skills/` + 运行时缓存
  `~/.claude/plugins/cache/yzmac-personal/tilelang/0.1.0/skills/`。源仓库 git commit(风格 `tilelang: ...`,本地仓无远程,只 commit)。
- 回复正文用中文,代码/注释/commit 用英文。

---

## 9. 关键命令

```bash
# 正确性(NPU)— 先 decode(快),再 8K prefill(慢,CPU golden 几分钟)
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "decode"
pytest sparse_attn_sharedkv_tilelang/test_sparse_attn_sharedkv.py -k "prefill" --runslow

# profile(NPU)— scfa_prefill,只跑 tilelang
msprof --output=./prof_scfa --aic-metrics=PipeUtilization \
  --application="python sparse_attn_sharedkv_perf_compare.py --scenarios scfa_prefill --only tilelang --warmup 5 --iters 3"
# 看产物里 kernel_details.csv / op_summary_*.csv 的 sparse_attn_sharedkv_kernel 行:
#   Duration / aicore_time / aiv_time / aic_mac_time / aiv_vec_time / aiv_scalar_time / aiv_mte2_time / aiv_mte3_time
#   gap = Duration − max(aicore_time, aiv_time) 是诊断的核心数字。

# 性能对比(NPU,both 两套对比)
python sparse_attn_sharedkv_perf_compare.py            # 默认三个 8K prefill 场景
python sparse_attn_sharedkv_perf_compare.py --only tilelang --scenarios scfa_prefill

# 本地(Mac,无 NPU)只能做语法/格式
export PATH=$HOME/.local/bin:$PATH
ruff format sparse_attn_sharedkv_tilelang/kernel.py && ruff check sparse_attn_sharedkv_tilelang/kernel.py
python -m py_compile sparse_attn_sharedkv_tilelang/kernel.py
```

kernel 名字已从 `main` 改成 `sparse_attn_sharedkv`,msprof 里显示 `sparse_attn_sharedkv_kernel`。

---

## 10. 回退点

- tag **`before_performance`** = 一切优化之前的基线(已 force-push 到最新基线状态)。
- 每刀都有独立 commit,可单独回退。cut1(已验证可用)= `eb28372`。
- 当前 main HEAD 含全部 3 刀(已验证正确)+ 探针。

---

## 11. 建议的新 session 开场白

> 接着优化 TileLang 版 sparse_attn_sharedkv 的性能。先读
> `sparse_attn_sharedkv_tilelang/PERF_HANDOFF.md` 全文,然后按 §5 实现三级软件流水
> (Ascend C 的 PreloadPipeline)。先把 TileLang 版的流水相位结构设计出来给我过目,再动主 kernel。
