# SparseAttnSharedKV — TileLang port

A TileLang implementation of the Ascend C operator at
`ops-transformer/experimental/attention/sparse_attn_sharedkv`. Targets
**Atlas A3 / Ascend 910_93** (1 AIC : 2 AIV cube/vector pairs).

The kernel covers all three production scenarios:

| Scenario | Inputs                                               | Code path                     |
| -------- | ---------------------------------------------------- | ----------------------------- |
| 1 (SWA)  | `ori_kv` only                                        | sliding-window pass only      |
| 2 (CFA)  | `ori_kv` + `cmp_kv`                                  | window + dense-cmp pass       |
| 3 (SCFA) | `ori_kv` + `cmp_kv` + `cmp_sparse_indices`           | window + sparse-cmp pass      |

All three share one online-softmax FlashAttn-v2 state seeded from
per-q-head **sinks**. Mathematically each query attends to the union of
the sliding-window ori tokens and the top-K sparse cmp tokens (plus a
virtual sink token whose V is zero); the chunked online softmax matches
the closed-form `softmax([scores, sink_h]) @ [V, 0]` to numerical
precision (verified by `test_golden_math_matches_single_shot_softmax`).

## Files

```
kernel.py    TileLang prim_func + JIT builder
api.py       High-level Python entry: layout dispatch, scenario routing
metadata.py  Python port of the companion SparseAttnSharedkvMetadata
             aicpu kernel (load-balancing scheduler + GenMetaData),
             called alongside sharedkv to mirror the Ascend C flow
golden.py    Python (CPU) golden reference + paged-KV data generators
test_sparse_attn_sharedkv.py
             pytest suite; CPU golden math test, metadata sanity
             checks, and NPU end-to-end cases (metadata + sharedkv)
README.md    this file
```

## metadata + sharedkv pairing

The Ascend C operator ships as **two ops** -- ``SparseAttnSharedkv``
plus the companion ``SparseAttnSharedkvMetadata``. The metadata op
runs on the AI CPU and returns a per-AIC FA task table + per-AIV FD
reduction table; the sharedkv kernel reads it to know which slice of
``(batch, head, S1G_block, S2_block)`` work to execute.

The TileLang port mirrors the same call shape:

```python
from metadata import sparse_attn_sharedkv_metadata
from api import sparse_attn_sharedkv

md = sparse_attn_sharedkv_metadata(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    cu_seqlens_q=cu_seqlens_q, seqused_kv=seqused_kv,
    batch_size=B, max_seqlen_q=T1, max_seqlen_kv=max(seqused_kv_list),
    cmp_topk=512, cmp_ratio=4,
    ori_mask_mode=4, cmp_mask_mode=3,
    ori_win_left=127, ori_win_right=0,
    layout_q="TND", layout_kv="PA_ND",
)
out = sparse_attn_sharedkv(q, ..., metadata=md, ...)
```

``metadata.py`` is a faithful Python port of
``sparse_attn_sharedkv_metadata_aicpu.cpp``: same ``BalanceSchedule``
algorithm (``AssignByBatch`` → ``AssignByRow`` → ``AssignByBlock``,
``supportFd=False`` default that matches the upstream source), same
``GenMetaData`` packing into a flat ``int32[1024]`` tensor laid out as
``faMetadata[36][8] || fdMetadata[72][8]``.

The TileLang ``sparse_attn_sharedkv`` kernel does its own
``T.Kernel(core_num)`` dispatch and does not consume ``metadata`` for
scheduling -- the parameter is accepted purely for API parity so the
test flow can be 1:1 with the Ascend C reference suite (`ops-
transformer/.../sparse_attn_sharedkv/tests/pytest/batch/
sparse_attn_sharedkv_process.py`).

## Quick start

```python
import torch, torch_npu  # noqa
from api import sparse_attn_sharedkv

out = sparse_attn_sharedkv(
    q,                              # bf16/fp16, [T1, N1, D] (TND) or [B, S1, N1, D]
    ori_kv=ori_kv,                  # [block_num, block_size, N2, D]
    cmp_kv=cmp_kv,                  # paged, or None for SWA-only
    cmp_sparse_indices=indices,     # int32, or None for CFA
    ori_block_table=ori_block_table,
    cmp_block_table=cmp_block_table,
    cu_seqlens_q=cu_seqlens_q,      # required for TND
    seqused_kv=seqused_kv,
    sinks=sinks,                    # fp32, [N1]
    softmax_scale=0.04419417,
    cmp_ratio=4,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_ND",
    topk_cmp=512,                   # K2; pass 0 for SWA
)
```

## Constraints (API-level)

Matching the original Ascend C kernel:

- `N1 == 64` (q heads)
- `N2 == 1`  (kv heads)
- `D == 512` (head dim)
- `ori_win_left == 127`, `ori_win_right == 0` (causal window of 128)
- `ori_mask_mode == 4`, `cmp_mask_mode == 3`
- `cmp_ratio ∈ {4, 128}`
- `layout_kv == "PA_ND"`
- `layout_q ∈ {"TND", "BSND"}`
- `dtype ∈ {bfloat16, float16}` (matched across q/ori_kv/cmp_kv/out)
- `topk_cmp` must be a multiple of `block_I = 64`
- `ori_block_size` must be a multiple of `block_I = 64`
- `return_softmax_lse == False` (lse output is unsupported)

## Implementation summary

* **One fused kernel** with two passes per work item, sharing the
  online-softmax state. Pass A walks the closed sliding window
  `[max(s_global - 127, 0), s_global]` in `ceil(window / 64) = 2`
  cube-sized chunks. Pass B walks the `K` sparse cmp indices in
  `K / 64` chunks. Each chunk is `BI = 64` KV tokens × `H_per_block = 64`
  q-heads × `D = 512`.
* **Sinks** seed the flash-v2 prev-state as `(row_max, row_sum) = (sink_h, 1.0)`
  instead of `(-inf, 0)`. End-to-end this adds `exp(sink_h - row_max)` to
  the denominator and nothing to the numerator — equivalent to a virtual
  KV token with V row of zeros.
* **Causal masks** are built on the vector side. The ori pass derives
  per-lane token ids with `T.tile.createvecindex`; the cmp pass DMAs
  them from `cmp_indices` (SCFA) or, for CFA, generates the dense
  `[0, K)` range with `createvecindex` as well. Both feed
  `T.tile.compare` → `T.tile.select(... -∞)`.
* **KV gather**: the kernel receives **paged** KV
  `[block_num, block_size, N2, D]` plus `ori_block_table` /
  `cmp_block_table`, and resolves the block table on the AIV — each lane
  maps a logical token id to `(physical_block, row)` and DMAs one `[D]`
  row into the per-chunk workspace; the cube then loads it as contiguous
  KV. Two AIVs split the 64 lanes in half. This mirrors the Ascend C
  `DataCopyPA` path (no host-side un-paging).
* **Cube ↔ vector** is split explicitly into `T.Scope("C")` /
  `T.Scope("V")` blocks with manual `T.set_cross_flag` /
  `T.wait_cross_flag` handshakes per chunk, on the default (non-pto)
  Ascend lowering path.
* **Persistent dispatch**: a `T.Kernel(core_num)` outer loop with an
  inner `T.serial(ceildiv(B*S_max, core_num))` slot loop keeps the
  workspace footprint per-physical-core (24 slots) rather than per
  work-item.

## Numerical accuracy

`test_golden_math_matches_single_shot_softmax` (CPU-only) verifies the
golden's chunked online softmax matches the closed-form
`softmax([scores, sink]) @ V` reference to `atol=2e-4` for an fp32 case
with a small sparse index set.

NPU end-to-end cases use the Ascend C reference suite's check criterion
(`_check_result` in `test_sparse_attn_sharedkv.py`, mirroring
`result_compare_method.check_result`): at least 99.5% of output elements
pass `np.isclose` with dtype-specific tolerance (bf16: `rtol=0.0078125,
atol=0.0001`; fp16: `rtol=0.005, atol=0.000025`), AND the worst
normalized relative error among failing elements stays below 10.

## Performance notes (deferred work)

* The Ascend C kernel uses an `S2 = 512` chunk with internal `N = 128`
  tiles and explicit double-buffered preload (depth 2). The TileLang
  port uses a flatter `BI = 64` chunking that emits one cube iteration
  per chunk. Same online-softmax math, simpler scheduling. Expect
  performance ~ within 1.5–2× of hand-tuned Ascend C as the starting
  point; profile + tighten before pushing further.
* The kernel uses manual `T.Scope` + cross-flag scheduling (à la
  `example_sparse_flash_attn_mask_pa.py`). Finer-grained pipelining --
  overlapping the next chunk's gather with the current chunk's cube ops
  -- is possible but unimplemented; profile-driven decision.

## Known limitations / TODO

* `return_softmax_lse` is unsupported (matches the upstream API).
* `seqused_q` is unsupported (matches the upstream API).
* `ori_sparse_indices` is unused (matches the upstream API).
* For scenario 1 (SWA-only) `topk_cmp == 0`, so there are no cmp chunks
  (`NI_cmp == 0`); `api` still passes a dummy zero `cmp_kv` to keep the
  kernel signature well-typed. The JIT cache key distinguishes scenarios.
* `cmp_ratio` is treated as a compile-time integer divisor. Only the
  documented `{4, 128}` values are exercised.
* TND inputs are consumed natively: the kernel addresses `Q` /
  `Output` / `cmp_indices` by a flat token id `q_prefix[b] + s` (no
  host-side TND→BSND padding). The persistent dispatch still walks a
  `batch * max_seq` grid and skips padded `(b, s)` slots, so on-device
  cost stays proportional to ``T_total`` only when ``S_max ≈ T_total / B``.

## Running tests

```bash
# CPU-only sanity check (no NPU needed, exercises only the golden).
pytest -q test_sparse_attn_sharedkv.py::test_golden_math_matches_single_shot_softmax

# Full NPU end-to-end (requires Ascend NPU + tilelang-ascend + torch_npu).
pytest -q test_sparse_attn_sharedkv.py
pytest -q test_sparse_attn_sharedkv.py -k scfa_decode

# Also run the large-S1 cases (S1=8192; the CPU golden takes minutes).
pytest -q test_sparse_attn_sharedkv.py --runslow
```

NPU cases skip automatically (`requires_npu` mark) when `torch_npu` is
absent. Large-S1 (`slow`) cases are skipped unless `--runslow` is given.
