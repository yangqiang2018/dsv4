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
golden.py    Python (CPU) golden reference + paged-KV data generators
test_sparse_attn_sharedkv.py
             pytest suite; CPU golden math test + NPU end-to-end cases
README.md    this file
```

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
  them from `cmp_indices`. Both feed `T.tile.compare` →
  `T.tile.select(... -∞)`.
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

NPU end-to-end cases assert `rtol=2e-2, atol=2e-2` against the golden,
matching the tolerance used by the original test suite.

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
* For the TND layout, the kernel internally pads to BSND `[B, S_max,
  N1, D]`. Batches with short sequences do useless work on padded
  slots; on-device cost stays proportional to ``T_total`` only when
  ``S_max ≈ T_total / B``.

## Running tests

```bash
# CPU-only sanity check (no NPU needed, exercises only the golden).
pytest -q test_sparse_attn_sharedkv.py::test_golden_math_matches_single_shot_softmax

# Full NPU end-to-end (requires Ascend NPU + tilelang-ascend + torch_npu).
pytest -q test_sparse_attn_sharedkv.py
pytest -q test_sparse_attn_sharedkv.py -k scfa_decode
```

NPU cases skip automatically (`requires_npu` mark) when `torch_npu` is
absent.
