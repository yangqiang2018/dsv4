"""TileLang implementation of SparseAttnSharedKV (Atlas A3 / Ascend 910_93).

Ports the Ascend C kernel at
``ops-transformer/experimental/attention/sparse_attn_sharedkv`` to TileLang.

A single fused kernel runs the sliding-window pass over ``ori_kv`` and the
top-K sparse pass over ``cmp_kv`` under one online-softmax state that is
seeded from per-q-head sinks. The same kernel covers all three scenarios:

* Scenario 1 (SWA only): ``topk_cmp == 0`` skips the cmp pass.
* Scenario 2 (CFA): the kernel generates the dense compressed-token
  indices on-device; ``cmp_indices`` is an unused placeholder.
* Scenario 3 (SCFA): full sparse compressed attention.

This uses the default (non-pto) Ascend C lowering path: explicit
``T.Scope("C")`` / ``T.Scope("V")`` cube/vector partitioning, manual
``T.set_cross_flag`` / ``T.wait_cross_flag`` handshakes, and manual
``T.annotate_address`` placement -- mirroring the verified
``example_sparse_flash_attn_mask_pa.py`` example. The per-chunk phases
(V0 gather / V1 softmax / V2 merge on the vector; MM1 / MM2 on the cube)
are software-pipelined: a skewed loop issues V0(t) / V1(t-1) / V2(t-2)
and MM1(t) / MM2(t-1) per step so cube and vector overlap across chunks.
The vector phases are emitted by nested helper functions (the verified
``emit_lane`` metaprogramming pattern) so the skewed loop can issue them
at different chunk indices without duplicating the bodies.

The kernel takes **paged** KV: ``ori_KV`` / ``cmp_KV`` are
``[block_num, block_size, n_kv_heads, D]`` with companion
``ori_block_table`` / ``cmp_block_table``. PageAttention block-table
resolution runs on the AI Core (vector): each lane maps a logical
token id to ``(physical_block, row)`` and DMAs the ``[D]`` KV row,
mirroring the Ascend C ``DataCopyPA`` path and the verified
``example_sparse_flash_attn_mask_pa`` gather.

``Q`` / ``Output`` / ``cmp_indices`` are flat ``[total_tokens, ...]``
tensors. Each ``(batch, seq)`` work item is mapped to a token id via a
per-batch ``q_prefix`` offset, so native TND inputs need no host-side
padding: TND passes ``q_prefix[b] = cu_seqlens_q[b]``, BSND passes
``q_prefix[b] = b * max_seq`` over a reshaped ``[B*max_seq, ...]`` view.
"""

import tilelang
from tilelang import language as T
from tvm import ir as tvm_ir
from tvm import tir as tvm_tir


def _sub_tile2(buf, row0, rows, col0, cols):
    return tvm_tir.BufferRegion(
        buf,
        [
            tvm_ir.Range.from_min_extent(row0, rows),
            tvm_ir.Range.from_min_extent(col0, cols),
        ],
    )


def _sub_tile(buf, row0, rows, cols):
    """Explicit tvm BufferRegion for a sub-tile (subscript slices collapse to
    BufferLoad and tile.* rejects them; binary_op/select accept BufferRegion,
    offset from region mins, constant or Var)."""
    return tvm_tir.BufferRegion(
        buf,
        [
            tvm_ir.Range.from_min_extent(row0, rows),
            tvm_ir.Range.from_min_extent(0, cols),
        ],
    )


# Disable AND clear TileLang's on-disk kernel cache. disable_cache()
# alone is not enough: the JIT cache key tracks the prim_func signature
# but not every body edit, so a kernel compiled from an earlier (buggy)
# revision is silently reused when only the body changes (e.g. an
# address-map tweak or an operand-order fix). clear_cache() wipes any
# stale artefact -- TileLang-Ascend's own test-suite clears the cache
# before every test for exactly this reason.
tilelang.disable_cache()
tilelang.cache.clear_cache()

# Atlas A3 cube/vector pair count.
DEFAULT_CORE_NUM = 24

# KV tile (cube N-split): how many KV tokens one cube gemm + one gather
# chunk processes. 128 matches the Ascend C kernel's N_SPLIT_SIZE=128.
# Larger BI => fewer chunks => less per-chunk handshake / scalar / barrier
# overhead and better cube utilization. Must divide topk_cmp and
# ori_block_size; at BI=128 the kv operand tile [BI, D] is 128KB, so the
# two cube gemms are manually K-split in the cube scope (gemm_v0 does NOT
# auto-tile to L0 -- it loads the whole operand, so the full tile would
# overflow the 64KB L0B) so each sub-gemm's L0B tile is 64KB. api.py
# imports this to size the scenario-specific cmp_indices placeholders.
DEFAULT_BLOCK_I = 128

# ---- SasMetaData layout mirroring sparse_attn_sharedkv_metadata.h. ----
# faMetadata[AIC_CORE_NUM][FA_METADATA_SIZE] is laid out first inside
# the flat int32[SAS_META_SIZE] metadata tensor produced by the Ascend C
# SparseAttnSharedkvMetadata aicpu kernel (see ``metadata.py`` for the
# Python port). Each AIC core reads its row to know which (bn2, m) work
# range it owns.
_SAS_META_SIZE = 1024
_FA_METADATA_SIZE = 8
_FA_CORE_ENABLE_INDEX = 0
_FA_BN2_START_INDEX = 1
_FA_M_START_INDEX = 2
_FA_S2_START_INDEX = 3
_FA_BN2_END_INDEX = 4
_FA_M_END_INDEX = 5
_FA_S2_END_INDEX = 6

# Cross-flag ids for the cube<->vector handshake (per chunk).
_FLAG_KV_READY = 0  # V -> C : gathered KV is in ws_kv
_FLAG_SCORE_READY = 1  # C -> V : Q@K^T is in ws_score
_FLAG_P_READY = 2  # V -> C : softmax P is in ws_p
_FLAG_PV_READY = 3  # C -> V : P@V is in ws_o
_FLAG_ITER_DONE = 4  # V -> C : vector finished the chunk


def _check_dtypes(dtype: str) -> None:
    if dtype not in ("bfloat16", "float16"):
        raise ValueError(f"dtype must be bfloat16 or float16, got {dtype!r}")


def build_sparse_attn_sharedkv(
    *,
    batch: int,
    max_seq: int,
    total_tokens: int,
    ori_block_num: int,
    ori_block_size: int,
    ori_table_len: int,
    cmp_block_num: int,
    cmp_block_size: int,
    cmp_table_len: int,
    n_heads: int = 64,
    n_kv_heads: int = 1,
    head_dim: int = 512,
    topk_cmp: int = 512,
    cmp_ratio: int = 4,
    scenario: int = 3,
    ori_win_left: int = 127,
    softmax_scale: float = 0.04419417,
    dtype: str = "bfloat16",
    block_I: int = DEFAULT_BLOCK_I,
    core_num: int = DEFAULT_CORE_NUM,
):
    """Build a JIT-compiled TileLang kernel for SparseAttnSharedKV.

    Arguments are all compile-time constants. Call the returned kernel
    object with the runtime tensors (flat ``[total_tokens, ...]`` Q +
    ``q_prefix`` + paged KV + block tables + sinks). :mod:`api` provides
    a high-level wrapper that maps layouts and synthesises the cmp
    scenario inputs.
    """
    _check_dtypes(dtype)
    assert n_heads == 64, "API constraint: n_heads must be 64"
    assert n_kv_heads == 1, "API constraint: n_kv_heads must be 1"
    assert head_dim == 512, "API constraint: head_dim must be 512"
    assert ori_win_left == 127, "API constraint: ori_win_left must be 127"
    assert topk_cmp >= 0
    assert topk_cmp % block_I == 0, "topk_cmp must be a multiple of block_I"
    assert scenario in (1, 2, 3), "scenario must be 1 (SWA), 2 (CFA) or 3 (SCFA)"
    assert batch > 0 and max_seq > 0 and total_tokens > 0
    assert ori_block_num > 0 and ori_block_size > 0 and ori_table_len > 0
    assert cmp_block_num > 0 and cmp_block_size > 0 and cmp_table_len > 0

    gqa_group = n_heads // n_kv_heads  # 64
    BI = block_I  # 128
    D = head_dim  # 512
    accum_dtype = "float"
    indices_dtype = "int32"

    # Sliding window: q-token attends to [s - win_left, s] (closed).
    ori_window_max = ori_win_left + 1  # 128
    NI_ori = (ori_window_max + BI - 1) // BI  # 1 for BI=128
    NI_cmp = topk_cmp // BI  # 4 for topk=512, BI=128
    NI_total = NI_ori + NI_cmp
    # KV-half width. MUST be a closure constant (computed here, outside the
    # prim_func body): a body-level assignment becomes a runtime Var, and
    # then GM slices like ws_kv[cid, 0:BI_half] can't build a constant-lane
    # Ramp ("int() argument ... not 'Var'"). Each cube gemm processes one
    # [BI_half, D] = 64KB KV half so the operand fits the 64KB L0B.
    BI_half = BI // 2  # 64
    # CFA: the cmp indices are the dense range [0, topk_cmp); the kernel
    # generates them per chunk with createvecindex instead of reading a
    # host-synthesized cmp_indices array (mirrors the Ascend C CFA path).
    is_cfa = scenario == 2
    # cube-direct KV for SWA and CFA: KV pulled GM->L1 by the cube itself (no
    # vector gather / ws_kv round-trip / KV_READY). Both feed contiguous chunks
    # -- SWA's ori sliding window and CFA's dense cmp range [0, topk_cmp). SCFA's
    # sparse topK indices stay on the vector gather path.
    # SWA-prefill needs the paged-block-boundary split: the window start ori_left
    # is not block-aligned, so a 16-row pass with rowc = g0 % block > block-16
    # straddles two paged blocks and must read each from its own block (AscendC
    # DataCopyPA form). The split is in the cube-direct load below; it relies on
    # the fork treating a runtime-extent GM->L1 dst as a sub-tile (skip the
    # whole-block clear) -- tilelang-ascend ascendc_pto 025ef5c. CFA's cmp range
    # is block-aligned (starts at token 0, block sizes are 16-multiples), so its
    # split is compile-time dead; it is applied for parity / non-aligned configs.
    cube_direct = (NI_cmp == 0) or is_cfa

    H_per_block = gqa_group  # 64
    v_block = H_per_block // 2  # 32 -- each AIV handles half the heads
    ub_len = max(32 // 4, v_block)  # 32-byte UB alignment for fp32 scalars
    # Mask is BI bits = BI//8 (=16) bytes. VEC ops require a 32-byte-aligned UB
    # operand, so a [2, BI//8] parity buffer's odd row (stride 16B) would be
    # unaligned. Pad each parity row to mask_w (round BI//8 up to 32B) so both
    # rows start on a 32B boundary; only the low BI//8 bytes carry the mask.
    mask_w = ((BI // 8 + 31) // 32) * 32  # 32 for BI=128
    # S2b.0 gather sub-tile: instead of staging all BI//2 (=64) gathered KV rows
    # in one 64KB UB buffer, gather GATHER_ROWS rows per pass into a ping-pong
    # half and write that half to ws_kv, N_GATHER_PASS passes. 16 rows = a 16KB
    # bf16 tile (matches the Ascend C kvMerg 16K ping-pong) and shrinks the
    # gather UB to [2*16, D] = 32KB, which (with the 16-head V2 merge tile) lets
    # S2b.1 un-alias the buffers and drop the within-core barriers. BI//2 must be
    # a multiple of GATHER_ROWS.
    GATHER_ROWS = 16
    assert (BI // 2) % GATHER_ROWS == 0, "BI//2 must be a multiple of GATHER_ROWS"
    N_GATHER_PASS = (BI // 2) // GATHER_ROWS  # 4 for BI=128
    # S2b.0 V2 merge sub-tile: rescale+add MERGE_HEADS heads per pass so the P@V
    # load buffer acc_o_ub shrinks to [16, D] = 32KB ([16,512]fp32), matching the
    # 32KB gather tile so the two fit un-aliased. v_block must be a multiple.
    MERGE_HEADS = 16
    assert v_block % MERGE_HEADS == 0, "v_block must be a multiple of MERGE_HEADS"
    N_MERGE_PASS = v_block // MERGE_HEADS  # 2 for v_block=32
    # Q4 hand-unrolls the cube_direct merge/normalize into 2 passes so the
    # double-buffer work tile can be picked by a Python int (a `for .. range()`
    # makes the index a TIR Var, which can't index a Python tuple of buffers and
    # would hit the :1499 UB-tile Var-parity trap). All shipped configs have
    # v_block == 2*MERGE_HEADS == 32, so this always holds.
    assert N_MERGE_PASS == 2, "Q4 cube_direct merge unroll assumes N_MERGE_PASS == 2"

    q_shape = [total_tokens, n_heads, D]
    out_shape = [total_tokens, n_heads, D]
    ori_kv_shape = [ori_block_num, ori_block_size, n_kv_heads, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, n_kv_heads, D]
    ori_bt_shape = [batch, ori_table_len]
    cmp_bt_shape = [batch, cmp_table_len]
    # cmp_indices holds only the real cmp top-K indices; cmp chunk `chunk`
    # (chunk in NI_ori..NI_total-1) addresses it at `(chunk - NI_ori) * BI`.
    # SWA has no cmp pass, so its dummy is a single BI-wide slot.
    indices_shape = [total_tokens, n_kv_heads, max(NI_cmp, 1) * BI]

    # ---- Manual address maps (bytes). Sized for BI=128. ----
    KB = 1024
    # gemm_v0 needs each operand to fit the 64KB L0B as a whole buffer (it
    # neither slices operands nor auto-tiles to L0), so the BI=128 kv is two
    # physical [64,512]=64KB halves and p is two [64,64] halves.
    # L1 (>=336KB). The KV halves are double-buffered as a [2, BI_half, D]
    # buffer indexed by chunk parity: MM2(t-1)'s P@V reads chunk t-1's KV from
    # kv_lo[(t-1)%2] while MM1(t)'s Q@K^T loads chunk t into kv_lo[t%2]. A
    # runtime-indexed BufferRegion (gemm_v0 accepts these) -- not separate named
    # buffers -- because the loop var is a TIR Var, so parity can't select a
    # Python buffer object. q 64KB @0; kv_lo[2] 128KB @64; kv_hi[2] 128KB @192;
    # p_lo 8KB @320; p_hi 8KB @328 -> 336KB <= 512KB.
    l1_addr = {
        "q_l1": 0,
        "kv_lo": 64 * KB,
        "kv_hi": 192 * KB,
        "p_lo": 320 * KB,
        "p_hi": 328 * KB,
    }
    l0c_addr = {"acc_s_l0c": 0, "acc_o_l0c": 0}  # disjoint phases ⇒ alias
    # UB (192KB). acc_s_* are [32,128]. S2b.0b sub-tiles the two 64KB chunk
    # buffers into 32KB tiles: kv_ub_multi (gather) and acc_o_ub (P@V merge)
    # are SEPARATE 32KB buffers (un-aliased, both live without barriers);
    # acc_o_half (epilogue cast) aliases kv_ub_multi. S2b.1d-alpha doubles
    # acc_s_ub_ to a [2*v_block, BI] = 32KB flat ping-pong (halves
    # pv*v_block..+v_block, same flat form as alpha / ws_kv rows) so 1d-beta
    # can prefetch the next chunk's score into the idle half; that no longer
    # fits between acc_s_half and kv_ub_multi, forcing this full repack.
    # Named peak = 178.3KB. The planner (ascend_memory_planning.cc) places the
    # HIDDEN tmp buffers it injects (reduce/select sharedTmpBuffer etc.) at
    # next_new_offset_ = top of the annotated layout -- the tail above the
    # named peak is NOT free slack. 186.3K left only 5.7K and the reduce
    # scratch ran past 192K ("VEC ub address out of bounds" device fault);
    # 178.3K restores a 13.7K tail like the verified pre-1d layout (176K+16K).
    # acc_s_half therefore aliases the head of acc_o_ub: V1's cast/ws_p-write
    # and V2's ws_o-load/merge are strictly serialized by the V1-end + V2-end
    # barrier_all in 1d-alpha. 1d-beta must keep the V1-end barrier or split
    # this alias. acc_o_ub stays SINGLE-buffered: x2 = +32KB -> over the wall
    # (matches AscendC, which ping-pongs the score inputBuff1 32K*2 but NOT
    # the output accumulator).
    ub_addr = {
        "acc_o": 0,  # [32,512]fp32 = 64KB -> 0..64KB
        "kv_ub_multi": 64 * KB,  # [2*16,512]bf16 = 32KB -> 64..96KB
        "acc_o_ub": 96 * KB,  # [16,512]fp32 = 32KB -> 96..128KB
        "acc_s_half": 96 * KB,  # [32,128]bf16 = 8KB, aliases acc_o_ub head (see above)
        "acc_s_ub_": 128 * KB,  # [2*32,128]fp32 = 32KB -> 128..160KB (1d ping-pong)
        "acc_s_ub": 160 * KB,  # [32,128]fp32 = 16KB -> 160..176KB
        # Per-row scalar vectors + index/mask scratch, packed from 176KB.
        "m_i": 176 * KB,
        "m_i_prev": 176 * KB + 128,
        "sumexp": 176 * KB + 256,
        "sumexp_i_ub": 176 * KB + 384,
        "sinks_ub": 176 * KB + 512,
        "lse_ub": 176 * KB + 640,
        "idx_int": 176 * KB + 768,  # [128]int32 = 512B
        "idx_float": 176 * KB + 1280,  # [128]fp32 = 512B
        # Mask double buffer [2, mask_w]: row padded to mask_w (=32B) so BOTH
        # parity rows are 32B aligned. A [2, BI//8] buffer's row stride is 16B,
        # so the odd-parity row mask_ub[1,:] starts at +16B -- not 32B aligned
        # -- and the VEC compare/and on odd chunks faults ("UB address accessed
        # by the VEC instruction is not aligned" on device). parity = chunk % 2
        # is a TIR Var (real TIR loop), so the row is picked by Var index, only
        # the low BI//8 bytes of each row carry the mask.
        "mask_ub": 176 * KB + 1792,  # [2,32]uint8 = 64B, rows 32B aligned
        "mask_ub_2": 176 * KB + 1856,  # [2,32]uint8 = 64B (V0 AND scratch)
        # alpha[2*ub_len]fp32 = 256B: the V1->V2 rescale-factor handoff, double
        # buffered by chunk parity -- V2(t-2) reads slot (t-2)%2 while V1(t-1)
        # writes slot (t-1)%2. FLAT 1D (not [2,ub_len]): the per-head rescale is
        # read as a SCALAR alpha[pv*ub_len + h_i] in T.tile.mul, and tile's
        # binary_op scalar path forwards only indices[0] to the intrinsic -- a
        # 2D alpha[pv, h_i] would silently drop h_i and read alpha.flat[pv] for
        # every head. A single flat index keeps the whole offset in indices[0].
        "alpha": 176 * KB + 2048,  # [2*ub_len]fp32 = 256B -> ..2304
        "mask_sel": 176 * KB + 2304,  # [32]uint8 whole buffer for select selMask
        "acc_o_half": 64 * KB,  # [32,512]bf16 = 32KB, aliases kv_ub_multi (epilogue)
    }

    # Output 0: attn_out [total_tokens, n_heads, D] (dtype)
    # Output 1: lse      [total_tokens, n_heads]     (fp32)
    # The kernel always writes lse (it is essentially free: the running
    # row_max / row_sum are already on the vector core; one ln + one add
    # + one [n_heads] DMA per work item). The api.py wrapper hides it
    # behind a ``return_softmax_lse`` switch -- this matches the Ascend
    # C contract (``softmax_lse`` is a REQUIRED output, gated at the
    # attribute level) and gives every caller (training reverse,
    # online-softmax composition, etc.) the value for free.
    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15, 16, 17])
    def _make():
        @T.prim_func
        def sparse_attn_sharedkv(
            Q: T.Tensor(q_shape, dtype),  # type: ignore[valid-type]
            ori_KV: T.Tensor(ori_kv_shape, dtype),  # type: ignore[valid-type]
            ori_block_table: T.Tensor(ori_bt_shape, indices_dtype),  # type: ignore[valid-type]
            cmp_KV: T.Tensor(cmp_kv_shape, dtype),  # type: ignore[valid-type]
            cmp_block_table: T.Tensor(cmp_bt_shape, indices_dtype),  # type: ignore[valid-type]
            cmp_indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
            q_prefix: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            actual_q_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            actual_kv_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            Sinks: T.Tensor([n_heads], accum_dtype),  # type: ignore[valid-type]
            Metadata: T.Tensor([_SAS_META_SIZE], indices_dtype),  # type: ignore[valid-type]
            Output: T.Tensor(out_shape, dtype),  # type: ignore[valid-type]
            LSE_out: T.Tensor([total_tokens, n_heads], accum_dtype),  # type: ignore[valid-type]
            # Workspaces are double-buffered by chunk parity (leading dim 2)
            # so the software pipeline can have chunk i and i-1/i-2 in flight
            # without the producer clobbering a buffer the consumer still reads.
            ws_kv: T.Tensor([core_num, 2, BI, D], dtype),  # type: ignore[valid-type]
            ws_score: T.Tensor([core_num, 2, H_per_block, BI], accum_dtype),  # type: ignore[valid-type]
            ws_p: T.Tensor([core_num, 2, H_per_block, BI], dtype),  # type: ignore[valid-type]
            ws_o: T.Tensor([core_num, 2, H_per_block, D], accum_dtype),  # type: ignore[valid-type]
            # S1/S2 (cross-slot pipeline): acc_o's persistent store moves UB -> GM.
            # S2 adds the leading slot-parity dim (2), indexed by slot%2, so slot k
            # and k+1's accumulators don't collide once S3 overlaps them -- INERT
            # under the still-serial slot loop (each slot is self-contained in its
            # parity, fully drained before the next). Mirrors ws_o's [core_num, 2,
            # H_per_block, D]; each AIV indexes its half by vid*v_block. cube_direct
            # (swa/cfa) only -- on the SCFA trace this arg is unreferenced.
            ws_acc_o: T.Tensor([core_num, 2, H_per_block, D], accum_dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- L1 / L0 (cube). ----
                q_l1 = T.alloc_L1([H_per_block, D], dtype)
                # kv / p are split into two physical halves so each gemm_v0
                # operand fits the 64KB L0B as a whole buffer. kv is also
                # double-buffered as [2, BI_half, D] indexed by chunk parity:
                # the pipelined cube reads chunk t-1's KV (for P@V) from
                # kv_lo[(t-1)%2] while chunk t's KV (for Q@K^T) lands in
                # kv_lo[t%2]. gemm_v0 takes the [BI_half, D] sub-region
                # kv_lo[parity] (a BufferRegion, which it accepts).
                kv_lo = T.alloc_L1([2, BI_half, D], dtype)
                kv_hi = T.alloc_L1([2, BI_half, D], dtype)
                p_lo = T.alloc_L1([H_per_block, BI_half], dtype)
                p_hi = T.alloc_L1([H_per_block, BI_half], dtype)
                acc_s_l0c = T.alloc_L0C([H_per_block, BI_half], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                # ---- UB (vector). ----
                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                # S1: cube_direct working tile for the GM-resident acc_o. The
                # merge/normalize load a MERGE_HEADS pass from ws_acc_o into this,
                # operate, store back -- so acc_o's 64KB UB is dead on the
                # cube_direct trace and this 32KB tile reuses its low half (annotated
                # below, cube_direct-only, so SCFA's IR never sees the alias). The
                # 64KB is genuinely reclaimed in S2. Allocated unconditionally
                # (tvmscript block-scopes a buffer declared inside `if cube_direct:`).
                # Q4: double-buffered. The merge/normalize sub-tile the v_block heads
                # into N_MERGE_PASS passes; with TWO work tiles (acc_o_work for even
                # passes, acc_o_work2 for odd) the next pass's GM load (MTE2) overlaps
                # this pass's rescale/div (VEC) + store (MTE3) under precise pipe flags
                # instead of S1's barrier_all walls (which serialized DMA-vs-VEC on the
                # bottleneck vector core -- the -7). The two 32KB tiles fill the freed
                # 64KB acc_o slot exactly (0..32KB, 32..64KB), cube_direct-only.
                acc_o_work = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_work2 = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                # P@V merge load buffer, sub-tiled (S2b.0) to MERGE_HEADS heads =
                # [16, D] = 32KB; V2 merges v_block heads in N_MERGE_PASS passes.
                acc_o_ub = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                m_i = T.alloc_ub([ub_len], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
                sinks_ub = T.alloc_ub([ub_len], accum_dtype)
                lse_ub = T.alloc_ub([ub_len], accum_dtype)
                # alpha: rescale factor exp(m_prev-m_new) handed from V1(chunk)
                # to V2(chunk), double-buffered by chunk parity so the pipelined
                # V1(t-1) and V2(t-2) use distinct halves in one step. FLAT 1D
                # [2*ub_len] (parity p occupies [p*ub_len, (p+1)*ub_len)): V2
                # reads it as a per-head SCALAR alpha[pv*ub_len + h_i], and the
                # tile binary_op scalar path forwards only indices[0], so a 2D
                # alpha[pv, h_i] would drop h_i and rescale every head by
                # alpha.flat[pv]. The flat index keeps the offset in indices[0].
                alpha = T.alloc_ub([2 * ub_len], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                # S2b.1d-alpha: [2*v_block, BI] flat ping-pong (halves
                # pv*v_block .. +v_block). 1d-alpha only repacks + doubles the
                # buffer; V1 still uses half pv1=(t-1)%2 throughout, so behavior
                # is unchanged under the V1-end barrier (the two halves are used
                # by alternating chunks, fully drained between them). 1d-beta
                # will prefetch chunk t's score into half t%2 while V1(t-1)
                # computes half (t-1)%2.
                acc_s_ub_ = T.alloc_ub([2 * v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)
                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                # Multi-row gather staging buffer, sub-tiled (S2b.0) into a
                # [2*GATHER_ROWS, D] = 32KB ping-pong: pass gp gathers GATHER_ROWS
                # KV rows into half (gp%2) with NO per-row barrier (disjoint dst
                # rows -> MTE2 pipelines them), then one barrier + one batched
                # write of that half to ws_kv. N_GATHER_PASS passes cover all
                # BI//2 rows. Still aliases acc_o_ub's address (gather at the
                # chunk head, P@V merge at the tail -- disjoint phases under the
                # S2a barriers); S2b.1 will un-alias once the merge tile shrinks
                # too. The ping-pong halves are rows [0,GATHER_ROWS) and
                # [GATHER_ROWS, 2*GATHER_ROWS).
                kv_ub_multi = T.alloc_ub([2 * GATHER_ROWS, D], dtype)
                # V1 softmax fused-op scratch (perf lever, replicate Ascend C
                # SoftmaxFlashV2): the cube_direct V1 softmax fuses the manual
                # max-subtract / exp / row-sum / running-state rescale chain into one
                # T.tile.softmax_flashv2. softmax_tmp is its uint8 working buffer
                # (SoftMaxFlashV2TilingFunc tiles within tmp.GetSize()); alpha_exp
                # receives the per-row rescale alpha = exp(prev_max - new_max), stashed
                # to the flat alpha[] for V2. Both cube_direct ONLY (SCFA keeps the
                # manual per-head chain). Allocated unconditionally (tvmscript
                # block-scopes a buffer declared inside `if cube_direct:`, so it would
                # be invisible at the annotate/use scopes); only cube_direct ANNOTATES
                # them (below) onto the idle kv_ub_multi -- softmax_tmp reuses the 16KB
                # the removed m_i_brd broadcast scratch held (the fused op does the
                # max-subtract internally, so no broadcast buffer is needed). The ALIAS
                # (not the alloc) is what perturbs SCFA (conservative syncs around its
                # gather, +4ms), so SCFA leaves both auto-placed, never aliased.
                softmax_tmp = T.alloc_ub([16 * KB], "uint8")
                alpha_exp = T.alloc_ub([ub_len], accum_dtype)
                # V2 rescale scratch (perf lever, replicate Ascend C RowMuls):
                # brcb writes the pass's MERGE_HEADS alpha scalars here as
                # [MERGE_HEADS, 8] (one 32B block per head), then row_muls reads
                # each row's block to scale acc_o. Tiny (MERGE_HEADS*8*4 = 512B),
                # auto-placed in the UB tail; cube_direct only (SCFA keeps the
                # per-head scalar mul, never references this).
                alpha_brd8 = T.alloc_ub([MERGE_HEADS, 8], accum_dtype)
                # Mask double buffer [2, mask_w]: V0(t) writes parity t%2 while
                # V1(t-1) reads parity (t-1)%2 in the same step. The row is
                # padded to mask_w (32B) so both parity rows are 32B aligned; a
                # [2, BI//8] row stride of 16B leaves the odd row unaligned and
                # the VEC compare/and on odd chunks faults. parity is a TIR Var
                # (the chunk loop is a real TIR loop), so rows are picked by Var
                # index mask_ub[pv, :] -- a Python-side buffer pick would hit
                # "tuple indices must be integers or slices, not Var".
                mask_ub = T.alloc_ub([2, mask_w], "uint8")
                # V0-internal AND scratch (written + consumed inside one V0);
                # kept [2, mask_w] so the bitwise_and operands are same-rank
                # padded rows. Only the pv0 row is ever used.
                mask_ub_2 = T.alloc_ub([2, mask_w], "uint8")
                # Whole-buffer mask for tile.select (selMask calls .access_ptr,
                # which a Var-indexed parity BufferRegion lacks); V1 copies the
                # current chunk's mask_ub row here each step. mask_w wide to
                # match the mask_ub row copy.
                mask_sel = T.alloc_ub([mask_w], "uint8")

                T.annotate_address(
                    {
                        q_l1: l1_addr["q_l1"],
                        kv_lo: l1_addr["kv_lo"],
                        kv_hi: l1_addr["kv_hi"],
                        p_lo: l1_addr["p_lo"],
                        p_hi: l1_addr["p_hi"],
                        acc_s_l0c: l0c_addr["acc_s_l0c"],
                        acc_o_l0c: l0c_addr["acc_o_l0c"],
                        acc_o: ub_addr["acc_o"],
                        acc_s_ub: ub_addr["acc_s_ub"],
                        acc_s_ub_: ub_addr["acc_s_ub_"],
                        acc_s_half: ub_addr["acc_s_half"],
                        m_i: ub_addr["m_i"],
                        m_i_prev: ub_addr["m_i_prev"],
                        sumexp: ub_addr["sumexp"],
                        sumexp_i_ub: ub_addr["sumexp_i_ub"],
                        sinks_ub: ub_addr["sinks_ub"],
                        lse_ub: ub_addr["lse_ub"],
                        idx_int: ub_addr["idx_int"],
                        idx_float: ub_addr["idx_float"],
                        alpha: ub_addr["alpha"],
                        # S2b.0b: kv_ub_multi (gather, 32KB) and acc_o_ub (merge,
                        # 32KB) are now separate buffers -- un-aliased so S2b.1 can
                        # keep both live without the within-core barriers.
                        kv_ub_multi: ub_addr["kv_ub_multi"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_ub_2: ub_addr["mask_ub_2"],
                        mask_sel: ub_addr["mask_sel"],
                        acc_o_ub: ub_addr["acc_o_ub"],
                        acc_o_half: ub_addr["acc_o_half"],
                    }
                )
                # softmax_tmp / alpha_brd8 / alpha_exp (cube_direct V1-softmax +
                # V2-rescale scratch) alias the cube_direct-idle kv_ub_multi. A
                # SEPARATE call keyed only here keeps them out of SCFA's IR entirely
                # (SCFA must not see the alias -- it adds conservative syncs around its
                # gather, measured +4ms). annotate_address accumulates: each call sets
                # addresses for its listed buffers, leaving the rest as placed by the
                # call above. A conditional dict-unpack inside the main literal is
                # rejected by the tvmscript parser, so this is a plain second call under
                # the compile-time cube_direct guard. Layout in the 32KB idle
                # kv_ub_multi: softmax_tmp [0, 16KB), alpha_brd8 [16KB, 16KB+512B),
                # alpha_exp [16KB+512B, +128B) -- all disjoint.
                if cube_direct:
                    T.annotate_address(
                        {
                            softmax_tmp: ub_addr["kv_ub_multi"],
                            alpha_brd8: ub_addr["kv_ub_multi"] + 16 * KB,
                            alpha_exp: ub_addr["kv_ub_multi"] + 16 * KB + 512,
                            # S1: acc_o is GM-resident on this trace, so its 64KB UB
                            # region is dead -- acc_o_work (32KB) reuses its low half.
                            # Q4: acc_o_work2 (32KB) takes the high half, so the two
                            # double-buffer tiles fill the whole dead 64KB acc_o slot.
                            acc_o_work: ub_addr["acc_o"],
                            acc_o_work2: ub_addr["acc_o"] + 32 * KB,
                        }
                    )

                # ---- Read this AIC core's metadata row. ----
                # Each row is FA_METADATA_SIZE int32 entries; layout
                # mirrors SasMetaData::faMetadata from the Ascend C
                # SparseAttnSharedkvMetadata aicpu kernel (see
                # ``metadata.py`` for the Python port and
                # ``sparse_attn_sharedkv_metadata.h`` for the canonical
                # struct definition).
                #
                # API constraints: n_kv_heads == 1 and
                # mBaseSize == groupSize == n_heads, so the (bn2, m)
                # work coordinate produced by the scheduler maps 1:1 to
                # this kernel's (batch, seq) work coordinate. Each
                # ``(bn2_idx, m_idx)`` corresponds to one ``(b_i, s_i)``
                # work item; the S2 dimension is fully covered by this
                # kernel's internal ``NI_total`` chunk loop and is
                # therefore not sliced across cores (supportFd defaults
                # to False in the aicpu source -- ``s2_start`` and
                # ``s2_end`` stay at the row's start/end for every
                # core).
                meta_base = cid * _FA_METADATA_SIZE
                core_enable = Metadata[meta_base + _FA_CORE_ENABLE_INDEX]
                bn2_start = Metadata[meta_base + _FA_BN2_START_INDEX]
                m_start = Metadata[meta_base + _FA_M_START_INDEX]
                bn2_end = Metadata[meta_base + _FA_BN2_END_INDEX]
                m_end = Metadata[meta_base + _FA_M_END_INDEX]
                # Linearize the (bn2, m) range to a pid range. The
                # scheduler's ``m`` index counts S1G rows within a batch
                # (= seq token id when groupSize == mBaseSize). The
                # planar pid space ``b * max_seq + s`` already accounts
                # for per-batch padding via the ``s_i < act_q`` guard
                # below, so the same guard skips padded pids inside the
                # assigned window automatically.
                linear_start = bn2_start * max_seq + m_start
                linear_end = bn2_end * max_seq + m_end

                # Static loop upper bound. The actual range walked per
                # core is ``linear_end - linear_start``; in the worst
                # case (one core owns the whole job, e.g. tiny decode
                # cases) that equals ``batch * max_seq``. Out-of-range
                # iterations are skipped by the pid range guard.
                total_work = batch * max_seq
                for slot in T.serial(total_work):
                    pid = linear_start + slot
                    if core_enable != 0 and pid < linear_end:
                        b_i = pid // max_seq
                        s_i = pid % max_seq
                        act_q = actual_q_len[b_i]
                        act_kv = actual_kv_len[b_i]
                        if s_i < act_q:
                            # Flat token id of this (batch, seq) work
                            # item. q_prefix carries the per-batch token
                            # offset: cu_seqlens_q[b] for native TND,
                            # b * max_seq for a reshaped BSND view.
                            t_i = q_prefix[b_i] + s_i
                            # Causal s-position in the kv sequence.
                            s_global = act_kv - act_q + s_i
                            ori_right = s_global
                            # Clamp window start to 0. T.if_then_else (a
                            # ternary) avoids the ambiguous C++ max()
                            # overload that T.max generates here.
                            ori_left_raw = s_global - ori_win_left
                            ori_left = T.if_then_else(ori_left_raw < 0, 0, ori_left_raw)
                            cmp_threshold = (s_global + 1) // cmp_ratio

                            # ================= CUBE =================
                            # Software-pipelined (skewed) cube loop. Step t runs
                            # MM1(t) [Q@K^T of chunk t] then MM2(t-1) [P@V of
                            # chunk t-1]. MM1 needs only chunk t's KV (ready early
                            # in the vector's step t); MM2 needs chunk t-1's P
                            # (produced later in the same step) -- so MM1 goes
                            # first and the cube never idles waiting on the vector
                            # (this is what eats the cube<->vector gap). ws_* are
                            # double-buffered by chunk parity; the L1 KV too, so
                            # MM2(t-1) reads chunk t-1's KV from kv_*[(t-1)%2]
                            # while MM1(t) loads chunk t into kv_*[t%2]. NI_total+1
                            # steps so MM2 drains the last chunk. No ITER_DONE
                            # back-flag: ws double-buffering + the forward-flag
                            # lattice make reuse safe (see PIPELINE_DESIGN.md S4).
                            with T.Scope("C"):
                                T.copy(Q[t_i, 0:n_heads, 0:D], q_l1)
                                T.barrier_all()
                                for t in range(NI_total + 1):
                                    if t < NI_total:
                                        # ---- MM1(t): Q@K^T of chunk t ----
                                        pa = t % 2
                                        # CUBE-DIRECT KV (AscendC form): contiguous
                                        # chunks (ori window; CFA dense cmp) are
                                        # pulled GM->L1 by cube itself, 16-row
                                        # blocks -- no vector gather, no ws_kv
                                        # round-trip, no KV_READY. Only SCFA's
                                        # sparse topK chunks keep the vector path.
                                        if cube_direct and t < NI_ori:
                                            # A 16-row pass may straddle a paged block
                                            # boundary: prefill ori_left is not block-
                                            # aligned, so g0 % block can be > block-16.
                                            # Common case (whole pass in one block) keeps
                                            # the compile-time 16-row copy; the straddle
                                            # pass splits at the boundary (AscendC
                                            # DataCopyPA form), each part from its own
                                            # block. The split's runtime-extent dst makes
                                            # the fork skip the whole-block clear (a
                                            # runtime extent is a sub-tile -> 025ef5c).
                                            for gp in range(BI_half // GATHER_ROWS):
                                                g0 = (
                                                    ori_left + t * BI + gp * GATHER_ROWS
                                                )
                                                bidx = g0 // ori_block_size
                                                rowc = g0 % ori_block_size
                                                if ori_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = ori_block_size - rowc
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            for gp in range(BI_half // GATHER_ROWS):
                                                g0 = (
                                                    ori_left
                                                    + t * BI
                                                    + BI_half
                                                    + gp * GATHER_ROWS
                                                )
                                                bidx = g0 // ori_block_size
                                                rowc = g0 % ori_block_size
                                                if ori_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = ori_block_size - rowc
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            T.barrier_all()
                                        elif cube_direct and is_cfa:
                                            # CFA cmp cube-direct: this chunk's
                                            # compressed token ids are the dense
                                            # range [(t-NI_ori)*BI, +BI), so the
                                            # cube pulls [BI, D] from cmp_KV GM->L1
                                            # itself (no vector gather / ws_kv /
                                            # KV_READY), exactly like the ori window
                                            # above. Same paged-boundary split: a
                                            # 16-row pass straddling a cmp block
                                            # reads each part from its own block.
                                            # The cmp range starts at token 0 with
                                            # 16-multiple block sizes, so the split
                                            # is compile-time dead for the shipped
                                            # configs (plain 16-row copy); kept for
                                            # parity / non-aligned cmp configs.
                                            for gp in range(BI_half // GATHER_ROWS):
                                                gc0 = (
                                                    t - NI_ori
                                                ) * BI + gp * GATHER_ROWS
                                                bidx = gc0 // cmp_block_size
                                                rowc = gc0 % cmp_block_size
                                                if cmp_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = cmp_block_size - rowc
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            for gp in range(BI_half // GATHER_ROWS):
                                                gc0 = (
                                                    (t - NI_ori) * BI
                                                    + BI_half
                                                    + gp * GATHER_ROWS
                                                )
                                                bidx = gc0 // cmp_block_size
                                                rowc = gc0 % cmp_block_size
                                                if cmp_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = cmp_block_size - rowc
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            T.barrier_all()
                                        else:
                                            T.wait_cross_flag(_FLAG_KV_READY)
                                            T.barrier_all()
                                            T.copy(
                                                ws_kv[cid, pa, 0:BI_half, 0:D],
                                                kv_lo[pa, :, :],
                                            )
                                            T.barrier_all()
                                            T.copy(
                                                ws_kv[cid, pa, BI_half:BI, 0:D],
                                                kv_hi[pa, :, :],
                                            )
                                            T.barrier_all()
                                        # MM1 cube debarrier (perf lever 1): the
                                        # two score gemms + their L0C->ws_score
                                        # drains are a strictly serial MAD->FIX->
                                        # MAD->FIX chain on the single acc_s_l0c.
                                        # barrier_all over-synced it (also drained
                                        # MTE2/MTE1), inflating the cube bubble
                                        # (~34% of aicore_time, swa profile). Each
                                        # internal sync is one targeted pipe flag:
                                        # m->fix (gemm drains to copy) and fix->m
                                        # (copy reads acc_s_l0c before the next
                                        # gemm overwrites it, WAR). The boundary
                                        # barrier_all is kept -- copy_hi(FIX) must
                                        # finish before MM2 reuses L0C (acc_s_l0c /
                                        # acc_o_l0c alias). Cube pipe flags live on
                                        # AIC, disjoint from the V-scope's AIV flag
                                        # ids.
                                        T.gemm_v0(
                                            q_l1,
                                            kv_lo[pa, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                        )
                                        T.set_flag("m", "fix", 0)  # gemm_lo -> copy_lo
                                        T.wait_flag("m", "fix", 0)
                                        T.copy(
                                            acc_s_l0c,
                                            ws_score[cid, pa, 0:H_per_block, 0:BI_half],
                                        )
                                        T.set_flag(
                                            "fix", "m", 1
                                        )  # copy_lo -> gemm_hi WAR
                                        T.wait_flag("fix", "m", 1)
                                        T.gemm_v0(
                                            q_l1,
                                            kv_hi[pa, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                        )
                                        T.set_flag("m", "fix", 2)  # gemm_hi -> copy_hi
                                        T.wait_flag("m", "fix", 2)
                                        T.copy(
                                            acc_s_l0c,
                                            ws_score[
                                                cid, pa, 0:H_per_block, BI_half:BI
                                            ],
                                        )
                                        T.barrier_all()
                                        T.set_cross_flag("FIX", _FLAG_SCORE_READY)
                                    if t >= 1:
                                        # ---- MM2(t-1): P@V of chunk t-1 ----
                                        # acc_s_l0c (MM1) and acc_o_l0c (MM2) alias
                                        # at L0C 0; MM1(t) drained its score to
                                        # ws_score (barriers above) before MM2
                                        # overwrites L0C. MM2 reads chunk t-1's KV
                                        # from the (t-1)%2 L1 buffers, loaded by
                                        # MM1(t-1) the previous step.
                                        pb = (t - 1) % 2
                                        # MM2 cube debarrier (same idea as MM1):
                                        # replace the over-syncing barrier_all on
                                        # the real cross-pipe deps with one targeted
                                        # flag, and drop the redundant same-pipe
                                        # barrier. Kept as barrier_all: the boundary
                                        # after the P_READY wait (p L1 reuse + the
                                        # acc_s_l0c/acc_o_l0c alias vs MM1), the
                                        # split between the two accumulating gemms
                                        # (same-pipe MAD ordering is automatic, but
                                        # the L0C accumulate is left guarded for
                                        # now), and the trailing boundary.
                                        T.wait_cross_flag(_FLAG_P_READY)
                                        T.barrier_all()
                                        T.copy(
                                            ws_p[cid, pb, 0:H_per_block, 0:BI_half],
                                            p_lo,
                                        )
                                        # copy_lo -> copy_hi: same MTE2 pipe, disjoint
                                        # dst buffers -> in-order, no barrier needed.
                                        T.copy(
                                            ws_p[cid, pb, 0:H_per_block, BI_half:BI],
                                            p_hi,
                                        )
                                        # p copies (MTE2) -> gemm (MAD) reads p (RAW).
                                        T.set_flag("mte2", "m", 0)
                                        T.wait_flag("mte2", "m", 0)
                                        # P@V = sum over the two KV halves;
                                        # init=False accumulates the second half.
                                        T.gemm_v0(
                                            p_lo, kv_lo[pb, :, :], acc_o_l0c, init=True
                                        )
                                        T.barrier_all()
                                        T.gemm_v0(
                                            p_hi, kv_hi[pb, :, :], acc_o_l0c, init=False
                                        )
                                        # gemm (MAD) -> copy acc_o_l0c (FIX) reads it (RAW).
                                        T.set_flag("m", "fix", 1)
                                        T.wait_flag("m", "fix", 1)
                                        T.copy(
                                            acc_o_l0c,
                                            ws_o[cid, pb, 0:H_per_block, 0:D],
                                        )
                                        T.barrier_all()
                                        T.set_cross_flag("FIX", _FLAG_PV_READY)

                            # ================ VECTOR ================
                            with T.Scope("V"):
                                # Seed online softmax from sinks.
                                T.copy(
                                    Sinks[vid * v_block : vid * v_block + v_block],
                                    m_i,
                                )
                                # S1: seed acc_o = 0. cube_direct (swa/cfa) keeps the
                                # accumulator in GM (ws_acc_o), so zero each MERGE_HEADS
                                # pass via the working tile and store to GM; SCFA keeps
                                # the UB accumulator. (Single-buffer GM is safe: the slot
                                # loop is still serial, so ws_acc_o[cid] is never in
                                # flight across slots -- that becomes S2's parity dim.)
                                if cube_direct:
                                    for hp in range(N_MERGE_PASS):
                                        hb = hp * MERGE_HEADS
                                        T.tile.fill(acc_o_work, 0.0)
                                        T.barrier_all()
                                        T.copy(
                                            acc_o_work,
                                            ws_acc_o[
                                                cid,
                                                slot % 2,
                                                vid * v_block + hb : vid * v_block
                                                + hb
                                                + MERGE_HEADS,
                                                :,
                                            ],
                                        )
                                    T.barrier_all()
                                else:
                                    T.tile.fill(acc_o, 0.0)
                                T.tile.fill(sumexp, 1.0)
                                T.barrier_all()

                                # Software-pipelined (skewed) vector loop -- INLINE phases in
                                # a TIR loop (the loop var t is a TIR Var, so guards and
                                # cross-flags are TIR conditionals -- Ascend C's PreloadPipeline
                                # likewise wraps CrossCoreWaitFlag in `if`s; kv/ws/mask/alpha are
                                # indexed by TIR parity). Step t: V0(t) gather, V1(t-1) softmax,
                                # V2(t-2) merge. Guards give the prologue (t<2) / epilogue
                                # (t>=NI). Forward-flag counts stay balanced at NI each.
                                # S2b.1d-beta WAR pre-set: both acc_s_ub_ halves start
                                # writable for the score prefetch (select(t-2) re-arms
                                # half t%2 from step 2 on; drain after the loop).
                                T.set_flag("v", "mte2", 0)
                                T.set_flag("v", "mte2", 1)
                                for t in range(NI_total + 2):
                                    # ---- V0(t): gather chunk t + build mask ----
                                    if t < NI_total:
                                        c0 = t
                                        pv0 = t % 2
                                        is_ori = c0 < NI_ori

                                        # ---- gather KV + build mask ----
                                        if is_ori:
                                            chunk_start = ori_left + c0 * BI
                                            T.tile.createvecindex(
                                                idx_int,
                                                chunk_start,
                                            )
                                            T.copy(idx_int, idx_float)
                                            T.barrier_all()
                                            T.tile.compare(
                                                mask_ub[pv0, :],
                                                idx_float,
                                                T.float32(ori_right),
                                                "LE",
                                            )
                                            T.barrier_all()
                                            # ORI KV is now cube-direct (GM->L1,
                                            # AscendC form): no vector gather.
                                            if not cube_direct:
                                                # Batched gather: issue all BI//2 row
                                                # DMAs into distinct rows of
                                                # kv_ub_multi with NO per-row barrier
                                                # (disjoint dst rows -> MTE2 pipelines
                                                # them), then one barrier + one
                                                # batched write-out. Out-of-window
                                                # tokens (g_idx > ori_right) are
                                                # gathered too: the window is a prefix
                                                # of a valid causal range (g_idx <=
                                                # s_global < act_kv) so the block-table
                                                # entry is always valid, and the
                                                # additive score mask sets their
                                                # column to -inf -- the gathered value
                                                # never contributes.
                                                for gp in range(N_GATHER_PASS):
                                                    pp = gp % 2
                                                    gh = pp * GATHER_ROWS
                                                    kv_row0 = (
                                                        vid * (BI // 2)
                                                        + gp * GATHER_ROWS
                                                    )
                                                    # S2b.1b: ping-pong the gather UB half
                                                    # so gather[gp] (MTE2 into half pp)
                                                    # overlaps write[gp-1] (MTE3 from the
                                                    # other half). WAR: half pp was last
                                                    # read by write[gp-2], so wait its
                                                    # back-flag before overwriting it.
                                                    if gp >= 2:
                                                        T.wait_flag("mte3", "mte2", pp)
                                                    # Per-row gather (proven 1d-beta
                                                    # form). FUSE-V0's single-DMA
                                                    # block-copy fast path was never
                                                    # prefill-verified and is the last
                                                    # window-dependent suspect.
                                                    for r in range(GATHER_ROWS):
                                                        g_idx = (
                                                            chunk_start + kv_row0 + r
                                                        )
                                                        ori_blk = ori_block_table[
                                                            b_i,
                                                            g_idx // ori_block_size,
                                                        ]
                                                        ori_row = g_idx % ori_block_size
                                                        T.copy(
                                                            ori_KV[
                                                                ori_blk,
                                                                ori_row,
                                                                0,
                                                                :,
                                                            ],
                                                            kv_ub_multi[gh + r, :],
                                                        )
                                                    # gather[gp](MTE2) -> write[gp](MTE3)
                                                    T.set_flag("mte2", "mte3", pp)
                                                    T.wait_flag("mte2", "mte3", pp)
                                                    T.copy(
                                                        kv_ub_multi[
                                                            gh : gh + GATHER_ROWS, :
                                                        ],
                                                        ws_kv[
                                                            cid,
                                                            pv0,
                                                            kv_row0 : kv_row0
                                                            + GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                    # write[gp] done -> half pp free (gp+2)
                                                    T.set_flag("mte3", "mte2", pp)
                                        else:
                                            if is_cfa:
                                                # CFA: this chunk's compressed
                                                # token ids are the dense range
                                                # [(c0-NI_ori)*BI, +BI).
                                                # Generate them on the vector
                                                # core -- no host index array.
                                                T.tile.createvecindex(
                                                    idx_int,
                                                    (c0 - NI_ori) * BI,
                                                )
                                            else:
                                                cmp_off = (c0 - NI_ori) * BI
                                                T.copy(
                                                    cmp_indices[
                                                        t_i,
                                                        0,
                                                        cmp_off : cmp_off + BI,
                                                    ],
                                                    idx_int,
                                                )
                                                # The cmp index load is an async
                                                # GM->UB DMA. Wait for it before
                                                # idx_int is read below, else the
                                                # UB->UB copy lands the previous
                                                # cmp chunk's DMA result in
                                                # idx_float (off-by-one chunk).
                                                T.barrier_all()
                                            T.copy(idx_int, idx_float)
                                            T.barrier_all()
                                            # mask = (idx >= 0) AND (idx < thr)
                                            T.tile.compare(
                                                mask_ub[pv0, :],
                                                idx_float,
                                                T.float32(-0.5),
                                                "GT",
                                            )
                                            T.tile.compare(
                                                mask_ub_2[pv0, :],
                                                idx_float,
                                                T.float32(cmp_threshold),
                                                "LT",
                                            )
                                            T.barrier_all()
                                            T.tile.bitwise_and(
                                                mask_ub[pv0, :],
                                                mask_ub[pv0, :],
                                                mask_ub_2[pv0, :],
                                            )
                                            T.barrier_all()
                                            # Batched sparse gather. cmp indices
                                            # are arbitrary (different blocks) so
                                            # the DMAs stay per-row, but they land
                                            # in distinct rows of kv_ub_multi with
                                            # no per-row barrier (MTE2 pipelines
                                            # them) + one batched write-out.
                                            # Invalid (-1 padding) indices are
                                            # clamped to 0 so the block-table
                                            # lookup stays in range; the score
                                            # mask (idx >= 0) zeroes their
                                            # contribution, so reading token 0's
                                            # KV is harmless (matches the old
                                            # fill-0 path numerically).
                                            # CFA cmp KV is now cube-direct (GM->L1,
                                            # AscendC form), same as ori above: when
                                            # cube_direct the cube pulls cmp itself,
                                            # so skip the vector gather + ws_kv write
                                            # (and the KV_READY handshake below). Only
                                            # SCFA's sparse topK keeps this path. The
                                            # mask above is still built -- the vector
                                            # softmax needs it regardless.
                                            if not cube_direct:
                                                for gp in range(N_GATHER_PASS):
                                                    pp = gp % 2
                                                    gh = pp * GATHER_ROWS
                                                    kv_row0 = (
                                                        vid * (BI // 2)
                                                        + gp * GATHER_ROWS
                                                    )
                                                    # S2b.1b ping-pong (see ori):
                                                    # gather[gp](MTE2,half pp) overlaps
                                                    # write[gp-1](MTE3); WAR back-flag.
                                                    if gp >= 2:
                                                        T.wait_flag("mte3", "mte2", pp)
                                                    for r in range(GATHER_ROWS):
                                                        cmp_idx = idx_int[kv_row0 + r]
                                                        safe_idx = T.if_then_else(
                                                            cmp_idx < 0, 0, cmp_idx
                                                        )
                                                        cmp_blk = cmp_block_table[
                                                            b_i,
                                                            safe_idx // cmp_block_size,
                                                        ]
                                                        cmp_row = (
                                                            safe_idx % cmp_block_size
                                                        )
                                                        T.copy(
                                                            cmp_KV[
                                                                cmp_blk, cmp_row, 0, :
                                                            ],
                                                            kv_ub_multi[gh + r, :],
                                                        )
                                                    # gather[gp](MTE2)->write[gp](MTE3)
                                                    T.set_flag("mte2", "mte3", pp)
                                                    T.wait_flag("mte2", "mte3", pp)
                                                    T.copy(
                                                        kv_ub_multi[
                                                            gh : gh + GATHER_ROWS, :
                                                        ],
                                                        ws_kv[
                                                            cid,
                                                            pv0,
                                                            kv_row0 : kv_row0
                                                            + GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                    # write[gp] done -> half pp free
                                                    T.set_flag("mte3", "mte2", pp)
                                        # S2b.1c: drain the 2 dangling back-flags (the
                                        # last two passes set mte3->mte2 with no
                                        # in-loop waiter) so kv_ub_multi's two halves
                                        # are free for the next chunk's gather. The
                                        # V0-end barrier is GONE now: dropping it lets
                                        # this chunk's gather MTE2/MTE3 keep draining
                                        # into V1(t-1)'s VEC window (the overlap we
                                        # want). set_cross_flag("MTE3") stays correct
                                        # without it -- it is pipe-ordered on MTE3, so
                                        # it still fires only after the ws_kv writes
                                        # land, and cube's KV_READY wait is unaffected.
                                        # Non-cube-direct (SCFA): the vector gather
                                        # ran for EVERY chunk (ori + cmp), so every
                                        # chunk set 2 mte3->mte2 back-flags and must
                                        # drain them -- gating this by t>=NI_ori
                                        # leaked +2 per slot and deadlocked prefill
                                        # (the event counter saturates over many
                                        # query slots; decode's single slot survived).
                                        # SWA/CFA are cube_direct: no vector gather
                                        # ran, so no back-flags and no KV_READY.
                                        if not cube_direct:
                                            T.wait_flag("mte3", "mte2", 0)
                                            T.wait_flag("mte3", "mte2", 1)
                                            T.set_cross_flag("MTE3", _FLAG_KV_READY)
                                    # ---- S2b.1d-beta prologue: prefetch chunk 0 score ----
                                    # Cold start: nothing to overlap yet, just land
                                    # chunk 0 in half 0 so V1(0) finds it at t=1.
                                    # Steady-state prefetch lives inside V1 below.
                                    if t == 0:
                                        T.wait_cross_flag(_FLAG_SCORE_READY)
                                        T.wait_flag("v", "mte2", 0)
                                        T.copy(
                                            ws_score[
                                                cid,
                                                0,
                                                vid * v_block : vid * v_block + v_block,
                                                :,
                                            ],
                                            acc_s_ub_[0:v_block, :],
                                        )
                                        T.set_flag("mte2", "v", 0)
                                    # ---- V1(t-1): online softmax of chunk t-1 ----
                                    if t >= 1:
                                        if t <= NI_total:
                                            pv1 = (t - 1) % 2
                                            # ---- masked score (S2b.1d-beta) ----
                                            # Chunk t-1's score is ALREADY in half pv1
                                            # (prefetched last step), so the old
                                            # fill-0 + select-0/-inf + add(score)
                                            # collapses into ONE select straight on the
                                            # score (in-window -> score, out -> -inf),
                                            # matching AscendC. RAW: the prefetch load
                                            # (MTE2) wrote half pv1 last step.
                                            T.wait_flag("mte2", "v", pv1)
                                            # select's selMask needs a whole
                                            # Buffer (it calls .access_ptr, which
                                            # a Var-indexed parity BufferRegion
                                            # lacks), so copy chunk t-1's mask
                                            # row into the whole mask_sel first.
                                            T.copy(mask_ub[pv1, :], mask_sel)
                                            # Per-row select (proven): the FUSE-V1
                                            # whole-tile broadcast-mask select was wrong
                                            # on partial windows -- decode never hits it
                                            # (last query = full window, mask all-ones),
                                            # prefill does (early queries mask -inf).
                                            for h_i in T.serial(v_block):
                                                T.tile.select(
                                                    acc_s_ub[h_i, :],
                                                    mask_sel,
                                                    acc_s_ub_[pv1 * v_block + h_i, :],
                                                    -T.infinity(accum_dtype),
                                                    "VSEL_TENSOR_SCALAR_MODE",
                                                )
                                            # WAR (eid = half): selects are done reading
                                            # half pv1; the prefetch 2 steps ahead may
                                            # overwrite it. Balanced by the pre-set pair
                                            # before the loop + the drain after it.
                                            T.set_flag("v", "mte2", pv1)
                                            T.copy(m_i, m_i_prev)
                                            T.tile.mul(
                                                acc_s_ub,
                                                acc_s_ub,
                                                softmax_scale,
                                            )

                                            if cube_direct:
                                                # Fused online softmax (replicate Ascend
                                                # C SoftmaxFlashV2Compute, swa_block_
                                                # vector.h:349-355): ONE op does the
                                                # max-subtract + exp + row-sum + running-
                                                # state rescale that the manual chain
                                                # (SCFA else) spreads over ~8 VEC passes
                                                # -- the §3.5 (3) vector-idiom lever.
                                                # SoftmaxFlashV2 double-buffers the
                                                # running max/sum (in != out): seed prev
                                                # max into m_i_prev (the copy above) and
                                                # prev sum into sumexp_i_ub here. Outputs:
                                                # new max -> m_i, new sum -> sumexp, P ->
                                                # acc_s_ub (in place), per-row alpha =
                                                # exp(prev - new) -> alpha_exp (stashed to
                                                # the flat alpha[] for V2, like the manual
                                                # m_i_prev stash). The sink seed (m_i <-
                                                # Sinks, sumexp <- 1.0) flows in as the
                                                # chunk-0 prev state, so the sink lands in
                                                # the denominator correctly. scores were
                                                # pre-scaled above (Ascend C runs Muls /
                                                # ElewiseCompute before SoftmaxFlashV2).
                                                T.copy(sumexp, sumexp_i_ub)
                                                T.tile.softmax_flashv2(
                                                    acc_s_ub,
                                                    sumexp,
                                                    m_i,
                                                    alpha_exp,
                                                    sumexp_i_ub,
                                                    m_i_prev,
                                                    softmax_tmp,
                                                    v_block,
                                                    BI,
                                                    BI,
                                                )
                                                T.copy(
                                                    alpha_exp,
                                                    alpha[
                                                        pv1 * ub_len : pv1 * ub_len
                                                        + ub_len
                                                    ],
                                                )
                                            else:
                                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                                # m_i = max(m_i_prev, m_i): dst must be
                                                # the LAST operand. T.tile.max(m_i, m_i,
                                                # m_i_prev) silently drops m_i_prev,
                                                # leaving m_i at the chunk-local max --
                                                # the running max in m_i_prev is lost
                                                # (chunk-2 ori->cmp divergence). dst-last
                                                # matches the verified online-softmax ref.
                                                T.tile.max(m_i, m_i_prev, m_i)
                                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                                T.tile.exp(m_i_prev, m_i_prev)
                                                # Stash alpha = exp(m_prev - m_new) into
                                                # this chunk's parity half of the flat
                                                # alpha buffer so V2 (2 steps later)
                                                # applies it; m_i_prev is overwritten by
                                                # the next chunk's V1.
                                                T.copy(
                                                    m_i_prev,
                                                    alpha[
                                                        pv1 * ub_len : pv1 * ub_len
                                                        + ub_len
                                                    ],
                                                )
                                                # SCFA max-subtract: per-head scalar-fed
                                                # sub (lockstep -- a broadcast/fused op
                                                # resonates here, so SCFA keeps the loop).
                                                for h_i in range(v_block):
                                                    T.tile.sub(
                                                        acc_s_ub[h_i, :],
                                                        acc_s_ub[h_i, :],
                                                        m_i[h_i],
                                                    )
                                                T.tile.exp(acc_s_ub, acc_s_ub)
                                                T.reduce_sum(
                                                    acc_s_ub,
                                                    sumexp_i_ub,
                                                    dim=-1,
                                                )
                                                T.tile.mul(sumexp, sumexp, m_i_prev)
                                                T.tile.add(
                                                    sumexp,
                                                    sumexp,
                                                    sumexp_i_ub,
                                                )
                                            # The acc_o rescale lives in V2 now (alpha was
                                            # stashed above); V1 never touches acc_o, so
                                            # V1(t-1) and V2(t-2) run in one pipeline step
                                            # without racing on the accumulator.

                                            # ---- prefetch chunk t score (1d-beta) ----
                                            # Issued AFTER the whole softmax VEC chain:
                                            # those ops are already in flight, so the
                                            # wait_cross stall (cube finishing MM1(t))
                                            # blocks only this load, and the MTE2 DMA
                                            # then runs parallel to the cast (VEC) and
                                            # ws_p write (MTE3) below. Putting it before
                                            # the softmax would serialize MM1(t) latency
                                            # in front of the VEC work -- the gap we are
                                            # removing.
                                            if t < NI_total:
                                                T.wait_cross_flag(_FLAG_SCORE_READY)
                                                # WAR (eid t%2): select(t-2) freed half
                                                # t%2 (pre-set covers steps 1..2).
                                                T.wait_flag("v", "mte2", t % 2)
                                                T.copy(
                                                    ws_score[
                                                        cid,
                                                        t % 2,
                                                        vid * v_block : vid * v_block
                                                        + v_block,
                                                        :,
                                                    ],
                                                    acc_s_ub_[
                                                        (t % 2) * v_block : (t % 2)
                                                        * v_block
                                                        + v_block,
                                                        :,
                                                    ],
                                                )
                                                # RAW (eid t%2): V1(t) selects from this
                                                # half next step.
                                                T.set_flag("mte2", "v", t % 2)

                                            # ---- cast P, publish for cube ----
                                            T.copy(acc_s_ub, acc_s_half)
                                            # RAW flag #3: cast (VEC) writes acc_s_half,
                                            # the ws_p write (MTE3) reads it.
                                            T.set_flag("v", "mte3", 0)
                                            T.wait_flag("v", "mte3", 0)
                                            T.copy(
                                                acc_s_half,
                                                ws_p[
                                                    cid,
                                                    pv1,
                                                    vid * v_block : vid * v_block
                                                    + v_block,
                                                    :,
                                                ],
                                            )
                                            # V1-end barrier KEPT: it drains this chunk's
                                            # gather (MTE2/MTE3) + softmax (VEC) before
                                            # V2, acting as the step boundary so the
                                            # single-buffered V1 scratch (acc_s_ub/ub_/
                                            # half, m_i*, sumexp, mask_sel) is free for
                                            # the next chunk -- no cross-chunk ping-pong
                                            # needed until S2b.1d.
                                            T.barrier_all()
                                            T.set_cross_flag("MTE3", _FLAG_P_READY)
                                    # ---- V2(t-2): merge chunk t-2 into the accumulator ----
                                    if t >= 2:
                                        pv2 = (t - 2) % 2
                                        # ---- wait P@V (chunk c2), merge output ----
                                        T.wait_cross_flag(_FLAG_PV_READY)
                                        T.barrier_all()
                                        # Output recurrence acc_o = alpha*acc_o + O,
                                        # sub-tiled (S2b.0) into N_MERGE_PASS passes of
                                        # MERGE_HEADS heads so the P@V load fits the 32KB
                                        # acc_o_ub. alpha = exp(m_prev-m_new) was stashed by
                                        # V1(c2) into the pv2 half of the flat alpha buffer;
                                        # per-head SCALAR alpha[pv2*ub_len+h] (a 2D
                                        # alpha[pv2,h] would lose h -- binary_op keeps only
                                        # indices[0]). Rescale precedes add; chunk 0 has
                                        # acc_o=0 so it is a no-op. mul+add use scalar-row
                                        # acc_o[hbase+h_i] (no Var-offset range slice).
                                        # Q4: debarrier the GM round-trip + double-buffer the
                                        # work tile so pass 1's GM load (MTE2) overlaps pass
                                        # 0's rescale (VEC) + store (MTE3) -- what S1's
                                        # barrier_all walls forbade (the DMA-vs-VEC
                                        # serialization that cost the -7 on the bottleneck
                                        # vector core). N_MERGE_PASS==2 is hand-unrolled: the
                                        # work-tile (acc_o_work even / acc_o_work2 odd) must
                                        # be picked by a Python int -- a `for .. range()`
                                        # makes the index a TIR Var (can't pick a Python
                                        # buffer object, and a Var-parity UB-tile operand
                                        # hits :1499). Per-head SCALAR alpha[pv2*ub_len+h]
                                        # (a 2D alpha[pv2,h] would drop h). SCFA keeps the
                                        # original TIR loop (UB accumulator + lockstep
                                        # barriers; debarrier resonates -> unchanged).
                                        if cube_direct:
                                            # ============ pass 0 (heads 0:MERGE_HEADS) ============
                                            # L_acc(0): GM ws_acc_o -> acc_o_work (MTE2).
                                            T.copy(
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block : vid * v_block
                                                    + MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_work,
                                            )
                                            # L_o(0): GM ws_o -> acc_o_ub (MTE2). acc_o_ub is
                                            # single-buffered (UB budget; AscendC also keeps
                                            # the P@V output single-buffered).
                                            T.copy(
                                                ws_o[
                                                    cid,
                                                    pv2,
                                                    vid * v_block : vid * v_block
                                                    + MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_ub,
                                            )
                                            # loads (MTE2, in-order) -> rescale (VEC) RAW,
                                            # buf-0 eid 2.
                                            T.set_flag("mte2", "v", 2)
                                            T.wait_flag("mte2", "v", 2)
                                            # rescale acc_o_work *= alpha (brcb + row_muls,
                                            # replicate Ascend C RowMuls) then per-head add
                                            # += acc_o_ub. brcb->row_muls->add are VEC->VEC
                                            # RAW -> light pipe_barrier("v") (Ascend C
                                            # swa_block_vector.h 689/692). chunk 0 has
                                            # acc_o=0 so the rescale is a no-op. The ADD
                                            # stays per-head: a wide read of acc_o_ub races
                                            # its next-pass reload (the fa63798 WAR).
                                            T.tile.brcb(
                                                alpha_brd8,
                                                alpha[
                                                    pv2 * ub_len : pv2 * ub_len
                                                    + MERGE_HEADS
                                                ],
                                                (MERGE_HEADS + 7) // 8,
                                                1,
                                                8,
                                            )
                                            T.pipe_barrier("v")
                                            T.tile.row_muls(
                                                acc_o_work,
                                                acc_o_work,
                                                alpha_brd8,
                                                MERGE_HEADS,
                                                D,
                                                D,
                                            )
                                            T.pipe_barrier("v")
                                            for h_i in range(MERGE_HEADS):
                                                T.tile.add(
                                                    acc_o_work[h_i, :],
                                                    acc_o_work[h_i, :],
                                                    acc_o_ub[h_i, :],
                                                )
                                            # add(0) (VEC) done reading acc_o_ub -> free it
                                            # for pass 1's L_o (WAR), eid 2 (set@0 / wait@1
                                            # -> balanced, no leak: the :1259 deadlock class).
                                            T.set_flag("v", "mte2", 2)
                                            # add(0) (VEC) -> store(0) (MTE3) RAW, buf-0 eid 1.
                                            T.set_flag("v", "mte3", 1)
                                            T.wait_flag("v", "mte3", 1)
                                            # S(0): acc_o_work -> GM ws_acc_o (MTE3). No
                                            # trailing barrier: buf reuse is next-slot only
                                            # (phase barriers), and the V2->tail GM RAW + buf
                                            # WAR are drained by the tail-entry barrier.
                                            T.copy(
                                                acc_o_work,
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block : vid * v_block
                                                    + MERGE_HEADS,
                                                    :,
                                                ],
                                            )
                                            # ====== pass 1 (heads MERGE_HEADS:2*MERGE_HEADS) ======
                                            # L_acc(1) -> acc_o_work2 (MTE2): distinct buffer,
                                            # so this overlaps pass 0's VEC/store (no WAR).
                                            T.copy(
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block + MERGE_HEADS : vid
                                                    * v_block
                                                    + 2 * MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_work2,
                                            )
                                            # WAR: pass 0's add must finish reading acc_o_ub
                                            # before this L_o overwrites it (eid 2).
                                            T.wait_flag("v", "mte2", 2)
                                            # L_o(1): GM ws_o -> acc_o_ub (MTE2).
                                            T.copy(
                                                ws_o[
                                                    cid,
                                                    pv2,
                                                    vid * v_block + MERGE_HEADS : vid
                                                    * v_block
                                                    + 2 * MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_ub,
                                            )
                                            # loads (MTE2) -> rescale (VEC) RAW, buf-1 eid 3.
                                            T.set_flag("mte2", "v", 3)
                                            T.wait_flag("mte2", "v", 3)
                                            T.tile.brcb(
                                                alpha_brd8,
                                                alpha[
                                                    pv2 * ub_len + MERGE_HEADS : pv2
                                                    * ub_len
                                                    + 2 * MERGE_HEADS
                                                ],
                                                (MERGE_HEADS + 7) // 8,
                                                1,
                                                8,
                                            )
                                            T.pipe_barrier("v")
                                            T.tile.row_muls(
                                                acc_o_work2,
                                                acc_o_work2,
                                                alpha_brd8,
                                                MERGE_HEADS,
                                                D,
                                                D,
                                            )
                                            T.pipe_barrier("v")
                                            for h_i in range(MERGE_HEADS):
                                                T.tile.add(
                                                    acc_o_work2[h_i, :],
                                                    acc_o_work2[h_i, :],
                                                    acc_o_ub[h_i, :],
                                                )
                                            # no WAR set (last pass; eid 2 already balanced).
                                            # add(1) (VEC) -> store(1) (MTE3) RAW, buf-1 eid 2.
                                            T.set_flag("v", "mte3", 2)
                                            T.wait_flag("v", "mte3", 2)
                                            # S(1): acc_o_work2 -> GM ws_acc_o (MTE3).
                                            T.copy(
                                                acc_o_work2,
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block + MERGE_HEADS : vid
                                                    * v_block
                                                    + 2 * MERGE_HEADS,
                                                    :,
                                                ],
                                            )
                                        else:
                                            for mp in range(N_MERGE_PASS):
                                                hbase = mp * MERGE_HEADS
                                                # SCFA: UB accumulator, unchanged (byte-
                                                # identical to S1/S2; lockstep keeps the
                                                # per-head barriered mul/add).
                                                T.copy(
                                                    ws_o[
                                                        cid,
                                                        pv2,
                                                        vid * v_block + hbase : vid
                                                        * v_block
                                                        + hbase
                                                        + MERGE_HEADS,
                                                        :,
                                                    ],
                                                    acc_o_ub,
                                                )
                                                T.barrier_all()
                                                for h_i in range(MERGE_HEADS):
                                                    T.barrier_all()
                                                    T.tile.mul(
                                                        acc_o[hbase + h_i, :],
                                                        acc_o[hbase + h_i, :],
                                                        alpha[
                                                            pv2 * ub_len + hbase + h_i
                                                        ],
                                                    )
                                                    T.barrier_all()
                                                    T.tile.add(
                                                        acc_o[hbase + h_i, :],
                                                        acc_o[hbase + h_i, :],
                                                        acc_o_ub[h_i, :],
                                                    )
                                                    T.barrier_all()

                                # 1d-beta drain: the last two selects set v->mte2
                                # with no in-loop waiter (no prefetch in the final
                                # 2 steps); consume them so no event leaks into
                                # the next kernel launch (balances the pre-set).
                                T.wait_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 1)
                                # ---- normalize and write back ----
                                # cube_direct (swa/cfa): S1 -- acc_o is GM-resident.
                                # Per MERGE_HEADS pass, load from ws_acc_o, divide by
                                # sumexp (normalize), cast fp32->bf16 into this pass's
                                # slice of acc_o_half; then one batched Output write.
                                # The normalize is FOLDED into writeback -- the
                                # normalized acc_o has no other consumer (LSE reads
                                # sumexp/m_i, not acc_o), so no store-back to ws_acc_o.
                                # S1 re-adds the load/div/cast RAW barriers that the UB
                                # debarrier dropped (the GM round-trip needs them);
                                # accepted S1 perf cost. SCFA keeps the barriered UB
                                # form (lockstep resonance), unchanged.
                                if cube_direct:
                                    # Q4: debarrier the normalize+writeback GM round-trip.
                                    # wbuf = double-buffer tile (acc_o_work even / acc_o_
                                    # work2 odd) so pass hp+1's GM load (MTE2) overlaps pass
                                    # hp's div+cast (VEC); the casts feed one batched Output
                                    # write. S1 walled every leg with barrier_all, which
                                    # serialized DMA-vs-VEC on the bottleneck vector core.
                                    # Entry barrier: V2's stores (MTE3 -> ws_acc_o) must
                                    # land before these loads (MTE2 <- ws_acc_o) [GM RAW],
                                    # and V2's work-tile reads drain before the bufs reuse.
                                    T.barrier_all()
                                    # N_MERGE_PASS==2 hand-unrolled (same reason as the V2
                                    # merge: the double-buffer tile must be picked by a
                                    # Python int). acc_o_half slices are disjoint per pass,
                                    # so the two casts feed one batched Output write.
                                    # ============ pass 0 (heads 0:MERGE_HEADS) ============
                                    # L(0): GM ws_acc_o -> acc_o_work (MTE2).
                                    T.copy(
                                        ws_acc_o[
                                            cid,
                                            slot % 2,
                                            vid * v_block : vid * v_block + MERGE_HEADS,
                                            :,
                                        ],
                                        acc_o_work,
                                    )
                                    # L(0) (MTE2) -> div(0) (VEC) RAW, buf-0 eid 2.
                                    T.set_flag("mte2", "v", 2)
                                    T.wait_flag("mte2", "v", 2)
                                    for h_i in range(MERGE_HEADS):
                                        T.tile.div(
                                            acc_o_work[h_i, :],
                                            acc_o_work[h_i, :],
                                            sumexp[h_i],
                                        )
                                    # div(0) -> cast(0): VEC->VEC RAW (light pipe_barrier).
                                    T.pipe_barrier("v")
                                    T.copy(
                                        acc_o_work,
                                        acc_o_half[0:MERGE_HEADS, :],
                                    )
                                    # ====== pass 1 (heads MERGE_HEADS:2*MERGE_HEADS) ======
                                    # L(1) -> acc_o_work2 (MTE2): distinct buffer, overlaps
                                    # pass 0's div+cast (VEC).
                                    T.copy(
                                        ws_acc_o[
                                            cid,
                                            slot % 2,
                                            vid * v_block + MERGE_HEADS : vid * v_block
                                            + 2 * MERGE_HEADS,
                                            :,
                                        ],
                                        acc_o_work2,
                                    )
                                    # L(1) (MTE2) -> div(1) (VEC) RAW, buf-1 eid 3.
                                    T.set_flag("mte2", "v", 3)
                                    T.wait_flag("mte2", "v", 3)
                                    for h_i in range(MERGE_HEADS):
                                        T.tile.div(
                                            acc_o_work2[h_i, :],
                                            acc_o_work2[h_i, :],
                                            sumexp[MERGE_HEADS + h_i],
                                        )
                                    T.pipe_barrier("v")
                                    T.copy(
                                        acc_o_work2,
                                        acc_o_half[MERGE_HEADS : 2 * MERGE_HEADS, :],
                                    )
                                    # casts (VEC) -> Output write (MTE3) RAW. Both casts
                                    # wrote disjoint acc_o_half slices (no inter-pass WAR);
                                    # one flag after the loop covers them (VEC in-order).
                                    T.set_flag("v", "mte3", 1)
                                    T.wait_flag("v", "mte3", 1)
                                    T.copy(
                                        acc_o_half,
                                        Output[
                                            t_i,
                                            vid * v_block : vid * v_block + v_block,
                                            :,
                                        ],
                                    )
                                else:
                                    for h_i in range(v_block):
                                        T.barrier_all()
                                        T.tile.div(
                                            acc_o[h_i, :],
                                            acc_o[h_i, :],
                                            sumexp[h_i],
                                        )
                                        T.barrier_all()
                                    T.copy(acc_o, acc_o_half)
                                    T.barrier_all()
                                    T.copy(
                                        acc_o_half,
                                        Output[
                                            t_i,
                                            vid * v_block : vid * v_block + v_block,
                                            :,
                                        ],
                                    )

                                # ---- LSE epilogue ----
                                # lse[h] = m_i[h] + ln(sumexp[h])
                                # FA v2 LogSumExp identity:
                                #   lse = max_running + ln(sum_running)
                                # where sum_running is the FA v2 row_sum
                                # *after* the sink seed -- so the sink
                                # contribution exp(sink - max) is folded
                                # in automatically. The Ascend C kernel
                                # writes this same value when
                                # return_softmax_lse=True (see
                                # sparse_attn_sharedkv/op_kernel/arch32/
                                # sparse_attn_sharedkv_swa_block_vector.h
                                # `ProcessLse`).
                                T.tile.ln(lse_ub, sumexp)
                                T.barrier_all()
                                T.tile.add(lse_ub, lse_ub, m_i)
                                T.barrier_all()
                                T.copy(
                                    lse_ub,
                                    LSE_out[
                                        t_i,
                                        vid * v_block : vid * v_block + v_block,
                                    ],
                                )

        # ============================================================
        # SWA-specialized cross-slot-pipelined kernel (S3b).
        # ============================================================
        # Returned ONLY for SWA (NI_total == 1 -> cube_direct, NI_ori == 1,
        # NI_cmp == 0). It replaces the per-slot `for slot` loop (each slot
        # self-contained, cube idle during the vector tail = the S1 -7) with
        # a flat gloop that skews work across slots: cube does MM1(g) ||
        # MM2(g-1), vector does V0(g) || V1(g-1) || V2+tail(g-2). The slot
        # g-2's GM round-trip (merge + normalize + writeback + LSE) is now
        # buried under slot g-1/g's VEC -- the cross-slot interleave Q4's
        # intra-slot debarrier could not reach. Mirrors the Ascend C
        # PreloadPipeline (a flat gloop with `if AIC{..} if AIV{..}` and
        # isValid gating; the +2 trailing steps = extraLoop drain).
        #
        # The existing `sparse_attn_sharedkv` is left byte-identical (it
        # serves CFA/SCFA). This one is SWA-only, so every chunk is the
        # single ori window (c0 == 0, t == 0): the `for t` loop collapses,
        # cube_direct is always True, and the cmp / SCFA branches are dead.
        @T.prim_func
        def sparse_attn_sharedkv_swa(
            Q: T.Tensor(q_shape, dtype),  # type: ignore[valid-type]
            ori_KV: T.Tensor(ori_kv_shape, dtype),  # type: ignore[valid-type]
            ori_block_table: T.Tensor(ori_bt_shape, indices_dtype),  # type: ignore[valid-type]
            cmp_KV: T.Tensor(cmp_kv_shape, dtype),  # type: ignore[valid-type]
            cmp_block_table: T.Tensor(cmp_bt_shape, indices_dtype),  # type: ignore[valid-type]
            cmp_indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
            q_prefix: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            actual_q_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            actual_kv_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            Sinks: T.Tensor([n_heads], accum_dtype),  # type: ignore[valid-type]
            Metadata: T.Tensor([_SAS_META_SIZE], indices_dtype),  # type: ignore[valid-type]
            Output: T.Tensor(out_shape, dtype),  # type: ignore[valid-type]
            LSE_out: T.Tensor([total_tokens, n_heads], accum_dtype),  # type: ignore[valid-type]
            ws_kv: T.Tensor([core_num, 2, BI, D], dtype),  # type: ignore[valid-type]
            ws_score: T.Tensor([core_num, 2, H_per_block, BI], accum_dtype),  # type: ignore[valid-type]
            ws_p: T.Tensor([core_num, 2, H_per_block, BI], dtype),  # type: ignore[valid-type]
            ws_o: T.Tensor([core_num, 2, H_per_block, D], accum_dtype),  # type: ignore[valid-type]
            ws_acc_o: T.Tensor([core_num, 2, H_per_block, D], accum_dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- L1 / L0 (cube). ----
                # q_l1 is gloop double-buffered ([2, H, D]): MM1(g) loads slot
                # g's Q into q_l1[g%2] while MM2(g-1) still reads slot g-1's
                # gemm operands. L1 repacked so nothing overlaps: q_l1 @0
                # (128KB), kv_lo @128KB, kv_hi @256KB, p_lo @384KB, p_hi @392KB
                # -> 400KB <= 512KB.
                q_l1 = T.alloc_L1([2, H_per_block, D], dtype)
                kv_lo = T.alloc_L1([2, BI_half, D], dtype)
                kv_hi = T.alloc_L1([2, BI_half, D], dtype)
                p_lo = T.alloc_L1([H_per_block, BI_half], dtype)
                p_hi = T.alloc_L1([H_per_block, BI_half], dtype)
                acc_s_l0c = T.alloc_L0C([H_per_block, BI_half], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                # ---- UB (vector). ---- (SWA subset of the main kernel; see
                # the byte-identical decls above for the full commentary.)
                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_work = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_work2 = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_ub = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                m_i = T.alloc_ub([ub_len], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
                sinks_ub = T.alloc_ub([ub_len], accum_dtype)
                lse_ub = T.alloc_ub([ub_len], accum_dtype)
                alpha = T.alloc_ub([2 * ub_len], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_ub_ = T.alloc_ub([2 * v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)
                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                kv_ub_multi = T.alloc_ub([2 * GATHER_ROWS, D], dtype)
                softmax_tmp = T.alloc_ub([16 * KB], "uint8")
                alpha_exp = T.alloc_ub([ub_len], accum_dtype)
                alpha_brd8 = T.alloc_ub([MERGE_HEADS, 8], accum_dtype)
                mask_ub = T.alloc_ub([2, mask_w], "uint8")
                mask_sel = T.alloc_ub([mask_w], "uint8")
                # S3b carry buffers: V1(g-1) writes slot g-1's new softmax
                # state (sumexp / m_i) into the (g-1)%2 half while V2+tail(g-2)
                # still reads slot g-2's OLD state -- single-buffered sumexp /
                # m_i would be clobbered, so save the per-slot state by parity
                # ([2]) and restore the single-buffer copy the tail's
                # ln/add/div consume (a flat [2] region copy is :1499-safe; the
                # tail's tile-ops then read the single-buffer *_rt, no Var
                # parity on a tile-op operand). sumexp_sv / m_i_sv (parity
                # save) are [2, ub_len]; sumexp_rt / m_i_rt (restore target)
                # are [ub_len].
                sumexp_sv = T.alloc_ub([2, ub_len], accum_dtype)
                m_i_sv = T.alloc_ub([2, ub_len], accum_dtype)
                sumexp_rt = T.alloc_ub([ub_len], accum_dtype)
                m_i_rt = T.alloc_ub([ub_len], accum_dtype)

                T.annotate_address(
                    {
                        # L1 REPACKED for the gloop-double-buffered q_l1
                        # ([2, H, D] = 128KB, vs the main kernel's single-buffer
                        # 64KB): q_l1 @0 (128KB), kv_lo @128KB (128KB), kv_hi
                        # @256KB (128KB), p_lo @384KB (8KB), p_hi @392KB (8KB)
                        # -> 400KB <= 512KB. The shared l1_addr dict (q_l1 @0,
                        # kv_lo @64KB) sizes the single-buffer q_l1 and would
                        # make kv_lo overlap the doubled q_l1, so use literals.
                        q_l1: 0,
                        kv_lo: 128 * KB,
                        kv_hi: 256 * KB,
                        p_lo: 384 * KB,
                        p_hi: 392 * KB,
                        acc_s_l0c: l0c_addr["acc_s_l0c"],
                        acc_o_l0c: l0c_addr["acc_o_l0c"],
                        acc_o: ub_addr["acc_o"],
                        acc_s_ub: ub_addr["acc_s_ub"],
                        acc_s_ub_: ub_addr["acc_s_ub_"],
                        acc_s_half: ub_addr["acc_s_half"],
                        m_i: ub_addr["m_i"],
                        m_i_prev: ub_addr["m_i_prev"],
                        sumexp: ub_addr["sumexp"],
                        sumexp_i_ub: ub_addr["sumexp_i_ub"],
                        sinks_ub: ub_addr["sinks_ub"],
                        lse_ub: ub_addr["lse_ub"],
                        idx_int: ub_addr["idx_int"],
                        idx_float: ub_addr["idx_float"],
                        alpha: ub_addr["alpha"],
                        kv_ub_multi: ub_addr["kv_ub_multi"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_sel: ub_addr["mask_sel"],
                        acc_o_ub: ub_addr["acc_o_ub"],
                        acc_o_half: ub_addr["acc_o_half"],
                        # cube_direct V1/V2 scratch alias the idle kv_ub_multi
                        # (same layout as the main kernel's second annotate).
                        softmax_tmp: ub_addr["kv_ub_multi"],
                        alpha_brd8: ub_addr["kv_ub_multi"] + 16 * KB,
                        alpha_exp: ub_addr["kv_ub_multi"] + 16 * KB + 512,
                        acc_o_work: ub_addr["acc_o"],
                        acc_o_work2: ub_addr["acc_o"] + 32 * KB,
                        # S3b carry buffers packed onto the UB tail after
                        # mask_sel (mask_sel ends at +2336). sumexp_sv /
                        # m_i_sv are 2*ub_len*4 = 256B each; sumexp_rt /
                        # m_i_rt are ub_len*4 = 128B each. New top = 176KB +
                        # 3104 (~179KB) keeps a ~13KB tail for hidden tmps.
                        sumexp_sv: ub_addr["mask_sel"] + 32,
                        m_i_sv: ub_addr["mask_sel"] + 32 + 256,
                        sumexp_rt: ub_addr["mask_sel"] + 32 + 512,
                        m_i_rt: ub_addr["mask_sel"] + 32 + 640,
                    }
                )

                # ---- Read this AIC core's metadata row. ----
                meta_base = cid * _FA_METADATA_SIZE
                core_enable = Metadata[meta_base + _FA_CORE_ENABLE_INDEX]
                bn2_start = Metadata[meta_base + _FA_BN2_START_INDEX]
                m_start = Metadata[meta_base + _FA_M_START_INDEX]
                bn2_end = Metadata[meta_base + _FA_BN2_END_INDEX]
                m_end = Metadata[meta_base + _FA_M_END_INDEX]
                linear_start = bn2_start * max_seq + m_start
                linear_end = bn2_end * max_seq + m_end
                total_work = batch * max_seq

                # ============ CUBE (flat gloop) ============
                # Step g: MM1(g) [Q@K^T of slot g] then MM2(g-1) [P@V of slot
                # g-1]. +2 trailing steps drain MM2/V1/V2 for the last slots.
                # Every cross-flag's set and its matching wait are guarded by
                # the SAME valid(slot): SCORE_READY set@MM1(g)/valid0,
                # wait@V0(g)/valid0; P_READY set@V1(g-1)/valid1,
                # wait@MM2(g-1)/valid1; PV_READY set@MM2(g-1)/valid1,
                # wait@V2(g-2)/valid2. Over g in [0, total_work+2) all three
                # phases iterate the same valid-slot set, so set-count ==
                # wait-count per flag -> no leak -> no prefill deadlock.
                with T.Scope("C"):
                    for g in T.serial(total_work + 2):
                        # ---- decode(g) (OOB-safe; off = g >= 0 always) ----
                        pid0 = linear_start + g
                        in_range0 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid0 < linear_end, pid0 >= linear_start, False
                            ),
                            False,
                        )
                        b0_safe = T.if_then_else(in_range0, pid0 // max_seq, 0)
                        s0 = pid0 % max_seq
                        act_q0 = actual_q_len[b0_safe]
                        act_kv0 = actual_kv_len[b0_safe]
                        valid0 = T.if_then_else(in_range0, s0 < act_q0, False)
                        t0 = q_prefix[b0_safe] + s0
                        s_global0 = act_kv0 - act_q0 + s0
                        ori_left0_raw = s_global0 - ori_win_left
                        ori_left0 = T.if_then_else(ori_left0_raw < 0, 0, ori_left0_raw)
                        # ---- decode(g-1) (off = g-1 can be < 0 in prologue;
                        # in_range1 requires pid1 >= linear_start, which folds
                        # in off >= 0) ----
                        pid1 = linear_start + g - 1
                        in_range1 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid1 < linear_end, pid1 >= linear_start, False
                            ),
                            False,
                        )
                        b1_safe = T.if_then_else(in_range1, pid1 // max_seq, 0)
                        s1 = pid1 % max_seq
                        act_q1 = actual_q_len[b1_safe]
                        valid1 = T.if_then_else(in_range1, s1 < act_q1, False)

                        # ---- MM1(g): Q@K^T of slot g ----
                        if valid0:
                            T.copy(Q[t0, 0:n_heads, 0:D], q_l1[g % 2, :, :])
                            T.barrier_all()
                            pa = g % 2
                            # CUBE-DIRECT KV (AscendC form): SWA's single ori
                            # window is pulled GM->L1 by the cube itself, 16-row
                            # blocks, with the paged-block-boundary split (the
                            # window start ori_left0 is not block-aligned). The
                            # chunk offset is 0 (SWA NI_ori == 1, the old t was
                            # always 0), so g0 = ori_left0 + gp*GATHER_ROWS for
                            # the lo half and + BI_half for the hi half.
                            for gp in range(BI_half // GATHER_ROWS):
                                g0 = ori_left0 + gp * GATHER_ROWS
                                bidx = g0 // ori_block_size
                                rowc = g0 % ori_block_size
                                if ori_block_size - rowc >= GATHER_ROWS:
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx],
                                            rowc : rowc + GATHER_ROWS,
                                            0,
                                            :,
                                        ],
                                        kv_lo[
                                            pa,
                                            gp * GATHER_ROWS : (gp + 1) * GATHER_ROWS,
                                            :,
                                        ],
                                    )
                                else:
                                    n0 = ori_block_size - rowc
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx],
                                            rowc : rowc + n0,
                                            0,
                                            :,
                                        ],
                                        kv_lo[
                                            pa,
                                            gp * GATHER_ROWS : gp * GATHER_ROWS + n0,
                                            :,
                                        ],
                                    )
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx + 1],
                                            0 : GATHER_ROWS - n0,
                                            0,
                                            :,
                                        ],
                                        kv_lo[
                                            pa,
                                            gp * GATHER_ROWS + n0 : (gp + 1)
                                            * GATHER_ROWS,
                                            :,
                                        ],
                                    )
                            for gp in range(BI_half // GATHER_ROWS):
                                g0 = ori_left0 + BI_half + gp * GATHER_ROWS
                                bidx = g0 // ori_block_size
                                rowc = g0 % ori_block_size
                                if ori_block_size - rowc >= GATHER_ROWS:
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx],
                                            rowc : rowc + GATHER_ROWS,
                                            0,
                                            :,
                                        ],
                                        kv_hi[
                                            pa,
                                            gp * GATHER_ROWS : (gp + 1) * GATHER_ROWS,
                                            :,
                                        ],
                                    )
                                else:
                                    n0 = ori_block_size - rowc
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx],
                                            rowc : rowc + n0,
                                            0,
                                            :,
                                        ],
                                        kv_hi[
                                            pa,
                                            gp * GATHER_ROWS : gp * GATHER_ROWS + n0,
                                            :,
                                        ],
                                    )
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx + 1],
                                            0 : GATHER_ROWS - n0,
                                            0,
                                            :,
                                        ],
                                        kv_hi[
                                            pa,
                                            gp * GATHER_ROWS + n0 : (gp + 1)
                                            * GATHER_ROWS,
                                            :,
                                        ],
                                    )
                            T.barrier_all()
                            # MM1 cube debarrier (perf lever 1): targeted pipe
                            # flags on the serial MAD->FIX->MAD->FIX chain on the
                            # single acc_s_l0c (see the main kernel for the full
                            # commentary). Cube pipe flags live on AIC, disjoint
                            # from the V-scope's AIV flag ids.
                            T.gemm_v0(
                                q_l1[g % 2, :, :],
                                kv_lo[pa, :, :],
                                acc_s_l0c,
                                transpose_B=True,
                                init=True,
                            )
                            T.set_flag("m", "fix", 0)  # gemm_lo -> copy_lo
                            T.wait_flag("m", "fix", 0)
                            T.copy(
                                acc_s_l0c,
                                ws_score[cid, pa, 0:H_per_block, 0:BI_half],
                            )
                            T.set_flag("fix", "m", 1)  # copy_lo -> gemm_hi WAR
                            T.wait_flag("fix", "m", 1)
                            T.gemm_v0(
                                q_l1[g % 2, :, :],
                                kv_hi[pa, :, :],
                                acc_s_l0c,
                                transpose_B=True,
                                init=True,
                            )
                            T.set_flag("m", "fix", 2)  # gemm_hi -> copy_hi
                            T.wait_flag("m", "fix", 2)
                            T.copy(
                                acc_s_l0c,
                                ws_score[cid, pa, 0:H_per_block, BI_half:BI],
                            )
                            T.barrier_all()
                            T.set_cross_flag("FIX", _FLAG_SCORE_READY)
                        # ---- MM2(g-1): P@V of slot g-1 ----
                        # MM2 references only ws_p / kv / ws_o by parity (cid +
                        # (g-1)%2), no decode scalars, so it needs only the
                        # parity. Guarded by valid1 (matches V1(g-1)'s P_READY
                        # set and V2(g-2)'s PV_READY wait over the loop).
                        if valid1:
                            pb = (g - 1) % 2
                            T.wait_cross_flag(_FLAG_P_READY)
                            T.barrier_all()
                            T.copy(
                                ws_p[cid, pb, 0:H_per_block, 0:BI_half],
                                p_lo,
                            )
                            T.copy(
                                ws_p[cid, pb, 0:H_per_block, BI_half:BI],
                                p_hi,
                            )
                            T.set_flag("mte2", "m", 0)
                            T.wait_flag("mte2", "m", 0)
                            T.gemm_v0(p_lo, kv_lo[pb, :, :], acc_o_l0c, init=True)
                            T.barrier_all()
                            T.gemm_v0(p_hi, kv_hi[pb, :, :], acc_o_l0c, init=False)
                            T.set_flag("m", "fix", 1)
                            T.wait_flag("m", "fix", 1)
                            T.copy(
                                acc_o_l0c,
                                ws_o[cid, pb, 0:H_per_block, 0:D],
                            )
                            T.barrier_all()
                            T.set_cross_flag("FIX", _FLAG_PV_READY)

                # ============ VECTOR (flat gloop) ============
                # Step g: V0(g) seed + mask + score-prefetch, V1(g-1) softmax,
                # V2+tail(g-2) merge + normalize + writeback + LSE. The score
                # ping-pong (eid 0/1) pre-set / drain is moved OUTSIDE the gloop
                # (per-slot it leaked across the now-merged steps). Each gloop
                # step has exactly one set/wait of each score-machine flag, so
                # the steady-state ping-pong stays balanced (pre-set arms the
                # cold start, drain consumes the final two dangling sets).
                with T.Scope("V"):
                    T.set_flag("v", "mte2", 0)
                    T.set_flag("v", "mte2", 1)
                    for g in T.serial(total_work + 2):
                        # ---- decode(g) ----
                        pid0 = linear_start + g
                        in_range0 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid0 < linear_end, pid0 >= linear_start, False
                            ),
                            False,
                        )
                        b0_safe = T.if_then_else(in_range0, pid0 // max_seq, 0)
                        s0 = pid0 % max_seq
                        act_q0 = actual_q_len[b0_safe]
                        act_kv0 = actual_kv_len[b0_safe]
                        valid0 = T.if_then_else(in_range0, s0 < act_q0, False)
                        s_global0 = act_kv0 - act_q0 + s0
                        ori_right0 = s_global0
                        ori_left0_raw = s_global0 - ori_win_left
                        ori_left0 = T.if_then_else(ori_left0_raw < 0, 0, ori_left0_raw)
                        # ---- decode(g-1) ----
                        pid1 = linear_start + g - 1
                        in_range1 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid1 < linear_end, pid1 >= linear_start, False
                            ),
                            False,
                        )
                        b1_safe = T.if_then_else(in_range1, pid1 // max_seq, 0)
                        s1 = pid1 % max_seq
                        act_q1 = actual_q_len[b1_safe]
                        valid1 = T.if_then_else(in_range1, s1 < act_q1, False)
                        # ---- decode(g-2) ----
                        pid2 = linear_start + g - 2
                        in_range2 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid2 < linear_end, pid2 >= linear_start, False
                            ),
                            False,
                        )
                        b2_safe = T.if_then_else(in_range2, pid2 // max_seq, 0)
                        s2 = pid2 % max_seq
                        act_q2 = actual_q_len[b2_safe]
                        valid2 = T.if_then_else(in_range2, s2 < act_q2, False)
                        t2 = q_prefix[b2_safe] + s2

                        # ---- V0(g): seed slot g's GM accumulator + build mask
                        # + prefetch score ----
                        # DEVIATION from the spec's literal V0 placement (see
                        # the report): the softmax max/sum seed (m_i <- Sinks,
                        # sumexp <- 1.0) is NOT seeded here. Seeding single-buffer
                        # m_i/sumexp in V0(g) is only correct if V0(g) runs in the
                        # same step that V1(g-1) consumes them; but V0(g) is
                        # skipped whenever slot g is padded/past-end while slot
                        # g-1 is still valid (every batch's last query), leaving
                        # m_i/sumexp holding slot g-2's stale softmax output. The
                        # Ascend C blueprint reads the seed from a persistent
                        # never-clobbered default buffer as the softmax INPUT
                        # (swa_block_vector.h InitSoftmaxDefaultBuffer +
                        # SoftmaxFlashV2Compute isFirstSInnerLoop), so the seed
                        # belongs WITH the softmax. SWA is single-chunk -> the
                        # seed is the softmax input for EVERY slot, slot-
                        # independent, so seeding it at the top of V1(g-1) (under
                        # valid1) is numerically identical and robust. Only the
                        # GM-accumulator zero-fill (genuinely per-slot-g, parity
                        # g%2) stays here.
                        if valid0:
                            pv0 = g % 2
                            # No GM-accumulator seed: SWA is single-chunk, so V2+
                            # tail normalizes ws_o directly and ws_acc_o is unused
                            # (a dead workspace arg, like SCFA tolerates -- the jit
                            # still auto-allocs it). This drops the per-slot zero-
                            # fill's GM stores entirely.
                            # ---- build mask for slot g (LE compare) ----
                            # SWA: the single ori chunk starts at ori_left0
                            # (c0 == 0, so no c0*BI term). No vector gather
                            # (cube_direct). mask_ub[g%2] is read by V1(g) next
                            # gloop step.
                            chunk_start = ori_left0
                            T.tile.createvecindex(idx_int, chunk_start)
                            T.copy(idx_int, idx_float)
                            T.barrier_all()
                            T.tile.compare(
                                mask_ub[pv0, :],
                                idx_float,
                                T.float32(ori_right0),
                                "LE",
                            )
                            T.barrier_all()
                            # ---- prefetch slot g's score into acc_s_ub_[g%2] ----
                            # Score for slot g lands in half g%2, consumed by
                            # V1(g) next gloop step. Replaces the in-V1
                            # steady-state prefetch (dead for SWA: t < NI_total
                            # is 0 < 1 only at the cold start; the V0 prologue
                            # form below runs every gloop step).
                            T.wait_cross_flag(_FLAG_SCORE_READY)
                            T.wait_flag("v", "mte2", g % 2)
                            T.copy(
                                ws_score[
                                    cid,
                                    g % 2,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                                acc_s_ub_[
                                    (g % 2) * v_block : (g % 2) * v_block + v_block,
                                    :,
                                ],
                            )
                            T.set_flag("mte2", "v", g % 2)

                        # ---- V1(g-1): online softmax of slot g-1 ----
                        if valid1:
                            pv1 = (g - 1) % 2
                            # Seed slot g-1's online-softmax PREV state here (NOT
                            # in V0): SWA is single-chunk, so the softmax input is
                            # always the slot-independent seed (m_i <- Sinks,
                            # sumexp <- 1.0), exactly the Ascend C default-buffer
                            # seed (swa_block_vector.h SoftmaxFlashV2Compute
                            # isFirstSInnerLoop). Seeding under valid1 makes it
                            # robust when V0(g) is skipped (slot g padded while
                            # slot g-1 is the batch's last valid query). The
                            # barrier_all drains the Sinks GM->UB DMA (MTE2)
                            # before m_i / sumexp are read into the softmax inputs
                            # below; it does not touch flag counters, so the
                            # score-machine v<->mte2 balance is unaffected
                            # (mirrors the original seed->barrier_all->loop
                            # discipline).
                            T.copy(
                                Sinks[vid * v_block : vid * v_block + v_block],
                                m_i,
                            )
                            T.tile.fill(sumexp, 1.0)
                            T.barrier_all()
                            # Slot g-1's score is in half pv1 (prefetched by
                            # V0(g-1) last step). RAW: that prefetch (MTE2) wrote
                            # half pv1.
                            T.wait_flag("mte2", "v", pv1)
                            T.copy(mask_ub[pv1, :], mask_sel)
                            for h_i in T.serial(v_block):
                                T.tile.select(
                                    acc_s_ub[h_i, :],
                                    mask_sel,
                                    acc_s_ub_[pv1 * v_block + h_i, :],
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                            # WAR (eid pv1): selects done reading half pv1; the
                            # V0 prefetch one gloop step ahead may overwrite it.
                            # Balanced by the pre-set pair before the loop + the
                            # drain after it.
                            T.set_flag("v", "mte2", pv1)
                            T.copy(m_i, m_i_prev)
                            T.tile.mul(acc_s_ub, acc_s_ub, softmax_scale)
                            # Fused online softmax (replicate Ascend C
                            # SoftmaxFlashV2Compute): one op does max-subtract +
                            # exp + row-sum + running-state rescale. Seed prev
                            # sum into sumexp_i_ub; outputs new max -> m_i, new
                            # sum -> sumexp, P -> acc_s_ub, alpha = exp(prev-new)
                            # -> alpha_exp (stashed to the flat alpha[] for V2).
                            T.copy(sumexp, sumexp_i_ub)
                            T.tile.softmax_flashv2(
                                acc_s_ub,
                                sumexp,
                                m_i,
                                alpha_exp,
                                sumexp_i_ub,
                                m_i_prev,
                                softmax_tmp,
                                v_block,
                                BI,
                                BI,
                            )
                            T.copy(
                                alpha_exp,
                                alpha[pv1 * ub_len : pv1 * ub_len + ub_len],
                            )
                            # ---- S3b carry SAVE: stash slot g-1's new softmax
                            # state by parity so V2+tail(g-2) keeps reading slot
                            # g-2's OLD single-buffer state until it restores its
                            # own. Flat [2] region copies (Var parity dst) are
                            # :1499-safe (no tile-op operand). ----
                            T.copy(sumexp, sumexp_sv[pv1, :])
                            T.copy(m_i, m_i_sv[pv1, :])
                            # ---- cast P, publish for cube ----
                            T.copy(acc_s_ub, acc_s_half)
                            # RAW flag: cast (VEC) writes acc_s_half, the ws_p
                            # write (MTE3) reads it (score-machine v->mte3 eid 0).
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(
                                acc_s_half,
                                ws_p[
                                    cid,
                                    pv1,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                            )
                            # V1-end barrier KEPT: drains this slot's softmax
                            # (VEC) before P_READY publishes; step boundary for
                            # the single-buffered V1 scratch (acc_s_ub/half,
                            # m_i*, mask_sel).
                            T.barrier_all()
                            T.set_cross_flag("MTE3", _FLAG_P_READY)

                        # ---- V2+tail(g-2): merge slot g-2 + normalize + LSE ----
                        if valid2:
                            pv2 = (g - 2) % 2
                            T.wait_cross_flag(_FLAG_PV_READY)
                            T.barrier_all()
                            # SWA is SINGLE-CHUNK, so the online-softmax output
                            # accumulator is trivially acc = 0*alpha + O = O -- O is
                            # exactly the P@V result the cube wrote to ws_o. There is
                            # NO cross-chunk accumulation, so there is no accumulator
                            # to carry: the Ascend C blueprint normalizes its O buffer
                            # (vec2ResGm) directly. So V2+tail loads ws_o, divides by
                            # the carried sumexp, writes Output -- dropping the entire
                            # ws_acc_o GM round-trip (S1's GM accumulator was the -7
                            # source AND redundant for SWA) and the now-no-op rescale.
                            # This ALSO fixes a correctness bug: ws_acc_o's 2-deep
                            # parity cannot survive the accumulator across the 3-gloop-
                            # step skew -- V0(g)'s zero of half g%2 is clobbered by
                            # V2(g-2)'s store to the SAME half (g%2 == (g-2)%2), and the
                            # re-zero that V2(g-2) actually relies on is V0(g+2)'s, which
                            # is SKIPPED for the last valid slots (g+2 past-end / padded)
                            # -> V2 would read a stale O(g-4). Reading ws_o (freshly
                            # written by MM2(g-2), gated by PV_READY) has none of this.
                            # carry RESTORE: slot g-2's saved softmax state -> the
                            # single-buffer *_rt the div/ln/add consume (single buffer
                            # keeps tile-ops off Var parity -> :1499-safe).
                            T.copy(sumexp_sv[pv2, :], sumexp_rt)
                            T.copy(m_i_sv[pv2, :], m_i_rt)
                            # restore (MTE2 UB->UB) -> div/ln (VEC) read *_rt: drain.
                            T.barrier_all()
                            # 2-pass debarriered normalize: pass 1's ws_o load (MTE2)
                            # overlaps pass 0's div (VEC). N_MERGE_PASS == 2 unrolled
                            # (the work tile must be picked by a Python int).
                            # ============ pass 0 (heads 0:MERGE_HEADS) ============
                            T.copy(
                                ws_o[
                                    cid,
                                    pv2,
                                    vid * v_block : vid * v_block + MERGE_HEADS,
                                    :,
                                ],
                                acc_o_work,
                            )
                            T.set_flag("mte2", "v", 2)
                            T.wait_flag("mte2", "v", 2)
                            for h_i in range(MERGE_HEADS):
                                T.tile.div(
                                    acc_o_work[h_i, :],
                                    acc_o_work[h_i, :],
                                    sumexp_rt[h_i],
                                )
                            T.pipe_barrier("v")
                            T.copy(
                                acc_o_work,
                                acc_o_half[0:MERGE_HEADS, :],
                            )
                            # ====== pass 1 (heads MERGE_HEADS:2*MERGE_HEADS) ======
                            T.copy(
                                ws_o[
                                    cid,
                                    pv2,
                                    vid * v_block + MERGE_HEADS : vid * v_block
                                    + 2 * MERGE_HEADS,
                                    :,
                                ],
                                acc_o_work2,
                            )
                            T.set_flag("mte2", "v", 3)
                            T.wait_flag("mte2", "v", 3)
                            for h_i in range(MERGE_HEADS):
                                T.tile.div(
                                    acc_o_work2[h_i, :],
                                    acc_o_work2[h_i, :],
                                    sumexp_rt[MERGE_HEADS + h_i],
                                )
                            T.pipe_barrier("v")
                            T.copy(
                                acc_o_work2,
                                acc_o_half[MERGE_HEADS : 2 * MERGE_HEADS, :],
                            )
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            T.copy(
                                acc_o_half,
                                Output[
                                    t2,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                            )
                            # ---- LSE epilogue: lse = m_i + ln(sumexp), reading
                            # the restored slot g-2 state (*_rt). ----
                            T.tile.ln(lse_ub, sumexp_rt)
                            T.barrier_all()
                            T.tile.add(lse_ub, lse_ub, m_i_rt)
                            T.barrier_all()
                            T.copy(
                                lse_ub,
                                LSE_out[
                                    t2,
                                    vid * v_block : vid * v_block + v_block,
                                ],
                            )
                    # Score-machine drain: the last two V0 prefetches set
                    # mte2->v with their V1 consumer in the next step; the final
                    # two V1 selects set v->mte2 with no V0 consumer left.
                    # Consume the 2 dangling v->mte2 (balances the pre-set).
                    T.wait_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 1)

        return sparse_attn_sharedkv_swa if NI_total == 1 else sparse_attn_sharedkv

    return _make()
