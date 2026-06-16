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
    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15, 16])
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
                # V1 softmax max-subtract broadcast scratch (perf lever 2).
                # m_i [v_block] -> m_i_brd [v_block, BI] so the per-head subtract
                # loop (v_block tiny scalar-fed VEC ops) collapses to one broadcast
                # + one full-tile sub (the reference attention idiom). Used ONLY
                # when cube_direct (swa/cfa); SCFA keeps the per-head loop (it needs
                # kv_ub_multi for the gather, and the broadcast resonates in its
                # lockstep -- tilelang-perf skill "broadcast row sub").
                # Allocated unconditionally (tvmscript block-scopes a buffer
                # declared inside `if cube_direct:`, so it would be invisible at the
                # annotate/broadcast scopes). The ALIAS, not the alloc, is what hurt
                # SCFA: annotating m_i_brd onto kv_ub_multi made the compiler add
                # conservative syncs around SCFA's gather (+4ms). So only cube_direct
                # annotates it onto the idle kv_ub_multi (below); for SCFA m_i_brd
                # stays unannotated (auto-placed, NOT aliased to the gather buffer).
                m_i_brd = T.alloc_ub([v_block, BI], accum_dtype)
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
                # m_i_brd (perf lever 2 broadcast scratch) aliases the
                # cube_direct-idle kv_ub_multi. A SEPARATE call keyed only here
                # keeps the buffer out of SCFA's IR entirely (SCFA must not see
                # the alias -- it adds conservative syncs around its gather,
                # measured +4ms). annotate_address accumulates: each call sets
                # addresses for its listed buffers, leaving the rest as placed
                # by the call above. A conditional dict-unpack inside the main
                # literal is rejected by the tvmscript parser, so this is a
                # plain second call under the compile-time cube_direct guard.
                if cube_direct:
                    T.annotate_address(
                        {
                            m_i_brd: ub_addr["kv_ub_multi"],
                            # alpha_brd8 (V2 rescale brcb scratch, 512B) -> upper
                            # half of the idle kv_ub_multi, disjoint from m_i_brd's
                            # lower 16KB. Without this it is auto-placed and collides
                            # with a live buffer (the brcb write corrupts it ->
                            # prefill red even though the NI_total=1 rescale is a
                            # 0*alpha no-op). cube_direct-only so SCFA never sees it.
                            alpha_brd8: ub_addr["kv_ub_multi"] + 16 * KB,
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

                                            T.reduce_max(acc_s_ub, m_i, dim=-1)
                                            # m_i = max(m_i_prev, m_i): the dst must
                                            # be the LAST operand. T.tile.max in the
                                            # form T.tile.max(m_i, m_i, m_i_prev)
                                            # silently drops m_i_prev, leaving m_i at
                                            # the chunk-local max -- the running max
                                            # carried in m_i_prev was lost, which is
                                            # the chunk-2 ori->cmp divergence. This
                                            # dst-last form matches the verified
                                            # example_online_softmax.py.
                                            T.tile.max(m_i, m_i_prev, m_i)
                                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                                            T.tile.exp(m_i_prev, m_i_prev)
                                            # Stash the rescale factor alpha = exp(m_prev
                                            # - m_new) into this chunk's parity HALF of the
                                            # flat alpha buffer so V2 of the same chunk
                                            # (which runs 2 pipeline steps later) applies
                                            # it; m_i_prev itself is overwritten by the
                                            # next chunk's V1. Var-offset 1D slice (same
                                            # form as the gather's ws_kv vid-slice write).
                                            T.copy(
                                                m_i_prev,
                                                alpha[
                                                    pv1 * ub_len : pv1 * ub_len + ub_len
                                                ],
                                            )

                                            # Softmax max-subtract. cube_direct
                                            # (swa/cfa): one broadcast m_i[v_block]->
                                            # [v_block,BI] + one full-tile sub (the
                                            # reference idiom), replacing v_block
                                            # scalar-fed VEC ops -> cuts the per-head
                                            # scalar loads (27% aiv_scalar, swa
                                            # profile). SCFA keeps the per-head loop
                                            # (kv_ub_multi busy; broadcast resonates
                                            # in lockstep).
                                            if cube_direct:
                                                T.tile.broadcast(m_i_brd, m_i)
                                                T.tile.sub(acc_s_ub, acc_s_ub, m_i_brd)
                                            else:
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
                                        for mp in range(N_MERGE_PASS):
                                            hbase = mp * MERGE_HEADS
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
                                            # Per-head rescale + merge. cube_direct
                                            # (swa/cfa, perf lever 2): drop the per-
                                            # head barrier_all -- the mul (rescale by
                                            # scalar alpha) and add are same-VEC-pipe
                                            # in-order, distinct/RAW rows, and acc_o_ub
                                            # was already drained by the barrier above
                                            # -- so the ~3*MERGE_HEADS barriers/pass (a
                                            # big slice of the vector bubble, swa
                                            # profile) are redundant. Ops stay per-head
                                            # single-row (acc_o[h,:]); the coalesced
                                            # range-slice add was reverted (see the
                                            # if-block below). SCFA keeps the
                                            # barriered form: debarrier resonates in
                                            # its lockstep (skill FUSE-V2 note).
                                            if cube_direct:
                                                # Rescale acc_o[h,:] *= alpha[h], then
                                                # per-head add. perf lever (§3.5 ③,
                                                # replicate Ascend C RowMuls): the
                                                # MERGE_HEADS scalar muls collapse to one
                                                # brcb (alpha slice -> [MERGE_HEADS,8]
                                                # block) + one row_muls (strided Mul that
                                                # broadcasts each row's block across all D
                                                # cols, issuing exactly D/64 repeats so the
                                                # full row is written). This is the CORRECT
                                                # rescale vectorization: the two earlier
                                                # tries both regressed -- fa63798 (one wide
                                                # range-slice ADD) hit a cross-pass WAR on
                                                # acc_o_ub (decode), b30b447 (AscendC
                                                # Broadcast alpha -> [MERGE_HEADS,D]) hit
                                                # AscendC Broadcast's wide-dst bug (last
                                                # 64-col block stale, prefill ~87%). The ADD
                                                # stays per-head: acc_o_ub IS overwritten by
                                                # the next pass's T.copy (MTE2); a wide read
                                                # races that copy (the fa63798 bug).
                                                abase = pv2 * ub_len + hbase
                                                T.tile.brcb(
                                                    alpha_brd8,
                                                    alpha[abase : abase + MERGE_HEADS],
                                                    (MERGE_HEADS + 7) // 8,
                                                    1,
                                                    8,
                                                )
                                                # Sync (matches Ascend C swa_block_vector.h
                                                # 689/692/694-696): brcb->row_muls and
                                                # row_muls->add are VEC->VEC RAW -> light
                                                # PIPE_V (pipe_barrier "v"). The add->next-
                                                # pass T.copy(ws_o, acc_o_ub) is a VEC->MTE2
                                                # WAR on acc_o_ub (the fa63798 hazard, which
                                                # the row_muls timing newly exposes) -> needs
                                                # a full barrier (cross-pipe), not PIPE_V.
                                                # Diagnostic: 3x barrier_all green; PIPE_V x2
                                                # alone = 97% (missed the WAR).
                                                T.pipe_barrier("v")
                                                T.tile.row_muls(
                                                    acc_o[
                                                        hbase : hbase + MERGE_HEADS, :
                                                    ],
                                                    acc_o[
                                                        hbase : hbase + MERGE_HEADS, :
                                                    ],
                                                    alpha_brd8,
                                                    MERGE_HEADS,
                                                    D,
                                                    D,
                                                )
                                                T.pipe_barrier("v")
                                                for h_i in range(MERGE_HEADS):
                                                    T.tile.add(
                                                        acc_o[hbase + h_i, :],
                                                        acc_o[hbase + h_i, :],
                                                        acc_o_ub[h_i, :],
                                                    )
                                                T.barrier_all()
                                            else:
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
                                # cube_direct (swa/cfa, perf lever 2): drop the
                                # per-head barriers -- the v_block divs are same-
                                # VEC-pipe in-order on distinct acc_o rows, so the
                                # 2*v_block barrier_all per slot are redundant
                                # (another vector-bubble source). SCFA keeps the
                                # barriered form (lockstep resonance).
                                if cube_direct:
                                    for h_i in range(v_block):
                                        T.tile.div(
                                            acc_o[h_i, :],
                                            acc_o[h_i, :],
                                            sumexp[h_i],
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

        return sparse_attn_sharedkv

    return _make()
