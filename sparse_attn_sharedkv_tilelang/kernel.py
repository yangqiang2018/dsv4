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

    H_per_block = gqa_group  # 64
    v_block = H_per_block // 2  # 32 -- each AIV handles half the heads
    ub_len = max(32 // 4, v_block)  # 32-byte UB alignment for fp32 scalars
    # Mask is BI bits = BI//8 (=16) bytes. VEC ops require a 32-byte-aligned UB
    # operand, so a [2, BI//8] parity buffer's odd row (stride 16B) would be
    # unaligned. Pad each parity row to mask_w (round BI//8 up to 32B) so both
    # rows start on a 32B boundary; only the low BI//8 bytes carry the mask.
    mask_w = ((BI // 8 + 31) // 32) * 32  # 32 for BI=128

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
    # UB (192KB). acc_s_* are [32,128] now (double the BI=64 size), so
    # everything below them shifts up. acc_o_ub / acc_o_half / kv_ub_multi
    # alias one 64KB region (disjoint phases). Peak = 176KB (16KB margin).
    ub_addr = {
        "acc_o": 0,  # [32,512]fp32 = 64KB -> 0..64KB
        "acc_s_ub": 64 * KB,  # [32,128]fp32 = 16KB -> 64..80KB
        "acc_s_ub_": 80 * KB,  # [32,128]fp32 = 16KB -> 80..96KB
        "acc_s_half": 96 * KB,  # [32,128]bf16 = 8KB -> 96..104KB
        # Per-row scalar vectors + index/mask scratch, packed from 104KB.
        "m_i": 104 * KB,
        "m_i_prev": 104 * KB + 128,
        "sumexp": 104 * KB + 256,
        "sumexp_i_ub": 104 * KB + 384,
        "sinks_ub": 104 * KB + 512,
        "lse_ub": 104 * KB + 640,
        "idx_int": 104 * KB + 768,  # [128]int32 = 512B
        "idx_float": 104 * KB + 1280,  # [128]fp32 = 512B
        # Mask double buffer [2, mask_w]: row padded to mask_w (=32B) so BOTH
        # parity rows are 32B aligned. A [2, BI//8] buffer's row stride is 16B,
        # so the odd-parity row mask_ub[1,:] starts at +16B -- not 32B aligned
        # -- and the VEC compare/and on odd chunks faults ("UB address accessed
        # by the VEC instruction is not aligned" on device). parity = chunk % 2
        # is a TIR Var (real TIR loop), so the row is picked by Var index, only
        # the low BI//8 bytes of each row carry the mask.
        "mask_ub": 104 * KB + 1792,  # [2,32]uint8 = 64B, rows 32B aligned
        "mask_ub_2": 104 * KB + 1856,  # [2,32]uint8 = 64B (V0 AND scratch)
        # alpha[2,ub_len]fp32 = 256B: the V1->V2 rescale-factor handoff,
        # double-buffered so V2(t-2) reads alpha[(t-2)%2] while V1(t-1)
        # writes alpha[(t-1)%2]. Row stride 128B is already 32B aligned.
        "alpha": 104 * KB + 2048,  # [2,ub_len]fp32 = 256B -> ..2304
        "mask_sel": 104 * KB + 2304,  # [32]uint8 whole buffer for select selMask
        "acc_o_ub": 112 * KB,  # [32,512]fp32 = 64KB -> 112..176KB
        "acc_o_half": 112 * KB,  # aliases acc_o_ub (disjoint phases)
        # kv_ub_multi [64,512]bf16=64KB also aliases acc_o_ub @112KB.
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
                acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                m_i = T.alloc_ub([ub_len], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
                sinks_ub = T.alloc_ub([ub_len], accum_dtype)
                lse_ub = T.alloc_ub([ub_len], accum_dtype)
                # alpha[2,ub_len]: rescale factor exp(m_prev-m_new) handed from
                # V1(chunk) to V2(chunk), double-buffered by chunk parity so the
                # pipelined V1(t-1) and V2(t-2) use distinct slots in one step.
                alpha = T.alloc_ub([2, ub_len], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)
                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                # Multi-row gather staging buffer. Each AIV DMAs its BI//2 KV
                # rows into distinct rows here with NO per-row barrier (the
                # DMAs target disjoint rows, so MTE2 pipelines them), then a
                # single barrier_all + one batched UB->workspace write. This
                # replaces the old single-row kv_ub, whose write-after-read
                # hazard forced a barrier per row. Aliases acc_o_ub's address
                # (gather runs at the chunk head; acc_o_ub's P@V merge at the
                # chunk tail -- disjoint phases, and the chunk-head mask
                # barrier separates the previous chunk's merge from this
                # chunk's gather), so the UB peak is unchanged.
                kv_ub_multi = T.alloc_ub([BI // 2, D], dtype)
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
                        # kv_ub_multi aliases acc_o_ub. In S2a the intra-vector
                        # barriers keep V0 gather and V2 merge serialized within
                        # a pipeline step, so they are still disjoint in time and
                        # the alias holds. (S2b will remove those barriers and
                        # must sub-tile / un-alias these two 64KB buffers.)
                        kv_ub_multi: ub_addr["acc_o_ub"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_ub_2: ub_addr["mask_ub_2"],
                        mask_sel: ub_addr["mask_sel"],
                        acc_o_ub: ub_addr["acc_o_ub"],
                        acc_o_half: ub_addr["acc_o_half"],
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
                                        T.wait_cross_flag(_FLAG_KV_READY)
                                        T.barrier_all()
                                        # Load gathered KV as two [BI_half,D]=64KB
                                        # halves into the t%2 L1 sub-buffers
                                        # kv_lo[pa, :, :]/kv_hi[pa, :, :] (BufferRegion
                                        # operands -- gemm_v0 accepts these).
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
                                        T.gemm_v0(
                                            q_l1,
                                            kv_lo[pa, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                        )
                                        T.barrier_all()
                                        T.copy(
                                            acc_s_l0c,
                                            ws_score[cid, pa, 0:H_per_block, 0:BI_half],
                                        )
                                        T.barrier_all()
                                        T.gemm_v0(
                                            q_l1,
                                            kv_hi[pa, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                        )
                                        T.barrier_all()
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
                                        T.wait_cross_flag(_FLAG_P_READY)
                                        T.barrier_all()
                                        T.copy(
                                            ws_p[cid, pb, 0:H_per_block, 0:BI_half],
                                            p_lo,
                                        )
                                        T.barrier_all()
                                        T.copy(
                                            ws_p[cid, pb, 0:H_per_block, BI_half:BI],
                                            p_hi,
                                        )
                                        T.barrier_all()
                                        # P@V = sum over the two KV halves;
                                        # init=False accumulates the second half.
                                        T.gemm_v0(
                                            p_lo, kv_lo[pb, :, :], acc_o_l0c, init=True
                                        )
                                        T.barrier_all()
                                        T.gemm_v0(
                                            p_hi, kv_hi[pb, :, :], acc_o_l0c, init=False
                                        )
                                        T.barrier_all()
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
                                            for bi_i in range(BI // 2):
                                                lane = bi_i + vid * (BI // 2)
                                                g_idx = chunk_start + lane
                                                ori_blk = ori_block_table[
                                                    b_i, g_idx // ori_block_size
                                                ]
                                                ori_row = g_idx % ori_block_size
                                                T.copy(
                                                    ori_KV[ori_blk, ori_row, 0, :],
                                                    kv_ub_multi[bi_i, :],
                                                )
                                            T.barrier_all()
                                            T.copy(
                                                kv_ub_multi[0 : BI // 2, :],
                                                ws_kv[
                                                    cid,
                                                    pv0,
                                                    vid * (BI // 2) : vid * (BI // 2)
                                                    + BI // 2,
                                                    :,
                                                ],
                                            )
                                            T.barrier_all()
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
                                            for bi_i in range(BI // 2):
                                                lane = bi_i + vid * (BI // 2)
                                                cmp_idx = idx_int[lane]
                                                safe_idx = T.if_then_else(
                                                    cmp_idx < 0, 0, cmp_idx
                                                )
                                                cmp_blk = cmp_block_table[
                                                    b_i, safe_idx // cmp_block_size
                                                ]
                                                cmp_row = safe_idx % cmp_block_size
                                                T.copy(
                                                    cmp_KV[cmp_blk, cmp_row, 0, :],
                                                    kv_ub_multi[bi_i, :],
                                                )
                                            T.barrier_all()
                                            T.copy(
                                                kv_ub_multi[0 : BI // 2, :],
                                                ws_kv[
                                                    cid,
                                                    pv0,
                                                    vid * (BI // 2) : vid * (BI // 2)
                                                    + BI // 2,
                                                    :,
                                                ],
                                            )
                                            T.barrier_all()
                                        T.set_cross_flag("MTE3", _FLAG_KV_READY)
                                    # ---- V1(t-1): online softmax of chunk t-1 ----
                                    if t >= 1:
                                        if t <= NI_total:
                                            pv1 = (t - 1) % 2
                                            # ---- additive mask (0 / -inf) ----
                                            T.tile.fill(acc_s_ub_, 0.0)
                                            T.barrier_all()
                                            # select's selMask needs a whole
                                            # Buffer (it calls .access_ptr, which
                                            # a Var-indexed parity BufferRegion
                                            # lacks), so copy chunk t-1's mask
                                            # row into the whole mask_sel first.
                                            T.copy(mask_ub[pv1, :], mask_sel)
                                            T.barrier_all()
                                            for h_i in T.serial(v_block):
                                                T.tile.select(
                                                    acc_s_ub[h_i, :],
                                                    mask_sel,
                                                    acc_s_ub_[h_i, :],
                                                    -T.infinity(accum_dtype),
                                                    "VSEL_TENSOR_SCALAR_MODE",
                                                )
                                                T.barrier_all()
                                            T.copy(m_i, m_i_prev)
                                            T.barrier_all()

                                            # ---- wait Q@K^T, online softmax ----
                                            T.wait_cross_flag(_FLAG_SCORE_READY)
                                            T.barrier_all()
                                            T.copy(
                                                ws_score[
                                                    cid,
                                                    pv1,
                                                    vid * v_block : vid * v_block
                                                    + v_block,
                                                    :,
                                                ],
                                                acc_s_ub_,
                                            )
                                            T.barrier_all()
                                            T.tile.add(
                                                acc_s_ub,
                                                acc_s_ub,
                                                acc_s_ub_,
                                            )
                                            T.barrier_all()
                                            T.tile.mul(
                                                acc_s_ub,
                                                acc_s_ub,
                                                softmax_scale,
                                            )
                                            T.barrier_all()

                                            T.reduce_max(acc_s_ub, m_i, dim=-1)
                                            T.barrier_all()
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
                                            T.barrier_all()
                                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                                            T.barrier_all()
                                            T.tile.exp(m_i_prev, m_i_prev)
                                            T.barrier_all()
                                            # Stash the rescale factor alpha = exp(m_prev
                                            # - m_new) into this chunk's parity slot so
                                            # V2 of the same chunk (which runs 2 pipeline
                                            # steps later) applies it; m_i_prev itself is
                                            # overwritten by the next chunk's V1.
                                            T.copy(m_i_prev, alpha[pv1, :])
                                            T.barrier_all()

                                            for h_i in range(v_block):
                                                T.barrier_all()
                                                T.tile.sub(
                                                    acc_s_ub[h_i, :],
                                                    acc_s_ub[h_i, :],
                                                    m_i[h_i],
                                                )
                                                T.barrier_all()
                                            T.tile.exp(acc_s_ub, acc_s_ub)
                                            T.barrier_all()
                                            T.reduce_sum(
                                                acc_s_ub,
                                                sumexp_i_ub,
                                                dim=-1,
                                            )
                                            T.barrier_all()
                                            T.tile.mul(sumexp, sumexp, m_i_prev)
                                            T.barrier_all()
                                            T.tile.add(
                                                sumexp,
                                                sumexp,
                                                sumexp_i_ub,
                                            )
                                            T.barrier_all()
                                            # The acc_o rescale lives in V2 now (alpha was
                                            # stashed above); V1 never touches acc_o, so
                                            # V1(t-1) and V2(t-2) run in one pipeline step
                                            # without racing on the accumulator.

                                            # ---- cast P, publish for cube ----
                                            T.copy(acc_s_ub, acc_s_half)
                                            T.barrier_all()
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
                                            T.barrier_all()
                                            T.set_cross_flag("MTE3", _FLAG_P_READY)
                                    # ---- V2(t-2): merge chunk t-2 into the accumulator ----
                                    if t >= 2:
                                        pv2 = (t - 2) % 2
                                        # ---- wait P@V (chunk c2), merge output ----
                                        T.wait_cross_flag(_FLAG_PV_READY)
                                        T.barrier_all()
                                        T.copy(
                                            ws_o[
                                                cid,
                                                pv2,
                                                vid * v_block : vid * v_block + v_block,
                                                :,
                                            ],
                                            acc_o_ub,
                                        )
                                        T.barrier_all()
                                        # Output recurrence acc_o = alpha*acc_o + O,
                                        # fused here. alpha = exp(m_prev - m_new) was
                                        # stashed by V1(c2) into alpha[pv2]; the parity
                                        # slot survives the 2-step pipeline latency.
                                        # Rescale MUST precede the add. Chunk 0 has
                                        # acc_o = 0 so the rescale is a no-op.
                                        for h_i in range(v_block):
                                            T.barrier_all()
                                            T.tile.mul(
                                                acc_o[h_i, :],
                                                acc_o[h_i, :],
                                                alpha[pv2, h_i],
                                            )
                                            T.barrier_all()
                                        T.tile.add(acc_o, acc_o, acc_o_ub)
                                        T.barrier_all()

                                # ---- normalize and write back ----
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
