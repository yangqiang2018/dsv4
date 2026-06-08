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
``example_sparse_flash_attn_mask_pa.py`` example. The whole kernel body
is flat (no helper calls inside the ``@T.prim_func``).

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
    block_I: int = 64,
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
    BI = block_I  # 64
    D = head_dim  # 512
    accum_dtype = "float"
    indices_dtype = "int32"

    # Sliding window: q-token attends to [s - win_left, s] (closed).
    ori_window_max = ori_win_left + 1  # 128
    NI_ori = (ori_window_max + BI - 1) // BI  # 2
    NI_cmp = topk_cmp // BI  # 8 for topk=512
    NI_total = NI_ori + NI_cmp
    # CFA: the cmp indices are the dense range [0, topk_cmp); the kernel
    # generates them per chunk with createvecindex instead of reading a
    # host-synthesized cmp_indices array (mirrors the Ascend C CFA path).
    is_cfa = scenario == 2

    H_per_block = gqa_group  # 64
    v_block = H_per_block // 2  # 32 -- each AIV handles half the heads
    ub_len = max(32 // 4, v_block)  # 32-byte UB alignment for fp32 scalars

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

    # ---- Manual address maps (bytes). ----
    KB = 1024
    l1_addr = {"q_l1": 0, "kv_l1": 64 * KB, "p_l1": 128 * KB}
    l0c_addr = {"acc_s_l0c": 0, "acc_o_l0c": 0}  # disjoint phases ⇒ alias
    ub_addr = {
        "acc_o": 0,
        "acc_s_ub": 64 * KB,
        "acc_s_ub_": 72 * KB,
        "acc_s_half": 80 * KB,
        "m_i": 84 * KB,
        "m_i_prev": 84 * KB + 128,
        "sumexp": 84 * KB + 256,
        "sumexp_i_ub": 84 * KB + 384,
        "sinks_ub": 84 * KB + 512,
        # lse_ub holds the per-head LogSumExp result before the GM
        # write-back. ub_len * sizeof(fp32) = 32 * 4 = 128 bytes; fits in
        # the 128-byte gap between sinks_ub (128 bytes used out of the
        # 256-byte slot) and idx_int.
        "lse_ub": 84 * KB + 640,
        "idx_int": 84 * KB + 768,
        "idx_float": 84 * KB + 1024,
        "mask_ub": 84 * KB + 2304,
        "mask_ub_2": 84 * KB + 2336,
        "acc_o_ub": 88 * KB,
        "acc_o_half": 88 * KB,  # aliases acc_o_ub (disjoint phases)
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
            ws_kv: T.Tensor([core_num, BI, D], dtype),  # type: ignore[valid-type]
            ws_score: T.Tensor([core_num, H_per_block, BI], accum_dtype),  # type: ignore[valid-type]
            ws_p: T.Tensor([core_num, H_per_block, BI], dtype),  # type: ignore[valid-type]
            ws_o: T.Tensor([core_num, H_per_block, D], accum_dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- L1 / L0 (cube). ----
                q_l1 = T.alloc_L1([H_per_block, D], dtype)
                kv_l1 = T.alloc_L1([BI, D], dtype)
                p_l1 = T.alloc_L1([H_per_block, BI], dtype)
                acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
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
                mask_ub = T.alloc_ub([BI // 8], "uint8")
                mask_ub_2 = T.alloc_ub([BI // 8], "uint8")

                T.annotate_address(
                    {
                        q_l1: l1_addr["q_l1"],
                        kv_l1: l1_addr["kv_l1"],
                        p_l1: l1_addr["p_l1"],
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
                        # kv_ub_multi aliases acc_o_ub (disjoint phases:
                        # chunk-head gather vs chunk-tail P@V merge).
                        kv_ub_multi: ub_addr["acc_o_ub"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_ub_2: ub_addr["mask_ub_2"],
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
                            with T.Scope("C"):
                                T.copy(Q[t_i, 0:n_heads, 0:D], q_l1)
                                T.barrier_all()
                                for _ in T.serial(NI_total):
                                    T.wait_cross_flag(_FLAG_KV_READY)
                                    T.barrier_all()
                                    T.copy(ws_kv[cid, 0:BI, 0:D], kv_l1)
                                    T.barrier_all()
                                    T.gemm_v0(
                                        q_l1,
                                        kv_l1,
                                        acc_s_l0c,
                                        transpose_B=True,
                                        init=True,
                                    )
                                    T.barrier_all()
                                    T.copy(
                                        acc_s_l0c,
                                        ws_score[cid, 0:H_per_block, 0:BI],
                                    )
                                    T.barrier_all()
                                    T.set_cross_flag("FIX", _FLAG_SCORE_READY)

                                    T.wait_cross_flag(_FLAG_P_READY)
                                    T.barrier_all()
                                    T.copy(
                                        ws_p[cid, 0:H_per_block, 0:BI],
                                        p_l1,
                                    )
                                    T.barrier_all()
                                    T.gemm_v0(
                                        p_l1,
                                        kv_l1,
                                        acc_o_l0c,
                                        init=True,
                                    )
                                    T.barrier_all()
                                    T.copy(
                                        acc_o_l0c,
                                        ws_o[cid, 0:H_per_block, 0:D],
                                    )
                                    T.barrier_all()
                                    T.set_cross_flag("FIX", _FLAG_PV_READY)
                                    T.wait_cross_flag(_FLAG_ITER_DONE)

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

                                for chunk in range(NI_total):
                                    is_ori = chunk < NI_ori

                                    # ---- gather KV + build mask ----
                                    if is_ori:
                                        chunk_start = ori_left + chunk * BI
                                        T.tile.createvecindex(
                                            idx_int,
                                            chunk_start,
                                        )
                                        T.copy(idx_int, idx_float)
                                        T.barrier_all()
                                        T.tile.compare(
                                            mask_ub,
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
                                            # [(chunk-NI_ori)*BI, +BI).
                                            # Generate them on the vector
                                            # core -- no host index array.
                                            T.tile.createvecindex(
                                                idx_int,
                                                (chunk - NI_ori) * BI,
                                            )
                                        else:
                                            cmp_off = (chunk - NI_ori) * BI
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
                                            mask_ub,
                                            idx_float,
                                            T.float32(-0.5),
                                            "GT",
                                        )
                                        T.tile.compare(
                                            mask_ub_2,
                                            idx_float,
                                            T.float32(cmp_threshold),
                                            "LT",
                                        )
                                        T.barrier_all()
                                        T.tile.bitwise_and(
                                            mask_ub,
                                            mask_ub,
                                            mask_ub_2,
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
                                                vid * (BI // 2) : vid * (BI // 2)
                                                + BI // 2,
                                                :,
                                            ],
                                        )
                                        T.barrier_all()
                                    T.set_cross_flag("MTE3", _FLAG_KV_READY)

                                    # ---- additive mask (0 / -inf) ----
                                    T.tile.fill(acc_s_ub_, 0.0)
                                    T.barrier_all()
                                    for h_i in T.serial(v_block):
                                        T.tile.select(
                                            acc_s_ub[h_i, :],
                                            mask_ub,
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
                                            vid * v_block : vid * v_block + v_block,
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

                                    for h_i in range(v_block):
                                        T.barrier_all()
                                        T.tile.mul(
                                            acc_o[h_i, :],
                                            acc_o[h_i, :],
                                            m_i_prev[h_i],
                                        )
                                        T.barrier_all()

                                    # ---- cast P, publish for cube ----
                                    T.copy(acc_s_ub, acc_s_half)
                                    T.barrier_all()
                                    T.copy(
                                        acc_s_half,
                                        ws_p[
                                            cid,
                                            vid * v_block : vid * v_block + v_block,
                                            :,
                                        ],
                                    )
                                    T.barrier_all()
                                    T.set_cross_flag("MTE3", _FLAG_P_READY)

                                    # ---- wait P@V, merge into output ----
                                    T.wait_cross_flag(_FLAG_PV_READY)
                                    T.barrier_all()
                                    T.copy(
                                        ws_o[
                                            cid,
                                            vid * v_block : vid * v_block + v_block,
                                            :,
                                        ],
                                        acc_o_ub,
                                    )
                                    T.barrier_all()
                                    T.tile.add(acc_o, acc_o, acc_o_ub)
                                    T.barrier_all()
                                    T.set_cross_flag("V", _FLAG_ITER_DONE)

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
