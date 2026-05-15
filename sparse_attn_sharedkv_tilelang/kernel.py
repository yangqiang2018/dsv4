"""TileLang implementation of SparseAttnSharedKV (Atlas A3 / Ascend 910_93).

Ports the Ascend C kernel at
``ops-transformer/experimental/attention/sparse_attn_sharedkv`` to TileLang.

A single fused kernel runs the sliding-window pass over ``ori_kv`` and the
top-K sparse pass over ``cmp_kv`` under one online-softmax state that is
seeded from per-q-head sinks. The same kernel covers all three scenarios:

* Scenario 1 (SWA only): ``topk_cmp == 0`` skips the cmp pass.
* Scenario 2 (CFA): caller synthesizes a causal ``cmp_sparse_indices``.
* Scenario 3 (SCFA): full sparse compressed attention.

The kernel takes a padded BSND layout for ``Q`` / ``Output``. :mod:`api`
converts TND inputs to BSND before invocation.
"""

import tilelang
from tilelang import language as T

# Atlas A3 cube/vector pair count.
DEFAULT_CORE_NUM = 24


def _check_dtypes(dtype: str) -> None:
    if dtype not in ("bfloat16", "float16"):
        raise ValueError(f"dtype must be bfloat16 or float16, got {dtype!r}")


def build_sparse_attn_sharedkv(
    *,
    n_heads: int = 64,
    n_kv_heads: int = 1,
    head_dim: int = 512,
    topk_cmp: int = 512,
    cmp_ratio: int = 4,
    ori_win_left: int = 127,
    ori_block_size: int = 128,
    cmp_block_size: int = 128,
    ori_block_num: int = 256,
    cmp_block_num: int = 64,
    softmax_scale: float = 0.04419417,
    dtype: str = "bfloat16",
    block_I: int = 64,
    core_num: int = DEFAULT_CORE_NUM,
):
    """Build a JIT-compiled TileLang kernel for SparseAttnSharedKV.

    Arguments are all compile-time constants. Call the returned kernel
    object with the runtime tensors (BSND-padded Q + paged KV + tables +
    sinks). :mod:`api` provides a high-level wrapper that converts
    layouts and synthesises the cmp scenario inputs.
    """
    _check_dtypes(dtype)
    assert n_heads == 64, "API constraint: n_heads must be 64"
    assert n_kv_heads == 1, "API constraint: n_kv_heads must be 1"
    assert head_dim == 512, "API constraint: head_dim must be 512"
    assert ori_win_left == 127, "API constraint: ori_win_left must be 127"
    assert topk_cmp >= 0
    assert topk_cmp % block_I == 0, "topk_cmp must be a multiple of block_I"
    assert ori_block_size > 0 and ori_block_size % block_I == 0
    assert cmp_block_size > 0

    gqa_group = n_heads // n_kv_heads  # 64
    BI = block_I
    D = head_dim
    accum_dtype = "float"
    indices_dtype = "int32"

    # Sliding window: q-token attends to [s - win_left, s] (closed).
    ori_window_max = ori_win_left + 1
    NI_ori = (ori_window_max + BI - 1) // BI  # 2 for window=128, BI=64
    NI_cmp = topk_cmp // BI  # 8 for topk=512, BI=64

    H_per_block = gqa_group  # 64
    v_block = H_per_block // 2  # 32 — each AIV handles half the heads
    # 32-byte UB alignment for scalar vectors (fp32 ⇒ ≥ 8 elements).
    ub_len = max(32 // 4, v_block)

    # Symbolic runtime dims.
    batch = T.symbolic("batch")
    max_seq = T.symbolic("max_seq")
    ori_table_len = T.symbolic("ori_table_len")
    cmp_table_len = T.symbolic("cmp_table_len")

    q_shape = [batch, max_seq, n_heads, D]
    out_shape = [batch, max_seq, n_heads, D]
    sinks_shape = [n_heads]
    ori_kv_shape = [ori_block_num, ori_block_size, n_kv_heads, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, n_kv_heads, D]
    indices_shape = [batch, max_seq, n_kv_heads, max(topk_cmp, 1)]

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[9], target="pto", pass_configs=pass_configs)
    def _make():
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore[valid-type]
            ori_KV: T.Tensor(ori_kv_shape, dtype),  # type: ignore[valid-type]
            cmp_KV: T.Tensor(cmp_kv_shape, dtype),  # type: ignore[valid-type]
            cmp_indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
            ori_block_table: T.Tensor([batch, ori_table_len], indices_dtype),  # type: ignore[valid-type]
            cmp_block_table: T.Tensor([batch, cmp_table_len], indices_dtype),  # type: ignore[valid-type]
            actual_q_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            actual_kv_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            Sinks: T.Tensor(sinks_shape, accum_dtype),  # type: ignore[valid-type]
            Output: T.Tensor(out_shape, dtype),  # type: ignore[valid-type]
            ws_kv: T.Tensor([core_num, BI, D], dtype),  # type: ignore[valid-type]
            ws_score: T.Tensor([core_num, H_per_block, BI], accum_dtype),  # type: ignore[valid-type]
            ws_p: T.Tensor([core_num, H_per_block, BI], dtype),  # type: ignore[valid-type]
            ws_o: T.Tensor([core_num, H_per_block, D], accum_dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- L1 / L0 (cube) ----
                q_l1 = T.alloc_L1([H_per_block, D], dtype)
                kv_l1 = T.alloc_L1([BI, D], dtype)
                p_l1 = T.alloc_L1([H_per_block, BI], dtype)
                acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                # ---- UB (vector) ----
                # acc_o is per-vid (half the heads).
                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)

                m_i = T.alloc_ub([ub_len], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
                sinks_ub = T.alloc_ub([n_heads], accum_dtype)

                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)

                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                kv_ub = T.alloc_ub([D], dtype)
                mask_ub = T.alloc_ub([BI // 8], "uint8")
                mask_ub_2 = T.alloc_ub([BI // 8], "uint8")

                # Persistent loop: each physical core consumes a strided
                # slice of (batch * max_seq) work items so the workspace
                # stays per-core-sized (24 slots) instead of per-work-item.
                total_work = batch * max_seq
                for slot in T.serial(T.ceildiv(total_work, core_num)):
                    pid = slot * core_num + cid
                    if pid < total_work:
                        b_i = pid // max_seq
                        s_i = pid % max_seq
                        act_q = actual_q_len[b_i]
                        act_kv = actual_kv_len[b_i]
                        if s_i < act_q:
                            # Causal s-position in the kv sequence.
                            s_global = act_kv - act_q + s_i

                            # --------- Seed online softmax from sinks. ---------
                            T.copy(Sinks, sinks_ub)
                            T.tile.fill(acc_o, 0.0)
                            T.tile.fill(sumexp, 1.0)
                            for h_i in range(v_block):
                                m_i[h_i] = sinks_ub[vid * v_block + h_i]

                            # Q tile lives in L1 for the whole kernel.
                            T.copy(Q[b_i, s_i, 0:n_heads, 0:D], q_l1)

                            # Window bounds (closed interval).
                            ori_right = s_global
                            ori_left_raw = s_global - ori_win_left
                            ori_left = T.if_then_else(ori_left_raw < 0, 0, ori_left_raw)

                            # --------- Sliding-window pass over ori_kv. ---------
                            for ori_chunk in T.serial(NI_ori):
                                chunk_start = ori_left + ori_chunk * BI
                                # mask[lane] = 1 iff (chunk_start + lane) <= ori_right.
                                T.tile.createvecindex(idx_int, chunk_start)
                                T.copy(idx_int, idx_float)
                                T.tile.compare(
                                    mask_ub,
                                    idx_float,
                                    T.float32(ori_right),
                                    "LE",
                                )

                                # Gather BI rows of ori_kv; two AIVs split.
                                for bi_i in range(BI // 2):
                                    lane = bi_i + vid * (BI // 2)
                                    g_idx = chunk_start + lane
                                    if g_idx <= ori_right:
                                        page = g_idx // ori_block_size
                                        block_phys = ori_block_table[b_i, page]
                                        in_off = g_idx % ori_block_size
                                        T.copy(
                                            ori_KV[block_phys, in_off, 0, :],
                                            kv_ub,
                                        )
                                    else:
                                        T.tile.fill(kv_ub, 0.0)
                                    T.copy(kv_ub, ws_kv[cid, lane, :])

                                # Cube: Q @ K^T.
                                T.copy(ws_kv[cid, :, :], kv_l1)
                                T.gemm_v0(
                                    q_l1,
                                    kv_l1,
                                    acc_s_l0c,
                                    transpose_B=True,
                                    init=True,
                                )
                                T.copy(acc_s_l0c, ws_score[cid, :, :])

                                # Vector: mask invalid lanes to -inf.
                                T.copy(
                                    ws_score[
                                        cid,
                                        vid * v_block : (vid + 1) * v_block,
                                        :,
                                    ],
                                    acc_s_ub_,
                                )
                                for h_i in range(v_block):
                                    T.tile.select(
                                        acc_s_ub[h_i, :],
                                        mask_ub,
                                        acc_s_ub_[h_i, :],
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                # Apply softmax_scale.
                                T.tile.mul(acc_s_ub, acc_s_ub, softmax_scale)

                                # Online softmax: rowmax / rowsum update.
                                T.copy(m_i, m_i_prev)
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)

                                for h_i in range(v_block):
                                    T.tile.sub(
                                        acc_s_ub[h_i, :],
                                        acc_s_ub[h_i, :],
                                        m_i[h_i],
                                    )
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                                T.tile.mul(sumexp, sumexp, m_i_prev)
                                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                                # Cast P and publish for the cube.
                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(
                                    acc_s_half,
                                    ws_p[
                                        cid,
                                        vid * v_block : (vid + 1) * v_block,
                                        :,
                                    ],
                                )

                                # Cube: P @ V (kv_l1 still holds the chunk).
                                T.copy(ws_p[cid, :, :], p_l1)
                                T.gemm_v0(
                                    p_l1,
                                    kv_l1,
                                    acc_o_l0c,
                                    init=True,
                                )
                                T.copy(acc_o_l0c, ws_o[cid, :, :])

                                # Rescale acc_o by alpha and add PV chunk.
                                for h_i in range(v_block):
                                    T.tile.mul(
                                        acc_o[h_i, :],
                                        acc_o[h_i, :],
                                        m_i_prev[h_i],
                                    )
                                T.copy(
                                    ws_o[
                                        cid,
                                        vid * v_block : (vid + 1) * v_block,
                                        :,
                                    ],
                                    acc_o_ub,
                                )
                                T.tile.add(acc_o, acc_o, acc_o_ub)

                            # --------- Sparse top-K pass over cmp_kv. ---------
                            # Causal threshold on the *cmp* index axis.
                            cmp_threshold = (s_global + 1) // cmp_ratio
                            for cmp_chunk in T.serial(NI_cmp):
                                # Load BI indices.
                                T.copy(
                                    cmp_indices[
                                        b_i,
                                        s_i,
                                        0,
                                        cmp_chunk * BI : (cmp_chunk + 1) * BI,
                                    ],
                                    idx_int,
                                )
                                T.copy(idx_int, idx_float)
                                # mask = (idx >= 0) AND (idx < cmp_threshold)
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
                                T.tile.bitwise_and(mask_ub, mask_ub, mask_ub_2)

                                for bi_i in range(BI // 2):
                                    lane = bi_i + vid * (BI // 2)
                                    raw = idx_int[lane]
                                    if raw >= 0:
                                        page = raw // cmp_block_size
                                        block_phys = cmp_block_table[b_i, page]
                                        in_off = raw % cmp_block_size
                                        T.copy(
                                            cmp_KV[block_phys, in_off, 0, :],
                                            kv_ub,
                                        )
                                    else:
                                        T.tile.fill(kv_ub, 0.0)
                                    T.copy(kv_ub, ws_kv[cid, lane, :])

                                T.copy(ws_kv[cid, :, :], kv_l1)
                                T.gemm_v0(
                                    q_l1,
                                    kv_l1,
                                    acc_s_l0c,
                                    transpose_B=True,
                                    init=True,
                                )
                                T.copy(acc_s_l0c, ws_score[cid, :, :])

                                T.copy(
                                    ws_score[
                                        cid,
                                        vid * v_block : (vid + 1) * v_block,
                                        :,
                                    ],
                                    acc_s_ub_,
                                )
                                for h_i in range(v_block):
                                    T.tile.select(
                                        acc_s_ub[h_i, :],
                                        mask_ub,
                                        acc_s_ub_[h_i, :],
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                T.tile.mul(acc_s_ub, acc_s_ub, softmax_scale)

                                T.copy(m_i, m_i_prev)
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)
                                for h_i in range(v_block):
                                    T.tile.sub(
                                        acc_s_ub[h_i, :],
                                        acc_s_ub[h_i, :],
                                        m_i[h_i],
                                    )
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                                T.tile.mul(sumexp, sumexp, m_i_prev)
                                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(
                                    acc_s_half,
                                    ws_p[
                                        cid,
                                        vid * v_block : (vid + 1) * v_block,
                                        :,
                                    ],
                                )

                                T.copy(ws_p[cid, :, :], p_l1)
                                T.gemm_v0(
                                    p_l1,
                                    kv_l1,
                                    acc_o_l0c,
                                    init=True,
                                )
                                T.copy(acc_o_l0c, ws_o[cid, :, :])

                                for h_i in range(v_block):
                                    T.tile.mul(
                                        acc_o[h_i, :],
                                        acc_o[h_i, :],
                                        m_i_prev[h_i],
                                    )
                                T.copy(
                                    ws_o[
                                        cid,
                                        vid * v_block : (vid + 1) * v_block,
                                        :,
                                    ],
                                    acc_o_ub,
                                )
                                T.tile.add(acc_o, acc_o, acc_o_ub)

                            # --------- Normalize and write back. ---------
                            for h_i in range(v_block):
                                T.tile.div(
                                    acc_o[h_i, :],
                                    acc_o[h_i, :],
                                    sumexp[h_i],
                                )
                            T.copy(acc_o, acc_o_half)
                            T.copy(
                                acc_o_half,
                                Output[
                                    b_i,
                                    s_i,
                                    vid * v_block : (vid + 1) * v_block,
                                    :,
                                ],
                            )

        return main

    return _make()
