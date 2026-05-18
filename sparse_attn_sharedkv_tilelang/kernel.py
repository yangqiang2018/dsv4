"""TileLang implementation of SparseAttnSharedKV (Atlas A3 / Ascend 910_93).

Ports the Ascend C kernel at
``ops-transformer/experimental/attention/sparse_attn_sharedkv`` to TileLang.

A single fused kernel runs the sliding-window pass over ``ori_kv`` and the
top-K sparse pass over ``cmp_kv`` under one online-softmax state that is
seeded from per-q-head sinks. The same kernel covers all three scenarios:

* Scenario 1 (SWA only): ``topk_cmp == 0`` skips the cmp pass.
* Scenario 2 (CFA): caller synthesizes a causal ``cmp_sparse_indices``.
* Scenario 3 (SCFA): full sparse compressed attention.

This uses the default (non-pto) Ascend C lowering path: explicit
``T.Scope("C")`` / ``T.Scope("V")`` cube/vector partitioning, manual
``T.set_cross_flag`` / ``T.wait_cross_flag`` handshakes, and manual
``T.annotate_address`` placement -- mirroring the verified
``example_sparse_flash_attn_mask_pa.py`` example. The whole kernel body
is flat (no helper calls inside the ``@T.prim_func``).

The kernel takes **logical** (un-paged) KV: ``ori_KV`` /  ``cmp_KV`` are
``[batch, max_s, n_kv_heads, D]``. PageAttention block-table resolution
is done host-side in :mod:`api` -- the kernel only does a single-level
indirect gather ``KV[b, idx, 0, :]``, matching the verified
``example_sparse_flash_attn_gqa_pto`` pattern (a two-level indirect
paged gather with a non-trivial block table mis-resolves on Ascend).

The kernel takes a padded BSND layout for ``Q`` / ``Output``; :mod:`api`
converts TND inputs to BSND before invocation.
"""

import tilelang
from tilelang import language as T

# Disable TileLang's on-disk kernel cache. Without this, a kernel
# compiled from an earlier (buggy) revision is silently reused even
# after the source changes -- the JIT cache key does not always track
# every source edit. Every TileLang-Ascend example disables the cache
# for exactly this reason.
tilelang.disable_cache()

# Atlas A3 cube/vector pair count.
DEFAULT_CORE_NUM = 24

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
    max_ori_s: int,
    max_cmp_s: int,
    n_heads: int = 64,
    n_kv_heads: int = 1,
    head_dim: int = 512,
    topk_cmp: int = 512,
    cmp_ratio: int = 4,
    ori_win_left: int = 127,
    softmax_scale: float = 0.04419417,
    dtype: str = "bfloat16",
    block_I: int = 64,
    core_num: int = DEFAULT_CORE_NUM,
):
    """Build a JIT-compiled TileLang kernel for SparseAttnSharedKV.

    Arguments are all compile-time constants. Call the returned kernel
    object with the runtime tensors (BSND-padded Q + logical KV + sinks).
    :mod:`api` provides a high-level wrapper that converts layouts,
    un-pages the KV and synthesises the cmp scenario inputs.
    """
    _check_dtypes(dtype)
    assert n_heads == 64, "API constraint: n_heads must be 64"
    assert n_kv_heads == 1, "API constraint: n_kv_heads must be 1"
    assert head_dim == 512, "API constraint: head_dim must be 512"
    assert ori_win_left == 127, "API constraint: ori_win_left must be 127"
    assert topk_cmp >= 0
    assert topk_cmp % block_I == 0, "topk_cmp must be a multiple of block_I"
    assert batch > 0 and max_seq > 0
    assert max_ori_s > 0 and max_cmp_s > 0

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

    H_per_block = gqa_group  # 64
    v_block = H_per_block // 2  # 32 -- each AIV handles half the heads
    ub_len = max(32 // 4, v_block)  # 32-byte UB alignment for fp32 scalars

    q_shape = [batch, max_seq, n_heads, D]
    out_shape = [batch, max_seq, n_heads, D]
    ori_kv_shape = [batch, max_ori_s, n_kv_heads, D]
    cmp_kv_shape = [batch, max_cmp_s, n_kv_heads, D]
    indices_shape = [batch, max_seq, n_kv_heads, max(topk_cmp, 1)]

    # ---- Manual address maps (bytes). ----
    KB = 1024
    l1_addr = {"q_l1": 0, "kv_l1": 64 * KB, "p_l1": 128 * KB}
    l0c_addr = {"acc_s_l0c": 0, "acc_o_l0c": 0}  # disjoint phases ⇒ alias
    # m_i / m_i_prev / sumexp are deliberately placed in an isolated
    # high-UB region, far from idx_int and from each other. The cmp pass
    # fills idx_int with a GM->UB DMA (the ori pass uses createvecindex,
    # never a DMA); per-chunk dumps showed that DMA corrupting the
    # online-softmax scalars that used to sit in idx_int's 768-byte
    # neighbourhood -- this was the chunk-2 (first cmp chunk) divergence.
    # m_i keeps a 2 KB pad after it as headroom for the reduce_max write.
    ub_addr = {
        "acc_o": 0,
        "acc_s_ub": 64 * KB,
        "acc_s_ub_": 72 * KB,
        "acc_s_half": 80 * KB,
        "sumexp_i_ub": 84 * KB + 384,
        "sinks_ub": 84 * KB + 512,
        "idx_int": 84 * KB + 768,
        "idx_float": 84 * KB + 1024,
        "kv_ub": 84 * KB + 1280,
        "mask_ub": 84 * KB + 2304,
        "mask_ub_2": 84 * KB + 2336,
        "acc_o_ub": 88 * KB,
        "acc_o_half": 88 * KB,  # aliases acc_o_ub (disjoint phases)
        "m_i": 152 * KB,
        "m_i_prev": 152 * KB + 2048,
        "sumexp": 152 * KB + 2048 + 128,
    }

    @tilelang.jit(out_idx=[7, 12, 13, 14, 15], workspace_idx=[8, 9, 10, 11])
    def _make():
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore[valid-type]
            ori_KV: T.Tensor(ori_kv_shape, dtype),  # type: ignore[valid-type]
            cmp_KV: T.Tensor(cmp_kv_shape, dtype),  # type: ignore[valid-type]
            cmp_indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
            actual_q_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            actual_kv_len: T.Tensor([batch], indices_dtype),  # type: ignore[valid-type]
            Sinks: T.Tensor([n_heads], accum_dtype),  # type: ignore[valid-type]
            Output: T.Tensor(out_shape, dtype),  # type: ignore[valid-type]
            ws_kv: T.Tensor([core_num, BI, D], dtype),  # type: ignore[valid-type]
            ws_score: T.Tensor([core_num, H_per_block, BI], accum_dtype),  # type: ignore[valid-type]
            ws_p: T.Tensor([core_num, H_per_block, BI], dtype),  # type: ignore[valid-type]
            ws_o: T.Tensor([core_num, H_per_block, D], accum_dtype),  # type: ignore[valid-type]
            dbg_acc_o: T.Tensor([NI_total, n_heads, D], accum_dtype),  # type: ignore[valid-type]
            dbg_m: T.Tensor([NI_total, n_heads], accum_dtype),  # type: ignore[valid-type]
            dbg_s: T.Tensor([NI_total, n_heads], accum_dtype),  # type: ignore[valid-type]
            dbg_pv: T.Tensor([NI_total, n_heads, D], accum_dtype),  # type: ignore[valid-type]
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
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)
                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                kv_ub = T.alloc_ub([D], dtype)
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
                        idx_int: ub_addr["idx_int"],
                        idx_float: ub_addr["idx_float"],
                        kv_ub: ub_addr["kv_ub"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_ub_2: ub_addr["mask_ub_2"],
                        acc_o_ub: ub_addr["acc_o_ub"],
                        acc_o_half: ub_addr["acc_o_half"],
                    }
                )

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
                            ori_right = s_global
                            # Clamp window start to 0. T.if_then_else (a
                            # ternary) avoids the ambiguous C++ max()
                            # overload that T.max generates here.
                            ori_left_raw = s_global - ori_win_left
                            ori_left = T.if_then_else(ori_left_raw < 0, 0, ori_left_raw)
                            cmp_threshold = (s_global + 1) // cmp_ratio

                            # ================= CUBE =================
                            with T.Scope("C"):
                                T.copy(Q[b_i, s_i, 0:n_heads, 0:D], q_l1)
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
                                        for bi_i in range(BI // 2):
                                            lane = bi_i + vid * (BI // 2)
                                            g_idx = chunk_start + lane
                                            T.barrier_all()
                                            if g_idx <= ori_right:
                                                T.barrier_all()
                                                T.copy(
                                                    ori_KV[b_i, g_idx, 0, :],
                                                    kv_ub,
                                                )
                                            else:
                                                T.tile.fill(kv_ub, 0.0)
                                            T.barrier_all()
                                            T.copy(
                                                kv_ub,
                                                ws_kv[cid, lane, :],
                                            )
                                            T.barrier_all()
                                    else:
                                        cmp_chunk = chunk - NI_ori
                                        T.copy(
                                            cmp_indices[
                                                b_i,
                                                s_i,
                                                0,
                                                cmp_chunk * BI : cmp_chunk * BI + BI,
                                            ],
                                            idx_int,
                                        )
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
                                        for bi_i in range(BI // 2):
                                            lane = bi_i + vid * (BI // 2)
                                            # Inline the UB-loaded index into
                                            # the T.copy, matching the verified
                                            # example_sparse_flash_attn_gqa_pto
                                            # gather. NOTE: assumes every cmp
                                            # index is >= 0 (true for all
                                            # current SCFA test cases -- the
                                            # generator never pads with -1).
                                            T.copy(
                                                cmp_KV[b_i, idx_int[lane], 0, :],
                                                kv_ub,
                                            )
                                            T.barrier_all()
                                            T.copy(
                                                kv_ub,
                                                ws_kv[cid, lane, :],
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
                                    T.tile.max(m_i, m_i, m_i_prev)
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
                                    # DEBUG: dump pre-normalization acc_o after
                                    # each chunk to localize the ori->cmp
                                    # online-softmax divergence.
                                    T.copy(
                                        acc_o,
                                        dbg_acc_o[
                                            chunk,
                                            vid * v_block : vid * v_block + v_block,
                                            0:D,
                                        ],
                                    )
                                    T.copy(
                                        acc_o_ub,
                                        dbg_pv[
                                            chunk,
                                            vid * v_block : vid * v_block + v_block,
                                            0:D,
                                        ],
                                    )
                                    T.copy(
                                        m_i,
                                        dbg_m[
                                            chunk,
                                            vid * v_block : vid * v_block + v_block,
                                        ],
                                    )
                                    T.copy(
                                        sumexp,
                                        dbg_s[
                                            chunk,
                                            vid * v_block : vid * v_block + v_block,
                                        ],
                                    )
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
                                        b_i,
                                        s_i,
                                        vid * v_block : vid * v_block + v_block,
                                        :,
                                    ],
                                )

        return main

    return _make()
