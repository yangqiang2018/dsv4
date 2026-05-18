"""High-level Python entry point for the TileLang SparseAttnSharedKV op.

Handles the three scenarios from the original Ascend C operator and the
TND/BSND layout dispatch. The kernel itself only sees a padded BSND
layout and a single combined ori-window + sparse-cmp code path; this
module converts to/from TND and synthesises dummy inputs for the
SWA-only and CFA scenarios so the same kernel can serve all three.

Usage::

    from api import sparse_attn_sharedkv

    out = sparse_attn_sharedkv(
        q,                       # [T1, N1, D] (TND) or [B, S1, N1, D]
        ori_kv=ori_kv,           # paged: [block_num, block_size, N2, D]
        cmp_kv=cmp_kv,           # paged or None
        cmp_sparse_indices=...,  # int32, or None
        ori_block_table=...,
        cmp_block_table=...,     # or None
        cu_seqlens_q=...,        # required for TND
        seqused_kv=...,          # actual ori_kv length per batch
        sinks=...,               # [N1] fp32
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from kernel import build_sparse_attn_sharedkv, DEFAULT_CORE_NUM

# Module-level kernel cache: key is the tuple of compile-time params.
_KERNEL_CACHE: dict = {}


def _torch_to_tilelang_dtype(t: torch.dtype) -> str:
    if t == torch.bfloat16:
        return "bfloat16"
    if t == torch.float16:
        return "float16"
    raise ValueError(f"unsupported torch dtype {t}")


def _get_kernel(
    *,
    batch: int,
    max_seq: int,
    max_ori_s: int,
    max_cmp_s: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    topk_cmp: int,
    cmp_ratio: int,
    ori_win_left: int,
    softmax_scale: float,
    dtype: str,
    core_num: int,
):
    key = (
        batch,
        max_seq,
        max_ori_s,
        max_cmp_s,
        n_heads,
        n_kv_heads,
        head_dim,
        topk_cmp,
        cmp_ratio,
        ori_win_left,
        round(softmax_scale, 8),
        dtype,
        core_num,
    )
    func = _KERNEL_CACHE.get(key)
    if func is None:
        func = build_sparse_attn_sharedkv(
            batch=batch,
            max_seq=max_seq,
            max_ori_s=max_ori_s,
            max_cmp_s=max_cmp_s,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            topk_cmp=topk_cmp,
            cmp_ratio=cmp_ratio,
            ori_win_left=ori_win_left,
            softmax_scale=softmax_scale,
            dtype=dtype,
            core_num=core_num,
        )
        _KERNEL_CACHE[key] = func
    return func


def _unpage_kv(
    paged_kv: torch.Tensor,  # [block_num, block_size, N2, D]
    block_table: torch.Tensor,  # [B, table_len] int32, -1 ⇒ unused
    max_logical_len: int,
) -> torch.Tensor:
    """Resolve a paged KV cache into a logical ``[B, max_logical_len, N2, D]``.

    PageAttention block-table resolution is done here on the host so the
    TileLang kernel only needs a single-level indirect gather. (A
    two-level paged gather with a non-trivial block table mis-resolves
    on Ascend; see kernel.py.)
    """
    block_num, block_size, N2, D = paged_kv.shape
    bt = block_table.cpu()
    B, table_len = bt.shape
    out = paged_kv.new_zeros((B, max_logical_len, N2, D))
    for b in range(B):
        for blk in range(table_len):
            phys = int(bt[b, blk])
            if phys < 0:
                continue
            s0 = blk * block_size
            if s0 >= max_logical_len:
                break
            s1 = min(s0 + block_size, max_logical_len)
            out[b, s0:s1, :, :] = paged_kv[phys, : s1 - s0, :, :]
    return out


def _tnd_to_bsnd_q(
    q_tnd: torch.Tensor, cu_seqlens_q: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    T1, N1, D = q_tnd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    B = len(seq_lens)
    S_max = max(seq_lens) if seq_lens else 0
    out = q_tnd.new_zeros((B, S_max, N1, D))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        out[b, :L, :, :] = q_tnd[start:end, :, :]
    return out, S_max


def _bsnd_to_tnd_out(
    out_bsnd: torch.Tensor, cu_seqlens_q: torch.Tensor
) -> torch.Tensor:
    B, S_max, N1, D = out_bsnd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    T1 = sum(seq_lens)
    out = out_bsnd.new_zeros((T1, N1, D))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        out[start:end, :, :] = out_bsnd[b, :L, :, :]
    return out


def _tnd_to_bsnd_indices(
    idx_tnd: torch.Tensor,  # [T1, N2, K]
    cu_seqlens_q: torch.Tensor,
    S_max: int,
) -> torch.Tensor:
    T1, N2, K = idx_tnd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    B = len(seq_lens)
    out = torch.full((B, S_max, N2, K), -1, dtype=idx_tnd.dtype, device=idx_tnd.device)
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        out[b, :L, :, :] = idx_tnd[start:end, :, :]
    return out


def _synthesize_dense_cmp_indices(
    *,
    B: int,
    S_max: int,
    N2: int,
    K: int,
    seqused_kv: torch.Tensor,
    act_q_lens: torch.Tensor,
    cmp_ratio: int,
) -> torch.Tensor:
    """Make a dense ``arange``-based cmp_sparse_indices for the CFA scenario.

    Each (b, s) row contains ``[0, 1, ..., K-1]`` truncated/padded so the
    causal threshold is respected by the mask in the kernel.
    """
    idx = (
        torch.arange(K, dtype=torch.int32)
        .view(1, 1, 1, K)
        .expand(B, S_max, N2, K)
        .contiguous()
    )
    return idx


def sparse_attn_sharedkv(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: torch.Tensor,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    core_num: int = DEFAULT_CORE_NUM,
    topk_cmp: Optional[int] = None,
    return_chunk_dump: bool = False,
) -> torch.Tensor:
    """Forward pass. Returns ``attention_out`` with the same layout as ``q``.

    The function detects the scenario from the optional arguments:

    * ``cmp_kv=None, cmp_sparse_indices=None`` → SWA (scenario 1).
    * ``cmp_kv=<tensor>, cmp_sparse_indices=None`` → CFA (scenario 2).
    * ``cmp_kv=<tensor>, cmp_sparse_indices=<tensor>`` → SCFA (scenario 3).
    """
    assert ori_mask_mode == 4, "only ori_mask_mode=4 supported"
    assert cmp_mask_mode == 3 or cmp_kv is None, "only cmp_mask_mode=3 supported"
    assert ori_win_right == 0, "only ori_win_right=0 supported"
    assert layout_kv == "PA_ND", "only layout_kv=PA_ND supported"
    assert layout_q in ("TND", "BSND"), f"unsupported layout_q={layout_q!r}"

    dtype = q.dtype
    tl_dtype = _torch_to_tilelang_dtype(dtype)

    # Resolve scenario.
    if cmp_kv is None:
        scenario = 1  # SWA
    elif cmp_sparse_indices is None:
        scenario = 2  # CFA
    else:
        scenario = 3  # SCFA

    # ---- Normalize Q / indices into BSND. ----
    if layout_q == "TND":
        assert cu_seqlens_q is not None, "cu_seqlens_q is required for TND"
        cu = cu_seqlens_q.to(torch.int32).cpu()
        q_bsnd, S_max = _tnd_to_bsnd_q(q, cu)
        B = cu.numel() - 1
        seq_lens = (cu[1:] - cu[:-1]).tolist()
        act_q_lens = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
        N1, D = q.shape[1], q.shape[2]
        if scenario == 3:
            cmp_sparse_indices_bsnd = _tnd_to_bsnd_indices(
                cmp_sparse_indices.to(torch.int32), cu, S_max
            ).to(q.device)
        else:
            cmp_sparse_indices_bsnd = None
    else:  # BSND
        q_bsnd = q
        B, S_max, N1, D = q.shape
        seq_lens = [S_max] * B
        act_q_lens = torch.full((B,), S_max, dtype=torch.int32, device=q.device)
        if cu_seqlens_q is not None:
            cu = cu_seqlens_q.to(torch.int32).cpu()
            seq_lens = (cu[1:] - cu[:-1]).tolist()
            act_q_lens = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
        if scenario == 3:
            cmp_sparse_indices_bsnd = cmp_sparse_indices.to(torch.int32)
        else:
            cmp_sparse_indices_bsnd = None

    seqused_kv_dev = seqused_kv.to(torch.int32).to(q.device)

    # ---- Resolve topk / scenario-specific tensors. ----
    if scenario == 3:
        N2 = cmp_sparse_indices_bsnd.shape[2]
        K = topk_cmp if topk_cmp is not None else cmp_sparse_indices_bsnd.shape[3]
        assert K == cmp_sparse_indices_bsnd.shape[3], (
            "topk_cmp does not match cmp_sparse_indices last dim"
        )
        cmp_indices_dev = cmp_sparse_indices_bsnd.to(q.device)
    elif scenario == 2:
        N2 = cmp_kv.shape[2]
        # CFA attends to ALL compressed tokens up to the per-row causal
        # threshold (NOT a fixed top-K). The dense synthesized indices
        # must therefore span the largest possible threshold, i.e.
        # floor(max(seqused_kv) / cmp_ratio). topk_cmp is meaningless for
        # CFA and is intentionally ignored here.
        max_cmp = int(seqused_kv.max().item()) // cmp_ratio
        K = max(64, ((max_cmp + 63) // 64) * 64)
        cmp_indices_dev = _synthesize_dense_cmp_indices(
            B=B,
            S_max=S_max,
            N2=N2,
            K=K,
            seqused_kv=seqused_kv,
            act_q_lens=act_q_lens,
            cmp_ratio=cmp_ratio,
        ).to(q.device)
    else:
        N2 = ori_kv.shape[2]
        K = 0
        cmp_indices_dev = torch.zeros(
            (B, S_max, N2, 1), dtype=torch.int32, device=q.device
        )

    if N1 != 64 or N2 != 1 or D != 512:
        raise ValueError(
            f"only N1=64, N2=1, D=512 supported (got N1={N1}, N2={N2}, D={D})"
        )

    # ---- Un-page the KV caches into a logical [B, S, N2, D] layout. ----
    cmp_ratio_eff = cmp_ratio if cmp_ratio is not None else 4
    max_ori_s = int(seqused_kv.max().item())
    max_cmp_s = max(1, max_ori_s // cmp_ratio_eff)

    ori_kv_logical = _unpage_kv(
        ori_kv.to(q.device), ori_block_table.to(torch.int32), max_ori_s
    )
    if cmp_kv is not None:
        cmp_kv_logical = _unpage_kv(
            cmp_kv.to(q.device),
            cmp_block_table.to(torch.int32),
            max_cmp_s,
        )
    else:
        # Dummy logical cmp KV so the kernel signature is well-typed.
        cmp_kv_logical = torch.zeros(
            (B, max_cmp_s, N2, D), dtype=dtype, device=q.device
        )

    # ---- JIT-compile kernel for these compile-time params. ----
    func = _get_kernel(
        batch=int(B),
        max_seq=int(S_max),
        max_ori_s=max_ori_s,
        max_cmp_s=max_cmp_s,
        n_heads=N1,
        n_kv_heads=N2,
        head_dim=D,
        topk_cmp=K,
        cmp_ratio=cmp_ratio_eff,
        ori_win_left=ori_win_left,
        softmax_scale=float(softmax_scale),
        dtype=tl_dtype,
        core_num=core_num,
    )

    # ---- Sinks on device, fp32. ----
    sinks_dev = sinks.to(torch.float32).to(q.device)

    # ---- Run kernel. Workspaces are auto-allocated via workspace_idx. ----
    out_bsnd, dbg_acc_o, dbg_m, dbg_s, dbg_pv, dbg_mprev, dbg_score = func(
        q_bsnd.contiguous(),
        ori_kv_logical.contiguous(),
        cmp_kv_logical.contiguous(),
        cmp_indices_dev.contiguous(),
        act_q_lens.contiguous(),
        seqused_kv_dev.contiguous(),
        sinks_dev.contiguous(),
    )

    # ---- Convert layout back. ----
    if layout_q == "TND":
        cu = cu_seqlens_q.to(torch.int32).cpu()
        out_tnd = _bsnd_to_tnd_out(out_bsnd, cu)
        if return_chunk_dump:
            return (
                out_tnd,
                dbg_acc_o,
                dbg_m,
                dbg_s,
                dbg_pv,
                dbg_mprev,
                dbg_score,
            )
        return out_tnd
    if return_chunk_dump:
        return (
            out_bsnd,
            dbg_acc_o,
            dbg_m,
            dbg_s,
            dbg_pv,
            dbg_mprev,
            dbg_score,
        )
    return out_bsnd
