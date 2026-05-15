"""Pytest suite for the TileLang SparseAttnSharedKV port.

Each case exercises one (scenario, layout, dtype) combination and
compares the TileLang kernel output against the Python golden in
:mod:`golden`.

Run on an Ascend NPU host with TileLang-Ascend installed::

    pytest -q test_sparse_attn_sharedkv.py
    pytest -q test_sparse_attn_sharedkv.py -k "scfa_prefill"
    pytest -q test_sparse_attn_sharedkv.py -k "swa_decode"

The CPU golden is the long pole; cases with ``S1`` in the thousands take
minutes to compute the reference. The :data:`SMALL_CASES` list below
restricts the default run to fast cases; opt in to the larger cases with
``-m slow``.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

# Make local modules importable when pytest runs from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import golden as G  # noqa: E402


# ---- Detect the NPU; tolerate CPU-only hosts. ----
# NOTE: we deliberately do NOT call torch.set_default_device("npu").
# Data generation and the golden run on CPU; only the kernel call is
# wrapped in `with torch.device("npu")`. Setting the default device
# globally makes torch.randperm(generator=<cpu-gen>) fail with a
# device-mismatch error.
def _try_set_npu():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available(), None
    except Exception as exc:
        return False, repr(exc)


HAS_NPU, _NPU_ERR = _try_set_npu()
# Print a diagnostic banner so a silent run is at least somewhat decipherable.
print(
    f"[test_sparse_attn_sharedkv] HAS_NPU={HAS_NPU} "
    f"(reason: {'OK' if HAS_NPU else _NPU_ERR})",
    flush=True,
)
requires_npu = pytest.mark.skipif(
    not HAS_NPU,
    reason=f"Ascend NPU not available ({_NPU_ERR})",
)


# ---- Test-case definitions (mirrors original sparse_attn_sharedkv_paramset). ----

SCENARIOS = {
    "scfa_decode": dict(
        scenario=3,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "scfa_prefill_small": dict(
        scenario=3,
        layout_q="TND",
        B=1,
        S1=128,
        T1=128,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 128],
        seqused_kv=[4096],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "swa_decode": dict(
        scenario=1,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        block_num1=65,
        block_num2=1,
        block_size1=128,
        block_size2=1,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        K=0,
    ),
    "swa_prefill_small": dict(
        scenario=1,
        layout_q="TND",
        B=1,
        S1=128,
        T1=128,
        N1=64,
        N2=1,
        D=512,
        block_num1=65,
        block_num2=1,
        block_size1=128,
        block_size2=1,
        cu_seqlens_q=[0, 128],
        seqused_kv=[4096],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        K=0,
    ),
    "cfa_decode": dict(
        scenario=2,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "scfa_bsnd_small": dict(
        scenario=3,
        layout_q="BSND",
        B=2,
        S1=16,
        T1=32,
        N1=64,
        N2=1,
        D=512,
        K=64,
        block_num1=128,
        block_num2=64,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 16, 32],
        seqused_kv=[2048, 2048],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
}

SMALL_CASES = [
    "scfa_decode",
    "scfa_prefill_small",
    "swa_decode",
    "swa_prefill_small",
    "cfa_decode",
    "scfa_bsnd_small",
]


def _build_case(cfg: dict, dtype: torch.dtype, seed: int = 42):
    """Generate inputs, paged KV, indices, and the CPU golden output.

    Returns a dict mirroring the original suite's contract.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    layout_q = cfg["layout_q"]
    B = cfg["B"]
    S1 = cfg["S1"]
    T1 = cfg.get("T1", S1 * B)
    N1, N2, D = cfg["N1"], cfg["N2"], cfg["D"]
    K = cfg.get("K", 0)
    cmp_ratio = cfg["cmp_ratio"]
    block_size1, block_num1 = cfg["block_size1"], cfg["block_num1"]
    block_size2, block_num2 = cfg["block_size2"], cfg["block_num2"]
    seqused_kv = cfg["seqused_kv"]
    cu_seqlens_q = torch.tensor(cfg["cu_seqlens_q"], dtype=torch.int32)
    softmax_scale = cfg["softmax_scale"]
    scenario = cfg["scenario"]

    # --- Q ---
    if layout_q == "TND":
        q = (torch.rand((T1, N1, D)) * 20 - 10).to(dtype)
    else:
        q = (torch.rand((B, S1, N1, D)) * 20 - 10).to(dtype)

    # --- ori_kv (paged) ---
    ori_pa, ori_bt, ori_k_bnsd = G.gen_ori_kv_paged(
        B=B,
        N2=N2,
        D=D,
        block_num=block_num1,
        block_size=block_size1,
        seqused_kv=seqused_kv,
        dtype=dtype,
        data_range=(-10, 10),
        rng=np.random.default_rng(seed),
    )

    # --- cmp_kv (paged) + indices (only for scenarios 2, 3) ---
    if scenario >= 2:
        cmp_pa, cmp_bt, cmp_k_bnsd, cmp_seqs = G.gen_cmp_kv_paged(
            B=B,
            N2=N2,
            D=D,
            block_num=block_num2,
            block_size=block_size2,
            seqused_kv=seqused_kv,
            cmp_ratio=cmp_ratio,
            dtype=dtype,
            data_range=(-5, 10),
            rng=np.random.default_rng(seed + 1),
        )
    else:
        cmp_pa, cmp_bt, cmp_k_bnsd = None, None, None

    if scenario == 3:
        rng = torch.Generator()
        rng.manual_seed(seed + 7)
        if layout_q == "TND":
            cmp_idx = G.gen_cmp_sparse_indices_tnd(
                B=B,
                T1=T1,
                N2=N2,
                K=K,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                cmp_ratio=cmp_ratio,
                cmp_mask_mode=3,
                rng=rng,
            )
        else:
            cmp_idx = G.gen_cmp_sparse_indices_bsnd(
                B=B,
                S1=S1,
                N2=N2,
                K=K,
                seqused_kv=seqused_kv,
                cmp_ratio=cmp_ratio,
                cmp_mask_mode=3,
                rng=rng,
            )
    else:
        cmp_idx = None

    # --- sinks ---
    sinks = (torch.rand(N1) * 2 - 1).to(torch.float32)

    # --- Build BNSD reference Q for the golden. ---
    if layout_q == "TND":
        q_bnsd_ref = G.tnd_to_bnsd_q(q, cu_seqlens_q)
        act_q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    else:
        q_bnsd_ref = q.permute(0, 2, 1, 3).contiguous()
        act_q_lens = [S1] * B

    # Convert sparse indices to BSND for the golden.
    if cmp_idx is not None:
        if layout_q == "TND":
            # Reuse the api helper logic inline (TND→BSND).
            S_max = max(act_q_lens)
            cmp_idx_bsnd = torch.full((B, S_max, N2, K), -1, dtype=torch.int32)
            for b in range(B):
                s_start = int(cu_seqlens_q[b].item())
                L = int(act_q_lens[b])
                cmp_idx_bsnd[b, :L, :, :] = cmp_idx[s_start : s_start + L, :, :]
        else:
            cmp_idx_bsnd = cmp_idx
    else:
        cmp_idx_bsnd = None

    cpu_ref = G.sparse_attn_sharedkv_golden_bnsd(
        q_bnsd_ref,
        ori_k_bnsd,
        sinks,
        act_q_lens=act_q_lens,
        act_kv_lens=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_k_bnsd=cmp_k_bnsd,
        cmp_sparse_indices=cmp_idx_bsnd if scenario == 3 else None,
        cmp_ratio=cmp_ratio if scenario >= 2 else None,
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        ori_mask_mode=cfg["ori_mask_mode"],
        cmp_mask_mode=cfg["cmp_mask_mode"],
    )

    # Convert golden back to caller layout.
    if layout_q == "TND":
        cpu_ref = G.bnsd_to_tnd_out(cpu_ref, cu_seqlens_q)
    else:
        cpu_ref = cpu_ref.permute(0, 2, 1, 3).contiguous()

    return dict(
        cfg=cfg,
        q=q,
        ori_pa=ori_pa,
        ori_bt=ori_bt,
        cmp_pa=cmp_pa,
        cmp_bt=cmp_bt,
        cmp_idx=cmp_idx,
        sinks=sinks,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=torch.tensor(seqused_kv, dtype=torch.int32),
        cpu_ref=cpu_ref,
    )


@requires_npu
@pytest.mark.parametrize("case_name", SMALL_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_attn_sharedkv(case_name, dtype):
    # Imported lazily so the CPU-only math test below can run on hosts
    # without tilelang installed.
    from api import sparse_attn_sharedkv

    cfg = SCENARIOS[case_name]
    # Data generation + golden run on CPU (default device).
    case = _build_case(cfg, dtype)

    # Move tensors to NPU.
    def _dev(t):
        if t is None:
            return None
        return t.npu().contiguous() if hasattr(t, "npu") else t

    # The kernel call (and TileLang's auto-allocated output / workspaces)
    # must run with the NPU as the default device.
    with torch.device("npu"):
        out = sparse_attn_sharedkv(
            _dev(case["q"]),
            ori_kv=_dev(case["ori_pa"]),
            cmp_kv=_dev(case["cmp_pa"]),
            cmp_sparse_indices=_dev(case["cmp_idx"]),
            ori_block_table=_dev(case["ori_bt"]),
            cmp_block_table=_dev(case["cmp_bt"]),
            cu_seqlens_q=_dev(case["cu_seqlens_q"])
            if cfg["layout_q"] == "TND"
            else None,
            seqused_kv=_dev(case["seqused_kv"]),
            sinks=_dev(case["sinks"]),
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"] if cfg["scenario"] >= 2 else None,
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=cfg["layout_q"],
            layout_kv="PA_ND",
            topk_cmp=cfg.get("K", 0),
        )
        torch.npu.synchronize()

    out_cpu = out.cpu()
    ref = case["cpu_ref"]
    torch.testing.assert_close(out_cpu, ref, rtol=2e-2, atol=2e-2)


# ---- Math-only test: golden vs single-shot softmax (no NPU required). ----


def test_golden_math_matches_single_shot_softmax():
    """Sanity check that the chunked online softmax in :mod:`golden`
    matches a single-shot ``softmax(scores ∪ sinks) @ V`` computation for
    a small, scenario-3 case. CPU-only.
    """
    torch.manual_seed(0)
    B, S1, N1, N2, D = 1, 4, 64, 1, 512
    cmp_ratio = 4
    K = 8
    seqused_kv = [128]
    softmax_scale = 1.0 / math.sqrt(D)

    q = (torch.rand((B, N1, S1, D)) * 2 - 1).to(torch.float32)
    ori_k_bnsd = (torch.rand((B, N2, seqused_kv[0], D)) * 2 - 1).to(torch.float32)
    cmp_k_bnsd = (torch.rand((B, N2, seqused_kv[0] // cmp_ratio, D)) * 2 - 1).to(
        torch.float32
    )
    sinks = (torch.rand(N1) * 0.1).to(torch.float32)

    # Build a deterministic small sparse index set.
    idx = torch.full((B, S1, N2, K), -1, dtype=torch.int32)
    for s in range(S1):
        thr = (seqused_kv[0] - S1 + s + 1) // cmp_ratio
        valid = max(thr, 0)
        if valid > 0:
            take = min(K, valid)
            idx[0, s, 0, :take] = torch.arange(take, dtype=torch.int32)

    chunked = G.sparse_attn_sharedkv_golden_bnsd(
        q,
        ori_k_bnsd,
        sinks,
        act_q_lens=[S1],
        act_kv_lens=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_k_bnsd=cmp_k_bnsd,
        cmp_sparse_indices=idx,
        cmp_ratio=cmp_ratio,
    )

    # Reference: per (s) row, build the same sparse K=V slice and do one-shot.
    ref = torch.zeros_like(q)
    for s in range(S1):
        s_global = seqused_kv[0] - S1 + s
        ori_left = max(s_global - 127, 0)
        ori_right = s_global + 1
        ori_k = ori_k_bnsd[0, 0, ori_left:ori_right, :]
        thr = (seqused_kv[0] - S1 + s + 1) // cmp_ratio
        # Same sparse selection logic as the chunked golden uses.
        raw = idx[0, s, 0]
        valid = (raw >= 0) & (raw < thr)
        sel = raw[valid].long()
        cmp_k = (
            cmp_k_bnsd[0, 0, sel, :] if sel.numel() else cmp_k_bnsd.new_zeros((0, D))
        )
        k_concat = torch.cat([ori_k, cmp_k], dim=0)  # [n, D]
        q_row = q[0, :, s, :]  # [N1, D]
        sm = G.sinks_softmax_reference(
            q_row.unsqueeze(0),
            k_concat.unsqueeze(0),
            sinks=sinks,
            softmax_scale=softmax_scale,
        ).squeeze(0)
        ref[0, :, s, :] = sm

    torch.testing.assert_close(chunked, ref, rtol=2e-4, atol=2e-4)
