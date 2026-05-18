"""Focused diagnostic for the failing SCFA cases.

Runs the simplest scenario-3 case (scfa_decode: B=1, S1=1) and dumps a
detailed kernel-vs-golden comparison, plus two localising probes:

* SWA cross-check: run the kernel with ``topk_cmp=0`` on the same Q /
  ori_kv. That output is the verified-correct ori-only result. The SCFA
  output should differ from it (cmp adds tokens); how it differs tells
  us whether the cmp pass corrupts or no-ops.
* cmp_kv-zeroed probe: with cmp_kv all zeros, the cmp tokens contribute
  exp(0-rowmax) mass to the denominator and 0 to the numerator. Kernel
  vs golden then isolates the cmp control-flow / softmax from the cmp
  KV *values*.

Run on the NPU host:  python3 debug_scfa.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import golden as G  # noqa: E402
from test_sparse_attn_sharedkv import SCENARIOS, _build_case  # noqa: E402


def _summary(name, t):
    t = t.float()
    finite = torch.isfinite(t)
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    print(
        f"  {name:18s} shape={tuple(t.shape)} "
        f"min={t[finite].min().item():+.4f} max={t[finite].max().item():+.4f} "
        f"mean={t[finite].mean().item():+.4f} nan={n_nan} inf={n_inf}"
    )


def _diff(name, a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    flat = d.flatten()
    n_bad = int((flat > 2e-2).sum())
    total = flat.numel()
    amax = flat.max().item()
    idx = int(flat.argmax())
    coord = []
    rem = idx
    for s in reversed(a.shape):
        coord.append(rem % s)
        rem //= s
    coord = tuple(reversed(coord))
    print(
        f"  {name:28s} mismatch={n_bad}/{total} ({100 * n_bad / total:.1f}%) "
        f"max|diff|={amax:.4f} @ {coord}  a={a[coord].item():+.4f} "
        f"b={b[coord].item():+.4f}"
    )


def main():
    if not hasattr(torch, "npu"):
        print("torch_npu not available; run this on the NPU host.")
        return

    from api import sparse_attn_sharedkv

    cfg = dict(SCENARIOS["scfa_decode"])
    dtype = torch.bfloat16
    case = _build_case(cfg, dtype)

    def dev(t):
        return None if t is None else t.npu().contiguous()

    common = dict(
        ori_kv=dev(case["ori_pa"]),
        ori_block_table=dev(case["ori_bt"]),
        cu_seqlens_q=dev(case["cu_seqlens_q"]),
        seqused_kv=dev(case["seqused_kv"]),
        sinks=dev(case["sinks"]),
        softmax_scale=cfg["softmax_scale"],
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q=cfg["layout_q"],
        layout_kv="PA_ND",
    )

    print("=== scfa_decode (B=1, S1=1, K=512, bf16) ===")

    # ---- 1. Full SCFA ----
    with torch.device("npu"):
        out_scfa = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_kv=dev(case["cmp_pa"]),
            cmp_sparse_indices=dev(case["cmp_idx"]),
            cmp_block_table=dev(case["cmp_bt"]),
            cmp_ratio=cfg["cmp_ratio"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            topk_cmp=cfg["K"],
            **common,
        )
        torch.npu.synchronize()
    out_scfa = out_scfa.cpu()
    ref_scfa = case["cpu_ref"]

    # ---- 2. SWA cross-check (topk_cmp=0, no cmp) ----
    with torch.device("npu"):
        out_swa = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_kv=None,
            cmp_sparse_indices=None,
            cmp_block_table=None,
            cmp_ratio=None,
            cmp_mask_mode=cfg["cmp_mask_mode"],
            topk_cmp=0,
            **common,
        )
        torch.npu.synchronize()
    out_swa = out_swa.cpu()

    # golden SWA (cmp disabled).
    q_bnsd_ref = G.tnd_to_bnsd_q(case["q"], case["cu_seqlens_q"])
    act_q = (case["cu_seqlens_q"][1:] - case["cu_seqlens_q"][:-1]).tolist()
    ori_k_bnsd = G.unpack_paged_kv(
        case["ori_pa"], case["ori_bt"], int(case["seqused_kv"].max())
    )
    ref_swa_bnsd = G.sparse_attn_sharedkv_golden_bnsd(
        q_bnsd_ref,
        ori_k_bnsd,
        case["sinks"],
        act_q_lens=act_q,
        act_kv_lens=case["seqused_kv"].tolist(),
        softmax_scale=cfg["softmax_scale"],
        cmp_k_bnsd=None,
        cmp_sparse_indices=None,
        cmp_ratio=None,
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
    )
    ref_swa = G.bnsd_to_tnd_out(ref_swa_bnsd, case["cu_seqlens_q"])

    print("\n-- tensors --")
    _summary("kernel SCFA", out_scfa)
    _summary("golden SCFA", ref_scfa)
    _summary("kernel SWA", out_swa)
    _summary("golden SWA", ref_swa)

    print("\n-- diffs --")
    _diff("kernel SCFA vs golden SCFA", out_scfa, ref_scfa)
    _diff("kernel SWA   vs golden SWA", out_swa, ref_swa)
    _diff("kernel SCFA vs kernel SWA", out_scfa, out_swa)
    _diff("golden SCFA vs golden SWA", ref_scfa, ref_swa)

    print("\n-- head-0 token-0 first 8 dims --")
    print("  kernel SCFA:", out_scfa.flatten()[:8].tolist())
    print("  golden SCFA:", ref_scfa.flatten()[:8].tolist())
    print("  kernel SWA :", out_swa.flatten()[:8].tolist())
    print("  golden SWA :", ref_swa.flatten()[:8].tolist())

    # ---- 3. cmp_kv-zeroed probe ----
    print("\n-- cmp_kv zeroed probe --")
    cmp_pa_zero = torch.zeros_like(case["cmp_pa"])
    with torch.device("npu"):
        out_z = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_kv=dev(cmp_pa_zero),
            cmp_sparse_indices=dev(case["cmp_idx"]),
            cmp_block_table=dev(case["cmp_bt"]),
            cmp_ratio=cfg["cmp_ratio"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            topk_cmp=cfg["K"],
            **common,
        )
        torch.npu.synchronize()
    out_z = out_z.cpu()

    cmp_k_zero_bnsd = torch.zeros(
        (1, 1, int(case["seqused_kv"].max()) // cfg["cmp_ratio"], cfg["D"]),
        dtype=dtype,
    )
    ref_z_bnsd = G.sparse_attn_sharedkv_golden_bnsd(
        q_bnsd_ref,
        ori_k_bnsd,
        case["sinks"],
        act_q_lens=act_q,
        act_kv_lens=case["seqused_kv"].tolist(),
        softmax_scale=cfg["softmax_scale"],
        cmp_k_bnsd=cmp_k_zero_bnsd,
        cmp_sparse_indices=case["cmp_idx"].unsqueeze(0)
        if case["cmp_idx"].dim() == 3
        else case["cmp_idx"],
        cmp_ratio=cfg["cmp_ratio"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
    )
    ref_z = G.bnsd_to_tnd_out(ref_z_bnsd, case["cu_seqlens_q"])
    _summary("kernel SCFA z", out_z)
    _summary("golden SCFA z", ref_z)
    _diff("kernel z vs golden z", out_z, ref_z)


if __name__ == "__main__":
    main()
