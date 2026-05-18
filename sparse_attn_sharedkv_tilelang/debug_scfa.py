"""Focused diagnostic for the failing SCFA cases.

Localises the SCFA mismatch with several probes on scfa_decode
(B=1, S1=1, the simplest scenario-3 case):

* Probe A -- full SCFA with the real random sparse indices. Includes a
  per-head mismatch breakdown.
* Probe B -- the SAME scenario-3 kernel fed *sequential* indices
  (arange). If B passes, the topk=512 kernel is fine and the bug is
  specific to random index values; if B fails, the kernel itself is
  broken regardless of indices.
* cmp_kv-zeroed probe -- isolates the cmp control-flow/softmax from the
  cmp KV values.

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
    print(
        f"  {name:16s} shape={tuple(t.shape)} "
        f"min={t[finite].min().item():+.3f} max={t[finite].max().item():+.3f} "
        f"mean={t[finite].mean().item():+.4f} "
        f"nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())}"
    )


def _diff(name, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs().flatten()
    n_bad = int((d > 2e-2).sum())
    idx = int(d.argmax())
    coord, rem = [], idx
    for s in reversed(a.shape):
        coord.append(rem % s)
        rem //= s
    coord = tuple(reversed(coord))
    print(
        f"  {name:30s} mismatch={n_bad}/{d.numel()} "
        f"({100 * n_bad / d.numel():.1f}%) max|diff|={d.max().item():.4f} "
        f"@ {coord} a={a[coord].item():+.4f} b={b[coord].item():+.4f}"
    )


def _per_head(name, a, b):
    """a, b: [1, n_heads, D]. Print per-head mismatch counts."""
    a, b = a.float(), b.float()
    d = (a - b).abs()
    n_heads = a.shape[1]
    bad_heads = []
    for h in range(n_heads):
        nbad = int((d[0, h] > 2e-2).sum())
        if nbad > 0:
            bad_heads.append((h, nbad, d[0, h].max().item()))
    print(f"  {name}: {len(bad_heads)}/{n_heads} heads have mismatches")
    for h, nbad, mx in bad_heads[:20]:
        print(f"    head {h:2d}: {nbad:3d}/{a.shape[2]} bad, max|diff|={mx:.3f}")
    if len(bad_heads) > 20:
        print(f"    ... and {len(bad_heads) - 20} more")


def _golden(case, cfg, cmp_idx_bsnd):
    """Recompute the BNSD golden for scfa_decode with given BSND indices."""
    q_bnsd = G.tnd_to_bnsd_q(case["q"], case["cu_seqlens_q"])
    act_q = (case["cu_seqlens_q"][1:] - case["cu_seqlens_q"][:-1]).tolist()
    ori_k_bnsd = G.unpack_paged_kv(
        case["ori_pa"], case["ori_bt"], int(case["seqused_kv"].max())
    )
    cmp_seqs = int(case["seqused_kv"].max()) // cfg["cmp_ratio"]
    cmp_k_bnsd = G.unpack_paged_kv(case["cmp_pa"], case["cmp_bt"], cmp_seqs)
    ref = G.sparse_attn_sharedkv_golden_bnsd(
        q_bnsd,
        ori_k_bnsd,
        case["sinks"],
        act_q_lens=act_q,
        act_kv_lens=case["seqused_kv"].tolist(),
        softmax_scale=cfg["softmax_scale"],
        cmp_k_bnsd=cmp_k_bnsd if cmp_idx_bsnd is not None else None,
        cmp_sparse_indices=cmp_idx_bsnd,
        cmp_ratio=cfg["cmp_ratio"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
    )
    return G.bnsd_to_tnd_out(ref, case["cu_seqlens_q"])


def main():
    if not hasattr(torch, "npu"):
        print("torch_npu not available; run this on the NPU host.")
        return

    from api import sparse_attn_sharedkv

    cfg = dict(SCENARIOS["scfa_decode"])
    K = cfg["K"]
    dtype = torch.bfloat16
    case = _build_case(cfg, dtype)

    def dev(t):
        return None if t is None else t.npu().contiguous()

    common = dict(
        ori_kv=dev(case["ori_pa"]),
        ori_block_table=dev(case["ori_bt"]),
        cmp_kv=dev(case["cmp_pa"]),
        cmp_block_table=dev(case["cmp_bt"]),
        cu_seqlens_q=dev(case["cu_seqlens_q"]),
        seqused_kv=dev(case["seqused_kv"]),
        sinks=dev(case["sinks"]),
        softmax_scale=cfg["softmax_scale"],
        cmp_ratio=cfg["cmp_ratio"],
        ori_mask_mode=cfg["ori_mask_mode"],
        cmp_mask_mode=cfg["cmp_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q="TND",
        layout_kv="PA_ND",
        topk_cmp=K,
    )

    print("=== scfa_decode (B=1, S1=1, K=%d, bf16) ===" % K)

    def run(cmp_idx_tnd):
        with torch.device("npu"):
            out = sparse_attn_sharedkv(
                dev(case["q"]),
                cmp_sparse_indices=dev(cmp_idx_tnd),
                **common,
            )
            torch.npu.synchronize()
        return out.cpu()

    # ---- Probe A: real random sparse indices. ----
    print("\n-- Probe A: real random sparse indices --")
    out_a = run(case["cmp_idx"])
    # golden with the BSND version of the random indices.
    cmp_idx_bsnd = case["cmp_idx"].unsqueeze(0)  # [1,T1,N2,K] = [B,S,N2,K]
    ref_a = _golden(case, cfg, cmp_idx_bsnd)
    _summary("kernel A", out_a)
    _summary("golden A", ref_a)
    _diff("kernel A vs golden A", out_a, ref_a)
    _per_head("per-head (A)", out_a, ref_a)

    # ---- Probe B: SAME kernel, sequential indices (arange). ----
    print("\n-- Probe B: scenario-3 path, sequential indices [0..K-1] --")
    seq_idx_tnd = torch.arange(K, dtype=torch.int32).view(1, 1, K).contiguous()
    out_b = run(seq_idx_tnd)
    ref_b = _golden(case, cfg, seq_idx_tnd.unsqueeze(0))
    _summary("kernel B", out_b)
    _summary("golden B", ref_b)
    _diff("kernel B vs golden B", out_b, ref_b)

    # ---- Probe C: SAME random indices, padded to K=2048. ----
    # This runs the topk=2048 kernel on the exact same effective 512
    # cmp tokens as Probe A. If C passes but A fails, the topk=512
    # kernel itself is broken; if C also fails, it is not the kernel
    # config.
    print("\n-- Probe C: Probe-A random indices padded to K=2048 --")
    K2 = 2048
    idx_c = torch.full((1, 1, K2), -1, dtype=torch.int32)
    idx_c[0, 0, :K] = case["cmp_idx"][0, 0, :]
    common_c = dict(common)
    common_c["topk_cmp"] = K2
    with torch.device("npu"):
        out_c = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_sparse_indices=dev(idx_c),
            **common_c,
        )
        torch.npu.synchronize()
    out_c = out_c.cpu()
    ref_c = _golden(case, cfg, idx_c.unsqueeze(0))
    _summary("kernel C", out_c)
    _diff("kernel C vs golden C", out_c, ref_c)
    _diff("kernel C vs golden A", out_c, ref_a)

    # ---- cmp_kv-zeroed probe. ----
    print("\n-- Probe Z: cmp_kv zeroed (real random indices) --")
    common_z = dict(common)
    common_z["cmp_kv"] = dev(torch.zeros_like(case["cmp_pa"]))
    with torch.device("npu"):
        out_z = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_sparse_indices=dev(case["cmp_idx"]),
            **common_z,
        )
        torch.npu.synchronize()
    out_z = out_z.cpu()
    # golden Z: zero cmp_k.
    q_bnsd = G.tnd_to_bnsd_q(case["q"], case["cu_seqlens_q"])
    act_q = (case["cu_seqlens_q"][1:] - case["cu_seqlens_q"][:-1]).tolist()
    ori_k_bnsd = G.unpack_paged_kv(
        case["ori_pa"], case["ori_bt"], int(case["seqused_kv"].max())
    )
    cmp_seqs = int(case["seqused_kv"].max()) // cfg["cmp_ratio"]
    ref_z_bnsd = G.sparse_attn_sharedkv_golden_bnsd(
        q_bnsd,
        ori_k_bnsd,
        case["sinks"],
        act_q_lens=act_q,
        act_kv_lens=case["seqused_kv"].tolist(),
        softmax_scale=cfg["softmax_scale"],
        cmp_k_bnsd=torch.zeros((1, 1, cmp_seqs, cfg["D"]), dtype=dtype),
        cmp_sparse_indices=case["cmp_idx"].unsqueeze(0),
        cmp_ratio=cfg["cmp_ratio"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
    )
    ref_z = G.bnsd_to_tnd_out(ref_z_bnsd, case["cu_seqlens_q"])
    _diff("kernel Z vs golden Z", out_z, ref_z)


if __name__ == "__main__":
    main()
