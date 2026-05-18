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
* Probe I -- sets cmp_kv[token i] = i and zeroes ori_kv, so each head's
  output reads out the softmax-weighted mean of the token indices the
  kernel actually gathered. Probe I-seq repeats it with sequential
  indices as a control. A per-head I/I-seq split pinpoints a gather that
  substitutes lane-position for the loaded index value.

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


def _head_match(name, kernel, golden):
    """For each badly-wrong kernel head, find the closest golden head.

    If kernel head h matches golden head h' != h, the kernel is mixing
    up heads (a head-permutation / wrong-slice bug).
    """
    k = kernel.float()[0]  # [n_heads, D]
    g = golden.float()[0]
    n_heads = k.shape[0]
    print(f"  {name}: closest-golden-head for each wrong kernel head")
    for h in range(n_heads):
        self_dist = (k[h] - g[h]).abs().mean().item()
        if self_dist <= 2e-2:
            continue
        dists = (k[h : h + 1] - g).abs().mean(dim=1)  # [n_heads]
        best = int(dists.argmin())
        print(
            f"    kernel head {h:2d}: self-dist={self_dist:.4f}  "
            f"closest golden head={best:2d} (dist={dists[best].item():.4f})"
        )


def _indexed_cmp_paged(cmp_pa, cmp_bt, block_size):
    """Paged cmp_kv whose *logical* token i carries the constant value i.

    Inverts the block table so that ``_unpage_kv`` (and the golden's
    ``unpack_paged_kv``) resolve logical token i to a [D] vector of all
    ``i``. bf16 rounds indices >256 to the nearest representable value;
    that is fine for spotting a mis-gather.
    """
    pa = torch.zeros(cmp_pa.shape, dtype=torch.float32)
    _, bsz, _, _ = cmp_pa.shape
    B, table_len = cmp_bt.shape
    off = torch.arange(bsz, dtype=torch.float32).view(bsz, 1, 1)
    for b in range(B):
        for blk in range(table_len):
            phys = int(cmp_bt[b, blk])
            if phys < 0:
                continue
            pa[phys] = blk * block_size + off
    return pa.to(cmp_pa.dtype)


def _per_head_idx(name, kernel, golden, q_bnsd):
    """Per-head gathered-index readout for the cmp_kv[i]=i probe.

    ``kernel`` / ``golden``: [T1, n_heads, D] with T1=1. With cmp_kv[i]=i
    and ori_kv=0 every head row is ~constant over D and equals the
    softmax-weighted mean of the cmp-token indices that head attended to.
    A per-head mismatch therefore means the kernel gathered a different
    cmp-token multiset than the golden expects.
    """
    k = kernel.float()[0]  # [n_heads, D]
    g = golden.float()[0]
    n_heads, _ = k.shape
    bad = []
    for h in range(n_heads):
        km = k[h].mean().item()
        gm = g[h].mean().item()
        if abs(km - gm) > 2e-2:
            bad.append((h, km, gm, k[h].std().item()))
    print(f"  {name}: {len(bad)}/{n_heads} heads mismatch (weighted-mean readout)")
    for h, km, gm, kstd in bad[:40]:
        qsum = q_bnsd[0, h, 0, :].float().sum().item()
        print(
            f"    head {h:2d}: kernel={km:+9.3f} golden={gm:+9.3f} "
            f"diff={km - gm:+8.3f} kstd_over_D={kstd:.3f} sum(q_h)={qsum:+.1f}"
        )


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
    _head_match("head-match (A)", out_a, ref_a)

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

    # ---- Probe K: cmp_kv = constant 1.0. ----
    # Every cmp token is identical, so a wrong gather is undetectable;
    # this probe exercises gemm/softmax with NONZERO cmp values.
    # K passes -> bug is gathering wrong *distinct* token values.
    # K fails  -> bug is gemm/softmax with nonzero cmp data.
    print("\n-- Probe K: cmp_kv = constant 1.0 (real random indices) --")
    common_k = dict(common)
    common_k["cmp_kv"] = dev(torch.ones_like(case["cmp_pa"]))
    with torch.device("npu"):
        out_k = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_sparse_indices=dev(case["cmp_idx"]),
            **common_k,
        )
        torch.npu.synchronize()
    out_k = out_k.cpu()
    cmp_seqs_k = int(case["seqused_kv"].max()) // cfg["cmp_ratio"]
    ref_k_bnsd = G.sparse_attn_sharedkv_golden_bnsd(
        G.tnd_to_bnsd_q(case["q"], case["cu_seqlens_q"]),
        G.unpack_paged_kv(
            case["ori_pa"], case["ori_bt"], int(case["seqused_kv"].max())
        ),
        case["sinks"],
        act_q_lens=(case["cu_seqlens_q"][1:] - case["cu_seqlens_q"][:-1]).tolist(),
        act_kv_lens=case["seqused_kv"].tolist(),
        softmax_scale=cfg["softmax_scale"],
        cmp_k_bnsd=torch.ones((1, 1, cmp_seqs_k, cfg["D"]), dtype=dtype),
        cmp_sparse_indices=case["cmp_idx"].unsqueeze(0),
        cmp_ratio=cfg["cmp_ratio"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
    )
    ref_k = G.bnsd_to_tnd_out(ref_k_bnsd, case["cu_seqlens_q"])
    _diff("kernel K vs golden K", out_k, ref_k)

    # ---- Raw value dump for the worst head. ----
    print("\n-- head 31 raw dump (dims 252-268), Probe A --")
    print(
        "  kernel head31:",
        [round(x, 2) for x in out_a[0, 31, 252:268].float().tolist()],
    )
    print(
        "  golden head31:",
        [round(x, 2) for x in ref_a[0, 31, 252:268].float().tolist()],
    )
    print(
        "  kernel head 0:",
        [round(x, 2) for x in out_a[0, 0, 252:268].float().tolist()],
    )
    print(
        "  golden head 0:",
        [round(x, 2) for x in ref_a[0, 0, 252:268].float().tolist()],
    )

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

    # ---- Probe I: cmp_kv[i] = i, ori_kv = 0 -- gathered-index readout. ----
    # Every logical cmp token i carries the constant value i across D and
    # ori_kv is zeroed. The kernel output for each head is then constant
    # across D and equals the cmp-softmax-weighted mean of the token
    # indices the kernel ACTUALLY gathered; the golden (same cmp_kv, true
    # indices) yields the weighted mean of the TRUE set. A per-head
    # mismatch pinpoints a wrong gather. CFA passing already proves the
    # gather is correct for `arange` indices, so Probe I-seq below is the
    # control: if I-seq matches the golden but Probe I (random) does not,
    # the gather is substituting lane-position for the loaded index value.
    q_bnsd_i = G.tnd_to_bnsd_q(case["q"], case["cu_seqlens_q"])
    act_q_i = (case["cu_seqlens_q"][1:] - case["cu_seqlens_q"][:-1]).tolist()
    cmp_seqs_i = int(case["seqused_kv"].max()) // cfg["cmp_ratio"]
    cmp_k_idx_bnsd = (
        torch.arange(cmp_seqs_i, dtype=torch.float32)
        .view(1, 1, cmp_seqs_i, 1)
        .expand(1, cfg["N2"], cmp_seqs_i, cfg["D"])
        .to(dtype)
    )
    ori_zero_bnsd = torch.zeros(
        (1, cfg["N2"], int(case["seqused_kv"].max()), cfg["D"]), dtype=dtype
    )
    common_i = dict(common)
    common_i["ori_kv"] = dev(torch.zeros_like(case["ori_pa"]))
    common_i["cmp_kv"] = dev(
        _indexed_cmp_paged(case["cmp_pa"], case["cmp_bt"], cfg["block_size2"])
    )

    def _golden_idx(cmp_idx_bsnd):
        ref_bnsd = G.sparse_attn_sharedkv_golden_bnsd(
            q_bnsd_i,
            ori_zero_bnsd,
            case["sinks"],
            act_q_lens=act_q_i,
            act_kv_lens=case["seqused_kv"].tolist(),
            softmax_scale=cfg["softmax_scale"],
            cmp_k_bnsd=cmp_k_idx_bnsd,
            cmp_sparse_indices=cmp_idx_bsnd,
            cmp_ratio=cfg["cmp_ratio"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
        )
        return G.bnsd_to_tnd_out(ref_bnsd, case["cu_seqlens_q"])

    print("\n-- Probe I: cmp_kv[i]=i, ori_kv=0 (gathered-index readout) --")
    with torch.device("npu"):
        out_i = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_sparse_indices=dev(case["cmp_idx"]),
            **common_i,
        )
        torch.npu.synchronize()
    out_i = out_i.cpu()
    ref_i = _golden_idx(case["cmp_idx"].unsqueeze(0))
    _diff("kernel I vs golden I", out_i, ref_i)
    _per_head_idx("per-head (I)", out_i, ref_i, q_bnsd_i)

    # ---- Probe I-seq: same readout, sequential indices (control). ----
    print("\n-- Probe I-seq: cmp_kv[i]=i, ori_kv=0, sequential indices --")
    seq_idx_tnd = (
        torch.arange(cfg["K"], dtype=torch.int32).view(1, 1, cfg["K"]).contiguous()
    )
    with torch.device("npu"):
        out_iseq = sparse_attn_sharedkv(
            dev(case["q"]),
            cmp_sparse_indices=dev(seq_idx_tnd),
            **common_i,
        )
        torch.npu.synchronize()
    out_iseq = out_iseq.cpu()
    ref_iseq = _golden_idx(seq_idx_tnd.unsqueeze(0))
    _diff("kernel I-seq vs golden I-seq", out_iseq, ref_iseq)


if __name__ == "__main__":
    main()
