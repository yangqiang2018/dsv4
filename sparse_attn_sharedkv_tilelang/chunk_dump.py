"""Per-chunk online-softmax dump -- localizes the ori->cmp divergence.

Established so far (gather_probe.py + debug_scfa.py + the acc_o dump):

* the cmp gather is correct (gather_probe: all OK);
* the cmp pass in isolation is correct (Probe I: ori_kv=0, bit-exact);
* SWA -- the ori pass alone -- is correct;
* the per-chunk acc_o dump shows chunk 0/1 (ori) bit-exact and the
  divergence appearing abruptly at chunk 2, the FIRST cmp chunk.

So the bug fires inside the first cmp chunk: it processes the non-zero
online-softmax state the ori chunks left behind (acc_o != 0, m_i ~ +85)
and corrupts acc_o by ~|acc_o_ori|.

This script dumps four per-chunk quantities -- post-chunk ``acc_o``,
``m_i``, ``sumexp`` and the chunk's own ``P@V`` -- and compares each
against a CPU replay of the exact 64-wide chunked online softmax. At
chunk 2 this tells us directly whether the running max ``m_i`` is wrong
(alpha mis-scales acc_o) or ``P@V`` is wrong (gemm2 / softmax probs).

Run on the NPU host:  python3 chunk_dump.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import golden as G  # noqa: E402
from test_sparse_attn_sharedkv import SCENARIOS, _build_case  # noqa: E402


def _simulate_chunks(case, cfg):
    """Replay the kernel's 64-wide chunked online softmax on CPU.

    Returns ``(acc_o, m_i, sumexp, pv)`` per chunk, matching the kernel's
    ``dbg_acc_o`` / ``dbg_m`` / ``dbg_s`` / ``dbg_pv`` dumps.
    """
    BI = 64
    D = cfg["D"]
    scale = cfg["softmax_scale"]
    cmp_ratio = cfg["cmp_ratio"]
    act_kv = int(case["seqused_kv"][0])
    s_global = act_kv - 1  # act_q = 1, s_i = 0
    ori_left = max(s_global - cfg["ori_win_left"], 0)
    ori_right = s_global
    threshold = (s_global + 1) // cmp_ratio
    ni_ori = 2
    ni_total = ni_ori + cfg["K"] // BI

    q = case["q"][0].float()  # [n_heads, D]
    n_heads = q.shape[0]
    m = case["sinks"].float().clone()  # [n_heads]
    ori_k = G.unpack_paged_kv(case["ori_pa"], case["ori_bt"], act_kv)[0, 0].float()
    cmp_k = G.unpack_paged_kv(case["cmp_pa"], case["cmp_bt"], act_kv // cmp_ratio)[
        0, 0
    ].float()
    cmp_idx = case["cmp_idx"][0, 0].long()  # [K]

    s = torch.ones(n_heads)
    o = torch.zeros(n_heads, D)
    dump_o = torch.zeros(ni_total, n_heads, D)
    dump_m = torch.zeros(ni_total, n_heads)
    dump_s = torch.zeros(ni_total, n_heads)
    dump_pv = torch.zeros(ni_total, n_heads, D)
    dump_mprev = torch.zeros(ni_total, n_heads)
    dump_score = torch.zeros(ni_total, n_heads, BI)
    dump_idxf = torch.zeros(ni_total, BI)
    for ci in range(ni_total):
        if ci < ni_ori:
            g0 = ori_left + ci * BI
            k_tile = ori_k[g0 : g0 + BI]
            valid = torch.arange(g0, g0 + BI) <= ori_right
            idxf = torch.arange(g0, g0 + BI, dtype=torch.float32)
        else:
            idx = cmp_idx[(ci - ni_ori) * BI : (ci - ni_ori) * BI + BI]
            k_tile = cmp_k[idx]
            valid = (idx >= 0) & (idx < threshold)
            idxf = idx.float()
        dump_idxf[ci] = idxf
        score = ((q @ k_tile.T) * scale).masked_fill(~valid.unsqueeze(0), float("-inf"))
        dump_score[ci] = score
        m_old = m.clone()
        dump_mprev[ci] = m_old
        m = torch.maximum(m, score.amax(dim=1))
        alpha = torch.exp(m_old - m)
        p = torch.exp(score - m.unsqueeze(1))
        s = alpha * s + p.sum(dim=1)
        pv = p.to(torch.bfloat16).to(torch.float32) @ k_tile
        o = o * alpha.unsqueeze(1) + pv
        dump_o[ci], dump_m[ci], dump_s[ci], dump_pv[ci] = o, m, s, pv
    return dump_o, dump_m, dump_s, dump_pv, dump_mprev, dump_score, dump_idxf


def main():
    if not hasattr(torch, "npu"):
        print("torch_npu not available; run this on the NPU host.")
        return

    from api import sparse_attn_sharedkv

    cfg = dict(SCENARIOS["scfa_decode"])
    case = _build_case(cfg, torch.bfloat16)

    def dev(t):
        return None if t is None else t.npu().contiguous()

    with torch.device("npu"):
        _, k_o, k_m, k_s, k_pv, k_mprev, k_score, k_idxf = sparse_attn_sharedkv(
            dev(case["q"]),
            ori_kv=dev(case["ori_pa"]),
            cmp_kv=dev(case["cmp_pa"]),
            cmp_sparse_indices=dev(case["cmp_idx"]),
            ori_block_table=dev(case["ori_bt"]),
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
            topk_cmp=cfg["K"],
            return_chunk_dump=True,
        )
        torch.npu.synchronize()
    k_o = k_o.float().cpu()
    k_m = k_m.float().cpu()
    k_s = k_s.float().cpu()
    k_pv = k_pv.float().cpu()
    k_mprev = k_mprev.float().cpu()
    k_score = k_score.float().cpu()
    k_idxf = k_idxf.float().cpu()

    r_o, r_m, r_s, r_pv, r_mprev, r_score, r_idxf = _simulate_chunks(case, cfg)
    ni_total = k_o.shape[0]

    # head 0 is bit-exact in Probe A; the rest are its worst bad heads.
    watch = [0, 2, 18, 31, 42, 52, 59, 62]
    print(f"=== per-chunk dump: kernel vs CPU reference (NI_total={ni_total}) ===")
    print("    chunk 0-1 = ori window, chunk 2+ = cmp")
    for ci in range(ni_total):
        tag = "ori " if ci < 2 else f"cmp{ci - 2}"
        do = (k_o[ci] - r_o[ci]).abs().max().item()
        dpv = (k_pv[ci] - r_pv[ci]).abs().max().item()
        dm = (k_m[ci] - r_m[ci]).abs().max().item()
        ds = (k_s[ci] - r_s[ci]).abs().max().item()
        print(
            f"  chunk {ci:2d} [{tag}]: acc_o|d|={do:9.3f}  pv|d|={dpv:9.3f}  "
            f"m|d|={dm:11.4f}  s|d|={ds:13.3f}"
        )

    print("\n  --- chunk 2 (cmp0) per-head detail ---")
    print("  (m: running max -> if kernel != ref the alpha rescale is wrong;")
    print("   pv: this chunk's P@V -> if |d| is large the softmax/gemm2 is wrong)")
    for h in watch:
        pvd = (k_pv[2, h] - r_pv[2, h]).abs().max().item()
        od = (k_o[2, h] - r_o[2, h]).abs().max().item()
        print(
            f"    h{h:2d}: mprev kernel={k_mprev[2, h]:9.3f} "
            f"ref={r_mprev[2, h]:9.3f}  ->  m kernel={k_m[2, h]:9.3f} "
            f"ref={r_m[2, h]:9.3f}  |  pv|d|={pvd:8.3f}  acc_o|d|={od:8.3f}"
        )

    print("\n  --- chunk 2 (cmp0) score analysis (reduce_max input) ---")
    print("  (mask-mismatch: kernel -inf where ref finite, or vice versa;")
    print("   score-bad: both lanes finite but |kernel-ref| > 0.5)")
    for h in watch:
        ks = k_score[2, h]
        rs = r_score[2, h]
        kfin = ks[torch.isfinite(ks)]
        rfin = rs[torch.isfinite(rs)]
        mask_mm = int((~torch.isfinite(ks) != ~torch.isfinite(rs)).sum())
        both = torch.isfinite(ks) & torch.isfinite(rs)
        score_bad = int(((ks - rs).abs()[both] > 0.5).sum())
        k_amax = kfin.max().item() if kfin.numel() else float("nan")
        r_amax = rfin.max().item() if rfin.numel() else float("nan")
        print(
            f"    h{h:2d}: score amax kernel={k_amax:9.3f} ref={r_amax:9.3f}  "
            f"mask-mismatch={mask_mm:2d}  score-bad={score_bad:2d}/{ks.numel()}"
        )

    print("\n  --- chunk 2 (cmp0) idx_float: kernel vs reference ---")
    ki = k_idxf[2]
    ri = r_idxf[2]
    d = (ki - ri).abs()
    print(
        f"    idx_float mismatch={int((d > 0.5).sum())}/{ki.numel()}  "
        f"max|diff|={d.max().item():.3f}"
    )
    print(f"    kernel min={ki.min().item():+.1f} max={ki.max().item():+.1f}")
    print(f"    ref    min={ri.min().item():+.1f} max={ri.max().item():+.1f}")
    print(f"    kernel[:8]={[round(x, 1) for x in ki[:8].tolist()]}")
    print(f"    ref   [:8]={[round(x, 1) for x in ri[:8].tolist()]}")


if __name__ == "__main__":
    main()
