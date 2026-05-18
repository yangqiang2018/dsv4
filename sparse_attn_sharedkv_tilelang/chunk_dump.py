"""Per-chunk online-softmax dump -- localizes the ori->cmp divergence.

Established so far (gather_probe.py + debug_scfa.py):

* the cmp gather is correct (gather_probe: all OK);
* the cmp pass in isolation is correct (Probe I: ori_kv=0, bit-exact);
* SWA -- the ori pass alone -- is correct;

yet Probe A (real ori + real cmp) fails ~9.9%. So the bug is the
ori->cmp interaction: the ori chunks leave a non-zero online-softmax
state (acc_o != 0, m_i ~ +85) and the cmp chunks mishandle it. Probe I
hides it because ori_kv=0 leaves acc_o == 0 and 0 * alpha == 0.

The kernel debug build (api: ``return_chunk_dump=True``) now dumps the
pre-normalization ``acc_o`` after every chunk into ``dbg_acc_o``. This
script reruns scfa_decode, pulls that dump, and compares it
chunk-by-chunk against a CPU reference that replays the exact 64-wide
chunked online softmax. The first chunk where a bad head (2/31/...)
diverges while head 0 stays clean pins where the merge goes wrong.

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

    Returns ``[NI_total, n_heads, D]`` -- the pre-normalization ``acc_o``
    after each chunk, matching what the kernel dumps into ``dbg_acc_o``.
    """
    BI = 64
    D = cfg["D"]
    scale = cfg["softmax_scale"]
    cmp_ratio = cfg["cmp_ratio"]
    act_kv = int(case["seqused_kv"][0])
    act_q = 1  # scfa_decode: S1 = 1
    s_global = act_kv - act_q  # s_i = 0
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
    dump = torch.zeros(ni_total, n_heads, D)
    for ci in range(ni_total):
        if ci < ni_ori:
            g0 = ori_left + ci * BI
            k_tile = ori_k[g0 : g0 + BI]
            valid = torch.arange(g0, g0 + BI) <= ori_right
        else:
            idx = cmp_idx[(ci - ni_ori) * BI : (ci - ni_ori) * BI + BI]
            k_tile = cmp_k[idx]
            valid = (idx >= 0) & (idx < threshold)
        score = ((q @ k_tile.T) * scale).masked_fill(~valid.unsqueeze(0), float("-inf"))
        m_old = m.clone()
        m = torch.maximum(m, score.amax(dim=1))
        alpha = torch.exp(m_old - m)
        p = torch.exp(score - m.unsqueeze(1))
        s = alpha * s + p.sum(dim=1)
        p_low = p.to(torch.bfloat16).to(torch.float32)
        o = o * alpha.unsqueeze(1) + p_low @ k_tile
        dump[ci] = o
    return dump


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
        _, dbg = sparse_attn_sharedkv(
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
    dbg = dbg.float().cpu()  # [NI_total, n_heads, D]

    ref = _simulate_chunks(case, cfg)
    ni_total = dbg.shape[0]

    # head 0 is bit-exact in Probe A; the rest are its worst bad heads.
    watch = [0, 2, 18, 31, 42, 52, 59, 62]
    print(
        f"=== per-chunk acc_o: kernel dump vs CPU reference (NI_total={ni_total}) ==="
    )
    print("    chunk 0-1 = ori window, chunk 2+ = cmp; per-head numbers are max|diff|")
    for ci in range(ni_total):
        d = (dbg[ci] - ref[ci]).abs()
        per_head = d.amax(dim=1)  # [n_heads]
        worst = int(per_head.argmax())
        tag = "ori " if ci < 2 else f"cmp{ci - 2}"
        cols = "  ".join(f"h{h}={per_head[h].item():8.3f}" for h in watch)
        print(
            f"  chunk {ci:2d} [{tag}]: max|diff|={d.max().item():9.3f} "
            f"@head{worst:2d} | {cols}"
        )


if __name__ == "__main__":
    main()
