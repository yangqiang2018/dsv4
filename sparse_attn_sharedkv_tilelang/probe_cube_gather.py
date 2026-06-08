"""Minimal probe: can the CUBE core do a two-step paged indirect gather
into L1?

This is the single risk gate for the "move KV gather to the cube side"
rewrite (so the vector core stops doing the gather + the ws_kv GM
round-trip, mirroring Ascend C's DataCopyPA). The main kernel currently
does the two-step paged gather on the VECTOR core (GM->UB, verified).
Whether TileLang's CUBE-side copy_gm_to_l1 accepts an indirect source
(physical block resolved from a block_table scalar load) is UNVERIFIED --
this probe isolates exactly that, nothing else.

Flow (cube-only, no vector, no cross-flag):
  1. for each i: logical = indices[i];
     phys = block_table[0, logical // block_size]; row = logical % block_size;
     copy KV[phys, row, 0, :] -> kv_l1[i, :]        <-- THE PROBE
  2. gemm(identity, kv_l1) -> acc_l0c               (I @ kv = kv; the only
     way to get L1 contents back to GM is via L0C / fixpipe, and it also
     proves the gathered data can feed a gemm -- what B actually needs)
  3. copy acc_l0c -> out

Scattered indices + a permuted block table make this the hard (cmp-like)
case, not the trivial block_table==0 one the examples used.

PASS  = compiles, runs, out matches the host gather (max_abs_diff ~ 0)
       -> cube-side paged gather is viable, the B rewrite can proceed.
FAIL  = compile error / aicore exception / wrong output
       -> cube can't do it, fall back to the cube<->vector overlap (A).

Run on an Ascend NPU host:
    python probe_cube_gather.py
"""

from __future__ import annotations

import sys

import torch

try:
    import tilelang
    from tilelang import language as T
except Exception as exc:  # pragma: no cover - host dependent
    print(f"[fatal] tilelang import failed: {exc!r}", file=sys.stderr)
    raise

# Dev-time: never reuse a stale compiled kernel while iterating on the probe.
tilelang.disable_cache()
tilelang.cache.clear_cache()


def build_probe(block_num, block_size, N, D, table_len, dtype):
    elem = 2  # bf16 / fp16 bytes
    # L1: kv_l1 [N, D] then ident_l1 [N, N]. L0C: acc [N, D] at 0.
    l1_addr = {"kv_l1": 0, "ident_l1": N * D * elem}

    @tilelang.jit(out_idx=[4])
    def _make():
        @T.prim_func
        def cube_gather_probe(
            KV: T.Tensor([block_num, block_size, 1, D], dtype),  # type: ignore[valid-type]
            block_table: T.Tensor([1, table_len], "int32"),  # type: ignore[valid-type]
            indices: T.Tensor([1, N], "int32"),  # type: ignore[valid-type]
            ident: T.Tensor([N, N], dtype),  # type: ignore[valid-type]
            out: T.Tensor([N, D], dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                kv_l1 = T.alloc_L1([N, D], dtype)
                ident_l1 = T.alloc_L1([N, N], dtype)
                acc = T.alloc_L0C([N, D], "float")
                T.annotate_address(
                    {
                        kv_l1: l1_addr["kv_l1"],
                        ident_l1: l1_addr["ident_l1"],
                        acc: 0,
                    }
                )
                with T.Scope("C"):
                    # --- THE PROBE: two-step paged gather, on the cube core. ---
                    for i in range(N):
                        logical = indices[0, i]
                        phys = block_table[0, logical // block_size]
                        row = logical % block_size
                        T.copy(KV[phys, row, 0, :], kv_l1[i, :])
                    T.barrier_all()
                    # --- extract L1 -> GM via I @ kv = kv (and prove the
                    #     gathered data is gemm-able, which B needs). ---
                    T.copy(ident, ident_l1)
                    T.barrier_all()
                    T.gemm_v0(ident_l1, kv_l1, acc, init=True)
                    T.barrier_all()
                    T.copy(acc, out)

        return cube_gather_probe

    return _make()


def main():
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - host dependent
        print(f"[fatal] torch_npu unavailable: {exc!r}", file=sys.stderr)
        return 2
    if not torch.npu.is_available():
        print(
            "[fatal] torch.npu.is_available() == False; need an NPU.", file=sys.stderr
        )
        return 2

    torch.manual_seed(0)
    block_num, block_size, N, D, table_len = 8, 64, 64, 512, 4
    dtype = torch.bfloat16

    KV = (torch.rand(block_num, block_size, 1, D) * 2 - 1).to(dtype)
    # Permuted (non-trivial) block table: logical block b -> physical perm[b].
    perm = torch.randperm(block_num)[:table_len].to(torch.int32)
    block_table = perm.reshape(1, table_len)
    # Scattered logical token ids over the valid range (cmp-like gather).
    indices = torch.randint(0, table_len * block_size, (1, N), dtype=torch.int32)
    ident = torch.eye(N, dtype=dtype)

    # Host golden: out[i] = KV[block_table[0, idx//bs], idx % bs, 0, :].
    golden = torch.zeros(N, D, dtype=dtype)
    for i in range(N):
        logical = int(indices[0, i])
        phys = int(block_table[0, logical // block_size])
        row = logical % block_size
        golden[i] = KV[phys, row, 0, :]

    kernel = build_probe(block_num, block_size, N, D, table_len, "bfloat16")
    with torch.device("npu"):
        out = kernel(KV.npu(), block_table.npu(), indices.npu(), ident.npu())
        torch.npu.synchronize()

    out_cpu = out.cpu().to(torch.float32)
    golden_f = golden.to(torch.float32)
    max_diff = (out_cpu - golden_f).abs().max().item()
    ok = torch.allclose(out_cpu, golden_f, rtol=1e-2, atol=1e-2)
    print(f"max_abs_diff = {max_diff:.6f}")
    if ok:
        print("PROBE PASS - cube-side paged gather works; the B rewrite can proceed.")
        return 0
    print("PROBE FAIL - output mismatch; cube gather is wrong (see diff).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
