"""Micro-benchmarks for sparse_attn_sharedkv per-op cost model (vector side).

Why: the S2b/S2c profile chain (handoff #3 §2) showed Duration regressing on
every schedule cut while aiv_scalar (~10 ms) and aiv_mte2 (~8-9 ms) stay
flat -- per-op fixed overhead, not scheduling. Each kernel below isolates one
candidate: split vs fused VEC ops, per-row select, fragmented vs merged DMA,
flag/barrier cost. Run on the NPU and compare us/op deltas to decide the
next cut.

Usage:
    python sparse_attn_sharedkv_tilelang/bench_microop.py
"""

import time

import tilelang
import torch
from tilelang import language as T

tilelang.disable_cache()
tilelang.cache.clear_cache()

H, W = 32, 128  # acc_s tile shape, fp32
D = 512
REPS = 2000  # in-kernel repeats per launch (TIR loop)
LAUNCH = 5  # timed launches; first 5 warm up
NOISE_KEY = "noise"


def _bench(fn, src, name, ops_per_rep, results):
    out = torch.zeros(1, dtype=torch.float32).npu()
    for _ in range(5):
        fn(src, out)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(LAUNCH):
        fn(src, out)
    torch.npu.synchronize()
    dt_us = (time.perf_counter() - t0) / LAUNCH * 1e6
    results[name] = (dt_us, ops_per_rep)
    print(f"{name:28s} {dt_us:9.1f} us/launch  ({REPS} reps x {ops_per_rep} ops)")


def _kernel(body):
    @tilelang.jit
    def _make():
        @T.prim_func
        def k(
            Src: T.Tensor([H, D], "float"),  # type: ignore[valid-type]
            Out: T.Tensor([1], "float"),  # type: ignore[valid-type]
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a = T.alloc_ub([H, W], "float")
                b = T.alloc_ub([H, W], "float")
                row = T.alloc_ub([D], "float")
                blk = T.alloc_ub([16, D], "float")
                msk = T.alloc_ub([32], "uint8")
                T.annotate_address(
                    {a: 0, b: 16 * 1024, row: 32 * 1024, blk: 40 * 1024, msk: 80 * 1024}
                )
                with T.Scope("V"):
                    T.tile.fill(a, 1.0)
                    T.tile.fill(b, 2.0)
                    T.barrier_all()
                    # Duplicate (fill) rejects uint8 -- build the all-ones
                    # mask via compare like the main kernel does.
                    T.tile.compare(msk, b[0, :], T.float32(0.0), "GT")
                    T.barrier_all()
                    body(a, b, row, blk, msk, Src)
                    T.barrier_all()
                    T.copy(a[0, 0:1], Out[0:1])

        return k

    return _make()


def main():
    src = torch.randn(H, D, dtype=torch.float32).npu()
    results = {}

    def noise(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            pass

    def fused_mul(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            T.tile.mul(a, a, b)  # 1 op covers H*W

    def split_mul(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            for h in range(H):
                T.tile.mul(a[h, :], a[h, :], b[h, :])  # H ops

    def split_mul_scalar(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            for h in range(H):
                T.tile.mul(a[h, :], a[h, :], b[h, 0])  # H scalar-muls (V2 form)

    def per_row_select(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            for h in range(H):
                T.tile.select(a[h, :], msk, b[h, :], -1.0, "VSEL_TENSOR_SCALAR_MODE")

    def dma_rows(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            for r in range(16):
                T.copy(Src[r, :], blk[r, :])  # 16 x 2KB DMA

    def dma_block(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            T.copy(Src[0:16, :], blk)  # 1 x 32KB DMA

    def flag_pair(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)

    def barrier(a, b, row, blk, msk, Src):
        for _ in T.serial(REPS):
            T.barrier_all()

    cases = [
        (NOISE_KEY, noise, 0),
        ("mul_fused_32x128", fused_mul, 1),
        ("mul_split_32rows", split_mul, H),
        ("mul_split_scalar_32rows", split_mul_scalar, H),
        ("select_perrow_32rows", per_row_select, H),
        ("dma_16x2KB_rows", dma_rows, 16),
        ("dma_1x32KB_block", dma_block, 1),
        ("flag_set_wait_pair", flag_pair, 1),
        ("barrier_all", barrier, 1),
    ]
    for name, body, ops in cases:
        _bench(_kernel(body), src, name, ops, results)

    base = results[NOISE_KEY][0]
    print(f"\n-- net per-op cost (minus {NOISE_KEY} {base:.1f} us) --")
    for name, (dt, ops) in results.items():
        if name == NOISE_KEY or ops == 0:
            continue
        print(f"{name:28s} {(dt - base) / (REPS * ops) * 1000:9.2f} ns/op")


if __name__ == "__main__":
    main()
