"""Verify the MHC head-compute-mix GM->UB tile copy on Ascend NPU.

Validates, in isolation, the data movement needed to port the DeepSeek
TileKernels MHC operator to Ascend before writing the full kernel:

    GPU kernel:  tile_kernels/mhc/head_compute_mix_kernel.py
    GPU test:    tests/mhc/test_head_compute_mix.py

The GPU forward kernel walks ``input_mix`` of shape
``(num_tokens, mhc_mult)`` in tiles of ``token_block_size`` tokens, so
each block needs a ``[token_block_size, mhc_mult]`` float32 tile in UB.
The GPU test only ever uses ``mhc_mult == 4``.

On Ascend that tile has a ``mhc_mult * 4 = 16``-byte inner row, below
the 32-byte UB / MTE alignment floor: a 2D ``[token_block_size, 4]`` UB
buffer cannot be walked row-by-row by the vector unit (16B row stride
-> ADDR_MISALIGN) and a strided ``[token_block_size, 4]`` DMA has a
non-32B-aligned row stride.

Fix: ``token_block_size`` *consecutive* tokens are a contiguous
``token_block_size * mhc_mult`` float32 span -- 128 float32 = 512 B for
the (32, 4) tile, an exact 32B multiple. Reshape ``input_mix`` to
``[num_blocks, token_block_size * mhc_mult]`` and DMA one flat row per
core: a single aligned burst, no sub-32B row anywhere.

Two kernels are round-tripped GM->UB->GM and bit-compared to the input:

    build_copy_flat      recommended flat-row copy; must be bit-exact.
    build_copy_naive_2d  naive [token_block_size, mhc_mult] copy with
                         16-byte rows -- run only under --naive so a
                         hardware trap cannot mask the flat result.

Run on the NPU host:
    python3 verify_mhc_copy.py
    python3 verify_mhc_copy.py --naive
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T
import torch

# Dev-time: never serve a stale JIT artefact compiled from an older
# kernel body -- the cache key does not track every body edit.
tilelang.disable_cache()
tilelang.cache.clear_cache()

DTYPE = "float32"
VEC_NUM = 2  # vector sub-cores per AI Core on Atlas A3


@tilelang.jit(out_idx=[-1])
def build_copy_flat(num_blocks: int, blk: int):
    """Recommended copy: one contiguous ``blk``-element row per core.

    ``blk`` (= token_block_size * mhc_mult) float32 are split evenly
    over the VEC_NUM vector sub-cores; each sub-core DMAs its
    ``blk // VEC_NUM`` segment GM->UB and straight back UB->GM. Every
    segment is a single 32B-aligned burst, so the sub-32B mhc_mult
    inner dimension never appears as a DMA row stride.
    """
    half = blk // VEC_NUM

    @T.prim_func
    def main(
        x: T.Tensor([num_blocks, blk], DTYPE),
        y: T.Tensor([num_blocks, blk], DTYPE),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            buf = T.alloc_ub([1, half], DTYPE)
            with T.Scope("V"):
                T.copy(x[cid, vid * half], buf)
                T.barrier_all()  # drain MTE2 before MTE3 reads buf
                T.copy(buf, y[cid, vid * half])

    return main


@tilelang.jit(out_idx=[-1])
def build_copy_naive_2d(num_tokens: int, mhc_mult: int, token_block_size: int):
    """Naive copy: a 2D ``[rows, mhc_mult]`` UB tile with 16-byte rows.

    This is what a literal port of the GPU tiling would write. With
    mhc_mult == 4 the inner row is 16B < 32B; kept only as a
    hardware-truth contrast for build_copy_flat.
    """
    num_blocks = num_tokens // token_block_size
    rows_per_vid = token_block_size // VEC_NUM

    @T.prim_func
    def main(
        x: T.Tensor([num_tokens, mhc_mult], DTYPE),
        y: T.Tensor([num_tokens, mhc_mult], DTYPE),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            ub = T.alloc_ub([rows_per_vid, mhc_mult], DTYPE)
            with T.Scope("V"):
                row_base = cid * token_block_size + vid * rows_per_vid
                T.copy(x[row_base, 0], ub)
                T.barrier_all()
                T.copy(ub, y[row_base, 0])

    return main


def _check(name: str, got: torch.Tensor, ref: torch.Tensor, mhc_mult: int) -> bool:
    """Bit-exact compare of a pure copy; print one diagnostic line."""
    got = got.cpu().float()
    ref = ref.cpu().float()
    exact = torch.equal(got, ref)
    diff = (got - ref).abs()
    n_bad = int((diff > 0).sum())
    detail = ""
    if not exact:
        flat = int(diff.flatten().argmax())
        tok, col = flat // mhc_mult, flat % mhc_mult
        gf, rf = got.flatten(), ref.flatten()
        detail = (
            f" -- worst @ token={tok} col={col}: "
            f"got={gf[flat].item():+.6g} ref={rf[flat].item():+.6g}; "
            f"max|diff|={diff.max().item():.6g}"
        )
    status = "PASS" if exact else "FAIL"
    print(f"  [{status}] {name}: {n_bad}/{diff.numel()} elements differ{detail}")
    return exact


def _make_inputs(num_tokens: int, mhc_mult: int, seed: int):
    """A monotone ramp (catches chunk-shift / lane-substitution) + randn."""
    torch.manual_seed(seed)
    ramp = torch.arange(num_tokens * mhc_mult, dtype=torch.float32).reshape(
        num_tokens, mhc_mult
    )
    rand = torch.randn(num_tokens, mhc_mult, dtype=torch.float32)
    return (("ramp", ramp), ("randn", rand))


def _run_flat(num_tokens: int, mhc_mult: int, token_block_size: int, seed: int) -> bool:
    assert num_tokens % token_block_size == 0, (
        f"num_tokens ({num_tokens}) must be divisible by "
        f"token_block_size ({token_block_size})"
    )
    blk = token_block_size * mhc_mult
    assert blk % (VEC_NUM * 8) == 0, (
        f"token_block_size*mhc_mult ({blk}) must be a multiple of "
        f"{VEC_NUM * 8} so each vector core's float32 segment is 32B-aligned"
    )
    num_blocks = num_tokens // token_block_size
    kernel = build_copy_flat(num_blocks, blk)

    ok = True
    for name, src in _make_inputs(num_tokens, mhc_mult, seed):
        x = src.reshape(num_blocks, blk).npu().contiguous()
        y = kernel(x)
        torch.npu.synchronize()
        got = y.reshape(num_tokens, mhc_mult)
        ok &= _check(f"flat {name}", got, src, mhc_mult)
    return ok


def _run_naive(
    num_tokens: int, mhc_mult: int, token_block_size: int, seed: int
) -> bool:
    assert num_tokens % token_block_size == 0
    assert token_block_size % VEC_NUM == 0
    kernel = build_copy_naive_2d(num_tokens, mhc_mult, token_block_size)

    ok = True
    for name, src in _make_inputs(num_tokens, mhc_mult, seed):
        x = src.npu().contiguous()
        y = kernel(x)
        torch.npu.synchronize()
        ok &= _check(f"naive-2d {name}", y, src, mhc_mult)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the MHC input_mix GM->UB tile copy on Ascend NPU.",
    )
    parser.add_argument("--mhc-mult", type=int, default=4)
    parser.add_argument("--token-block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--naive",
        action="store_true",
        help="also run the unaligned [block, mhc_mult] 2D copy",
    )
    args = parser.parse_args()

    if not hasattr(torch, "npu"):
        print("torch_npu not available; run this on the NPU host.")
        return

    mm, tbs = args.mhc_mult, args.token_block_size
    # GPU test grid (tests/mhc/test_head_compute_mix.py): n0 in {1, 2},
    # n1 in {1024, 4096} -> num_tokens = n0 * n1.
    shapes = [1024, 2048, 4096, 8192]

    print(
        f"=== MHC input_mix GM->UB tile copy "
        f"(mhc_mult={mm}, token_block_size={tbs}) ==="
    )
    print(f"    [token_block_size, mhc_mult] inner row = {mm}*4 = {mm * 4}B")
    flat_bytes = tbs * mm * 4
    print(
        f"    flat per-core span = {tbs * mm}*4 = {flat_bytes}B "
        f"(32B-aligned: {flat_bytes % 32 == 0})"
    )

    print("\n-- recommended: flat contiguous-row copy --")
    all_ok = True
    for nt in shapes:
        print(f" num_tokens={nt}:")
        all_ok &= _run_flat(nt, mm, tbs, args.seed)

    if args.naive:
        print("\n-- contrast: naive 2D [block, mhc_mult] copy (16B rows) --")
        print("   an ADDR_MISALIGN trap aborts the process; a mis-lowered")
        print("   DMA shows up as FAIL lines -- either confirms the flat fix.")
        for nt in shapes:
            print(f" num_tokens={nt}:")
            try:
                _run_naive(nt, mm, tbs, args.seed)
            except Exception as exc:
                print(
                    f"  [ERROR] naive-2d num_tokens={nt}: {type(exc).__name__}: {exc}"
                )
    else:
        print("\n(run with --naive to also test the unaligned 2D copy)")

    print()
    if all_ok:
        print(
            "RESULT: flat-row GM->UB->GM copy is bit-exact for every shape. "
            "Use the [num_blocks, token_block_size*mhc_mult] reshape for the "
            "MHC input_mix tile load."
        )
    else:
        print("RESULT: flat-row copy MISMATCH -- see the FAIL line(s) above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
