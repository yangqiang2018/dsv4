"""Verify the MHC head-compute-mix GM->UB tile copy on Ascend NPU.

Validates, in isolation, the data movement needed to port the DeepSeek
TileKernels MHC operator to Ascend before writing the full kernel:

    GPU kernel:  tile_kernels/mhc/head_compute_mix_kernel.py
    GPU test:    tests/mhc/test_head_compute_mix.py

The GPU forward kernel walks input_mix of shape (num_tokens, mhc_mult)
in tiles of token_block_size tokens; each block needs a
[token_block_size, mhc_mult] float32 tile in UB. The GPU test only ever
uses mhc_mult == 4.

On Ascend a [token_block_size, 4] tile has a 4*4 = 16-byte inner row,
below the 32-byte UB / MTE alignment floor. The fix: token_block_size
*consecutive* tokens form a contiguous token_block_size * mhc_mult
float32 span -- 128 float32 = 512 B for the (32, 4) tile, an exact 32B
multiple. input_mix is reshaped to a 2D [M, N] tensor with
N = token_block_size * mhc_mult (an aligned flat row) and
M = num_tokens // token_block_size, then round-tripped GM->UB->GM and
bit-compared against the input.

The kernel is a copy-only clone of examples/reduce/example_reduce_min.py
(a verified Ascend example): identical @tilelang.jit / T.Kernel /
T.Scope / T.copy structure, with the reduction replaced by a copy-back.

Run on the NPU host:
    python3 verify_mhc_copy.py
"""

import argparse

import tilelang
from tilelang import language as T
import torch

tilelang.disable_cache()
tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[1])
def copy_kernel(M, N, block_M, dtype="float"):
    """GM->UB->GM round-trip of an [M, N] float tensor.

    A copy-only clone of example_reduce_min.py: the M rows are tiled
    block_M per core and split over VEC_NUM=2 vector sub-cores; each
    sub-core DMAs its sub_block_M rows into UB and straight back out.
    With N a multiple of 8, every row is a 32B-aligned burst.
    """
    m_num = M // block_M
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            row_base = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
                T.barrier_all()
                T.copy(a_ub, B[row_base : row_base + sub_block_M, :])

    return main


def _check(name, got, ref, mhc_mult):
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


def _make_inputs(num_tokens, mhc_mult, seed):
    """A monotone ramp (catches chunk-shift / lane substitution) + randn."""
    torch.manual_seed(seed)
    ramp = torch.arange(num_tokens * mhc_mult, dtype=torch.float32).reshape(
        num_tokens, mhc_mult
    )
    rand = torch.randn(num_tokens, mhc_mult, dtype=torch.float32)
    return (("ramp", ramp), ("randn", rand))


def _run(num_tokens, mhc_mult, token_block_size, seed):
    assert num_tokens % token_block_size == 0, (
        f"num_tokens ({num_tokens}) must be divisible by "
        f"token_block_size ({token_block_size})"
    )
    n = token_block_size * mhc_mult  # one aligned flat row
    m = num_tokens // token_block_size  # number of flat rows
    assert n % 8 == 0, f"flat row of {n} float32 must be 32B-aligned"
    assert m >= 2 and m % 2 == 0, f"flat-row count {m} must be even and >= 2"

    # block_M: even, divides m, kept small so m_num stays modest.
    block_m = 2
    for cand in (32, 16, 8, 4):
        if cand <= m and m % cand == 0:
            block_m = cand
            break

    kernel = copy_kernel(m, n, block_m)

    ok = True
    for name, src in _make_inputs(num_tokens, mhc_mult, seed):
        a = src.reshape(m, n).npu().contiguous()
        b = kernel(a)
        torch.npu.synchronize()
        got = b.reshape(num_tokens, mhc_mult)
        ok &= _check(name, got, src, mhc_mult)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify the MHC input_mix GM->UB tile copy on Ascend NPU.",
    )
    parser.add_argument("--mhc-mult", type=int, default=4)
    parser.add_argument("--token-block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
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
    flat = tbs * mm
    print(
        f"    flat row = token_block_size*mhc_mult = {flat} float32 "
        f"= {flat * 4}B (32B-aligned: {flat * 4 % 32 == 0})"
    )

    all_ok = True
    for nt in shapes:
        print(f" num_tokens={nt}:")
        all_ok &= _run(nt, mm, tbs, args.seed)

    print()
    if all_ok:
        print(
            "RESULT: flat-row GM->UB->GM copy is bit-exact for every shape. "
            "Reshape input_mix to [num_blocks, token_block_size*mhc_mult] "
            "for the MHC tile load."
        )
    else:
        print("RESULT: flat-row copy MISMATCH -- see the FAIL line(s) above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
