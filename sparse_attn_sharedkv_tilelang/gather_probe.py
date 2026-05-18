"""Isolated cmp-gather probe for the SCFA mismatch.

The full SparseAttnSharedKV kernel mis-handles the compressed-KV pass:
SWA (no cmp pass) is correct, but every ``debug_scfa.py`` probe that
feeds the cmp pass *distinct* data fails ~9% -- Probe A (random sparse
indices), Probe B (sequential indices) and Probe C alike. Probe K/Z
(uniform cmp data) pass only because a wrong gather is invisible when
every cmp token carries the same value. The error is therefore
independent of the index *values*; it is structural.

This file strips the kernel down to ONLY the cmp gather. The kernel
below is vector-only and reproduces ``kernel.py``'s cmp gather loop
byte-for-byte::

    T.copy(cmp_indices[b, s, 0, c*BI : c*BI+BI], idx_int)
    T.barrier_all()
    for bi_i in range(BI // 2):
        lane = bi_i + vid * (BI // 2)
        T.copy(cmp_KV[b, idx_int[lane], 0, :], kv_ub)
        T.barrier_all()
        T.copy(kv_ub, <dest>[..., lane, :])
        T.barrier_all()

The only change vs the real kernel is the destination: gathered rows go
straight to the output tensor instead of into the ``ws_kv`` workspace
(both are affine UB->GM copies, so this does not hide a gather bug).

Interpretation:

* If this probe reproduces the mismatch, the gather copy with a
  UB-loaded index is itself broken -- the bug is the gather, full stop.
* If this probe is clean, the gather copy is fine and the bug lives
  downstream (the ws_kv handoff, the cube's L1 copy, or the gemm).

The ``cmp_kv[i] = i`` tests make each gathered row reveal the token
index the kernel actually fetched, so a mismatch names the exact
(chunk, lane) and the wrong index.

A second section (``_poison_test``) adds one ``T.tile.createvecindex``
to the otherwise-identical kernel. The baseline cmp gather is known
good; the full kernel's cmp-pass index ``T.copy`` silently no-ops even
after fix (1) gave it a private buffer. The only structural feature the
full kernel then still shares with this probe and the baseline does not
is a createvecindex elsewhere in the V scope. This section isolates
whether that op is what breaks a subsequent GM->UB index copy.

Run on the NPU host:  python3 gather_probe.py
"""

# NOTE: deliberately no `from __future__ import annotations` here. PEP 563
# turns the @T.prim_func parameter annotations into strings, and the
# TileLang TIR parser then rejects them ("expected Object but got str").
import torch

import tilelang
from tilelang import language as T

# Same reasoning as kernel.py: a stale on-disk kernel must not be reused
# across source edits.
tilelang.disable_cache()

DEFAULT_CORE_NUM = 24


def build_gather_probe(
    *,
    batch: int,
    max_seq: int,
    max_cmp_s: int,
    topk: int,
    head_dim: int = 512,
    block_I: int = 64,
    core_num: int = DEFAULT_CORE_NUM,
    dtype: str = "bfloat16",
    with_createvecindex: bool = False,
):
    """Build the isolated vector-only cmp-gather kernel.

    Output ``[batch, max_seq, topk, head_dim]``: ``out[b, s, k, :]`` is
    the row the kernel gathered for sparse slot ``k`` of work item
    ``(b, s)`` -- i.e. ``cmp_KV[b, cmp_indices[b, s, 0, k], 0, :]`` if
    the gather is correct.

    With ``with_createvecindex=True`` the kernel runs one
    ``T.tile.createvecindex`` (into a separate UB buffer) at the top of
    the V scope, before the cmp-copy loop -- mirroring kernel.py's ori
    chunks under fix (1). It then also returns ``cvi_out``, the dumped
    createvecindex result, which both keeps the op alive (no DCE) and
    proves it executed. This isolates whether a createvecindex anywhere
    in the V scope poisons a subsequent GM->UB index ``T.copy``.
    """
    assert topk % block_I == 0, "topk must be a multiple of block_I"
    BI = block_I
    D = head_dim
    NI = topk // BI
    indices_dtype = "int32"

    cmp_kv_shape = [batch, max_cmp_s, 1, D]
    indices_shape = [batch, max_seq, 1, topk]
    out_shape = [batch, max_seq, topk, D]
    cvi_shape = [batch, max_seq, BI]

    # Recognisable base: cvi_out reads back as [cvi_base, cvi_base+1, ...],
    # proof the createvecindex actually ran rather than being optimised out.
    cvi_base = 4096

    if with_createvecindex:

        @tilelang.jit(out_idx=[2, 3])
        def _make():
            @T.prim_func
            def main(
                cmp_KV: T.Tensor(cmp_kv_shape, dtype),  # type: ignore[valid-type]
                cmp_indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
                Output: T.Tensor(out_shape, dtype),  # type: ignore[valid-type]
                cvi_out: T.Tensor(cvi_shape, indices_dtype),  # type: ignore[valid-type]
            ):
                with T.Kernel(core_num, is_npu=True) as (cid, vid):
                    idx_int = T.alloc_ub([BI], indices_dtype)
                    idx_cvi = T.alloc_ub([BI], indices_dtype)
                    kv_ub = T.alloc_ub([D], dtype)
                    T.annotate_address({idx_int: 0, idx_cvi: 1024, kv_ub: 2048})

                    total_work = batch * max_seq
                    for slot in T.serial(T.ceildiv(total_work, core_num)):
                        pid = slot * core_num + cid
                        if pid < total_work:
                            b_i = pid // max_seq
                            s_i = pid % max_seq
                            with T.Scope("V"):
                                # One createvecindex into a private buffer,
                                # before the cmp-copy loop -- mirrors the
                                # ori chunks of kernel.py under fix (1).
                                # The cvi_out dump keeps it live (no DCE)
                                # and proves it executed.
                                T.tile.createvecindex(idx_cvi, cvi_base)
                                T.barrier_all()
                                T.copy(idx_cvi, cvi_out[b_i, s_i, :])
                                T.barrier_all()
                                for ci in range(NI):
                                    T.copy(
                                        cmp_indices[
                                            b_i,
                                            s_i,
                                            0,
                                            ci * BI : ci * BI + BI,
                                        ],
                                        idx_int,
                                    )
                                    T.barrier_all()
                                    for bi_i in range(BI // 2):
                                        lane = bi_i + vid * (BI // 2)
                                        T.copy(
                                            cmp_KV[b_i, idx_int[lane], 0, :],
                                            kv_ub,
                                        )
                                        T.barrier_all()
                                        T.copy(
                                            kv_ub,
                                            Output[b_i, s_i, ci * BI + lane, :],
                                        )
                                        T.barrier_all()

            return main

        return _make()

    @tilelang.jit(out_idx=[2])
    def _make():
        @T.prim_func
        def main(
            cmp_KV: T.Tensor(cmp_kv_shape, dtype),  # type: ignore[valid-type]
            cmp_indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
            Output: T.Tensor(out_shape, dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                idx_int = T.alloc_ub([BI], indices_dtype)
                kv_ub = T.alloc_ub([D], dtype)
                T.annotate_address({idx_int: 0, kv_ub: 2048})

                total_work = batch * max_seq
                for slot in T.serial(T.ceildiv(total_work, core_num)):
                    pid = slot * core_num + cid
                    if pid < total_work:
                        b_i = pid // max_seq
                        s_i = pid % max_seq
                        with T.Scope("V"):
                            for ci in range(NI):
                                T.copy(
                                    cmp_indices[
                                        b_i,
                                        s_i,
                                        0,
                                        ci * BI : ci * BI + BI,
                                    ],
                                    idx_int,
                                )
                                T.barrier_all()
                                for bi_i in range(BI // 2):
                                    lane = bi_i + vid * (BI // 2)
                                    T.copy(
                                        cmp_KV[b_i, idx_int[lane], 0, :],
                                        kv_ub,
                                    )
                                    T.barrier_all()
                                    T.copy(
                                        kv_ub,
                                        Output[b_i, s_i, ci * BI + lane, :],
                                    )
                                    T.barrier_all()

        return main

    return _make()


def _ref_gather(cmp_kv: torch.Tensor, cmp_idx: torch.Tensor) -> torch.Tensor:
    """Trivial torch reference: out[b, s, k, :] = cmp_kv[b, idx, 0, :]."""
    B, S, _, K = cmp_idx.shape
    D = cmp_kv.shape[-1]
    out = cmp_kv.new_zeros((B, S, K, D))
    for b in range(B):
        for s in range(S):
            idx = cmp_idx[b, s, 0, :].long()
            out[b, s, :, :] = cmp_kv[b, idx, 0, :]
    return out


def _report(name, kernel_out, ref, cmp_idx, block_I=64):
    """Per-(chunk, lane) mismatch report. A pure gather is bitwise exact."""
    k = kernel_out.float()
    r = ref.float()
    B, S, K, D = k.shape
    bad = []
    for kk in range(K):
        d = (k[0, 0, kk] - r[0, 0, kk]).abs().max().item()
        if d > 1e-3:
            bad.append((kk, d))
    status = "OK" if not bad else "MISMATCH"
    print(f"  [{status}] {name}: {len(bad)}/{K} gathered rows wrong")
    for kk, d in bad[:48]:
        chunk, lane = kk // block_I, kk % block_I
        exp_idx = int(cmp_idx[0, 0, 0, kk])
        # With cmp_kv[i]=i, row[0] is the (bf16-rounded) gathered index.
        print(
            f"    k={kk:4d} (chunk {chunk:2d} lane {lane:2d}): "
            f"max|diff|={d:9.3f}  want_idx={exp_idx:5d}  "
            f"row[0] kernel={k[0, 0, kk, 0].item():+10.2f} "
            f"ref={r[0, 0, kk, 0].item():+10.2f}"
        )
    if len(bad) > 48:
        print(f"    ... and {len(bad) - 48} more")


def _run(build_kwargs, cmp_kv, cmp_idx):
    func = build_gather_probe(**build_kwargs)

    def dev(t):
        return t.npu().contiguous()

    with torch.device("npu"):
        out = func(dev(cmp_kv), dev(cmp_idx))
        torch.npu.synchronize()
    return out.cpu()


def _poison_test():
    """Decisive isolation test for the SCFA chunk-2 bug.

    The full kernel's cmp-pass GM->UB index ``T.copy`` silently no-ops,
    and kernel.py fix (1) (a private cmp index buffer) did not fix it --
    so the cause is not which buffer createvecindex writes. This runs
    the known-good baseline gather and an *identical* kernel with one
    createvecindex added (separate buffer), on the same inputs:

    * baseline OK + variant OK  -> createvecindex is exonerated; the
      kernel.py poison is some other structural feature.
    * baseline OK + variant MISMATCH -> a createvecindex anywhere in the
      V scope breaks a later GM->UB copy; fix (2) (delete it) is right.

    ``cvi_out[:8]`` must read back as ``[4096..4103]``; if not, the
    createvecindex was optimised away and the variant proves nothing.
    """
    print("\n==== createvecindex poison test (topk=512, random idx) ====")
    B, MAXS, D = 1, 1, 512
    MAX_CMP = 2048
    topk = 512
    cmp_kv_idx = (
        torch.arange(MAX_CMP, dtype=torch.float32)
        .view(1, MAX_CMP, 1, 1)
        .expand(1, MAX_CMP, 1, D)
        .to(torch.bfloat16)
    )
    rand_idx = (
        torch.randperm(MAX_CMP)[:topk].to(torch.int32).view(1, 1, 1, topk).contiguous()
    )
    ref = _ref_gather(cmp_kv_idx, rand_idx)
    kw = dict(batch=B, max_seq=MAXS, max_cmp_s=MAX_CMP, topk=topk, head_dim=D)

    def dev(t):
        return t.npu().contiguous()

    func0 = build_gather_probe(**kw)
    with torch.device("npu"):
        out0 = func0(dev(cmp_kv_idx), dev(rand_idx))
        torch.npu.synchronize()
    _report("baseline  (no createvecindex)", out0.cpu(), ref, rand_idx)

    func1 = build_gather_probe(with_createvecindex=True, **kw)
    with torch.device("npu"):
        out1, cvi = func1(dev(cmp_kv_idx), dev(rand_idx))
        torch.npu.synchronize()
    _report("variant   (+ createvecindex)", out1.cpu(), ref, rand_idx)
    cvi8 = cvi.cpu()[0, 0, :8].tolist()
    print(f"    cvi_out[:8] = {cvi8}  (expect [4096..4103] => createvecindex ran)")


def main():
    if not hasattr(torch, "npu"):
        print("torch_npu not available; run this on the NPU host.")
        return

    torch.manual_seed(0)
    B, MAXS, D = 1, 1, 512
    MAX_CMP = 2048  # = scfa_decode floor(8193 / cmp_ratio=4)

    # cmp_KV variants. idx-encoded: token i carries the value i across D,
    # so a gathered row directly reads out the index the kernel fetched.
    cmp_kv_idx = (
        torch.arange(MAX_CMP, dtype=torch.float32)
        .view(1, MAX_CMP, 1, 1)
        .expand(1, MAX_CMP, 1, D)
        .to(torch.bfloat16)
    )
    cmp_kv_rand = (torch.rand(1, MAX_CMP, 1, D) * 20 - 10).to(torch.bfloat16)

    for topk in (512, 2048):
        print(f"\n==== isolated cmp gather, topk={topk} ({topk // 64} chunks) ====")
        kw = dict(batch=B, max_seq=MAXS, max_cmp_s=MAX_CMP, topk=topk, head_dim=D)
        rand_idx = (
            torch.randperm(MAX_CMP)[:topk]
            .to(torch.int32)
            .view(1, 1, 1, topk)
            .contiguous()
        )
        seq_idx = torch.arange(topk, dtype=torch.int32).view(1, 1, 1, topk).contiguous()

        # Test 1 -- idx readout, random indices. The decisive test: a
        # mismatch names the (chunk, lane) and the wrong index gathered.
        out = _run(kw, cmp_kv_idx, rand_idx)
        _report(
            "idx-readout / random idx",
            out,
            _ref_gather(cmp_kv_idx, rand_idx),
            rand_idx,
        )

        # Test 2 -- idx readout, sequential indices.
        out = _run(kw, cmp_kv_idx, seq_idx)
        _report(
            "idx-readout / sequential idx",
            out,
            _ref_gather(cmp_kv_idx, seq_idx),
            seq_idx,
        )

        # Test 3 -- real-style distinct data, random indices.
        out = _run(kw, cmp_kv_rand, rand_idx)
        _report(
            "random data / random idx",
            out,
            _ref_gather(cmp_kv_rand, rand_idx),
            rand_idx,
        )

    _poison_test()


if __name__ == "__main__":
    main()
