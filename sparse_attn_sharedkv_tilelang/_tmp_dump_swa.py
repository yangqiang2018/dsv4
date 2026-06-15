"""THROWAWAY diagnostic (delete after use).

Builds the swa_prefill_fast kernel through the real api path, pulls the compiled
handle out of api._KERNEL_CACHE, and prints the generated Ascend C around every
Broadcast call -- to localize the lever-3A (V2 merge broadcast+wide-mul)
regression. Run on the container:

    python sparse_attn_sharedkv_tilelang/_tmp_dump_swa.py

Paste the whole stdout back. The full source is also written to /tmp/kern_swa.cc.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from test_sparse_attn_sharedkv import (  # noqa: E402
    _build_case,
    _call_metadata_then_sharedkv,
)
from test_sparse_attn_sharedkv_fast import FAST_SCENARIOS  # noqa: E402
from api import sparse_attn_sharedkv, _KERNEL_CACHE  # noqa: E402
from metadata import sparse_attn_sharedkv_metadata  # noqa: E402

cfg = FAST_SCENARIOS["swa_prefill_fast"]
case = _build_case(cfg, torch.bfloat16)
try:
    # This builds + caches the kernel (and runs it; the wrong numeric result is
    # irrelevant -- we only need the compiled source).
    _call_metadata_then_sharedkv(
        case, cfg, sparse_attn_sharedkv, sparse_attn_sharedkv_metadata
    )
except Exception as exc:  # noqa: BLE001 -- a runtime hiccup is fine, build is what we need
    print("call raised (ignored, only the build matters):", repr(exc))

print(f"\n_KERNEL_CACHE size = {len(_KERNEL_CACHE)}")
found = False
for func in _KERNEL_CACHE.values():
    src = func.get_kernel_source()
    lines = src.splitlines()
    hits = [i for i, ln in enumerate(lines) if "roadcast" in ln]
    if not hits:
        continue
    found = True
    with open("/tmp/kern_swa.cc", "w") as f:
        f.write(src)
    print(
        f"=== wrote /tmp/kern_swa.cc ({len(lines)} lines), {len(hits)} Broadcast hit(s) ===\n"
    )
    for i in hits:
        print(f"----- around line {i} -----")
        for j in range(max(0, i - 2), min(len(lines), i + 11)):
            print(f"{j:>6}: {lines[j].rstrip()}")
        print()

if not found:
    print("NO Broadcast found in any cached kernel.")
    if _KERNEL_CACHE:
        src = next(iter(_KERNEL_CACHE.values())).get_kernel_source()
        with open("/tmp/kern_swa.cc", "w") as f:
            f.write(src)
        print("wrote /tmp/kern_swa.cc for manual inspection")
