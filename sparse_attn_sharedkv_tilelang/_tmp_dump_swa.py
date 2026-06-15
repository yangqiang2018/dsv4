"""THROWAWAY diagnostic (delete after use).

Builds swa_prefill_fast, pulls the kernel from api._KERNEL_CACHE, and prints the
UB buffer ADDRESS MAP (every GetWithOffset) + the brcb / row_muls / Broadcast
calls -- to find what alpha_brd8 collides with. Run on the container:

    python sparse_attn_sharedkv_tilelang/_tmp_dump_swa.py

Paste the whole stdout back. Full source also written to /tmp/kern_swa.cc.
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
    _call_metadata_then_sharedkv(
        case, cfg, sparse_attn_sharedkv, sparse_attn_sharedkv_metadata
    )
except Exception as exc:  # noqa: BLE001 -- only need the build
    print("call raised (ignored, only the build matters):", repr(exc))

print(f"\n_KERNEL_CACHE size = {len(_KERNEL_CACHE)}")
for func in _KERNEL_CACHE.values():
    src = func.get_kernel_source()
    lines = src.splitlines()
    if not any("GetWithOffset" in ln for ln in lines):
        continue
    with open("/tmp/kern_swa.cc", "w") as f:
        f.write(src)
    print(f"=== wrote /tmp/kern_swa.cc ({len(lines)} lines) ===\n")
    print(
        "--- UB ADDRESS MAP (GetWithOffset: name = GetWithOffset<T>(count, byteoffset)) ---"
    )
    for i, ln in enumerate(lines):
        if "GetWithOffset" in ln:
            print(f"{i:>5}: {ln.strip()}")
    print("\n--- brcb / row_muls / Broadcast calls ---")
    for i, ln in enumerate(lines):
        if "brcb" in ln or "row_muls" in ln or "Broadcast" in ln:
            print(f"{i:>5}: {ln.strip()}")
    break
