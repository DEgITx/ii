#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generates reference data for the block-quantization codecs (src/tensor/quant.cpp).

For every GGUF block type that gguf-py (the Python package of llama.cpp) implements:
  <type>_blocks / <type>_dequant   random blocks (finite scales) and their dequantization by gguf-py;
  <type>_input / <type>_quant      random values and their quantization by gguf-py (types gguf-py can quantize).

Usage:
    python scripts/gen_quant_references.py            # writes tests/data/quant/references.npz
    python scripts/gen_quant_references.py --check    # fails if the committed file is out of date

Requires numpy and gguf (pip install gguf). The output is deterministic.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import quants

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "data" / "quant" / "references.npz"

# ii dtype name -> (gguf type, block elements, block bytes, offsets of f16 scale fields, offset of an f32 scale)
TYPES = {
    "q4_0": (T.Q4_0, 32, 18, [0], None),
    "q4_1": (T.Q4_1, 32, 20, [0, 2], None),
    "q5_0": (T.Q5_0, 32, 22, [0], None),
    "q5_1": (T.Q5_1, 32, 24, [0, 2], None),
    "q8_0": (T.Q8_0, 32, 34, [0], None),
    "q2_k": (T.Q2_K, 256, 84, [80, 82], None),
    "q3_k": (T.Q3_K, 256, 110, [108], None),
    "q4_k": (T.Q4_K, 256, 144, [0, 2], None),
    "q5_k": (T.Q5_K, 256, 176, [0, 2], None),
    "q6_k": (T.Q6_K, 256, 210, [208], None),
}
QUANTIZABLE = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0"}


def build() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    arrays: dict[str, np.ndarray] = {}
    for name, (qtype, elements, nbytes, f16_offsets, f32_offset) in TYPES.items():
        blocks = 8 if elements == 256 else 32
        raw = rng.integers(0, 256, size=(blocks, nbytes), dtype=np.uint8)
        for offset in f16_offsets:
            scales = rng.uniform(-0.05, 0.05, size=blocks).astype(np.float16)
            raw[:, offset : offset + 2] = scales.view(np.uint8).reshape(blocks, 2)
        if f32_offset is not None:
            scales = rng.uniform(-0.05, 0.05, size=blocks).astype(np.float32)
            raw[:, f32_offset : f32_offset + 4] = scales.view(np.uint8).reshape(blocks, 4)
        flat = raw.reshape(-1)
        arrays[f"{name}_blocks"] = flat
        arrays[f"{name}_dequant"] = quants.dequantize(flat, qtype).astype(np.float32).reshape(-1)
        if name in QUANTIZABLE:
            values = (rng.standard_normal(blocks * elements) * rng.uniform(0.1, 3.0)).astype(np.float32)
            values[:elements] = 0.0  # an all-zero block
            values[elements : 2 * elements] = 1.0  # a constant block
            arrays[f"{name}_input"] = values
            arrays[f"{name}_quant"] = quants.quantize(values.reshape(1, -1), qtype).reshape(-1)
    return arrays


def serialize(arrays: dict[str, np.ndarray]) -> bytes:
    """A deterministic .npz (fixed timestamps, sorted names, DEFLATE)."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the committed file is out of date")
    args = parser.parse_args()
    data = serialize(build())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != data:
            print(f"{OUTPUT.relative_to(ROOT)} is out of date; run scripts/gen_quant_references.py", file=sys.stderr)
            return 1
        print(f"{OUTPUT.relative_to(ROOT)} is up to date")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(data)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
