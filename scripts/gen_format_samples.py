#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Writes small GGUF and safetensors files with the reference implementations (gguf-py, safetensors), so the C++
readers are tested against files written by the libraries the ecosystem uses.

Usage:
    python scripts/gen_format_samples.py            # writes tests/data/formats/sample.{gguf,safetensors}
    python scripts/gen_format_samples.py --check    # fails if the committed files are out of date

Requires numpy, gguf and safetensors. The output is deterministic.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter, quants
from safetensors.numpy import save

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "data" / "formats"


def gguf_bytes() -> bytes:
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.gguf"
        writer = GGUFWriter(str(path), arch="llama")
        writer.add_name("sample")
        writer.add_uint32("llama.block_count", 2)
        writer.add_float32("llama.rope.freq_base", 10000.0)
        writer.add_bool("sample.flag", True)
        writer.add_int64("sample.negative", -5)
        writer.add_array("sample.strings", ["a", "bc", ""])
        writer.add_array("sample.ints", [1, -2, 3])
        writer.add_array("sample.floats", [0.5, -1.25])
        writer.add_string("sample.unicode", "привет ✓")
        f32 = np.arange(12, dtype=np.float32).reshape(3, 4)
        writer.add_tensor("f32", f32)
        writer.add_tensor("f16", (np.arange(8, dtype=np.float32) / 4).astype(np.float16).reshape(2, 4))
        values = rng.standard_normal(2 * 64).astype(np.float32).reshape(2, 64)
        q8 = quants.quantize(values, GGMLQuantizationType.Q8_0)
        writer.add_tensor("q8_0", q8, raw_dtype=GGMLQuantizationType.Q8_0)
        writer.add_tensor("i32", np.array([7, -8, 9], dtype=np.int32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path.read_bytes()


def safetensors_bytes() -> bytes:
    tensors = {
        "a.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b.bias": np.array([1, 2, 3, 4], dtype=np.float16),
        "c.ids": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "scalar": np.array(2.5, dtype=np.float64),
    }
    return save(tensors, metadata={"format": "pt"})  # one key: the order of several varies between runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the committed files are out of date")
    args = parser.parse_args()
    files = {"sample.gguf": gguf_bytes(), "sample.safetensors": safetensors_bytes()}
    status = 0
    for name, data in files.items():
        path = OUTPUT / name
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                print(f"{path.relative_to(ROOT)} is out of date; run scripts/gen_format_samples.py", file=sys.stderr)
                status = 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"wrote {path.relative_to(ROOT)} ({len(data)} bytes)")
    return status


if __name__ == "__main__":
    sys.exit(main())
