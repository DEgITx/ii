# SPDX-License-Identifier: MIT
"""Runs an ONNX model with ONNX Runtime and with the ii CLI on the same inputs and compares the outputs.

    python scripts/compare_onnxruntime.py model.onnx --ii build/.../ii [--dims batch=1] [--inputs in.npz]

Inputs are taken from --inputs (an .npz archive) or generated: floats from a normal distribution, integers in
[0, --int-high), booleans at random. Symbolic dimensions take the values of --dims (default 1). Prints, per output,
the maximum absolute and relative error and the cosine similarity, and exits with 1 when an output exceeds the
tolerance (--rtol / --atol, relative to the output's largest magnitude).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


ONNX_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def parse_dims(text):
    dims = {}
    for item in filter(None, (text or "").split(",")):
        name, value = item.split("=")
        dims[name] = int(value)
    return dims


def make_inputs(session, dims, seed, int_high):
    rng = np.random.default_rng(seed)
    inputs = {}
    for info in session.get_inputs():
        shape = [d if isinstance(d, int) else dims.get(d, 1) for d in info.shape]
        dtype = ONNX_TO_NUMPY.get(info.type)
        if dtype is None:
            raise SystemExit(f"unsupported input type {info.type} of '{info.name}'")
        if np.issubdtype(dtype, np.floating):
            inputs[info.name] = rng.standard_normal(shape).astype(dtype)
        elif dtype == np.bool_:
            inputs[info.name] = rng.integers(0, 2, shape).astype(np.bool_)
        else:
            inputs[info.name] = rng.integers(0, int_high, shape).astype(dtype)
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--ii", required=True, help="path of the ii executable")
    parser.add_argument("--dims", default="", help="symbolic dimensions, e.g. batch=1,sequence=128")
    parser.add_argument("--fix-dims", action="store_true",
                        help="also fix the symbolic dimensions when ii loads the model (static shapes)")
    parser.add_argument("--inputs", help="an .npz archive of inputs (otherwise random)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--int-high", type=int, default=100, help="upper bound of random integer inputs")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="tolerance relative to the output's largest magnitude")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--backends", default="", help="ii backends (default: all)")
    parser.add_argument("--save", help="directory to keep inputs and both outputs in")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    import onnxruntime as ort

    options = ort.SessionOptions()
    if args.threads:
        options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, options, providers=["CPUExecutionProvider"])
    dims = parse_dims(args.dims)
    if args.inputs:
        inputs = dict(np.load(args.inputs))
    else:
        inputs = make_inputs(session, dims, args.seed, args.int_high)

    start = time.perf_counter()
    expected = session.run(None, inputs)
    ort_ms = (time.perf_counter() - start) * 1000
    names = [o.name for o in session.get_outputs()]

    work = args.save or tempfile.mkdtemp(prefix="ii_compare_")
    os.makedirs(work, exist_ok=True)
    input_file = os.path.join(work, "inputs.npz")
    output_file = os.path.join(work, "ii_outputs.npz")
    np.savez(input_file, **inputs)
    np.savez(os.path.join(work, "ort_outputs.npz"), **dict(zip(names, expected)))
    command = [os.path.abspath(args.ii), "run", args.model, "--inputs", input_file, "--save-outputs", output_file,
               "--repeat", "1", "--warmup", "0"]
    if args.dims and args.fix_dims:
        command += ["--dims", args.dims]
    if args.backends:
        command += ["--backends", args.backends]
    if args.threads:
        command += ["--threads", str(args.threads)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, result.stderr, sep="\n")
        return 1
    actual = dict(np.load(output_file))

    failed = False
    print(f"{os.path.basename(args.model)} (ONNX Runtime {ort.__version__}, first run {ort_ms:.1f} ms)")
    for name, want in zip(names, expected):
        got = actual.get(name)
        if got is None:
            print(f"  {name}: missing in ii outputs")
            failed = True
            continue
        if got.shape != want.shape:
            print(f"  {name}: shape {got.shape} vs {want.shape}")
            failed = True
            continue
        if want.dtype == np.bool_ or np.issubdtype(want.dtype, np.integer):
            mismatches = int(np.count_nonzero(got != want))
            print(f"  {name}: {want.dtype}{list(want.shape)}, {mismatches} mismatching elements")
            failed |= mismatches != 0
            continue
        w = want.astype(np.float64).ravel()
        g = got.astype(np.float64).ravel()
        scale = float(np.max(np.abs(w))) if w.size else 0.0
        abs_err = float(np.max(np.abs(g - w))) if w.size else 0.0
        cosine = float(np.dot(w, g) / (np.linalg.norm(w) * np.linalg.norm(g))) if w.size and np.any(w) else 1.0
        ok = abs_err <= args.atol + args.rtol * scale
        line = (f"  {name}: {want.dtype}{list(want.shape)}, max |error| {abs_err:.3g} "
                f"({abs_err / scale if scale else 0:.3g} of max |value| {scale:.3g}), cosine {cosine:.8f}")
        if want.ndim >= 2 and want.shape[-1] > 1:
            top_w = np.argmax(want.reshape(-1, want.shape[-1]), axis=-1)
            top_g = np.argmax(got.reshape(-1, got.shape[-1]), axis=-1)
            line += f", argmax agreement {np.mean(top_w == top_g) * 100:.1f}%"
        print(line + ("" if ok else "  <-- exceeds tolerance"))
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
