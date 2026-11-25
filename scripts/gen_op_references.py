#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generates reference results for the OpSet conformance tests.

Every case is a one-operation model in the ii IR text format plus input and expected output tensors. Expected values
are computed with NumPy / PyTorch in float64 and rounded once to the output type, so they are the correctly rounded
results the reference backend must reproduce.

Usage:
    python scripts/gen_op_references.py            # writes tests/data/op_references.json
    python scripts/gen_op_references.py --check    # fails if the committed file is out of date

Requires numpy, torch (CPU build is enough) and gguf (the llama.cpp package, for block-quantized tensors). The
output is deterministic.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "data" / "op_references.json"

# ---------------------------------------------------------------------------------------------------------------------
# Tensors and types


class BF16:
    """A bfloat16 tensor (NumPy has no bfloat16): float values rounded to bf16, stored as float32."""

    def __init__(self, values: np.ndarray):
        t = torch.from_numpy(np.asarray(values, dtype=np.float64)).to(torch.bfloat16)
        self.values = t.to(torch.float32).numpy()
        self.bits = t.view(torch.int16).numpy()
        self.shape = self.values.shape


NP_TO_II = {
    np.dtype(np.float32): "f32",
    np.dtype(np.float16): "f16",
    np.dtype(np.float64): "f64",
    np.dtype(np.int8): "i8",
    np.dtype(np.uint8): "u8",
    np.dtype(np.int16): "i16",
    np.dtype(np.uint16): "u16",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.int64): "i64",
    np.dtype(np.uint64): "u64",
    np.dtype(np.bool_): "bool",
}


class QBlocks:
    """A block-quantized tensor (GGUF layout): raw blocks, the ii dtype name and the logical shape."""

    def __init__(self, dtype: str, shape: tuple, raw: np.ndarray):
        self.dtype_name = dtype
        self.shape = tuple(shape)
        self.raw = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1)


def ii_dtype(t) -> str:
    if isinstance(t, QBlocks):
        return t.dtype_name
    return "bf16" if isinstance(t, BF16) else NP_TO_II[t.dtype]


def tensor_type(t) -> str:
    return f"{ii_dtype(t)}[{', '.join(str(d) for d in t.shape)}]"


def encode(t) -> dict:
    if isinstance(t, QBlocks):
        return {"dtype": t.dtype_name, "shape": list(t.shape), "base64": base64.b64encode(t.raw.tobytes()).decode()}
    raw = t.bits if isinstance(t, BF16) else np.ascontiguousarray(t)
    data = raw.astype(raw.dtype.newbyteorder("<"), copy=False).tobytes()
    return {"dtype": ii_dtype(t), "shape": list(t.shape), "base64": base64.b64encode(data).decode("ascii")}


class DType:
    """An attribute value that is a dtype name."""

    def __init__(self, name: str):
        self.name = name


class Dims:
    """An attribute value that is a dims list (may contain -1)."""

    def __init__(self, values):
        self.values = list(values)


def format_attr(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        text = repr(value)
        return text if any(c in text for c in ".en") else text + ".0"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, DType):
        return value.name
    if isinstance(value, Dims):
        return "[" + ", ".join(str(v) for v in value.values) + "]"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_attr(v) for v in value) + "]"
    raise TypeError(f"unsupported attribute {value!r}")


# ---------------------------------------------------------------------------------------------------------------------
# Cases

CASES: list[dict] = []


def add(name: str, op: str, inputs: list, outputs: list, attrs: dict | None = None, rtol: float = 1e-6,
        atol: float = 1e-6, absent: tuple[int, ...] = ()):
    """Adds a case. `absent` lists input positions passed as `_` (inputs are the present ones, in order)."""
    attrs = attrs or {}
    lines = []
    args = []
    k = 0
    for position in range(len(inputs) + len(absent)):
        if position in absent:
            args.append("_")
            continue
        lines.append(f"  input %in{k} : {tensor_type(inputs[k])}")
        args.append(f"%in{k}")
        k += 1
    results = ", ".join(f"%out{i} : {tensor_type(o)}" for i, o in enumerate(outputs))
    attr_text = ""
    if attrs:
        attr_text = " {" + ", ".join(f"{key} = {format_attr(attrs[key])}" for key in sorted(attrs)) + "}"
    lines.append(f"  {results} = {op}({', '.join(args)}){attr_text}")
    for i in range(len(outputs)):
        lines.append(f"  output out{i} = %out{i}")
    graph = "graph {\n" + "\n".join(lines) + "\n}\n"
    CASES.append({
        "name": name,
        "graph": graph,
        "inputs": [encode(t) for t in inputs],
        "outputs": [encode(t) for t in outputs],
        "rtol": rtol,
        "atol": atol,
    })


RNG = np.random.default_rng(20261125)


def rand(*shape, low=-2.0, high=2.0, dtype=np.float32) -> np.ndarray:
    return RNG.uniform(low, high, size=shape).astype(dtype)


def randint(*shape, low=-10, high=10, dtype=np.int32) -> np.ndarray:
    return RNG.integers(low, high + 1, size=shape).astype(dtype)


def f64(x) -> np.ndarray:
    return np.asarray(x.values if isinstance(x, BF16) else x, dtype=np.float64)


def like(x, values):
    """Rounds float64 `values` to the type of `x`."""
    if isinstance(x, BF16):
        return BF16(values)
    return np.asarray(values).astype(x.dtype)


def t64(x) -> torch.Tensor:
    return torch.from_numpy(f64(x))


# --- Unary ----------------------------------------------------------------------------------------------------------

def gen_unary():
    from scipy.special import erf

    float_ops = {
        "neg": (lambda x: -x, None),
        "abs": (np.abs, None),
        "sign": (np.sign, None),
        "exp": (np.exp, None),
        "log": (np.log, (0.05, 4.0)),
        "sqrt": (np.sqrt, (0.0, 4.0)),
        "rsqrt": (lambda x: 1.0 / np.sqrt(x), (0.05, 4.0)),
        "recip": (lambda x: 1.0 / x, (0.1, 3.0)),
        "sin": (np.sin, None),
        "cos": (np.cos, None),
        "tanh": (np.tanh, None),
        "sigmoid": (lambda x: 1.0 / (1.0 + np.exp(-x)), None),
        "erf": (erf, None),
        "relu": (lambda x: np.maximum(x, 0.0), None),
        "floor": (np.floor, None),
        "ceil": (np.ceil, None),
        "round": (np.round, None),
    }
    for op, (fn, bounds) in float_ops.items():
        low, high = bounds or (-3.0, 3.0)
        x = rand(4, 5, low=low, high=high)
        add(f"{op}_f32", op, [x], [like(x, fn(f64(x)))])
        h = rand(7, low=low, high=high, dtype=np.float16)
        add(f"{op}_f16", op, [h], [like(h, fn(f64(h)))], rtol=1e-3, atol=1e-3)
    b = BF16(rand(6, low=-3, high=3))
    add("exp_bf16", "exp", [b], [BF16(np.exp(f64(b)))], rtol=1e-2, atol=1e-2)
    d = rand(5, dtype=np.float64)
    add("tanh_f64", "tanh", [d], [np.tanh(d)], rtol=1e-14, atol=1e-14)
    # Halfway cases of round (ties to even).
    halves = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 0.49999997], dtype=np.float32)
    add("round_ties", "round", [halves], [np.round(f64(halves)).astype(np.float32)])
    i = randint(10, low=-100, high=100)
    add("neg_i32", "neg", [i], [-i])
    add("abs_i32", "abs", [i], [np.abs(i)])
    add("relu_i32", "relu", [i], [np.maximum(i, 0)])
    add("sign_i64", "sign", [i.astype(np.int64)], [np.sign(i).astype(np.int64)])
    m = np.array([True, False, True], dtype=np.bool_)
    add("not", "not", [m], [~m])


# --- Binary -----------------------------------------------------------------------------------------------------------

def int_trunc_div(a, b):
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    q = np.zeros(np.broadcast(a, b).shape, dtype=np.int64)
    nz = np.broadcast_to(b != 0, q.shape)
    aa, bb = np.broadcast_arrays(a, b)
    q[nz] = np.sign(aa[nz]) * np.sign(bb[nz]) * (np.abs(aa[nz]) // np.abs(bb[nz]))
    return q


def gen_binary():
    a = rand(3, 1, 4)
    b = rand(5, 1)
    arith = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "max": np.maximum,
        "min": np.minimum,
    }
    for op, fn in arith.items():
        add(f"{op}_broadcast_f32", op, [a, b], [like(a, fn(f64(a), f64(b)))])
    divisor = rand(5, 1, low=0.5, high=2.0) * np.where(rand(5, 1) > 0, 1, -1).astype(np.float32)
    add("div_f32", "div", [a, divisor], [like(a, f64(a) / f64(divisor))])
    base = rand(3, 4, low=0.1, high=3.0)
    exponent = rand(4, low=-2.0, high=2.0)
    add("pow_f32", "pow", [base, exponent], [like(base, np.power(f64(base), f64(exponent)))])
    fa = rand(12, low=-7, high=7)
    fb = rand(12, low=0.5, high=3) * np.where(rand(12) > 0, 1, -1).astype(np.float32)
    add("mod_f32", "mod", [fa, fb], [like(fa, np.mod(f64(fa), f64(fb)))], rtol=1e-5, atol=1e-5)
    add("fmod_f32", "mod", [fa, fb], [like(fa, np.fmod(f64(fa), f64(fb)))], {"fmod": True}, rtol=1e-5, atol=1e-5)
    h1 = rand(2, 3, dtype=np.float16)
    h2 = rand(3, dtype=np.float16)
    add("mul_f16", "mul", [h1, h2], [like(h1, f64(h1) * f64(h2))], rtol=1e-3, atol=1e-3)

    ia = randint(4, 5, low=-50, high=50)
    ib = randint(5, low=1, high=9) * np.where(randint(5) > 0, 1, -1).astype(np.int32)
    add("add_i32", "add", [ia, ib], [ia + ib])
    add("mul_i32", "mul", [ia, ib], [ia * ib])
    add("div_i32", "div", [ia, ib], [int_trunc_div(ia, ib).astype(np.int32)])
    add("mod_i32", "mod", [ia, ib], [np.mod(ia, ib)])
    add("fmod_i32", "mod", [ia, ib], [np.fmod(ia, ib)], {"fmod": True})
    zeros = np.array([7, -7, 0], dtype=np.int32)
    add("div_by_zero_i32", "div", [zeros, np.zeros(3, dtype=np.int32)], [np.zeros(3, dtype=np.int32)])
    pb = randint(6, low=-4, high=4, dtype=np.int64)
    pe = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    add("pow_i64", "pow", [pb, pe], [pb ** pe])
    wrap_a = np.array([120, -120, 100], dtype=np.int8)
    wrap_b = np.array([10, -10, 100], dtype=np.int8)
    add("add_wraps_i8", "add", [wrap_a, wrap_b], [(wrap_a.astype(np.int16) + wrap_b).astype(np.int8)])

    ca = randint(3, 4, low=-3, high=3)
    cb = randint(4, low=-3, high=3)
    for op, fn in {"eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal, "gt": np.greater,
                   "ge": np.greater_equal}.items():
        add(f"{op}_i32", op, [ca, cb], [fn(ca, cb)])
    fx = np.array([1.0, np.nan, -np.inf, 2.0], dtype=np.float32)
    fy = np.array([1.0, np.nan, -np.inf, 3.0], dtype=np.float32)
    add("eq_nan_f32", "eq", [fx, fy], [np.equal(fx, fy)])
    la = RNG.integers(0, 2, size=(2, 3)).astype(np.bool_)
    lb = RNG.integers(0, 2, size=(3,)).astype(np.bool_)
    add("and", "and", [la, lb], [la & lb])
    add("or", "or", [la, lb], [la | lb])
    add("xor", "xor", [la, lb], [la ^ lb])


# --- Other elementwise ------------------------------------------------------------------------------------------------

def gen_other_elementwise():
    cond = RNG.integers(0, 2, size=(2, 1, 3)).astype(np.bool_)
    a = rand(4, 3)
    b = rand(1)
    add("where", "where", [cond, a, b], [np.where(cond, a, b)])
    x = rand(10, low=-300, high=300)
    add("cast_f32_i32", "cast", [x], [np.trunc(x).astype(np.int32)], {"to": DType("i32")})
    add("cast_f32_u8", "cast", [x], [np.clip(np.trunc(x), 0, 255).astype(np.uint8)], {"to": DType("u8")})
    g = rand(8, low=-1000, high=1000)
    add("cast_f32_f16", "cast", [g], [g.astype(np.float16)], {"to": DType("f16")})
    add("cast_f32_bf16", "cast", [g], [BF16(f64(g))], {"to": DType("bf16")})
    add("cast_f64_f16", "cast", [g.astype(np.float64) / 3], [(g.astype(np.float64) / 3).astype(np.float16)],
        {"to": DType("f16")})
    add("cast_f32_bool", "cast", [np.array([0, 1.5, -0.0, np.nan], dtype=np.float32)],
        [np.array([False, True, False, True])], {"to": DType("bool")})
    ints = randint(6, low=-100000, high=100000)
    add("cast_i32_f32", "cast", [ints], [ints.astype(np.float32)], {"to": DType("f32")})
    add("cast_bool_i64", "cast", [np.array([True, False])], [np.array([1, 0], dtype=np.int64)], {"to": DType("i64")})
    c = rand(10, low=-10, high=10)
    add("clamp_f32", "clamp", [c], [np.clip(c, -1.5, 2.0)], {"min": -1.5, "max": 2.0})
    add("clamp_min_only", "clamp", [c], [np.maximum(c, 0.0).astype(np.float32)], {"min": 0.0})
    ci = randint(10)
    add("clamp_i32", "clamp", [ci], [np.clip(ci, -3, 4)], {"min": -3.5, "max": 4.5})


# --- Reductions -------------------------------------------------------------------------------------------------------

def gen_reductions():
    x = rand(3, 4, 5)
    reducers = {"reduce_sum": np.sum, "reduce_mean": np.mean, "reduce_max": np.max, "reduce_min": np.min,
                "reduce_prod": np.prod}
    for op, fn in reducers.items():
        for axes, keep in (([1], False), ([0, 2], True), ([], False), ([-1], True)):
            axis = tuple(axes) if axes else None
            expected = fn(f64(x), axis=axis, keepdims=keep)
            suffix = "_".join(str(a).replace("-", "m") for a in axes) or "all"
            add(f"{op}_{suffix}_{'keep' if keep else 'drop'}", op, [x], [like(x, expected)],
                {"axes": axes, "keepdims": keep}, rtol=1e-5)
    xi = randint(3, 4, low=-5, high=5)
    add("reduce_sum_i32", "reduce_sum", [xi], [np.sum(xi, axis=1, dtype=np.int32)], {"axes": [1]})
    h = rand(4, 16, dtype=np.float16)
    add("reduce_sum_f16", "reduce_sum", [h], [like(h, np.sum(f64(h), axis=1))], {"axes": [1]}, rtol=2e-3, atol=2e-3)
    a = rand(4, 6)
    add("argmax", "argmax", [a], [np.argmax(a, axis=1)], {"axis": 1})
    add("argmin_keep", "argmin", [a], [np.argmin(a, axis=0)[None, :]], {"axis": 0, "keepdims": True})
    ties = np.array([[1, 3, 3, 0], [2, 2, 1, 2]], dtype=np.float32)
    add("argmax_ties_first", "argmax", [ties], [np.array([1, 0])], {"axis": 1})
    add("argmax_ties_last", "argmax", [ties], [np.array([2, 3])], {"axis": 1, "select_last_index": True})
    nans = np.array([[1, np.nan, 5, np.nan]], dtype=np.float32)
    add("argmax_nan", "argmax", [nans], [np.argmax(nans, axis=1)], {"axis": 1})
    add("argmin_nan", "argmin", [nans], [np.argmin(nans, axis=1)], {"axis": 1})


# --- Linear algebra, convolution, pooling ---------------------------------------------------------------------------

def gen_linalg():
    a = rand(3, 4)
    b = rand(4, 5)
    add("matmul_2d", "matmul", [a, b], [like(a, f64(a) @ f64(b))])
    ab = rand(2, 1, 3, 4)
    bb = rand(5, 4, 2)
    add("matmul_batched_broadcast", "matmul", [ab, bb], [like(ab, f64(ab) @ f64(bb))])
    at = rand(4, 3)
    bt = rand(5, 4)
    add("matmul_transposed", "matmul", [at, bt], [like(at, f64(at).T @ f64(bt).T)],
        {"transpose_a": True, "transpose_b": True})
    v = rand(4)
    add("matmul_vector_matrix", "matmul", [v, b], [like(v, f64(v) @ f64(b))])
    add("matmul_matrix_vector", "matmul", [a, v], [like(a, f64(a) @ f64(v))])
    add("matmul_vector_vector", "matmul", [v, v], [like(v, np.array(f64(v) @ f64(v)))])
    ia = randint(2, 3)
    ib = randint(3, 2)
    add("matmul_i32", "matmul", [ia, ib], [ia @ ib])
    ha = rand(3, 8, dtype=np.float16)
    hb = rand(8, 2, dtype=np.float16)
    add("matmul_f16", "matmul", [ha, hb], [like(ha, f64(ha) @ f64(hb))], rtol=2e-3, atol=2e-3)

    def conv_case(name, x, w, bias, stride, padding, dilation, groups, attrs):
        ref = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[x.ndim - 2]
        y = ref(t64(x), t64(w), None if bias is None else t64(bias), stride, padding, dilation, groups)
        inputs = [x, w] + ([bias] if bias is not None else [])
        add(name, "conv", inputs, [like(x, y.numpy())], attrs, rtol=1e-5, atol=1e-5)

    x = rand(1, 3, 8, 7)
    w = rand(4, 3, 3, 3)
    bias = rand(4)
    conv_case("conv2d", x, w, bias, 1, 0, 1, 1, {})
    conv_case("conv2d_stride_pad", x, w, None, 2, 1, 1, 1, {"strides": [2, 2], "pads": [1, 1, 1, 1]})
    conv_case("conv2d_dilation", x, w, bias, 1, 2, 2, 1, {"dilations": [2, 2], "pads": [2, 2, 2, 2]})
    xg = rand(2, 4, 6, 6)
    wg = rand(6, 2, 3, 3)
    conv_case("conv2d_groups", xg, wg, rand(6), 1, 1, 1, 2, {"groups": 2, "pads": [1, 1, 1, 1]})
    xd = rand(1, 3, 9)
    wd = rand(3, 1, 3)
    conv_case("conv1d_depthwise", xd, wd, None, 2, 0, 1, 3, {"groups": 3, "strides": [2]})
    x3 = rand(1, 2, 4, 4, 4)
    w3 = rand(2, 2, 2, 2, 2)
    conv_case("conv3d", x3, w3, None, 1, 0, 1, 1, {})
    # Asymmetric padding (only expressible with explicit pads): pad the input with numpy first.
    xa = rand(1, 1, 5, 5)
    wa = rand(1, 1, 3, 3)
    padded = np.pad(f64(xa), ((0, 0), (0, 0), (0, 2), (1, 0)))
    ya = F.conv2d(torch.from_numpy(padded), t64(wa))
    add("conv2d_asymmetric_pads", "conv", [xa, wa], [like(xa, ya.numpy())], {"pads": [0, 1, 2, 0]}, rtol=1e-5,
        atol=1e-5)
    # SAME_UPPER: output ceil(7 / 2) = 4 needs 3 padding elements, the extra one at the end.
    xs = rand(1, 1, 7)
    ws = rand(1, 1, 4)
    ys = F.conv1d(torch.from_numpy(np.pad(f64(xs), ((0, 0), (0, 0), (1, 2)))), t64(ws), stride=2)
    add("conv1d_same_upper", "conv", [xs, ws], [like(xs, ys.numpy())], {"auto_pad": "same_upper", "strides": [2]},
        rtol=1e-5, atol=1e-5)

    xt = rand(1, 4, 3, 3)
    wt = rand(4, 3, 3, 3)
    yt = F.conv_transpose2d(t64(xt), t64(wt), None, stride=2, padding=1, output_padding=1)
    add("conv_transpose2d", "conv_transpose", [xt, wt], [like(xt, yt.numpy())],
        {"strides": [2, 2], "pads": [1, 1, 1, 1], "output_padding": [1, 1]}, rtol=1e-5, atol=1e-5)
    xtg = rand(1, 4, 4)
    wtg = rand(4, 1, 3)
    btg = rand(2)
    ytg = F.conv_transpose1d(t64(xtg), t64(wtg), t64(btg), stride=1, groups=2, dilation=2)
    add("conv_transpose1d_groups", "conv_transpose", [xtg, wtg, btg], [like(xtg, ytg.numpy())],
        {"groups": 2, "dilations": [2]}, rtol=1e-5, atol=1e-5)

    xp = rand(1, 2, 5, 6)
    yp = F.max_pool2d(t64(xp), 2, 2, 0, 1, ceil_mode=True)
    add("max_pool2d_ceil", "max_pool", [xp], [like(xp, yp.numpy())], {"kernel": [2, 2], "strides": [2, 2],
                                                                          "ceil_mode": True})
    yp2 = F.max_pool2d(t64(xp), 3, 1, 1, 2)
    add("max_pool2d_pad_dilation", "max_pool", [xp], [like(xp, yp2.numpy())],
        {"kernel": [3, 3], "pads": [1, 1, 1, 1], "dilations": [2, 2]})
    ya1 = F.avg_pool2d(t64(xp), 3, 2, 1, ceil_mode=False, count_include_pad=True)
    add("avg_pool2d_include_pad", "avg_pool", [xp], [like(xp, ya1.numpy())],
        {"kernel": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "count_include_pad": True})
    ya2 = F.avg_pool2d(t64(xp), 3, 2, 1, ceil_mode=True, count_include_pad=False)
    add("avg_pool2d_exclude_pad_ceil", "avg_pool", [xp], [like(xp, ya2.numpy())],
        {"kernel": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": True})
    ya3 = F.avg_pool2d(t64(xp), 2, 2, 1, ceil_mode=True, count_include_pad=True)
    add("avg_pool2d_include_pad_ceil", "avg_pool", [xp], [like(xp, ya3.numpy())],
        {"kernel": [2, 2], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": True, "count_include_pad": True})
    x1 = rand(2, 3, 10)
    add("max_pool1d", "max_pool", [x1], [like(x1, F.max_pool1d(t64(x1), 3, 2).numpy())],
        {"kernel": [3], "strides": [2]})


# --- Data movement ----------------------------------------------------------------------------------------------------

def gen_movement():
    x = rand(2, 3, 4)
    add("reshape", "reshape", [x], [x.reshape(4, 6)], {"shape": Dims([4, -1])})
    add("permute", "permute", [x], [np.transpose(x, (2, 0, 1))], {"perm": [2, 0, 1]})
    add("slice", "slice", [x], [x[:, 1:3, ::2]], {"axes": [1, 2], "starts": [1, 0], "ends": [3, 100],
                                                  "steps": [1, 2]})
    add("slice_negative", "slice", [x], [x[:, ::-1, -1:-4:-2]], {"axes": [1, 2], "starts": [-1, -1],
                                                                 "ends": [-100, -4], "steps": [-1, -2]})
    add("slice_empty", "slice", [x], [x[:, 2:1]], {"axes": [1], "starts": [2], "ends": [1]})
    e = rand(3, 1)
    add("expand", "expand", [e], [np.broadcast_to(e, (2, 3, 4)).copy()], {"shape": Dims([2, 1, 4])})
    c1 = rand(2, 3)
    c2 = rand(2, 1)
    c3 = rand(2, 2)
    add("concat", "concat", [c1, c2, c3], [np.concatenate([c1, c2, c3], axis=1)], {"axis": -1})
    add("split", "split", [x], [x[:, 0:1], x[:, 1:3]], {"axis": 1, "sizes": [1, 2]})
    p = rand(2, 4)
    add("pad_constant", "pad", [p], [np.pad(p, ((1, 0), (2, 1)), constant_values=1.5)],
        {"pads": [1, 2, 0, 1], "value": 1.5})
    add("pad_reflect", "pad", [p], [np.pad(p, ((1, 1), (3, 2)), mode="reflect")],
        {"pads": [1, 3, 1, 2], "mode": "reflect"})
    add("pad_edge", "pad", [p], [np.pad(p, ((0, 2), (1, 0)), mode="edge")], {"pads": [0, 1, 2, 0], "mode": "edge"})
    add("pad_crop", "pad", [p], [p[:, 1:3]], {"pads": [0, -1, 0, -1]})
    data = rand(5, 4)
    idx = np.array([[0, 4], [-1, 2]], dtype=np.int64)
    add("gather_axis0", "gather", [data, idx], [np.take(data, idx, axis=0)])
    idx32 = np.array([3, 0, 0], dtype=np.int32)
    add("gather_axis1_i32", "gather", [data, idx32], [np.take(data, idx32, axis=1)], {"axis": 1})
    sd = rand(3, 4)
    si = np.array([[1, 0, 2, 2], [0, 2, 1, 1]], dtype=np.int64)
    su = rand(2, 4)
    expected = torch.from_numpy(sd.copy()).scatter(0, torch.from_numpy(si), torch.from_numpy(su)).numpy()
    add("scatter", "scatter", [sd, si, su], [expected])
    sid = np.array([[1, 1, 0, 0]], dtype=np.int64)
    sud = rand(1, 4)
    for reduction, torch_reduce in (("add", "sum"), ("mul", "prod"), ("max", "amax"), ("min", "amin")):
        r = torch.from_numpy(f64(sd).copy()).scatter_reduce(0, torch.from_numpy(sid), t64(sud), torch_reduce)
        add(f"scatter_{reduction}", "scatter", [sd, sid, sud], [like(sd, r.numpy())], {"reduction": reduction})
    t = rand(2, 3)
    add("tile", "tile", [t], [np.tile(t, (2, 3))], {"repeats": [2, 3]})
    add("flip", "flip", [x], [np.flip(x, axis=(0, 2)).copy()], {"axes": [0, -1]})
    cs = rand(3, 4)
    add("cumsum", "cumsum", [cs], [like(cs, np.cumsum(f64(cs), axis=1))], {"axis": 1})
    excl = np.cumsum(f64(cs), axis=0) - f64(cs)
    add("cumsum_exclusive", "cumsum", [cs], [like(cs, excl)], {"axis": 0, "exclusive": True})
    rev = np.flip(np.cumsum(np.flip(f64(cs), 1), axis=1), 1)
    add("cumsum_reverse", "cumsum", [cs], [like(cs, rev)], {"axis": 1, "reverse": True})
    ci = randint(5)
    add("cumsum_i32", "cumsum", [ci], [np.cumsum(ci).astype(np.int32)], {"axis": 0})


# --- Creation ---------------------------------------------------------------------------------------------------------

def philox4x32_10(counter: int, key: int) -> list[int]:
    m0, m1, w0, w1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    mask = 0xFFFFFFFF
    c = [counter & mask, (counter >> 32) & mask, 0, 0]
    k = [key & mask, (key >> 32) & mask]
    for _ in range(10):
        p0 = m0 * c[0]
        p1 = m1 * c[2]
        c = [((p1 >> 32) ^ c[1] ^ k[0]) & mask, p1 & mask, ((p0 >> 32) ^ c[3] ^ k[1]) & mask, p0 & mask]
        k = [(k[0] + w0) & mask, (k[1] + w1) & mask]
    return c


def unit(word: int) -> float:
    return (word >> 8) * (1.0 / 16777216.0)


def gen_creation():
    add("iota", "iota", [], [np.broadcast_to(np.arange(4, dtype=np.float64) * 0.5 + 1, (3, 4)).astype(np.float32)],
        {"shape": Dims([3, 4]), "dtype": DType("f32"), "axis": 1, "start": 1.0, "step": 0.5})
    add("iota_i64", "iota", [], [np.broadcast_to(np.arange(3, dtype=np.int64)[:, None], (3, 2)).copy()],
        {"shape": Dims([3, 2]), "axis": 0})
    add("fill", "fill", [], [np.full((2, 3), 7, dtype=np.int32)], {"shape": Dims([2, 3]), "dtype": DType("i32"),
                                                                   "value": 7.9})
    n = 10
    seed = 1234
    uniform = [unit(philox4x32_10(i // 4, seed)[i % 4]) for i in range(n)]
    add("random_uniform", "random_uniform", [], [np.array([-1 + 3 * u for u in uniform], dtype=np.float32)],
        {"shape": Dims([n]), "low": -1.0, "high": 2.0, "seed": seed})
    normal = []
    for i in range(n):
        words = philox4x32_10(i // 2, seed)
        pair = 2 * (i % 2)
        u1 = 1.0 - unit(words[pair])
        u2 = unit(words[pair + 1])
        normal.append(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))
    add("random_normal", "random_normal", [], [np.array([0.5 + 2 * z for z in normal], dtype=np.float32)],
        {"shape": Dims([n]), "mean": 0.5, "stddev": 2.0, "seed": seed}, rtol=1e-6, atol=1e-6)


# --- Quantization -----------------------------------------------------------------------------------------------------

def gen_quantization():
    x = rand(2, 3, 4, low=-5, high=5)
    scale = np.array(0.05, dtype=np.float32)
    zp = np.array(3, dtype=np.int8)
    q = np.clip(np.round(f64(x) / f64(scale)) + 3, -128, 127).astype(np.int8)
    add("quantize_per_tensor", "quantize", [x, scale, zp], [q])
    scales = np.array([0.02, 0.05, 0.1], dtype=np.float32)
    zps = np.array([0, 128, 255], dtype=np.uint8)
    qa = np.clip(np.round(f64(x) / f64(scales)[None, :, None]) + zps.astype(np.float64)[None, :, None], 0, 255)
    add("quantize_per_axis_u8", "quantize", [x, scales, zps], [qa.astype(np.uint8)], {"dtype": DType("u8")})
    qi = randint(2, 3, low=-128, high=127, dtype=np.int8)
    dz = np.array([1, -2, 0], dtype=np.int8)
    ds = np.array([0.5, 0.25, 2.0], dtype=np.float32)
    add("dequantize_per_axis", "dequantize", [qi, ds, dz],
        [((qi.astype(np.float64) - dz[None, :]) * f64(ds)[None, :]).astype(np.float32)])
    add("dequantize_no_zero_point", "dequantize", [qi, np.array(0.125, dtype=np.float32)],
        [(qi.astype(np.float64) * 0.125).astype(np.float32)])


# --- Composites -------------------------------------------------------------------------------------------------------

def gen_composites():
    x = rand(3, 7)
    add("gelu", "gelu", [x], [like(x, F.gelu(t64(x)).numpy())])
    add("gelu_tanh", "gelu", [x], [like(x, F.gelu(t64(x), approximate="tanh").numpy())], {"approximate": "tanh"})
    add("silu", "silu", [x], [like(x, F.silu(t64(x)).numpy())])
    add("softmax", "softmax", [x], [like(x, F.softmax(t64(x), dim=-1).numpy())])
    add("softmax_axis0", "softmax", [x], [like(x, F.softmax(t64(x), dim=0).numpy())], {"axis": 0})
    add("log_softmax", "log_softmax", [x], [like(x, F.log_softmax(t64(x), dim=1).numpy())], {"axis": 1})
    big = np.array([[1000, 1001, 999]], dtype=np.float32)
    add("softmax_large", "softmax", [big], [like(big, F.softmax(t64(big), dim=-1).numpy())])
    h = rand(2, 9, dtype=np.float16)
    add("softmax_f16", "softmax", [h], [like(h, F.softmax(t64(h), dim=-1).numpy())], rtol=2e-3, atol=2e-3)

    ln = rand(2, 3, 4)
    g = rand(3, 4)
    b = rand(3, 4)
    y = F.layer_norm(t64(ln), (3, 4), t64(g), t64(b), eps=1e-5)
    add("layer_norm", "layer_norm", [ln, g, b], [like(ln, y.numpy())], {"axis": 1}, rtol=1e-5, atol=1e-5)
    g1 = rand(4)
    y1 = F.layer_norm(t64(ln), (4,), t64(g1), None, eps=1e-3)
    add("layer_norm_scale_only", "layer_norm", [ln, g1], [like(ln, y1.numpy())], {"epsilon": 1e-3}, rtol=1e-5,
        atol=1e-5, absent=(2,))
    rn = rand(3, 8)
    rs = rand(8)
    rms = f64(rn) / np.sqrt(np.mean(f64(rn) ** 2, axis=-1, keepdims=True) + 1e-6) * f64(rs)
    add("rms_norm", "rms_norm", [rn, rs], [like(rn, rms)], rtol=1e-5, atol=1e-5)

    lx = rand(2, 3, 4)
    lw = rand(5, 4)
    lb = rand(5)
    add("linear", "linear", [lx, lw, lb], [like(lx, F.linear(t64(lx), t64(lw), t64(lb)).numpy())], rtol=1e-5,
        atol=1e-5)
    lw2 = rand(4, 5)
    add("linear_no_transpose", "linear", [lx, lw2], [like(lx, (f64(lx) @ f64(lw2)))], {"transpose_w": False},
        rtol=1e-5, atol=1e-5)

    def sdpa_reference(q, k, v, mask=None, causal=False, scale=None):
        q, k, v = t64(q), t64(k), t64(v)
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        sq, skv = q.shape[2], k.shape[2]
        bias = torch.zeros(sq, skv, dtype=torch.float64)
        if causal:
            allowed = torch.arange(skv)[None, :] <= torch.arange(sq)[:, None] + (skv - sq)
            bias = bias.masked_fill(~allowed, float("-inf"))
        if mask is not None:
            m = torch.from_numpy(np.asarray(mask))
            bias = bias + (torch.where(m, 0.0, float("-inf")) if m.dtype == torch.bool else m.to(torch.float64))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale).numpy()

    q = rand(1, 2, 3, 4)
    k = rand(1, 2, 5, 4)
    v = rand(1, 2, 5, 3)
    add("sdpa", "sdpa", [q, k, v], [like(q, sdpa_reference(q, k, v))], rtol=1e-5, atol=1e-5)
    qg = rand(2, 4, 3, 8)
    kg = rand(2, 2, 5, 8)
    vg = rand(2, 2, 5, 8)
    add("sdpa_gqa_causal", "sdpa", [qg, kg, vg], [like(qg, sdpa_reference(qg, kg, vg, causal=True, scale=0.3))],
        {"causal": True, "scale": 0.3}, rtol=1e-5, atol=1e-5)
    mask = RNG.integers(0, 2, size=(1, 1, 3, 5)).astype(np.bool_)
    mask[..., 0] = True  # no fully masked rows
    add("sdpa_bool_mask", "sdpa", [q, k, v, mask], [like(q, sdpa_reference(q, k, v, mask=mask))], rtol=1e-5,
        atol=1e-5)
    additive = rand(3, 5)
    add("sdpa_additive_mask", "sdpa", [q, k, v, additive], [like(q, sdpa_reference(q, k, v, mask=additive))],
        rtol=1e-5, atol=1e-5)


# --- Image and selection (stage 3) ----------------------------------------------------------------------------------

def gen_image_selection():
    x = rand(1, 2, 4, 5)
    t = t64(x)

    def resize_case(name, out_hw, torch_mode, attrs, align_corners=None, rtol=1e-6, atol=1e-6):
        kwargs = {"size": out_hw, "mode": torch_mode}
        if align_corners is not None:
            kwargs["align_corners"] = align_corners
        y = F.interpolate(t, **kwargs).numpy()
        attrs = dict(attrs)
        attrs["shape"] = [1, 2, out_hw[0], out_hw[1]]
        add(name, "resize", [x], [like(x, y)], attrs, rtol=rtol, atol=atol)

    # PyTorch "nearest" picks floor(o * in / out): asymmetric coordinates rounded down.
    resize_case("resize_nearest_up", [8, 10], "nearest", {"coordinate_mode": "asymmetric", "nearest_mode": "floor"})
    resize_case("resize_nearest_down", [3, 2], "nearest", {"coordinate_mode": "asymmetric", "nearest_mode": "floor"})
    # "nearest-exact" rounds half-pixel coordinates, ties up.
    resize_case("resize_nearest_exact", [7, 9], "nearest-exact",
                {"coordinate_mode": "half_pixel", "nearest_mode": "round_prefer_ceil"})
    resize_case("resize_linear_half_pixel", [7, 9], "bilinear", {"mode": "linear"}, align_corners=False,
                rtol=1e-5, atol=1e-6)
    resize_case("resize_linear_align_corners", [6, 3], "bilinear",
                {"mode": "linear", "coordinate_mode": "align_corners"}, align_corners=True, rtol=1e-5, atol=1e-6)
    resize_case("resize_cubic", [6, 8], "bicubic", {"mode": "cubic"}, align_corners=False, rtol=1e-5, atol=1e-5)
    xi = randint(2, 3, 4, dtype=np.int32)
    yi = F.interpolate(torch.from_numpy(xi.astype(np.float64)), size=[8], mode="nearest").numpy().astype(np.int32)
    add("resize_nearest_int_1d", "resize", [xi], [yi],
        {"shape": [2, 3, 8], "coordinate_mode": "asymmetric", "nearest_mode": "floor"})

    # topk: distinct values, so that the order is unique.
    v = RNG.permutation(30).astype(np.float32).reshape(3, 10) / 4 - 3
    values, indices = torch.topk(torch.from_numpy(v), 4, dim=-1, largest=True, sorted=True)
    add("topk_largest", "topk", [v], [values.numpy(), indices.numpy().astype(np.int64)], {"k": 4})
    values, indices = torch.topk(torch.from_numpy(v), 3, dim=0, largest=False, sorted=True)
    add("topk_smallest_axis0", "topk", [v], [values.numpy(), indices.numpy().astype(np.int64)],
        {"k": 3, "axis": 0, "largest": False})
    vi = RNG.permutation(12).astype(np.int64).reshape(2, 6)
    values, indices = torch.topk(torch.from_numpy(vi), 6, dim=1)
    add("topk_i64_all", "topk", [vi], [values.numpy(), indices.numpy().astype(np.int64)], {"k": 6})


# --- LLM operations (stage 4): block-quantized weights, rope, slice_update, sdpa with a KV cache -----------------

# ii dtype -> (gguf type, block elements, block bytes, offsets of f16 scale fields)
BLOCK_TYPES = {
    "q4_0": ("Q4_0", 32, 18, [0]),
    "q8_0": ("Q8_0", 32, 34, [0]),
    "q4_k": ("Q4_K", 256, 144, [0, 2]),
    "q5_k": ("Q5_K", 256, 176, [0, 2]),
    "q6_k": ("Q6_K", 256, 210, [208]),
}


def random_blocks(dtype: str, rows: int, cols: int) -> tuple[QBlocks, np.ndarray]:
    """Random blocks with finite, small scales and their dequantization (float32, [rows, cols])."""
    from gguf import GGMLQuantizationType, quants

    name, elements, nbytes, f16_offsets = BLOCK_TYPES[dtype]
    blocks = rows * cols // elements
    raw = RNG.integers(0, 256, size=(blocks, nbytes), dtype=np.uint8)
    for offset in f16_offsets:
        scales = RNG.uniform(-0.05, 0.05, size=blocks).astype(np.float16)
        raw[:, offset : offset + 2] = scales.view(np.uint8).reshape(blocks, 2)
    values = quants.dequantize(raw.reshape(-1), GGMLQuantizationType[name]).astype(np.float32).reshape(rows, cols)
    return QBlocks(dtype, (rows, cols), raw), values


def rope_reference(x, cos, sin, interleaved):
    x = f64(x)
    half = cos.shape[-1]
    cos = np.broadcast_to(f64(cos), x.shape[:-1] + (half,))
    sin = np.broadcast_to(f64(sin), x.shape[:-1] + (half,))
    y = x.copy()
    if interleaved:
        a, b = x[..., 0 : 2 * half : 2], x[..., 1 : 2 * half : 2]
        y[..., 0 : 2 * half : 2] = a * cos - b * sin
        y[..., 1 : 2 * half : 2] = b * cos + a * sin
    else:
        a, b = x[..., :half], x[..., half : 2 * half]
        y[..., :half] = a * cos - b * sin
        y[..., half : 2 * half] = b * cos + a * sin
    return y


def gen_llm():
    for dtype in ("q4_0", "q8_0", "q4_k", "q6_k"):
        cols = 256 if dtype.endswith("_k") else 64
        blocks, values = random_blocks(dtype, 3, cols)
        add(f"dequantize_{dtype}", "dequantize", [blocks], [values])
    for dtype in ("q4_k", "q8_0"):
        cols = 256 if dtype.endswith("_k") else 64
        w, wv = random_blocks(dtype, 5, cols)
        x = rand(2, 3, cols)
        add(f"matmul_{dtype}", "matmul", [x, w], [like(x, f64(x) @ f64(wv).T)], {"transpose_b": True}, rtol=1e-5,
            atol=1e-5)
        bias = rand(5)
        add(f"linear_{dtype}", "linear", [x, w, bias], [like(x, f64(x) @ f64(wv).T + f64(bias))], rtol=1e-5,
            atol=1e-5)
    xh = rand(4, 256, dtype=np.float16)
    w, wv = random_blocks("q5_k", 3, 256)
    add("matmul_f16_q5_k", "matmul", [xh, w], [like(xh, f64(xh) @ f64(wv).T)], {"transpose_b": True}, rtol=2e-3,
        atol=2e-3)
    table, table_values = random_blocks("q6_k", 6, 256)
    ids = np.array([[5, 0], [2, 5]], dtype=np.int32)
    add("gather_q6_k_rows", "gather", [table, ids], [QBlocks("q6_k", (2, 2, 256), table.raw.reshape(6, -1)[ids])])

    # rope: x [B, S, H, D] with cos / sin [S, 1, R]
    x = rand(2, 3, 2, 8)
    angles = f64(rand(3, 1, 4, low=0.0, high=6.0))
    cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    add("rope", "rope", [x, cos, sin], [like(x, rope_reference(x, cos, sin, False))], rtol=1e-5, atol=1e-5)
    add("rope_interleaved", "rope", [x, cos, sin], [like(x, rope_reference(x, cos, sin, True))],
        {"interleaved": True}, rtol=1e-5, atol=1e-5)
    partial_cos, partial_sin = cos[..., :2].copy(), sin[..., :2].copy()
    add("rope_partial", "rope", [x, partial_cos, partial_sin],
        [like(x, rope_reference(x, partial_cos, partial_sin, False))], rtol=1e-5, atol=1e-5)

    # slice_update: a KV cache [B, H, C, D] receives [B, H, S, D] at a position
    cache = rand(1, 2, 6, 3)
    update = rand(1, 2, 2, 3)
    expected = cache.copy()
    expected[:, :, 3:5] = update
    add("slice_update", "slice_update", [cache, update, np.array(3, dtype=np.int64)], [expected], {"axis": 2})
    expected0 = cache.copy()
    expected0[:, :, 0:2] = update
    add("slice_update_i32_start", "slice_update", [cache, update, np.array(0, dtype=np.int32)], [expected0],
        {"axis": -2})

    # sdpa over a preallocated cache: only kv_len keys are used; causal positions follow kv_len.
    def cache_reference(q, k, v, kv_len, causal, window=0, scale=None):
        qt, kt, vt = t64(q), t64(k)[:, :, :kv_len], t64(v)[:, :, :kv_len]
        groups = qt.shape[1] // kt.shape[1]
        kt = kt.repeat_interleave(groups, dim=1)
        vt = vt.repeat_interleave(groups, dim=1)
        sq = qt.shape[2]
        position = torch.arange(sq)[:, None] + (kv_len - sq)
        j = torch.arange(kv_len)[None, :]
        allowed = torch.ones(sq, kv_len, dtype=torch.bool)
        if causal:
            allowed &= j <= position
        if window:
            allowed &= j > position - window
        bias = torch.zeros(sq, kv_len, dtype=torch.float64).masked_fill(~allowed, float("-inf"))
        return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=bias, scale=scale).numpy()

    q = rand(1, 4, 2, 8)
    k = rand(1, 2, 7, 8)
    v = rand(1, 2, 7, 8)
    kv_len = np.array(5, dtype=np.int64)
    add("sdpa_kv_len_causal", "sdpa", [q, k, v, kv_len], [like(q, cache_reference(q, k, v, 5, True))],
        {"causal": True}, rtol=1e-5, atol=1e-5, absent=(3,))
    add("sdpa_kv_len", "sdpa", [q, k, v, np.array(4, dtype=np.int32)], [like(q, cache_reference(q, k, v, 4, False))],
        rtol=1e-5, atol=1e-5, absent=(3,))
    qw = rand(1, 4, 4, 8)
    add("sdpa_window", "sdpa", [qw, k, v, kv_len], [like(qw, cache_reference(qw, k, v, 5, True, window=2))],
        {"causal": True, "window": 2}, rtol=1e-5, atol=1e-5, absent=(3,))

    def softcap_reference(q, k, v, cap):
        qt, kt, vt = t64(q), t64(k), t64(v)
        scores = qt @ kt.transpose(-1, -2) / math.sqrt(qt.shape[-1])
        scores = cap * torch.tanh(scores / cap)
        sq, skv = scores.shape[-2:]
        allowed = torch.arange(skv)[None, :] <= torch.arange(sq)[:, None] + (skv - sq)
        scores = scores.masked_fill(~allowed, float("-inf"))
        return (torch.softmax(scores, dim=-1) @ vt).numpy()

    qs = rand(1, 2, 3, 4, low=-4, high=4)
    ks = rand(1, 2, 5, 4, low=-4, high=4)
    vs = rand(1, 2, 5, 4)
    add("sdpa_softcap", "sdpa", [qs, ks, vs], [like(qs, softcap_reference(qs, ks, vs, 2.0))],
        {"causal": True, "softcap": 2.0}, rtol=1e-5, atol=1e-5)

    # 16-bit weights with f32 activations
    xa = rand(3, 2, 16)
    wh = rand(16, 5, dtype=np.float16)
    add("matmul_f32_f16_weights", "matmul", [xa, wh], [like(xa, f64(xa) @ f64(wh))], rtol=1e-5, atol=1e-5)
    wb = BF16(rand(7, 16))
    bias = rand(7)
    add("linear_f32_bf16_weights", "linear", [xa, wb, bias], [like(xa, f64(xa) @ f64(wb).T + f64(bias))], rtol=1e-5,
        atol=1e-5)

    # A 16-bit KV cache with f32 queries
    kh = rand(1, 2, 7, 8, dtype=np.float16)
    vh = rand(1, 2, 7, 8, dtype=np.float16)
    add("sdpa_f16_cache", "sdpa", [q, kh, vh, kv_len], [like(q, cache_reference(q, kh, vh, 5, True))],
        {"causal": True}, rtol=1e-5, atol=1e-5, absent=(3,))


def generate() -> str:
    CASES.clear()
    gen_unary()
    gen_binary()
    gen_other_elementwise()
    gen_reductions()
    gen_linalg()
    gen_movement()
    gen_creation()
    gen_quantization()
    gen_composites()
    gen_image_selection()
    gen_llm()
    names = [c["name"] for c in CASES]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise SystemExit(f"duplicate case names: {sorted(duplicates)}")
    document = {
        "generator": "scripts/gen_op_references.py",
        "numpy": np.__version__,
        "torch": torch.__version__.split("+")[0],
        "cases": CASES,
    }
    return json.dumps(document, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the committed file is out of date")
    args = parser.parse_args()
    text = generate()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        # Library versions may differ between machines; compare the cases only.
        if json.loads(current or "{}").get("cases") != json.loads(text)["cases"]:
            print(f"{OUTPUT.relative_to(ROOT)} is out of date; run scripts/gen_op_references.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(CASES)} cases to {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
