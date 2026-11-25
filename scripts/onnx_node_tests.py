# SPDX-License-Identifier: MIT
"""Generates the ONNX operator conformance cases (from the `onnx` Python package) for `ii_onnx_tests`.

Every case becomes a directory in the layout of the ONNX backend test data:

    <out>/<test name>/model.onnx
    <out>/<test name>/test_data_set_0/input_0.pb, ..., output_0.pb, ...

Usage:
    pip install onnx
    python scripts/onnx_node_tests.py --out build/onnx-node-tests
    build/.../ii_onnx_tests build/onnx-node-tests
"""

import argparse
import os
import shutil
import sys
import warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory (replaced)")
    parser.add_argument("--filter", default="", help="only cases whose name contains this text")
    args = parser.parse_args()

    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
        from onnx.backend.test.case import node
    except ImportError as error:
        print(f"error: {error}; install the 'onnx' package", file=sys.stderr)
        return 1

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    cases = node.collect_testcases(None)
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    written = 0
    for case in cases:
        if args.filter and args.filter not in case.name:
            continue
        if case.model is None or not case.data_sets:
            continue
        directory = os.path.join(args.out, case.name)
        os.makedirs(directory)
        onnx.save(case.model, os.path.join(directory, "model.onnx"))
        for index, (inputs, outputs) in enumerate(case.data_sets):
            data_dir = os.path.join(directory, f"test_data_set_{index}")
            os.makedirs(data_dir)
            for kind, values, infos in (("input", inputs, case.model.graph.input),
                                        ("output", outputs, case.model.graph.output)):
                for i, value in enumerate(values):
                    if not isinstance(value, np.ndarray) or value.dtype.kind in "OUSc":
                        continue  # sequences, optionals, strings, complex: not tensors ii can read
                    name = infos[i].name if i < len(infos) else ""
                    try:
                        tensor = numpy_helper.from_array(value, name)
                    except Exception:  # noqa: BLE001 - exotic dtypes (float8, int4, ...)
                        continue
                    with open(os.path.join(data_dir, f"{kind}_{i}.pb"), "wb") as f:
                        f.write(tensor.SerializeToString())
        written += 1
    print(f"wrote {written} cases (onnx {onnx.__version__}, opset {onnx.defs.onnx_opset_version()}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
