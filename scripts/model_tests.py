# SPDX-License-Identifier: MIT
"""Runs the model tests: every model of tests/models/manifest.json (fetched into the cache by fetch_models.py) is
run with ONNX Runtime and with ii on the same inputs (scripts/compare_onnxruntime.py); detection models also run
examples/yolo on a test image and must find the expected classes.

    python scripts/fetch_models.py --cache CACHE --suite pr
    python scripts/model_tests.py --cache CACHE --ii build/<preset>/bin/Release/ii \
        [--yolo build/<preset>/bin/Release/ii_example_yolo] [--suite pr|nightly]
"""

import argparse
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fetch_models  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=os.environ.get("II_MODEL_CACHE", "build/models-cache"))
    parser.add_argument("--ii", required=True, help="the ii executable")
    parser.add_argument("--yolo", help="the ii_example_yolo executable (detection models)")
    parser.add_argument("--suite", choices=["pr", "nightly"], default="pr")
    parser.add_argument("--only", help="run one model by name")
    args = parser.parse_args()
    cache = pathlib.Path(args.cache).expanduser().resolve()
    compare = pathlib.Path(__file__).resolve().parent / "compare_onnxruntime.py"

    failures = []
    for model in fetch_models.models(args.suite, args.only):
        path = cache / f"{model['name']}.onnx"
        if not path.exists():
            failures.append(f"{model['name']}: not fetched ({path})")
            continue
        command = [sys.executable, str(compare), str(path), "--ii", args.ii]
        if model.get("dims"):
            command += ["--dims", model["dims"]]
        if "int_high" in model:
            command += ["--int-high", str(model["int_high"])]
        result = subprocess.run(command)
        if result.returncode != 0:
            failures.append(f"{model['name']}: outputs differ from ONNX Runtime")
        detect = model.get("detect")
        if detect and args.yolo:
            image = cache / pathlib.Path(detect["image_url"]).name
            run = subprocess.run([os.path.abspath(args.yolo), str(path), str(image)], capture_output=True, text=True)
            print(run.stdout, end="")
            found = {line.split()[0] for line in run.stdout.splitlines()[1:] if line.strip()}
            missing = [name for name in detect["expect"] if name not in found]
            if run.returncode != 0 or missing:
                failures.append(f"{model['name']}: detection failed (missing {missing}) {run.stderr.strip()}")
    for failure in failures:
        print(f"FAILED {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
