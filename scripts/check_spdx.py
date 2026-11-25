#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Checks that every tracked source file carries an SPDX license identifier in its first lines."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTENSIONS = {".cpp", ".hpp", ".h", ".cc", ".c", ".py", ".cmake", ".in"}
NAMES = {"CMakeLists.txt"}
EXEMPT = {"cmake/ii.pc.in"}  # pkg-config files cannot contain comments before the variables reliably
HEADER = "SPDX-License-Identifier: MIT"


def main() -> int:
    output = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    missing = []
    for line in output.splitlines():
        path = ROOT / line
        if line in EXEMPT or not path.exists():
            continue
        if path.suffix not in EXTENSIONS and path.name not in NAMES:
            continue
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
        if not any(HEADER in text for text in head):
            missing.append(line)
    for line in missing:
        print(f"missing '{HEADER}': {line}")
    if missing:
        return 1
    print("all source files have SPDX headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
