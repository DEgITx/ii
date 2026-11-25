#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Checks (or fixes) C++ formatting of all tracked sources with clang-format.

Usage:
    python scripts/check_format.py            # report files that need formatting, exit 1 if any
    python scripts/check_format.py --fix      # reformat files in place
    python scripts/check_format.py --clang-format clang-format-19

CI pins the clang-format version (see .github/workflows/ci.yml) because formatting differs between versions.
"""

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTENSIONS = {".cpp", ".hpp", ".h", ".cc", ".c", ".in"}
DIRECTORIES = ("include", "src", "tools", "tests", "bench", "examples")


def tracked_sources() -> list[pathlib.Path]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", *DIRECTORIES],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    files = []
    for line in output.splitlines():
        path = ROOT / line
        if path.suffix in EXTENSIONS and path.name != "version.h.in" and path.exists():
            files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="reformat files in place")
    parser.add_argument("--clang-format", default="clang-format", help="clang-format executable")
    args = parser.parse_args()

    files = tracked_sources()
    if args.fix:
        subprocess.run([args.clang_format, "-i", *map(str, files)], check=True)
        print(f"formatted {len(files)} files")
        return 0

    unformatted = []
    for path in files:
        result = subprocess.run([args.clang_format, "--dry-run", "--Werror", str(path)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            unformatted.append(path.relative_to(ROOT))
    for path in unformatted:
        print(f"needs formatting: {path}")
    if unformatted:
        print(f"\n{len(unformatted)} file(s) need formatting; run: python scripts/check_format.py --fix")
        return 1
    print(f"all {len(files)} files are formatted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
