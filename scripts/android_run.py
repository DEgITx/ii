#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Runs a cross-compiled Android executable on a device or emulator through adb.

CMake uses this script as CMAKE_CROSSCOMPILING_EMULATOR for Android builds, so `ctest` runs tests, examples and fuzz
replays on the device exactly as it runs them on the host:

    python scripts/android_run.py <executable> [args...]

* The executable (and shared libraries next to it) is pushed to /data/local/tmp/ii/ — only when changed.
* Arguments and II_* environment variables that name existing host files or directories are pushed as well and
  rewritten to their device paths (test data, fuzz corpora, plugins).
* The exit code of the program on the device is returned.

Environment: ANDROID_SERIAL selects the device when several are connected (standard adb behaviour);
II_ANDROID_DIR overrides the device directory.
"""

import hashlib
import os
import pathlib
import posixpath
import shlex
import subprocess
import sys

DEVICE_ROOT = os.environ.get("II_ANDROID_DIR", "/data/local/tmp/ii")
CACHE = {}


def adb(*args, check=True, capture=False):
    command = ["adb", *args]
    result = subprocess.run(command, check=False, capture_output=capture, text=capture)
    if check and result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        sys.exit(f"android_run: '{' '.join(command)}' failed with exit code {result.returncode} {detail}")
    return result


def digest(path: pathlib.Path) -> str:
    if path.is_file():
        stat = path.stat()
        return f"{stat.st_size}-{stat.st_mtime_ns}"
    h = hashlib.sha1()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            h.update(f"{item.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns};".encode())
    return h.hexdigest()


def push(host: pathlib.Path) -> str:
    """Pushes a file or directory (if it changed since the last push) and returns its device path."""
    host = host.resolve()
    key = hashlib.sha1(str(host).encode()).hexdigest()[:12]
    device = posixpath.join(DEVICE_ROOT, "files", key, host.name)
    if device in CACHE:
        return device
    stamp = digest(host)
    stamp_file = device + ".stamp"
    current = adb("shell", f"cat {shlex.quote(stamp_file)} 2>/dev/null", check=False, capture=True).stdout.strip()
    if current != stamp:
        adb("shell", f"rm -rf {shlex.quote(device)} && mkdir -p {shlex.quote(posixpath.dirname(device))}")
        adb("push", str(host), device, capture=True)
        if host.is_file():
            adb("shell", f"chmod 755 {shlex.quote(device)}")
        adb("shell", f"echo {shlex.quote(stamp)} > {shlex.quote(stamp_file)}")
    CACHE[device] = device
    return device


def map_argument(value: str) -> str:
    if value and not value.startswith("-"):
        path = pathlib.Path(value)
        if path.is_absolute() and path.exists():
            return push(path)
    return value


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    executable = pathlib.Path(sys.argv[1])
    device_exe = push(executable)
    library_dirs = []
    for library in sorted(executable.resolve().parent.glob("*.so")):
        library_dirs.append(posixpath.dirname(push(library)))
    args = [map_argument(arg) for arg in sys.argv[2:]]

    env = []
    for name, value in sorted(os.environ.items()):
        if name.startswith("II_") and name != "II_ANDROID_DIR":
            env.append(f"{name}={shlex.quote(map_argument(value))}")
    if library_dirs:
        env.append("LD_LIBRARY_PATH=" + ":".join(sorted(set(library_dirs))))
    for name in ("GTEST_FILTER", "GTEST_COLOR", "GTEST_SHUFFLE", "GTEST_REPEAT"):
        if name in os.environ:
            env.append(f"{name}={shlex.quote(os.environ[name])}")

    work_dir = posixpath.dirname(device_exe)
    command = f"cd {shlex.quote(work_dir)} && " + " ".join(env + [shlex.quote(device_exe)] + [shlex.quote(a) for a in args])
    # `adb shell` returns the exit status of the remote command (adb >= 1.0.36 with shell protocol v2).
    result = subprocess.run(["adb", "shell", command], check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
