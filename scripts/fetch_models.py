# SPDX-License-Identifier: MIT
"""Downloads the models listed in tests/models/manifest.json into a cache directory and verifies their SHA-256.

Models given as a `source` with `"export": "ultralytics"` are PyTorch checkpoints exported to ONNX with the
`ultralytics` package (pip install ultralytics). Language models ("llms") are a GGUF file or a Hugging Face directory
(a list of files); the test text for perplexity is fetched with them. Already present and verified files are not
downloaded again.

    python scripts/fetch_models.py --cache ~/.cache/ii-models [--suite pr|nightly] [--only name]
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "models" / "manifest.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, expected_sha: str, target: pathlib.Path) -> None:
    if target.exists() and sha256(target) == expected_sha:
        return
    print(f"downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ii-model-tests"})
    with urllib.request.urlopen(request, timeout=600) as response, tempfile.NamedTemporaryFile(
            dir=target.parent, delete=False) as out:
        shutil.copyfileobj(response, out)
        temporary = pathlib.Path(out.name)
    actual = sha256(temporary)
    if actual != expected_sha:
        temporary.unlink()
        raise SystemExit(f"{url}: SHA-256 {actual} does not match the manifest ({expected_sha})")
    temporary.replace(target)


def export_ultralytics(checkpoint: pathlib.Path, target: pathlib.Path) -> None:
    if target.exists():
        return
    print(f"exporting {checkpoint.name} to ONNX with ultralytics", flush=True)
    code = (
        "import sys; from ultralytics import YOLO; "
        "YOLO(sys.argv[1]).export(format='onnx', opset=17, imgsz=640, dynamic=False, simplify=False)"
    )
    subprocess.run([sys.executable, "-c", code, str(checkpoint)], check=True, cwd=checkpoint.parent)
    exported = checkpoint.with_suffix(".onnx")
    if exported != target:
        exported.replace(target)


def models(suite: str, only: str | None, section: str = "models"):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    order = {"pr": 0, "nightly": 1}
    for model in manifest.get(section, []):
        if order[model["suite"]] > order[suite]:
            continue
        if only and model["name"] != only:
            continue
        yield model


def fetch(model: dict, cache: pathlib.Path) -> pathlib.Path:
    target = cache / f"{model['name']}.onnx"
    if "url" in model:
        download(model["url"], model["sha256"], target)
    else:
        source = model["source"]
        checkpoint = cache / pathlib.Path(source["url"]).name
        download(source["url"], source["sha256"], checkpoint)
        if model.get("export") != "ultralytics":
            raise SystemExit(f"{model['name']}: unknown export recipe")
        export_ultralytics(checkpoint, target)
    detect = model.get("detect")
    if detect:
        image = cache / pathlib.Path(detect["image_url"]).name
        if not image.exists():
            request = urllib.request.Request(detect["image_url"], headers={"User-Agent": "ii-model-tests"})
            with urllib.request.urlopen(request, timeout=300) as response:
                image.write_bytes(response.read())
    return target


def fetch_llm(model: dict, cache: pathlib.Path) -> pathlib.Path:
    """A GGUF file (cache/<name>.gguf) or a Hugging Face directory (cache/<name>/)."""
    if "files" in model:
        directory = cache / model["name"]
        directory.mkdir(parents=True, exist_ok=True)
        for file in model["files"]:
            download(file["url"], file["sha256"], directory / file["path"])
        return directory
    target = cache / f"{model['name']}.gguf"
    download(model["url"], model["sha256"], target)
    return target


def fetch_text(cache: pathlib.Path) -> pathlib.Path:
    """The perplexity test text (wikitext-2 test split), extracted from its archive."""
    import zipfile

    text = json.loads(MANIFEST.read_text(encoding="utf-8"))["text"]
    archive = cache / pathlib.Path(text["url"]).name
    download(text["url"], text["sha256"], archive)
    target = cache / pathlib.Path(text["member"]).name
    if not target.exists():
        with zipfile.ZipFile(archive) as z:
            target.write_bytes(z.read(text["member"]))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=os.environ.get("II_MODEL_CACHE", "build/models-cache"))
    parser.add_argument("--suite", choices=["pr", "nightly"], default="pr")
    parser.add_argument("--only", help="fetch one model by name")
    args = parser.parse_args()
    cache = pathlib.Path(args.cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for model in models(args.suite, args.only):
        path = fetch(model, cache)
        print(f"{model['name']}: {path}")
    llms = list(models(args.suite, args.only, "llms"))
    for model in llms:
        print(f"{model['name']}: {fetch_llm(model, cache)}")
    if any("perplexity_chunks" in model for model in llms):
        print(f"text: {fetch_text(cache)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
