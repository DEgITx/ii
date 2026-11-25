#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Checks the language models of tests/models/manifest.json ("llms") with the ii CLI.

Per model, as listed in the manifest:
  chat               ii generate --chat must answer the prompt with the expected word (greedy);
  logits_tolerance   logits of every position of a fixed text against transformers (Hugging Face directories);
  greedy_tokens      greedy generation must equal transformers' token for token;
  perplexity_chunks  perplexity on wikitext-2 (chunks of 512 tokens) against transformers: within perplexity_tolerance
                     (relative) for the model itself, or at most perplexity_max_increase above its float `reference`.

    python scripts/fetch_models.py --cache ~/.cache/ii-models --suite nightly
    python scripts/llm_tests.py --cache ~/.cache/ii-models --suite nightly --ii build/<preset>/bin/Release/ii

The transformers checks need torch and transformers; reference perplexities are cached in the model cache.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "models" / "manifest.json"
COMPARE = ROOT / "scripts" / "compare_transformers.py"
TEXT = ("The Eiffel Tower is located in Paris, the capital of France. It was built in 1889 for the World's Fair and "
        "was at the time the tallest structure in the world.")


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def model_path(cache: pathlib.Path, model: dict) -> pathlib.Path:
    return cache / model["name"] if "files" in model else cache / f"{model['name']}.gguf"


def perplexity_of(output: str) -> float:
    match = re.search(r"PPL = ([0-9.]+)", output)
    if match is None:
        raise RuntimeError(f"no perplexity in the output:\n{output}")
    return float(match.group(1))


def reference_perplexity(cache: pathlib.Path, reference: pathlib.Path, text: pathlib.Path, chunks: int) -> float:
    """transformers' perplexity of the reference model, cached next to the models."""
    store = cache / "reference_perplexities.json"
    known = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    key = f"{reference.name}/{chunks}"
    if key not in known:
        output = run([sys.executable, str(COMPARE), "--perplexity", str(text), "--reference", str(reference), "--ctx",
                      "512", "--chunks", str(chunks)])
        known[key] = perplexity_of(output)
        store.write_text(json.dumps(known, indent=1), encoding="utf-8")
    return known[key]


def check(model: dict, models: dict, cache: pathlib.Path, ii: str) -> list[str]:
    failures = []
    path = model_path(cache, model)
    name = model["name"]
    if "chat" in model:
        output = run([ii, "generate", str(path), "--chat", "-p", model["chat"]["prompt"], "-n", "32", "--temp", "0"])
        ok = model["chat"]["expect"].lower() in output.lower()
        print(f"{name}: chat {'ok' if ok else 'FAILED'}")
        if not ok:
            failures.append(f"{name}: the answer does not contain '{model['chat']['expect']}':\n{output}")
    with tempfile.TemporaryDirectory() as tmp:
        if "logits_tolerance" in model:
            logits = pathlib.Path(tmp) / "logits.npz"
            run([ii, "generate", str(path), "-p", TEXT, "--save-logits", str(logits)])
            result = subprocess.run([sys.executable, str(COMPARE), str(logits), "--reference", str(path), "--text", TEXT,
                                     "--tolerance", str(model["logits_tolerance"])], capture_output=True, text=True)
            print(f"{name}: logits: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''}")
            if result.returncode != 0:
                failures.append(f"{name}: logits differ from transformers:\n{result.stdout}\n{result.stderr}")
        if "greedy_tokens" in model:
            tokens = pathlib.Path(tmp) / "tokens.npz"
            run([ii, "generate", str(path), "--raw", "-p", TEXT, "--temp", "0", "-n", str(model["greedy_tokens"]),
                 "--save-tokens", str(tokens)])
            result = subprocess.run([sys.executable, str(COMPARE), str(tokens), "--reference", str(path), "--generated"],
                                    capture_output=True, text=True)
            print(f"{name}: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else 'greedy: no output'}")
            if result.returncode != 0:
                failures.append(f"{name}: greedy generation differs from transformers:\n{result.stdout}")
    if "perplexity_chunks" in model:
        text = cache / "wiki.test.raw"
        chunks = model["perplexity_chunks"]
        ours = perplexity_of(run([ii, "perplexity", str(path), "-f", str(text), "--chunks", str(chunks)]))
        reference_model = models[model.get("reference", name)]
        reference = reference_perplexity(cache, model_path(cache, reference_model), text, chunks)
        if "reference" in model:
            increase = ours / reference - 1
            ok = increase <= model["perplexity_max_increase"]
            print(f"{name}: perplexity {ours:.4f}, float reference {reference:.4f} (+{100 * increase:.2f}%)")
        else:
            ok = abs(ours / reference - 1) <= model["perplexity_tolerance"]
            print(f"{name}: perplexity {ours:.4f}, transformers {reference:.4f}")
        if not ok:
            failures.append(f"{name}: perplexity {ours:.4f} against {reference:.4f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ii", required=True, help="the ii executable")
    parser.add_argument("--suite", choices=["pr", "nightly"], default="pr")
    parser.add_argument("--only", help="check one model by name")
    args = parser.parse_args()
    cache = pathlib.Path(args.cache).expanduser().resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    models = {m["name"]: m for m in manifest.get("llms", [])}
    order = {"pr": 0, "nightly": 1}
    failures = []
    for model in models.values():
        if order[model["suite"]] > order[args.suite] or (args.only and model["name"] != args.only):
            continue
        try:
            failures += check(model, models, cache, str(pathlib.Path(args.ii).resolve()))
        except RuntimeError as error:
            failures.append(f"{model['name']}: {error}")
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
