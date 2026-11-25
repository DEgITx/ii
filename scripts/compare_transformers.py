#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compares the logits of a language model in ii with Hugging Face transformers.

ii writes the prompt tokens and the logits of every position (`ii generate MODEL -p TEXT --save-logits out.npz`);
this script runs the reference model (a Hugging Face directory or hub id) in float32 on the same tokens and reports
the largest difference relative to the largest logit, the agreement of the most likely tokens and the mean KL
divergence. With --text it also checks that ii's tokens equal the reference tokenizer's.

    python scripts/compare_transformers.py out.npz --reference path/to/hf-model [--text "..."] [--tolerance 1e-3]

Greedy generation: `ii generate MODEL -p TEXT --temp 0 -n 64 --save-tokens gen.npz`, then
    python scripts/compare_transformers.py gen.npz --reference MODEL --generated
compares ii's tokens with transformers' greedy continuation of the same prompt (reports the first difference).

Perplexity: `python scripts/compare_transformers.py --perplexity wiki.test.raw --reference MODEL --ctx 512
--chunks 20` computes it the way `ii perplexity` and llama.cpp's llama-perplexity do (chunks of --ctx tokens, the
first token of each replaced by BOS, the second half scored).

Exit status 1 when --tolerance or --min-agreement is given and not met, or the generations differ.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logits", nargs="?", help="the .npz written by ii generate --save-logits / --save-tokens")
    parser.add_argument("--reference", required=True, help="Hugging Face model directory or hub id")
    parser.add_argument("--text", help="the prompt, to compare the tokenization")
    parser.add_argument("--tolerance", type=float, help="maximum relative difference of the logits")
    parser.add_argument("--min-agreement", type=float, help="minimum fraction of equal top-1 tokens")
    parser.add_argument("--generated", action="store_true", help="the .npz holds prompt and generated tokens")
    parser.add_argument("--perplexity", help="a text file: compute its perplexity with the reference model")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=-1)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.perplexity:
        return perplexity(args)
    if args.generated:
        data = np.load(args.logits)
        prompt = torch.from_numpy(data["prompt"].astype(np.int64))[None]
        ours = data["generated"].tolist()
        model = AutoModelForCausalLM.from_pretrained(args.reference, dtype=torch.float32, attn_implementation="eager")
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=len(ours), do_sample=False, temperature=None, top_p=None,
                                 repetition_penalty=1.0, min_new_tokens=len(ours),
                                 top_k=None)
        reference = out[0, prompt.shape[1]:].tolist()
        same = 0
        while same < min(len(ours), len(reference)) and ours[same] == reference[same]:
            same += 1
        print(f"greedy generation: {same} of {len(ours)} tokens equal" +
              ("" if same == len(ours) else f" (first difference at {same}: ii {ours[same]}, "
                                            f"transformers {reference[same] if same < len(reference) else None})"))
        return 0 if same == len(ours) else 1

    data = np.load(args.logits)
    tokens = data["tokens"].astype(np.int64)
    ours = data["logits"].astype(np.float64)
    status = 0
    if args.text is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.reference)
        expected = tokenizer(args.text)["input_ids"]
        if list(expected) != tokens.tolist():
            print(f"tokens differ:\n  ii:           {tokens.tolist()}\n  transformers: {list(expected)}")
            status = 1
        else:
            print(f"tokens match ({len(expected)})")
    model = AutoModelForCausalLM.from_pretrained(args.reference, dtype=torch.float32, attn_implementation="eager")
    with torch.no_grad():
        reference = model(torch.from_numpy(tokens)[None]).logits[0].double().numpy()
    vocab = min(reference.shape[1], ours.shape[1])
    reference, ours = reference[:, :vocab], ours[:, :vocab]
    scale = np.abs(reference).max(axis=1, keepdims=True)
    relative = (np.abs(ours - reference) / scale).max()
    agreement = float((ours.argmax(axis=1) == reference.argmax(axis=1)).mean())

    def log_softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

    p_log = log_softmax(reference)
    kl = float((np.exp(p_log) * (p_log - log_softmax(ours))).sum(axis=1).mean())
    print(f"positions {len(tokens)}, max relative difference {relative:.3e}, top-1 agreement {agreement:.3f}, "
          f"mean KL {kl:.3e}")
    if args.tolerance is not None and relative > args.tolerance:
        status = 1
    if args.min_agreement is not None and agreement < args.min_agreement:
        status = 1
    return status


def perplexity(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.reference)
    model = AutoModelForCausalLM.from_pretrained(args.reference, dtype=torch.float32, attn_implementation="eager")
    text = open(args.perplexity, encoding="utf-8").read()
    tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
    n_ctx = args.ctx
    chunks = len(tokens) // n_ctx if args.chunks < 0 else min(args.chunks, len(tokens) // n_ctx)
    bos = tokenizer.bos_token_id
    first = n_ctx // 2
    nll = 0.0
    count = 0
    for c in range(chunks):
        chunk = list(tokens[c * n_ctx:(c + 1) * n_ctx])
        if bos is not None:
            chunk[0] = bos
        with torch.no_grad():
            logits = model(torch.tensor([chunk])).logits[0].double()
        log_probs = torch.log_softmax(logits, dim=-1)
        for j in range(first, n_ctx - 1):
            nll -= float(log_probs[j, tokens[c * n_ctx + j + 1]])
            count += 1
        print(f"[{c + 1}]{np.exp(nll / count):.4f} ", end="", flush=True)
    print(f"\nFinal estimate: PPL = {np.exp(nll / count):.4f} ({count} tokens scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
