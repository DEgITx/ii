#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Collects the chat templates of popular models and renders them with transformers' Jinja environment, as reference
cases for ii's template interpreter (tests/data/llm/chat_templates.json).

Templates come from GGUF files (tokenizer.chat_template) and Hugging Face tokenizer_config.json files given on the
command line. Every template is rendered for a few conversations, with and without the generation prompt; a template
that raises (e.g. a system message where the model has none) is recorded as an expected error.

    python scripts/gen_chat_template_cases.py model.gguf ... tokenizer_config.json ...

Requires transformers and gguf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "data" / "llm" / "chat_templates.json"

CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3? Answer with \"6\" or 'six'."},
    ],
    [
        {"role": "user", "content": "Привет! 日本語 🙂\nSecond line."},
        {"role": "assistant", "content": "  Hi there.  "},
        {"role": "user", "content": "Bye"},
    ],
]
CONTEXT = {"bos_token": "<s>", "eos_token": "</s>", "date_string": "26 Jul 2024"}


def templates_from(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".gguf":
        from gguf import GGUFReader

        field = GGUFReader(str(path)).fields.get("tokenizer.chat_template")
        if field is None:
            return []
        return [(path.stem, bytes(field.parts[field.data[0]]).decode("utf-8"))]
    config = json.loads(path.read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if isinstance(template, list):
        template = next((t["template"] for t in template if t.get("name") == "default"), template[0]["template"])
    return [(path.stem, template)] if template else []


def main() -> int:
    from transformers.utils.chat_template_utils import _compile_jinja_template

    cases = []
    for argument in sys.argv[1:]:
        for name, template in templates_from(Path(argument)):
            compiled = _compile_jinja_template(template)
            for c, messages in enumerate(CONVERSATIONS):
                for generation in (True, False):
                    case = {"name": f"{name}/{c}/{'gen' if generation else 'nogen'}", "template": template,
                            "messages": messages, "add_generation_prompt": generation}
                    try:
                        case["expected"] = compiled.render(messages=messages, add_generation_prompt=generation,
                                                           **CONTEXT)
                    except Exception as error:  # noqa: BLE001 - templates raise their own exceptions
                        case["error"] = str(error)
                    cases.append(case)
            print(f"{name}: {len(CONVERSATIONS) * 2} cases")
    OUTPUT.write_text(json.dumps({"context": CONTEXT, "cases": cases}, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
