#!/usr/bin/env python
"""Build a `redteam eval`-ready config for Stage 5 (base vs dpo-tuned).

Combines:
  - tests/redteam/defaultTest from a `redteam generate` output file (e.g. the
    deduped held-out attacks, which already carry native promptfoo:redteam:*
    assert blocks + metadata.purpose per test)
  - prompts/providers from a hand-written 2-provider comparison config (e.g.
    base + dpo-tuned at one fixed sampling condition)

This keeps Stage 5 on `promptfoo redteam eval`, symmetric with how the
training pairs were built (Step 1/2), instead of extracting tests into a
separate plain JSON file for `promptfoo eval`. Grading itself is identical
either way (native asserts execute the same regardless of which subcommand
invokes them) — this only changes which command runs it.

Local, no GPU, no API calls. Usage:
    python scripts/prepare_redteam_eval.py \
        --tests-from promptfoo/redteam.general.heldout.deduped.yaml \
        --providers-from promptfoo/promptfooconfig.eval.adv.yaml \
        --out promptfoo/redteam.general.heldout.eval.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-from", type=Path, required=True,
                     help="redteam generate output (has tests/redteam/defaultTest)")
    ap.add_argument("--providers-from", type=Path, required=True,
                     help="hand-written config to take prompts/providers from")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tests_data = yaml.safe_load(args.tests_from.read_text(encoding="utf-8"))
    providers_data = yaml.safe_load(args.providers_from.read_text(encoding="utf-8"))

    merged = dict(tests_data)
    merged["prompts"] = providers_data["prompts"]
    merged["providers"] = providers_data["providers"]
    if "defaultTest" in providers_data:
        merged["defaultTest"] = providers_data["defaultTest"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  tests:     {len(merged.get('tests', []))}")
    print(f"  providers: {[p.get('label', p.get('id')) for p in merged['providers']]}")
    print(f"  prompts:   {[p.get('label', p.get('id')) for p in merged['prompts']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
