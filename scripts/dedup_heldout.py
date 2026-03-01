#!/usr/bin/env python
"""Held-out purity — drop any generated held-out attack whose query text
overlaps (case-insensitive, exact match) with a training-set query.

Replaces build_eval_set.py's old dedup logic (which read the single
promptfoo/redteam.yaml + tests_redteam.json pair from the pilot pipeline). The
scale-up runs adversarial TRAINING and adversarial HELD-OUT generation as two
SEPARATE `redteam generate` passes (promptfooconfig.general.train.yaml vs
promptfooconfig.general.heldout.yaml), so this script dedups the held-out
generate output directly against the training generate output.

Both inputs are `redteam generate -o ...` YAML files (a `tests:` list plus the
`prompts`/`providers`/`redteam`/`defaultTest` blocks copied from the config).
Output is the held-out file with overlapping tests removed and every other
top-level key preserved, so `promptfoo redteam eval -c <output>` runs directly
on it with no further conversion.

Local, no GPU, no API calls. Usage:
    python scripts/dedup_heldout.py \
        --train promptfoo/redteam.general.train.yaml \
        --heldout promptfoo/redteam.general.heldout.yaml \
        --out promptfoo/redteam.general.heldout.deduped.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "promptfoo" / "redteam.general.train.yaml"
HELDOUT = ROOT / "promptfoo" / "redteam.general.heldout.yaml"
OUT = ROOT / "promptfoo" / "redteam.general.heldout.deduped.yaml"
TESTS_JSON = ROOT / "promptfoo" / "heldout_adv_tests.json"


def query_set(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = set()
    for t in data.get("tests") or []:
        q = (t.get("vars") or {}).get("query")
        if isinstance(q, str) and q.strip():
            out.add(q.strip().lower())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Dedup held-out attacks against training attacks")
    ap.add_argument("--train", type=Path, nargs="+", default=[TRAIN],
                    help="one or more training-generation `redteam generate` output files "
                         "(e.g. the main run plus any top-up runs)")
    ap.add_argument("--heldout", type=Path, default=HELDOUT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tests-json", type=Path, default=TESTS_JSON,
                    help="also write the deduped tests as a plain JSON list "
                         "(vars+assert only) for promptfooconfig.eval.adv.yaml's `tests:` field")
    args = ap.parse_args()

    missing = [str(p) for p in args.train if not p.exists()]
    if missing or not args.heldout.exists():
        print(f"[ERROR] missing input(s): train missing={missing} heldout={args.heldout.exists()}")
        return 1

    train_queries: set[str] = set()
    for p in args.train:
        train_queries |= query_set(p)
    heldout_data = yaml.safe_load(args.heldout.read_text(encoding="utf-8"))
    tests = heldout_data.get("tests") or []

    kept, dropped = [], Counter()
    seen = set()
    for t in tests:
        q = (t.get("vars") or {}).get("query")
        if not isinstance(q, str) or not q.strip():
            dropped["missing_query"] += 1
            continue
        key = q.strip().lower()
        if key in train_queries:
            dropped["overlaps_training"] += 1
            continue
        if key in seen:
            dropped["duplicate_within_heldout"] += 1
            continue
        seen.add(key)
        kept.append(t)

    # Native redteam graders require metadata.purpose on every test (used to
    # build the grading context) — some generated tests don't carry it, so
    # backfill from the same redteam.purpose used at generation time.
    purpose = (heldout_data.get("redteam") or {}).get("purpose", "")
    for t in kept:
        meta = t.setdefault("metadata", {})
        if not meta.get("purpose"):
            meta["purpose"] = purpose

    heldout_data["tests"] = kept
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(heldout_data, sort_keys=False), encoding="utf-8")

    # Plain JSON list (vars+assert+metadata only) for promptfooconfig.eval.adv.yaml's
    # `tests:` field -- keeps the native promptfoo:redteam:* graders baked into each
    # test's assert block, so base-vs-tuned eval still uses native redteam grading.
    args.tests_json.write_text(
        json.dumps([{k: t[k] for k in ("vars", "assert", "metadata") if k in t} for t in kept],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"training queries loaded: {len(train_queries)}")
    print(f"held-out tests before dedup: {len(tests)}")
    print(f"held-out tests after dedup:  {len(kept)}")
    print("dropped:", dict(dropped))
    print(f"-> {args.out}")
    print(f"-> {args.tests_json}")
    print(f"\nRun: promptfoo redteam eval -c {args.out.name} -o ../data/promptfoo_results_general_heldout.json"
          f"\n  (or: promptfoo eval -c promptfooconfig.eval.adv.yaml -o ../data/eval_results_general_adv.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
