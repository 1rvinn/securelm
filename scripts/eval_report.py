#!/usr/bin/env python
"""Stage 5 — summarise the base-vs-tuned held-out eval into a before/after table.

Scale-up ("general" track) update: the adversarial and benign held-out sets are
now graded by two SEPARATE pipelines (native redteam grading vs our custom
comply-rubric — see PHASE1_implementation_handoff.md item F) and therefore live
in two separate result files. This script now reads and MERGES them:
  --adv     data/eval_results_general_adv.json     (promptfooconfig.eval.adv.yaml)
  --benign  data/eval_results_general_benign.json   (promptfooconfig.eval.benign.yaml)

Per-category adversarial breakdown uses metadata.pluginId (native redteam-eval
rows don't carry category in vars) via PLUGIN_MAP, scoped to reporting only —
does not affect training (build_pairs.py deliberately does NOT need this).

Legacy single-file mode (pilot-era promptfooconfig.eval.yaml output, one file
covering both adversarial + benign) still works via --results, unchanged.

Local, no GPU. Usage:
    python scripts/eval_report.py                                    # new dual-file mode (general track)
    python scripts/eval_report.py --adv data/eval_results_general_adv.json --benign data/eval_results_general_benign.json
    python scripts/eval_report.py --results data/eval_results.json    # legacy pilot single-file mode
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADV_RESULTS = ROOT / "data" / "eval_results_general_adv.json"
BENIGN_RESULTS = ROOT / "data" / "eval_results_general_benign.json"

# Friendly category names for the "general" track's 6 plugins — reporting
# only, kept in sync with scripts/build_pairs.py's PLUGIN_MAP conventions and
# data/benign_prompts.jsonl's category slugs.
PLUGIN_MAP = {
    "harmful:chemical-biological-weapons": "chemical-biological-weapons",
    "harmful:cybercrime": "cybercrime",
    "harmful:illegal-activities": "illegal-activities",
    "harmful:illegal-drugs": "illegal-drugs",
    "harmful:weapons:ied": "weapons-ied",
    "harmful:child-exploitation": "child-exploitation",
}


def load_rows(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("results")
    if isinstance(rows, dict):
        rows = rows.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Could not find a results list in {path}")
    return rows


def adv_category(row: dict) -> str:
    meta = row.get("metadata") or {}
    plugin = meta.get("pluginId", "")
    v = row.get("vars") or {}
    return PLUGIN_MAP.get(plugin) or v.get("category") or "?"


def accumulate(stats: dict, rows: list[dict], group_fn) -> None:
    for r in rows:
        model = (r.get("provider") or {}).get("label", "?")
        group = group_fn(r)
        passed = bool(r.get("success"))
        stats[model][group][0] += int(passed)
        stats[model][group][1] += 1


def print_table(stats: dict) -> None:
    models = sorted(stats)
    groups = sorted({g for m in stats for g in stats[m]})

    def rate(model, group):
        p, n = stats[model][group]
        return (100.0 * p / n) if n else float("nan"), n

    col_w = 18
    header = f"{'group':<38}" + "".join(f"{m:>{col_w}}" for m in models) + f"{'delta(t-b)':>14}"
    print(header)
    print("-" * len(header))
    for g in groups:
        cells, vals = "", {}
        for m in models:
            pct, n = rate(m, g)
            vals[m] = pct
            cells += f"{pct:.1f}% (n={n})".rjust(col_w)
        delta = ""
        if "base" in vals and any("dpo" in m for m in models):
            tuned_key = next(m for m in models if "dpo" in m)
            delta = f"{vals[tuned_key] - vals['base']:+.1f}pp".rjust(14)
        print(f"{g:<38}{cells}{delta}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adv", type=Path, default=ADV_RESULTS)
    ap.add_argument("--benign", type=Path, default=BENIGN_RESULTS)
    ap.add_argument("--results", type=Path, default=None,
                    help="legacy pilot single-file mode (overrides --adv/--benign)")
    args = ap.parse_args()

    stats: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    if args.results is not None:
        rows = load_rows(args.results)

        def group_fn(r):
            v = r.get("vars") or {}
            expected = v.get("expected_behavior", "")
            return "BENIGN (helpfulness)" if expected == "comply" else f"adversarial:{v.get('category', '?')}"

        accumulate(stats, rows, group_fn)
        print(f"\n=== Held-out eval (legacy single-file mode): {args.results.name} ===\n")
    else:
        if not args.adv.exists() and not args.benign.exists():
            print(f"[ERROR] neither {args.adv} nor {args.benign} exist. Run the Stage 5 evals first "
                  "(promptfooconfig.eval.adv.yaml + promptfooconfig.eval.benign.yaml), "
                  "or pass --results for legacy pilot-era single-file mode.")
            return 1
        if args.adv.exists():
            accumulate(stats, load_rows(args.adv), lambda r: f"adversarial:{adv_category(r)}")
        else:
            print(f"[WARN] adversarial results not found, skipping: {args.adv}")
        if args.benign.exists():
            accumulate(stats, load_rows(args.benign), lambda r: "BENIGN (helpfulness)")
        else:
            print(f"[WARN] benign results not found, skipping: {args.benign}")
        print(f"\n=== Held-out eval (general track): {args.adv.name} + {args.benign.name} ===\n")

    print("Pass-rate by model (higher is better in every row):\n")
    print_table(stats)

    print("\nReading it:")
    print("  - adversarial:* rows  -> pass = model REFUSED the attack. Tuned should be >= base.")
    print("  - BENIGN row          -> pass = model HELPED. Watch for a DROP (over-refusal regression).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
