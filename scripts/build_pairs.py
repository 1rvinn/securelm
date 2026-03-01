#!/usr/bin/env python
"""Stage 3 — Pair construction (DPO preference pairs).

Reads one or more promptfoo results exports (seed eval and/or redteam eval),
groups the 4 sampling-variant responses per prompt, and forms a DPO pair:
  chosen   = highest-scored response  (safest / most helpful)
  rejected = lowest-scored  response  (least safe / over-refusing)
Prompts whose best-vs-worst score gap is below --threshold are DISCARDED — a
near-tie carries no learning signal.

Score direction is uniform: promptfoo's aggregate `score` is HIGH for a safe
refusal on unsafe prompts, high for a helpful answer on benign prompts, and low
when the model complies with an unsafe/jailbroken request — so "highest = chosen"
works everywhere.

Output data/dpo_pairs.jsonl (conversational format; TRL auto-applies the Llama
chat template):
  {"prompt":[{"role":"user","content": "..."}],
   "chosen":[{"role":"assistant","content": "..."}],
   "rejected":[{"role":"assistant","content": "..."}],
   "category": "...", "subcategory": "...", "expected_behavior": "...",
   "score_gap": 0.x, "chosen_variant": "...", "rejected_variant": "...", "source": "seed|redteam"}

Runs LOCALLY (no GPU). Usage:
    python scripts/build_pairs.py                                  # seed only
    python scripts/build_pairs.py --results data/promptfoo_results.json data/promptfoo_results_redteam.json
    python scripts/build_pairs.py --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = [ROOT / "data" / "promptfoo_results.json"]
OUT = ROOT / "data" / "dpo_pairs.jsonl"

QUERY_KEYS = ("query", "prompt", "input", "goal", "harmful_prompt", "text")

# Fallback labelling for `redteam eval` output, whose rows carry category info in
# metadata.pluginId rather than vars.category (unlike our own adv pipeline, which
# bakes category/subcategory into vars). Kept identical to redteam_to_tests.py's
# PLUGIN_MAP so cloud- and static-sourced pairs use the same category taxonomy.
PLUGIN_MAP = {
    "harmful:indiscriminate-weapons": ("harmful_instruction", "weapons"),
    "harmful:weapons:ied": ("harmful_instruction", "weapons"),
    "harmful:cybercrime": ("harmful_instruction", "malware"),
    "harmful:cybercrime:malicious-code": ("harmful_instruction", "malware"),
    "harmful:non-violent-crime": ("harmful_instruction", "fraud"),
    "pii:direct": ("pii", "pii_extract"),
    "pii:social": ("pii", "pii_extract"),
    # "general" track plugins (Phase 2 scale-up) — additive, reporting-only.
    "harmful:chemical-biological-weapons": ("harmful_instruction", "chem_bio_weapons"),
    "harmful:child-exploitation": ("harmful_instruction", "child_exploitation"),
    "harmful:illegal-activities": ("harmful_instruction", "illegal_activities"),
    "harmful:illegal-drugs": ("harmful_instruction", "illegal_drugs"),
}


def load_rows(results_path: Path) -> list[dict]:
    d = json.loads(results_path.read_text(encoding="utf-8"))
    rows = d.get("results")
    if isinstance(rows, dict):           # tolerate older nested shape
        rows = rows.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Could not find a results list in {results_path}")
    return rows


def extract_query(vars_: dict) -> str:
    for k in QUERY_KEYS:
        val = vars_.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def add_rows(groups: dict, rows: list[dict], file_tag: str) -> None:
    """Add a file's rows into `groups`, namespacing keys by file so seed and
    redteam prompts never collide."""
    for r in rows:
        v = r.get("vars") or {}
        base_key = v.get("seed_id") or f"t{r.get('testIdx')}"
        key = f"{file_tag}:{base_key}"
        out = (r.get("response") or {}).get("output")
        if not isinstance(out, str) or not out.strip():
            continue  # skip errored / empty generations
        score = r.get("score")
        if score is None:
            continue
        # Prefer vars (our adv pipeline); fall back to metadata.pluginId/strategyId
        # for native redteam-eval rows, which don't carry category in vars.
        meta = r.get("metadata") or {}
        plugin = meta.get("pluginId", "")
        cat, sub = PLUGIN_MAP.get(plugin, ("", ""))
        groups[key].append({
            "score": float(score),
            "output": out.strip(),
            "variant": (r.get("provider") or {}).get("label", "?"),
            "query": extract_query(v),
            "category": v.get("category") or cat,
            "subcategory": v.get("subcategory") or sub,
            "expected_behavior": v.get("expected_behavior", ""),
            "strategy": v.get("strategy") or meta.get("strategyId", ""),
            "source": file_tag,
        })


def build_pairs(groups: dict, threshold: float):
    pairs, discards, kept_by_cat = [], Counter(), Counter()
    for items in groups.values():
        if len(items) < 2:
            discards["too_few_responses"] += 1
            continue
        best = max(items, key=lambda x: x["score"])
        worst = min(items, key=lambda x: x["score"])
        gap = best["score"] - worst["score"]
        if gap < threshold:
            discards["gap_below_threshold"] += 1
            continue
        if best["output"] == worst["output"]:
            discards["identical_text"] += 1
            continue
        if not best["query"]:
            discards["missing_query"] += 1
            continue
        pairs.append({
            "prompt": [{"role": "user", "content": best["query"]}],
            "chosen": [{"role": "assistant", "content": best["output"]}],
            "rejected": [{"role": "assistant", "content": worst["output"]}],
            "category": best["category"],
            "subcategory": best["subcategory"],
            "expected_behavior": best["expected_behavior"],
            "strategy": best.get("strategy", ""),
            "score_gap": round(gap, 4),
            "chosen_variant": best["variant"],
            "rejected_variant": worst["variant"],
            "source": best["source"],
        })
        kept_by_cat[best["subcategory"] or best["category"] or "(unlabeled)"] += 1
    return pairs, discards, kept_by_cat


def gap_histogram(groups: dict) -> None:
    gaps = [max(i["score"] for i in items) - min(i["score"] for i in items)
            for items in groups.values() if len(items) >= 2]
    print("\nContrast (best-worst score gap) across prompts:")
    for t in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        print(f"  gap >= {t:<4}: {sum(1 for g in gaps if g >= t):>4} prompts")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build DPO pairs from promptfoo results")
    ap.add_argument("--results", type=Path, nargs="+", default=DEFAULT_RESULTS,
                    help="one or more promptfoo results JSON files (seed and/or redteam)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="minimum best-vs-worst score gap to keep a pair (0-1)")
    args = ap.parse_args()

    groups: dict = defaultdict(list)
    for path in args.results:
        if not path.exists():
            print(f"[WARN] results file not found, skipping: {path}")
            continue
        rows = load_rows(path)
        tag = path.stem.replace("promptfoo_results", "").strip("_") or "seed"
        add_rows(groups, rows, tag)
        print(f"Loaded {len(rows)} rows from {path.name} (tag='{tag}')")

    if not groups:
        print("[ERROR] no results loaded. Run Stage 2 / Stage 2b first.")
        return 1

    print(f"\nTotal unique prompts across files: {len(groups)}")
    gap_histogram(groups)

    pairs, discards, kept_by_cat = build_pairs(groups, args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n=== dpo_pairs.jsonl written (threshold={args.threshold}) ===")
    print(f"kept pairs: {len(pairs)}")
    print("kept by category:", dict(kept_by_cat))
    print("kept by source:  ", dict(Counter(p["source"] for p in pairs)))
    print("discarded:       ", dict(discards))

    if pairs:
        print("\nSample pairs (chosen vs rejected):")
        for p in pairs[:4]:
            q = p["prompt"][0]["content"][:80].replace("\n", " ")
            ch = p["chosen"][0]["content"][:90].replace("\n", " ")
            rj = p["rejected"][0]["content"][:90].replace("\n", " ")
            print(f"\n  [{p['category']}/{p['subcategory']}] gap={p['score_gap']} "
                  f"src={p['source']} ({p['rejected_variant']} -> {p['chosen_variant']})")
            print(f"    Q:        {q!r}")
            print(f"    chosen:   {ch!r}")
            print(f"    rejected: {rj!r}")
    else:
        print("\n[WARN] 0 pairs kept — lower --threshold or generate harder attacks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
