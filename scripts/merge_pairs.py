#!/usr/bin/env python
"""Merge step — combine the adversarial DPO pairs (build_pairs.py output) and
the synthetic benign DPO pairs (build_benign_pairs.py output) into ONE final
training file for the "general" fine-tune.

Additive only: both source files are left intact. Output is deterministically
shuffled (fixed seed) so training doesn't see all-adversarial-then-all-benign
in file order.

Local, no GPU. Usage:
    python scripts/merge_pairs.py
    python scripts/merge_pairs.py --adv data/dpo_pairs_general_adv.jsonl \
        --benign data/benign_pairs_general.jsonl --out data/dpo_pairs_general.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "data" / "dpo_pairs_general_adv.jsonl"
BENIGN = ROOT / "data" / "benign_pairs_general.jsonl"
OUT = ROOT / "data" / "dpo_pairs_general.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] not found, skipping: {path}")
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge adversarial + benign DPO pairs")
    ap.add_argument("--adv", type=Path, default=ADV)
    ap.add_argument("--benign", type=Path, default=BENIGN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    adv_pairs = load_jsonl(args.adv)
    benign_pairs = load_jsonl(args.benign)
    if not adv_pairs and not benign_pairs:
        print("[ERROR] both input files empty/missing — nothing to merge.")
        return 1

    merged = adv_pairs + benign_pairs
    random.Random(args.seed).shuffle(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in merged:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"=== {args.out} written ===")
    print(f"adversarial pairs: {len(adv_pairs)}")
    print(f"benign pairs:      {len(benign_pairs)}")
    print(f"total merged:      {len(merged)}")
    print("by source:", dict(Counter(p.get("source", "?") for p in merged)))
    print("by expected_behavior:", dict(Counter(p.get("expected_behavior", "?") for p in merged)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
