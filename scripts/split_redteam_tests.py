#!/usr/bin/env python3
"""Split a `redteam generate` output file into N chunk configs.

Each chunk keeps the original config (providers, prompts, defaultTest, etc.)
and gets a contiguous slice of the `tests` list, so each chunk can be run as
its own `promptfoo redteam eval -c <chunk> -o <result>` invocation — letting a
long generation run be broken into pod-sized (e.g. ~1hr) sessions. The
resulting result files are merged later by build_pairs.py, which already
accepts multiple --results files.

Usage:
  python scripts/split_redteam_tests.py --input promptfoo/redteam.general.train.yaml \
    --chunks 6 --out-prefix promptfoo/redteam.general.train
"""
import argparse
import math
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.input).read_text())
    tests = data["tests"]
    n = len(tests)
    chunk_size = math.ceil(n / args.chunks)

    written = []
    for i in range(args.chunks):
        chunk = tests[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        chunk_data = dict(data)
        chunk_data["tests"] = chunk
        out_path = Path(f"{args.out_prefix}.chunk{i + 1}.yaml")
        out_path.write_text(yaml.dump(chunk_data, sort_keys=False))
        written.append((out_path, len(chunk)))

    print(f"Total tests: {n}")
    for path, count in written:
        print(f"  {path} -> {count} tests")


if __name__ == "__main__":
    main()
