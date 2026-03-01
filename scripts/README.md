# scripts/

Pipeline scripts. Every script has a docstring with its exact usage. Below is the
map of which belong to the **final (iter-3) pipeline** vs. **iter-1 legacy**
(kept for provenance; superseded — see [`../docs/experiments.md`](../docs/experiments.md)).

## Final pipeline (iter-3)

Run in this order — see [`../README.md#reproduce`](../README.md#reproduce) and
[`../docs/methodology.md`](../docs/methodology.md) for the surrounding
`promptfoo` commands.

| Script | Stage | Does |
|---|---|---|
| `split_redteam_tests.py` | 1 | Split a large `redteam generate` output into N chunk configs for pod-sized eval sessions. |
| `build_pairs.py` | 3 | Group the 4 sampling variants per prompt → DPO pair (chosen = safest, rejected = least-safe); discard low-contrast pairs. |
| `build_benign_pairs.py` | 3b | Generate + QA benign "helpful > refuse" pairs from `data/benign_prompts.jsonl`. |
| `merge_pairs.py` | 3b | Merge + shuffle adversarial and benign pairs into the final training file. |
| `train_dpo.py` | 4 | DPO training (TRL + LoRA). `--no-quantize` for bf16 (the final choice); default is QLoRA/4-bit. |
| `dedup_heldout.py` | 5 | Drop held-out attacks whose text overlaps the training set (held-out purity). |
| `build_benign_heldout_tests.py` | 5 | Convert `data/benign_heldout.jsonl` → promptfoo benign eval tests. |
| `prepare_redteam_eval.py` | 5 | Assemble a `redteam eval`-ready config (deduped held-out tests + a 2-provider base-vs-dpo comparison). |
| `eval_report.py` | 5 | Merge the adversarial + benign result files into one before/after table. |

## Iter-1 legacy (superseded)

These implemented the original seed-prompt / PII-track pipeline that iter-2
retired once it was clear all useful signal came from jailbroken prompts. Kept so
the earlier iterations in the write-up are reproducible.

| Script | Was |
|---|---|
| `build_seed_prompts.py` | Collected 240 seed prompts (BeaverTails / Do-Not-Answer / AdvBench + a self-built PII set). |
| `seed_to_promptfoo_tests.py` | Converted seed prompts → promptfoo tests with refuse/comply rubrics. |
| `redteam_to_tests.py` | Parsed early `redteam.yaml` into our own tests/rubrics (before the switch to native red-team grading). |
| `build_eval_set.py` | Old held-out eval-set builder (replaced by `dedup_heldout.py` + the eval configs). |

## Environments

`build_pairs.py`, `merge_pairs.py`, `dedup_heldout.py`, `split_redteam_tests.py`,
`prepare_redteam_eval.py`, `build_benign_heldout_tests.py` run **locally, no GPU**
(`requirements-local.txt`). `train_dpo.py` needs the **GPU box**
(`requirements-gpu.txt`). `build_benign_pairs.py` needs the vLLM endpoint up **and**
`OPENAI_API_KEY` set (it grades as it generates).
