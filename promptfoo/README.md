# promptfoo/

[promptfoo](https://promptfoo.dev) configs for adversarial generation and
evaluation, the prompt templates, and the generated attack corpora. All configs
are commented with their exact `promptfoo` invocation.

⚠️ The `redteam.general.*.yaml` files contain real jailbreak prompts — see
[`../DATA.md`](../DATA.md).

## Prompt templates — `prompts/`

| File | Role |
|---|---|
| `with_sys.json` | User prompt **plus** the safety system prompt. Used by the two "safe" sampling variants. |
| `no_sys.json` | User prompt only (no system prompt). Used by the two "no-sys" variants. |

## Configs

### Generation (`redteam generate` + `redteam eval`)

| Config | Role |
|---|---|
| `promptfooconfig.general.train.yaml` | **Training** generation. 6 plugins × 7 strategies, per-plugin `numTests` (inverse-yield-weighted for balanced pairs). |
| `promptfooconfig.general.train.cybercrime_topup.yaml` | Extra cybercrime attacks (that category under-yielded). |
| `promptfooconfig.general.heldout.yaml` | **Held-out** generation. Same plugins/strategies, `numTests=24` uniform (comparable per-category eval sizes). |
| `promptfooconfig.general.pilot.yaml` | The iter-2 pilot (`numTests=5`) used to pick strategies + per-plugin weights. |

### Evaluation (`promptfoo eval`, base vs DPO-tuned)

| Config | Role |
|---|---|
| `promptfooconfig.eval.adv.yaml` | Held-out **adversarial** eval; native red-team grading; consumes `heldout_adv_tests.json`. |
| `promptfooconfig.eval.benign.yaml` | Held-out **benign** eval; custom comply-rubric; consumes `benign_heldout_tests.json`. |

Both eval configs hold the decoding condition fixed (with-sys, temp 0.3) for both
models, so any difference is attributable to the DPO adapter.

## Generated corpora & eval test sets

| File | Content |
|---|---|
| `redteam.general.train.yaml` | The full training attack set (`redteam generate` output). |
| `redteam.general.train.cybercrime_topup.yaml` | Cybercrime top-up attacks. |
| `redteam.general.heldout.deduped.yaml` | Held-out attacks after `dedup_heldout.py` (purity vs training). |
| `redteam.general.pilot.yaml` | Pilot attack set. |
| `heldout_adv_tests.json` | Held-out adversarial eval tests (consumed by `promptfooconfig.eval.adv.yaml`). |
| `benign_heldout_tests.json` | Held-out benign eval tests (consumed by `promptfooconfig.eval.benign.yaml`). |

**Not committed** (regenerable, large): the six `redteam.general.train.chunkN.yaml`
splits (produced by `split_redteam_tests.py` from `redteam.general.train.yaml`),
the pre-dedup held-out file, and all raw `promptfoo_results_*.json` /
`eval_results_*.json` dumps — the distilled outcome lives in
[`../results/`](../results/results.md).

## Grading key

Generation, grading, and the dynamic strategies' attacker-LLM calls
(`jailbreak:likert`, `math-prompt`) all run on **your own** `OPENAI_API_KEY`
(pinned via `defaultTest.options.provider: openai:gpt-4o`), not promptfoo's remote
proxy — so the judge is fixed and costs are controlled. Export it first:

```bash
cp ../.env.example ../.env   # fill in OPENAI_API_KEY
set -a && source ../.env && set +a
```
