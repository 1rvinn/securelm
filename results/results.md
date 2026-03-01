# Results (iter-3, full run)

Distilled from the held-out evaluation. Raw promptfoo result dumps
(`eval_results_general_adv.json` ~34 MB, `eval_results_general_benign.json` ~4 MB)
are regenerable intermediates and are not committed; the numbers below are the
extract. Machine-readable copy: [`results.json`](results.json).

**Setup.** Base `Llama-3.1-8B-Instruct` vs the DPO-tuned LoRA adapter, both at one
fixed decoding condition (safety system prompt, temperature 0.3). Adversarial
attacks graded by promptfoo's native red-team graders; benign prompts graded by a
custom GPT-4o comply-rubric (native red-team grading cannot measure over-refusal).
All prompts are **held out** — generated separately and de-duplicated against the
entire training set.

- Held-out adversarial attacks: **1,135** (2,270 base-vs-DPO evals)
- Held-out benign prompts: **150** (300 evals)

## Safety — % of adversarial prompts refused (↑ better)

| Category | Base | DPO-tuned | Δ | N |
|---|---|---|---|---|
| Cybercrime | 47.7% | 97.2% | **+49.5pp** | 109 |
| Chem-bio weapons | 56.8% | 99.0% | **+42.2pp** | 192 |
| Weapons / IED | 65.3% | 98.6% | **+33.3pp** | 216 |
| Illegal activities | 67.2% | 99.5% | **+32.3pp** | 186 |
| Illegal drugs | 76.9% | 99.1% | **+22.2pp** | 216 |
| Child exploitation | 84.7% | 99.1% | **+14.4pp** | 216 |

## Over-refusal — % of benign prompts answered (↑ better)

| Base | DPO-tuned | Δ |
|---|---|---|
| 100.0% (n=150) | 93.3% | **−6.7pp** |

Category breakdown of the 10 benign refusals:

| Category | Failures | Rate |
|---|---|---|
| Weapons / IED | 5/25 | 20% |
| Child exploitation | 2/25 | 8% |
| Chem-bio weapons | 1/25 | 4% |
| Illegal drugs | 1/25 | 4% |
| Illegal activities | 1/25 | 4% |
| Cybercrime | 0/25 | 0% |

## Training dynamics

bf16 LoRA, 1 epoch, ~19 min on one A100, 1,698 pairs:

| Metric | Start | End |
|---|---|---|
| Loss | 0.69 | 0.27 |
| `rewards/accuracies` | 0.51 | 0.94 |
| `rewards/margins` | 0.001 | 2.80 |

## Reading

Safety gains are large and decisively above noise across all six categories
(n = 109–216 each). The over-refusal cost is real but small in aggregate and
**concentrated** in `weapons:ied` — the category trained hardest — which is the
central helpfulness/safety tradeoff this project set out to measure rather than
hide. See [`../README.md#limitations`](../README.md#limitations).
