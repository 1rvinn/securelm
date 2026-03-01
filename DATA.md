# Data manifest & responsible-use notice

## ⚠️ Responsible use

This repository contains **real adversarial (jailbreak) prompts** and, in the
preference-training data, **real harmful model completions** stored as `rejected`
examples — including content in a child-exploitation harm category. This material
is published so that a safety fine-tuning pipeline can be **reproduced and
audited** end-to-end.

It is intended **exclusively** for:

- defensive safety research and red-team evaluation,
- reproducing or extending the guardrail fine-tuning in this repo,
- studying jailbreak strategies in order to defend against them.

It is **not** intended for, and must not be used for, eliciting or distributing
harmful content. The harmful completions are present only as *negative* training
targets — the entire point of the pipeline is to train a model **away** from them.
By using this data you accept responsibility for using it lawfully and ethically.

If you are packaging this for a wider or less-controlled audience, consider
gating the adversarial pair files (`dpo_pairs_general*.jsonl`) behind an
access request and shipping only the benign data + pipeline publicly.

---

## What's included

### `data/` — training & evaluation prompt sets

| File | Rows | Content | Safe to read casually? |
|---|---|---|---|
| `benign_prompts.jsonl` | 198 | Legitimate requests thematically adjacent to the six harm categories (training). | ✅ yes |
| `benign_heldout.jsonl` | 150 | Held-out legitimate requests, for the over-refusal eval. | ✅ yes |
| `dpo_pairs_general_benign.jsonl` | 197 | Benign preference pairs: *helpful answer* (chosen) vs *refusal* (rejected). | ✅ yes |
| `dpo_pairs_general_adv.jsonl` | 1,501 | **Adversarial** preference pairs. `rejected` fields contain harmful completions. | ⚠️ dual-use |
| `dpo_pairs_general.jsonl` | 1,698 | Final merged training set (adversarial + benign, shuffled). | ⚠️ dual-use |

**Preference-pair schema** (`prompt` / `chosen` / `rejected` in conversational
form, plus metadata):

```json
{
  "prompt":   [{"role": "user",      "content": "<the (possibly jailbroken) request>"}],
  "chosen":   [{"role": "assistant", "content": "<safest sampled response — a refusal>"}],
  "rejected": [{"role": "assistant", "content": "<least-safe sampled response>"}],
  "category": "harmful_instruction",
  "subcategory": "cybercrime",
  "expected_behavior": "refuse",
  "strategy": "jailbreak:likert",
  "score_gap": 1.0,
  "source": "..."
}
```

For benign pairs, `chosen` is a genuine helpful answer and `rejected` is a
refusal template — the inverse contrast, which is what stops the model from
learning to refuse everything.

### `promptfoo/` — attack corpora & eval test sets

| File | Content |
|---|---|
| `redteam.general.train.yaml` | Full generated adversarial prompt set used for training. |
| `redteam.general.train.cybercrime_topup.yaml` | Extra cybercrime attacks (that category under-yielded pairs). |
| `redteam.general.heldout.deduped.yaml` | Held-out attacks, de-duplicated against the training set. |
| `redteam.general.pilot.yaml` | The small pilot run's attacks. |
| `heldout_adv_tests.json` | Held-out adversarial eval test set (base vs DPO-tuned). |
| `benign_heldout_tests.json` | Held-out benign eval test set. |

These `.yaml` files contain raw jailbreak prompts — same dual-use caveat applies.

---

## What's deliberately **excluded**

- **Raw promptfoo result dumps** (`promptfoo_results_*.json`, `eval_results_*.json`,
  ~400 MB total). These are regenerable intermediates; the distilled outcome is in
  [`results/`](results/).
- **Secrets** (`.env`) — only `.env.example` (a template with empty values) is
  committed. Never commit a filled-in `.env`.
- **Iteration-1 / pilot scratch data** (superseded PII-track seeds, early
  `dpo_pairs.jsonl`, `*_SCRATCH.jsonl`) — the [`docs/experiments.md`](docs/experiments.md)
  narrative covers what they were without shipping the bytes.

## Provenance

Adversarial prompts were generated with [promptfoo](https://promptfoo.dev)
red-team plugins; responses were sampled from `meta-llama/Llama-3.1-8B-Instruct`
served via vLLM under four decoding conditions; safety/compliance was graded by
GPT-4o (and promptfoo's native red-team graders). Benign prompts were
hand-curated by the author. Full mechanics: [`docs/methodology.md`](docs/methodology.md).
