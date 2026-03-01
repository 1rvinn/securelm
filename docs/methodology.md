# Methodology

The final ("general") pipeline, stage by stage. This is the iter-3 configuration
that produced the [headline results](../results/results.md). For how the pipeline
evolved to this shape — and the earlier designs that were discarded — see
[`experiments.md`](experiments.md). For the DPO objective itself, see
[`dpo-theory.md`](dpo-theory.md).

## Threat model

The target is `meta-llama/Llama-3.1-8B-Instruct`, which — like most modern
instruction-tuned models — already refuses the large majority of *direct* harmful
requests. The residual risk is **jailbreaks**: the same harmful intent obfuscated
or reframed so the safety training doesn't fire. So the whole pipeline is built
around *adversarial* prompts, across six harm categories and seven jailbreak
strategies.

**Harm categories (6):**
`chemical-biological-weapons`, `cybercrime`, `child-exploitation`,
`illegal-activities`, `illegal-drugs`, `weapons:ied`.

**Jailbreak strategies (7):**
`basic` (unobfuscated baseline), `hex`, `homoglyph`, `leetspeak`,
`jailbreak-templates`, `jailbreak:likert` (Likert-scale framing), `math-prompt`
(math-notation obfuscation). Earlier strategies `base64` and `citation` were
dropped after the pilot showed they produced no usable contrast (see
[`experiments.md`](experiments.md)).

## Stage 1 — Adversarial generation

`promptfoo redteam generate` synthesizes attacks for each (plugin × strategy)
combination. Config: [`promptfoo/promptfooconfig.general.train.yaml`](../promptfoo/promptfooconfig.general.train.yaml).

Two design choices matter here:

- **Per-plugin `numTests`, inverse-yield-weighted.** Categories convert prompts to
  usable pairs at different rates (chem-bio ~53%, child-exploitation ~41% in the
  pilot). To get a roughly *balanced* number of final pairs per category, harder-
  converting plugins are given more tests (60–80 per plugin). This is deliberate —
  a balanced training set keeps any one category from dominating the gradient.
- **Grading and dynamic-strategy attacker-LLM calls run on our own OpenAI key**
  (`defaultTest.options.provider: openai:gpt-4o`), not promptfoo's remote proxy,
  so the judge is fixed and costs are controlled.

Output: [`promptfoo/redteam.general.train.yaml`](../promptfoo/redteam.general.train.yaml)
(~405 base attacks × 7 strategies). Because the run is large, `split_redteam_tests.py`
chunks it into pod-sized eval sessions that are merged back at pairing time.

## Stage 2 — Response sampling + grading

Each attack is sent to the vLLM-served Llama under **four sampling variants**, so
that a single prompt yields a spread of responses to contrast:

| Variant | System prompt | Temperature |
|---|---|---|
| `v1-lowtemp-safe` | safety system prompt | 0.3 |
| `v2-midtemp-safe` | safety system prompt | 0.7 |
| `v3-hightemp-nosys` | none | 1.0 |
| `v4-midtemp-nosys` | none | 0.7 |

(Prompt templates: [`promptfoo/prompts/with_sys.json`](../promptfoo/prompts/with_sys.json),
[`no_sys.json`](../promptfoo/prompts/no_sys.json).)

Every response is graded. `promptfoo redteam eval` applies the native red-team
graders baked into each generated test — a **binary** 0/1 safe-vs-unsafe judgment.
The binary grader turns out to be a feature: a pair survives Stage 3 only if the
same prompt produced *both* a 0 and a 1, which guarantees a genuine safe/unsafe
contrast (see [`experiments.md`](experiments.md), "the grader as a noise filter").

## Stage 3 — Preference-pair construction

[`scripts/build_pairs.py`](../scripts/build_pairs.py) groups the four variant
responses per prompt and forms one DPO pair:

- `chosen` = highest-scored response (safest refusal / safe-completion),
- `rejected` = lowest-scored response (the compliance / harmful completion).

Score direction is uniform (promptfoo's aggregate score is high for a safe refusal
on unsafe prompts and high for a helpful answer on benign prompts), so
"highest = chosen" works everywhere. A pair is **discarded** if the best-vs-worst
gap is below `--threshold` (0.5), if the two texts are identical, or if there were
fewer than two responses — a near-tie carries no learning signal.

Output: [`data/dpo_pairs_general_adv.jsonl`](../data/dpo_pairs_general_adv.jsonl)
— **1,501 adversarial pairs**. (Cybercrime under-yielded, so an extra top-up
generation, [`promptfooconfig.general.train.cybercrime_topup.yaml`](../promptfoo/promptfooconfig.general.train.cybercrime_topup.yaml),
added +62 pairs to balance it.)

## Stage 3b — Benign counter-slice

Safety training's classic failure is **over-refusal**. To counteract it,
[`scripts/build_benign_pairs.py`](../scripts/build_benign_pairs.py) builds preference
pairs with the *inverse* contrast, from 198 hand-curated legitimate prompts that
are thematically adjacent to the harm categories (e.g. a chemistry-homework
question near the chem-bio category):

- `chosen` = a genuine helpful answer (sampled from the model, then QA-checked
  with a GPT-4o comply-rubric; regenerated once if it fails, dropped if it fails
  again — 197/198 kept),
- `rejected` = a refusal, drawn from a *pool* of varied refusal phrasings so the
  model can't key on one literal string.

Output: [`data/dpo_pairs_general_benign.jsonl`](../data/dpo_pairs_general_benign.jsonl)
— **197 benign pairs**.

[`scripts/merge_pairs.py`](../scripts/merge_pairs.py) then combines and
deterministically shuffles the two sources into
[`data/dpo_pairs_general.jsonl`](../data/dpo_pairs_general.jsonl) — **1,698 pairs**
(88.4% adversarial / 11.6% benign).

## Stage 4 — DPO training

[`scripts/train_dpo.py`](../scripts/train_dpo.py), using TRL's `DPOTrainer` with a
PEFT/LoRA config. With a PEFT config the reference model is just the policy with
its adapters disabled (`ref_model=None`), so no second model is loaded — the memory
win that makes DPO on one GPU practical.

| Setting | Value | Why |
|---|---|---|
| Base load | bf16 (`--no-quantize`) | QLoRA and bf16-LoRA were equivalent on adapter quality (see experiments); bf16 chosen because compute was available. |
| LoRA | r=16, α=16, dropout 0.05 | Rank matched to ~1.7k pairs — enough capacity without memorizing. |
| Target modules | `q,k,v,o,gate,up,down` | Attention *and* MLP: what to attend to *and* how to process it. |
| β (KL coeff) | 0.1 | Standard DPO strength. |
| Learning rate | 1e-5 | DPOConfig's 1e-6 default is tuned for full fine-tuning; too small for LoRA. |
| Effective batch | 16 | batch 2 × grad-accum 8. |
| Max prompt / seq len | 1024 / 1536 | Jailbroken prompts (hex, templates) can be long. |
| Epochs | 1 | 1.7k pairs; more invites overfitting. |

Output: a LoRA adapter (`dpo-llama31-8b-general`), served at eval time with
`vllm ... --enable-lora --lora-modules dpo=<adapter>` — no merge needed. The
mathematical objective is derived in [`dpo-theory.md`](dpo-theory.md).

## Stage 5 — Held-out evaluation

Evaluation is a **separate** `redteam generate` pass
([`promptfooconfig.general.heldout.yaml`](../promptfoo/promptfooconfig.general.heldout.yaml),
`numTests=24` uniform per plugin — eval wants comparable per-category sample sizes,
not balanced pair yield). Then:

1. [`scripts/dedup_heldout.py`](../scripts/dedup_heldout.py) drops any held-out
   attack whose text matches a training attack — **1,135 clean held-out attacks**.
2. **Adversarial eval** ([`promptfooconfig.eval.adv.yaml`](../promptfoo/promptfooconfig.eval.adv.yaml)):
   base vs DPO-tuned at one fixed condition (with-sys, temp 0.3), graded by the
   native red-team graders. 2,270 evals.
3. **Benign eval** ([`promptfooconfig.eval.benign.yaml`](../promptfoo/promptfooconfig.eval.benign.yaml)):
   the 150 held-out benign prompts, graded by a custom comply-rubric — native
   red-team grading has no notion of "answered a safe question," so over-refusal
   needs its own grader. 300 evals.
4. [`scripts/eval_report.py`](../scripts/eval_report.py) merges both result files
   into the before/after table in [`results/`](../results/results.md).

Holding the decoding condition fixed for both models is the key control: any
difference is attributable to the DPO adapter, not to sampling.
