# Experimental journey

How the pipeline reached its final shape. This is the distilled story of three
iterations; the raw, timestamped version is in [`../my_notes.md`](../my_notes.md).
The value here is less the final numbers than *what each iteration ruled in or out*.

---

## Iter 1 — the pilot: establishing the problem

**Goal:** stand up an end-to-end DPO safety loop and see whether it moves the
needle at all.

**Setup.** Three guardrail categories (harmful-instruction refusal, PII protection,
jailbreak resistance), seeded from public datasets (BeaverTails, Do-Not-Answer,
AdvBench) plus a self-built PII set. Prompts were graded by a GPT-4o rubric,
responses sampled under four conditions, pairs built by max-vs-min score, and a
QLoRA DPO adapter trained on 344 pairs.

**What broke — and the key finding.** Direct harmful prompts had a **93.75% pass
rate**: the base model already refuses them, so there was almost no `chosen`/
`rejected` contrast to learn from. An adversarial-augmentation stage
(`redteam generate` with jailbreak strategies) raised the failure rate to 56% and
was clearly where the signal lived.

**Result:** benign helpfulness held (+2.5pp, no over-refusal ✅) but the safety
gain was within noise (+2.7pp on jailbreaks; the PII slice at n=9 was
uninterpretable). Loss 0.70 → 0.64, reward accuracy 0.46 → 0.83.

**Lessons carried forward:**
1. **The direct-prompt stages are dead weight** — nearly all useful pairs come
   from jailbroken prompts. Later iterations dropped seed collection entirely.
2. **Pair quality, not model tweaks, is the bottleneck.** 344 weak pairs can't
   produce a decisive gain regardless of hyperparameters.
3. A real benign counter-set is needed to *measure* (and later prevent)
   over-refusal, not a 40-prompt afterthought.

---

## Iter 2 — experimentation: fixing the method one lever at a time

Each idea was tried in isolation so its effect could be attributed.

**a. Switch pair-generation grading to promptfoo's native red-team eval.** The
native grader is **binary** (0/1). That sounds like a downgrade, but it acts as a
*noise filter*: a prompt only yields a pair when it produced *both* a safe (1) and
an unsafe (0) response — an automatic guarantee of genuine contrast. Pair quality
improved markedly. (Cost: the `jailbreak:meta` strategy was slow and was turned
off.)

**b. Delete the direct-prompt stages.** Confirmed harmless — the redteam-only
pipeline worked fine, validating iter-1's lesson 1.

**c. Choose guardrails deliberately.** Dropped the PII track. Settled on six harm
categories and ran a **pilot** (`numTests=5`) to measure per-category behavior:
306 tests → 66.5% pass. Two findings shaped the real run:
   - `base64` and `citation` strategies produced **no contrast** → dropped, leaving
     the final seven strategies.
   - Categories convert prompts→pairs at **different rates** → the training config
     uses *per-plugin* `numTests` (inverse-yield-weighted) so each category yields
     a balanced pair count.

**d. QLoRA vs bf16-LoRA.** Ran both, back-to-back, same data and hyperparameters
([`../my_notes.md`](../my_notes.md) and the noquant/LoRA comparison). Result: loss,
reward accuracy, and reward margins were all **within run-to-run noise** — because
quantization only affects the *frozen* base weights, not the LoRA adapter (the only
thing that trains, and used un-quantized at inference). The one real difference was
peak GPU memory: bf16 used ~1.85× QLoRA's (22.9 GB vs 12.4 GB). Both fit an A100, so
**bf16 LoRA was chosen** — marginally higher reward accuracy, slightly faster (no
dequant step), and compute was available. This ruled precision *out* as a lever for
the safety-gain problem, refocusing effort on data scale.

**e. Build a real benign dataset — and train on it.** The decision that mattered:
benign prompts are used **both** in training and eval. If they were eval-only, the
model could learn to always refuse and still "pass" adversarial eval. Benign pairs
(*helpful answer* > *refusal*) were added to the training mix so the gradient sees
a helpfulness signal too.

**f. Make eval symmetric with generation.** Since pairs are graded by red-team
eval, the adversarial *eval* was moved to red-team eval as well (fair comparison);
benign eval stays on a custom comply-rubric, since no red-team plugin generates
benign prompts.

**g. Size the real run.** ~1.5k pairs was judged the sweet spot — enough signal,
below the point where a single 8B adapter starts overfitting.

---

## Iter 3 — the full run

Everything above, at scale:

1. `redteam generate` (6 plugins × 7 strategies, per-plugin `numTests`) → **3,357
   tests**, split into six chunks for pod-sized eval sessions.
2. `build_pairs.py` → **1,439 pairs**; cybercrime under-yielded, so a top-up
   generation added +62 → **1,501 adversarial pairs**.
3. `build_benign_pairs.py` → 198 curated prompts graded for helpfulness → **197
   benign pairs** (1 dropped).
4. `merge_pairs.py` → **1,698 training pairs** (88.4% adversarial / 11.6% benign).
5. DPO: **bf16 LoRA, 1 epoch, ~19 min**. Loss 0.69 → 0.27, reward accuracy
   0.51 → 0.94, reward margins 0.001 → 2.80 — a decisive separation, unlike iter 1.
6. Held-out eval: a *separate* generate pass → deduped to **1,135 clean held-out
   attacks** + 150 held-out benign. 2,270 adversarial evals + 300 benign evals.

**Result** ([full tables](../results/results.md)): safety **+14 to +50pp across all
six categories** (n = 109–216, decisively above noise) at a benign-helpfulness cost
of **−6.7pp** — real, and concentrated in `weapons:ied` (20% over-refusal vs ≤8%
elsewhere).

**The clean confirmation of iter 1's diagnosis:** the difference between a
within-noise +2.7pp (344 weak pairs) and a decisive +14–50pp (1,698 contrast-
filtered pairs) is almost entirely **pair quantity and quality** — not model size,
quantization, or hyperparameters.

---

## Open threads

- **Over-refusal is uneven.** `weapons:ied` was trained hardest and over-refuses
  most — a per-category β or benign-ratio adjustment is the obvious next lever.
- **In-distribution eval.** Held-out attacks share the trained plugins/strategies;
  a *by-strategy* hold-out would test cross-family transfer.
- **Capability regression unmeasured.** An MMLU-style check is still owed.
- **Domain track.** A finance-specialized guardrail set was scoped but not run.
