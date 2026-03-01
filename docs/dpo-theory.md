# DPO from first principles

Why Direct Preference Optimization is the right tool here, and where its loss
comes from. This is the cleaned-up version of the derivation in
[`../my_notes.md`](../my_notes.md); the diagrams are the author's own working.

## The setup

DPO trains directly on preference triples — no separate reward model, no RL loop.
Each training example is a prompt `x` with two candidate responses: a preferred
(**chosen**) `y_c` and a dispreferred (**rejected**) `y_r`. In this project those
come from the pairing stage: for a jailbroken prompt, `y_c` is the model's safest
sampled response and `y_r` its least-safe one.

![DPO uses only the prompt, chosen, and rejected fields](../assets/image.png)

## From preference to loss

**1. Model the preference as a softmax over rewards.** The probability of
preferring `y_c` over `y_r` is the softmax of their rewards, which for two options
collapses to a sigmoid of the reward *difference*:

![Preference probability as a softmax of rewards](../assets/image%201.png)

$$P(y_c \succ y_r \mid x) = \sigma\big(r(x, y_c) - r(x, y_r)\big)$$

**2. Maximize that probability** — ideally to 1 — which, via cross-entropy loss
$-y_{\text{true}}\log y_{\text{pred}}$, gives:

$$\mathcal{L}(\theta, \theta_{\text{ref}}) = -\log \sigma\big(r(x, y_c) - r(x, y_r)\big)$$

where $\theta$ are the policy parameters and $\theta_{\text{ref}}$ the frozen
reference parameters.

**3. Define the reward implicitly** as the KL-scaled log-ratio between policy and
reference:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\theta_{\text{ref}}}(y \mid x)}$$

$\beta$ is the KL-divergence coefficient. This measures how much *more* likely the
policy is to produce `y` than the reference is. There is no separately-trained
reward model — the reward *is* this ratio.

**4. What is $\pi(y \mid x)$?** The probability the model assigns to the whole
response `y` given prompt `x`. For a single token it's one softmax entry; for a
multi-token response it's the product over positions (a sum in log-space):

![Sequence probability for a single token](../assets/image%202.png)

![Sequence probability for a multi-token output](../assets/image%203.png)

**5. Substitute** the reward back into the loss:

![The full DPO loss after substitution](../assets/image%204.png)

## What the gradient does

To reduce the loss, the term inside the sigmoid must grow — i.e. the chosen reward
must rise and the rejected reward must fall:

- $\pi_\theta(y_c \mid x)$ **up** relative to $\pi_{\theta_{\text{ref}}}(y_c \mid x)$
  — the policy becomes *more* likely than the reference to give the safe response.
- $\pi_\theta(y_r \mid x)$ **down** relative to $\pi_{\theta_{\text{ref}}}(y_r \mid x)$
  — the policy becomes *less* likely than the reference to give the harmful one.

The reference (base model, adapters disabled) is the anchor: $\beta$ controls how
far the policy is allowed to drift from it. Too high and nothing moves; too low and
the model can over-fit the preferences and drift into over-refusal — which is
exactly the tension the benign counter-slice and the β=0.1 choice manage.

## Why this fits guardrail tuning

The preference signal is naturally binary — a response either refuses the harmful
request or it doesn't — so ranking (chosen ≻ rejected) is a more honest encoding
than a scalar reward would be. DPO's implicit reward means no reward model to
train or mis-specify, and the reference-anchored KL term is precisely the knob for
"be safer, but don't regress into refusing everything." The
[training-run dynamics](../results/results.md#training-dynamics) —
`rewards/margins` climbing 0.001 → 2.80 while `rewards/accuracies` reaches 0.94 —
are this gradient separating chosen from rejected over the course of one epoch.

## References

- Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290) (2023)
- Cameron Wolfe, [DPO explainer](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
