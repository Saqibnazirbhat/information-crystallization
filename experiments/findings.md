# Preliminary findings — what shaped the design

These three exploratory runs tested early versions of bottom-up minimization with various score functions, granularities, and metrics. Their job was to identify which design choices in the final algorithm ([`../APPROACH.md`](../APPROACH.md)) are load-bearing and which are not.

## Setup

| Run | Script | Model | Granularity | Reference | Metric |
|---|---|---|---|---|---|
| Per-scalar | [`test_strict.py`](test_strict.py) | GPT-2 small (124M) | per-scalar | zero | argmax |
| Per-component | [`test_charitable.py`](test_charitable.py) | Pythia-160M (162M) | per-component (heads + MLP neurons) | mean over baseline corpus | argmax |
| Direct-effect | [`test_directeffect.py`](test_directeffect.py) | Pythia-410M (405M) | per-component | mean | argmax + top-3 + KL |

Conditions across runs:
- **A** — `\|∇ log P(x̂\|X)\|` (gradient saliency)
- **C** — variance baseline (prompt-agnostic)
- **D** — direct effect: `\|(a − a_mean) · ∂L/∂a\|` (Syed/Nanda 2024)
- **E** — bottom-up structure with DE seed + DE-on-restricted growth
- **F** — variance seed + DE-on-restricted growth

12 prompts × 4 query types per run.

## Key results

### Per-scalar run (GPT-2 small)

| Condition | Sparse argmax preservation found? | Comment |
|---|---|---|
| A — `\|∇ log P\|` | **No** at any tested fraction (down to 10% kept) | Loss-gradient ranking selects against parameters needed for output projection |
| C — magnitude | Yes but not sparse | Mean keep-frac at preservation = 0.80 (range 0.50–0.99) |
| Bottom-up with `\|∇\|` | **No** | Trajectory grew 0.5% → 60% → jumped to 100% (saturation; tied-zero gradients on remaining candidates make ranking degenerate) |

### Per-component run (Pythia-160M, mean reference)

| Condition | argmax preserved at any frac ≤ 0.50 |
|---|---|
| A — `\|∇\|` | 0 / 12 |
| C — variance | 3 / 12 |
| Bottom-up with `\|∇\|` | 2 / 12 (both at 25–49% kept, both default-token wins) |

### Direct-effect run (Pythia-410M, three metrics)

Wins out of 12:

| | argmax | top-3 | KL < 1.0 |
|---|--:|--:|--:|
| A — `\|∇\|` | 0 | 1\* | 0 |
| C — variance | 3 | 4 | 5 |
| **D — direct effect** | **2** | **4** | **8** |
| E — bottom-up with DE seed + DE grow | 3 | — | — |
| F — variance seed + DE grow | 2 | — | — |

\* the lone A win is a `" the"` default-token completion.

## Five design lessons

### 1. Loss-gradient magnitude is the wrong score
`|∇ log P(x̂|X)|` underperforms in 36/36 prompts across the three runs. Likely cause: low-gradient parameters include components whose ablation has *non-linear* effects — e.g., unembedding rows for tokens not in context. Direct effect (Syed/Nanda 2024), which weights gradient by the activation difference between counterfactuals, dominates on every metric in the third run. **The final algorithm uses direct effect.**

### 2. Argmax preservation is too strict for noisy small models
Pythia-410M margins range 0.06–2.26 (median ~0.5). Many "argmax flips" under partial ablation are noise. The KL-buffer metric (KL < 1.0) reveals real structure where argmax sees only failure: 8/12 prompts under direct-effect scoring preserve KL at 50% kept. **The final algorithm uses logit-margin coverage, not argmax preservation.**

### 3. Per-scalar is the wrong granularity
Magnitude pruning needs ~80% of GPT-2 small's parameters kept to preserve argmax. There is no semantic story per scalar. **The final algorithm operates on SAE features over the residual stream.**

### 4. Counterfactual contrast removes default-token artifacts
The 0.025-keep-frac "win" on `" the"` in the direct-effect run is unigram bias, not a discovered circuit. Direct effect on the *logit difference* between counterfactuals removes this kind of artifact. **The final algorithm requires a contrast prompt pair (X, X').**

### 5. Sparse argmax-preserving subsets often do not exist on small models
Across 36 prompts × 5 conditions, no setting found a sparse argmax-preserving subnetwork outside default-token cases. Best non-trivial keep-fraction observed was 0.36 (~36,000 components on Pythia-410M). On the strict per-scalar run, magnitude needed 50–99% kept; loss-gradient ranking never preserved at all. This is informative: predictions on small models are often genuinely *distributed*, not sparse. **The final algorithm reports `"distributed"` as a first-class output when no faithfulness-gated S reaches the margin-coverage target.**

## What did hold up

- **Bottom-up structure (seed → grow → prune).** With direct-effect scoring, bottom-up algorithms preserve argmax on 4/12 prompts at 36–56% kept on Pythia-410M. Strictly better than the per-scalar run. The skeleton is fine; the score function and the metric were the bugs.
- **Joint (parameter, input) framing.** Components and input positions are coupled through the residual stream; treating them as a single object is the right intuition.
- **Margin-preservation as the criterion.** Theoretically right (preserving sign of δ is strictly weaker than matching the full distribution); empirically loose enough to admit partial explanations.

## Reproducing

```bash
pip install torch transformers numpy

cd experiments
python test_strict.py             # ~50 min, GPT-2 small
python test_charitable.py         # ~3 min,  Pythia-160M
python test_directeffect.py       # ~12 min, Pythia-410M
```

Outputs land in `experiments/outputs/` (gitignored). Logs are written alongside each script.

## What's next

The findings above motivate Margin Crystallization (see [`../APPROACH.md`](../APPROACH.md)). The next implementation step is the v0 prototype on Pythia-410M with a small custom top-k SAE over residual stream layer 12.
