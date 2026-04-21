# Seed-Phase Results on GPT-2 Small

**Sample:** N = 17 hand-curated sequences, stratified into four query types: syntactic continuation (5), factual recall (5), negation-heavy (4), long-range syntactic dependency (3).
**Model:** GPT-2 small (124M parameters, 12 transformer blocks).
**Implementation:** [`seed_phase.ipynb`](seed_phase.ipynb).

N = 17 is a feasibility check, not a statistically significant study. Sub-bucket sizes are small — the long-range bucket contains only 3 sequences.

## Finding 1 — Parameter seed j\*: inconclusive, with a nuanced reading

In 17 / 17 sequences, the seed parameter tensor was `transformer.wte.weight`.

### The inconclusive part

This result is **methodologically uninformative for the S<sub>L</sub> vs S<sub>supp</sub> decomposition**. GPT-2 ties `lm_head.weight` to `transformer.wte.weight` — the input-embedding matrix and the unembedding projection share the same parameter tensor. Any gradient flowing back through the unembedding lands in `wte.weight`, so the experiment cannot distinguish "seed localizes in the input embedding" from "seed localizes in the unembedding projection." A clean test requires either (a) a model without weight-tying — e.g., Pythia-160M, OPT — or (b) an experimental design that separately attributes gradient mass to the embedding-path and unembedding-path through `wte.weight`.

### A stronger reading that survives the artifact

The fact that 17/17 seeds landed on *one specific parameter tensor* — out of ~140 distinct tensors across 12 transformer blocks — is not uninformative. It says the direct gradient of log P(x̂ | X) is concentrated on the final linear projection into logit space, and that this direct contribution dominates the residual-stream-mediated contributions at the seed step. If that concentration generalizes to untied architectures, it implies that S<sub>L</sub> (which contains the unembedding by construction) is the *empirically dominant* component of the minimal set, and that S<sub>supp</sub> serves as a refinement rather than an equal partner. This reading is consistent with the paper's framing but goes beyond what the weight-tied experiment can formally establish.

## Finding 2 — Input-position seed t\* (meaningful; refutes suffix-only heuristic)

No weight-tying confound here. Across the 17 sequences:

| Statistic | Value |
|---|---|
| t\* is the final (last) input token | **11.8%** (2 / 17) |
| t\* is in the first 3 tokens | **35.3%** (6 / 17) |
| Mean distance of t\* from the end of the sequence | **3.88 tokens** |

Stratified by query type:

| Query type | n | Mean distance of t\* from end |
|---|---|---|
| factual | 5 | 2.00 |
| syntactic | 5 | 2.80 |
| negation | 4 | 4.75 |
| **long-range** | **3** | **7.67** |

**Reading:** the suffix-only heuristic (the paper's initial conjecture that T is approximately the suffix `{x_{n-k}, ..., x_n}` with small k) is inconsistent with this sample. Only about one in eight sequences places t\* on the final token. Long-range sequences put t\* near the subject noun at the beginning of the sequence (`scientists`, `cats`, subject-of-`is`). This is exactly the failure mode the paper's revised formulation predicts, and supports the formalized Conjecture on attention-concentration identification.

**Caveat:** the long-range bucket has n = 3. A mean of 7.67 from three samples is a direction, not a measurement. The qualitative claim (that t\* tracks the subject noun) survives on individual inspection of the three sequences; the numerical mean should not be treated as an estimate. Scaling to N ≥ 100 per bucket is a prerequisite for any statistical claim.

## The mathematical displays

Logit margin:

$$\delta \;=\; f_\theta(X)[\hat{x}] \;-\; \max_{v \neq \hat{x}} f_\theta(X)[v]$$

Seed-phase definitions used in the notebook:

$$j^* \;=\; \arg\max_j\, |g_j|, \quad g \;=\; \nabla_\theta \log P_\theta(\hat{x} \mid X)$$

$$t^* \;=\; \arg\max_t\, \bigl\| \partial f_\theta / \partial x_t \bigr\|_2$$

(Display-math blocks render reliably on GitHub; inline `$...$` next to hyphens or `\text{}` sometimes does not — these results are written with both concerns in mind.)

## Relation to prior work

- **Attribution Patching** (Syed et al., 2023) uses gradient-based approximations for the same reason this paper does: to avoid the exponential ablation cost. Attribution patching scores every parameter once; this paper grows a sparse set from one seed. Complementary, not competing.
- **ACDC** (Conmy et al., 2023) is the canonical automated-circuit-discovery baseline, operating top-down.
- **Integrated Gradients** (Sundararajan et al., 2017) is the foundational gradient-attribution axiomatic framework; the seed step here uses a non-integrated single-point gradient.

## What these results support and do not support

**Support (preliminary):**

- The revised T-identification claim: attention concentration over positional recency (qualitatively, on the long-range bucket).
- The general feasibility of the seed step on real transformer models (one backward pass, interpretable outputs, deterministic).
- A stronger-reading hypothesis that S<sub>L</sub> is the empirically dominant component of the minimal parameter set.

**Do not support:**

- The S<sub>L</sub>-vs-S<sub>supp</sub> split itself — untestable on weight-tied GPT-2.
- Any claim about the full (S, T) minimal set — Phases 2–4 (growth, prune, verify) are not implemented.
- Any claim of statistical significance — N = 17 is a sanity check, not a study.
- Generalization to frontier-scale models — untested.

## Immediate next steps

1. **Re-run the j\* experiment on Pythia-160M.** Pythia is the closest analogue of GPT-2 small in scale and has untied embeddings. This gives a clean S<sub>L</sub>-vs-S<sub>supp</sub> test and a direct check on the "direct vs. residual-path gradient" stronger reading above.
2. **Scale to N = 100+ per query type** and apply a permutation test on the positional distribution of t\* to convert the current directional claim into a statistical one.
3. **Implement Phase 2 (greedy growth of S from the seed)** and measure |S| at the argmax-flip boundary. This is the next algorithmic milestone.
