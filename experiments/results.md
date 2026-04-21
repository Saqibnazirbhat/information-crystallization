# Seed-Phase Results on GPT-2 Small

**Sample:** N = 17 hand-curated sequences, stratified into four query types: syntactic continuation (5), factual recall (5), negation-heavy (4), long-range syntactic dependency (3).
**Model:** GPT-2 small (124M parameters, 12 transformer blocks).
**Implementation:** [`seed_phase.ipynb`](seed_phase.ipynb).

N = 17 is a feasibility check, not a statistically significant study.

## Finding 1 — Parameter seed $j^*$ (inconclusive due to weight-tying)

In 17 / 17 sequences, the seed parameter tensor was `transformer.wte.weight`.

This result is **methodologically uninformative** because GPT-2 ties `lm_head.weight` to `transformer.wte.weight` — the input-embedding matrix and the unembedding projection share the same parameter tensor. Any gradient flowing back through the unembedding lands in `wte.weight`, so the experiment cannot distinguish "seed localizes in the input embedding" from "seed localizes in the unembedding projection."

The result is **consistent with the $S_L$ conjecture** (the unembedding is in $S_L$ by construction), but does not test it cleanly. A proper test of the $S_L$-vs-$S_{\text{supp}}$ decomposition requires either (a) a model without weight-tying — e.g., Pythia, OPT — or (b) an experimental design that separately attributes gradient mass to the embedding-path and the unembedding-path through `wte.weight`. Both are open next steps.

## Finding 2 — Input-position seed $t^*$ (meaningful; refutes suffix-only heuristic)

No weight-tying confound here. Across the 17 sequences:

| Statistic | Value |
|---|---|
| $t^*$ is the final (last) input token | **11.8%** (2 / 17) |
| $t^*$ is in the first 3 tokens | **35.3%** (6 / 17) |
| Mean distance of $t^*$ from the end of the sequence | **3.88 tokens** |

Stratified by query type:

| Query type | Mean distance of $t^*$ from end |
|---|---|
| factual | 2.0 |
| syntactic | 2.8 |
| negation | 4.75 |
| **long-range** | **7.67** |

**Reading:** the suffix-only heuristic (the paper's initial conjecture that $T \approx \{x_{n-k}, \ldots, x_n\}$ with small $k$) is refuted on this sample. Only about one in eight sequences places $t^*$ on the final token. Long-range sequences in particular put $t^*$ near the subject noun at the beginning of the sequence (`scientists`, `cats`, subject-of-`is`). This is exactly the failure mode the paper's revised formulation predicts (Section 4.2, failure modes of the suffix assumption), and supports the corrected claim that $t^*$ is identified by attention weight concentration rather than by positional recency.

## What these results support and do not support

**Support (preliminary):**

- The revised $T$-identification claim: attention concentration over positional recency.
- The general feasibility of the seed step on real transformer models (one backward pass, interpretable outputs, deterministic).

**Do not support:**

- The $S_L$-vs-$S_{\text{supp}}$ decomposition — untestable on weight-tied GPT-2.
- Any claim about the full $(S, T)$ minimal set — Phases 2–4 (growth, prune, verify) are not implemented.
- Any claim of statistical significance — N = 17 is a sanity check, not a study.
- Generalization to frontier-scale models — untested.

## Immediate next steps

1. Re-run the parameter-seed experiment on a model **without weight-tying** (Pythia-160M is the closest analogue of GPT-2 small in scale and is untied). This gives a clean $S_L$ test.
2. Expand to N = 100+ per query type so the $t^*$ distribution claim has statistical weight.
3. Implement Phase 2 (greedy growth of $S$ from the seed) and measure $|S|$ at the argmax-flip boundary.
