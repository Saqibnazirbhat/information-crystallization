# Experiments

Preliminary empirical work for *A Joint Minimization Framework for Transformer Interpretability*.

## Current status

Only the **seed phase** (Phase 1 of Information Crystallization) has been implemented and tested. Phases 2 (greedy accretion), 3 (crystallization / prune), and 4 (verification loop) remain unimplemented. The paper's GPT-2 ablation protocol proposes N=1,000 sequences; this notebook runs N=17 as a sanity check on the seed computation only.

## Files

| File | What it does |
|------|-------------|
| [`seed_phase.ipynb`](seed_phase.ipynb) | Runs Phase 1 on GPT-2 small for 17 hand-curated sequences stratified by query type. Reports the parameter seed $j^*$ layer location and the input seed $t^*$ sequence position. |

## Requirements

```
torch
transformers
numpy
pandas
matplotlib
```

CPU runtime: ~2–4 minutes total. GPU: <30 seconds.

## Reproducibility

Deterministic (no sampling). Re-running the notebook produces identical outputs.

## Known limitations

- N=17 is not statistically significant. This is a feasibility check, not a result.
- GPT-2 small (12 layers, 124M params) may not generalize to larger models.
- Only the seed step is tested. Growth and prune phases would require substantially more engineering.
- The Sufficient Condition (Lipschitz-like perturbation bound) remains unproved.
