# Experiments

Preliminary empirical work for *A Joint Minimization Framework for Transformer Interpretability*.

## Current status

Only the **seed phase** (Phase 1 of Information Crystallization) has been implemented and tested. Phases 2 (greedy accretion), 3 (crystallization / prune), and 4 (verification loop) remain unimplemented. The paper's GPT-2 ablation protocol proposes N = 1,000 sequences; this notebook runs N = 17 as a feasibility check on the seed computation only.

## Files

| File | What it does |
|------|-------------|
| [`seed_phase.ipynb`](seed_phase.ipynb) | Runs Phase 1 on GPT-2 small for 17 hand-curated sequences stratified by query type. Reports the parameter seed $j^*$ layer location and the input seed $t^*$ sequence position. |
| [`results.md`](results.md) | Narrative report of the findings, including the weight-tying caveat on $j^*$ and the attention-concentration signal on $t^*$. |

## Requirements

```
torch
transformers
numpy
pandas
matplotlib
```

Pinned versions are printed by the first code cell of the notebook. CPU runtime: ~2–4 minutes total. GPU: <30 seconds.

## Reproducibility

- `torch.manual_seed(0)` and `np.random.seed(0)` at the top of the notebook.
- Model loaded in `eval()` mode; no dropout.
- Pure forward + backward; no sampling.
- Re-running produces identical outputs.

## Known limitations

- **N = 17 is not statistically significant.** This is a feasibility check, not a study. The long-range bucket has only n = 3.
- **Weight-tying confound.** GPT-2 ties `lm_head.weight` to `transformer.wte.weight`. The $j^*$ result therefore cannot cleanly test the paper's $S_L \cup S_{\text{supp}}$ decomposition. A replication on an untied architecture (Pythia-160M, OPT-125M) is the planned next experiment.
- **Seed step only.** Growth and prune phases would require substantially more engineering.
- **GPT-2 small (12 layers, 124M params)** may not generalize to larger models.
- **The Sufficient Condition** (Lipschitz-like perturbation bound) remains unproved.
