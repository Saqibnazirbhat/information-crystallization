# Experiments

Preliminary empirical work that shaped the *Margin Crystallization* algorithm specification in [`../APPROACH.md`](../APPROACH.md).

## Files

| File | What it does |
|------|-------------|
| [`findings.md`](findings.md) | Consolidated results across the three test scripts, with the design lessons each one produced |
| [`results.md`](results.md) | Early notebook results on GPT-2 small (seed-phase exploration) |
| [`seed_phase.ipynb`](seed_phase.ipynb) | Original seed-phase notebook on GPT-2 small for 17 hand-curated sequences |
| [`test_strict.py`](test_strict.py) | Per-scalar test on GPT-2 small with zero-reference ablation |
| [`test_charitable.py`](test_charitable.py) | Per-component test on Pythia-160M with mean-activation reference |
| [`test_directeffect.py`](test_directeffect.py) | Direct-effect scoring test on Pythia-410M with three metrics (argmax + top-3 + KL) |

## Requirements

```
torch
transformers
numpy
```

CPU-only is fine. Runtimes:
- `seed_phase.ipynb`: ~2–4 min
- `test_strict.py`: ~50 min
- `test_charitable.py`: ~3 min
- `test_directeffect.py`: ~12 min

## Reproducibility
- All scripts seed `torch.manual_seed(0)`.
- Models loaded in `eval()` mode, no dropout, pure forward + backward, no sampling.
- Re-running produces identical outputs modulo HF Hub transient errors.
- Outputs land in `experiments/outputs/` (gitignored).

## Known limitations
- Sample sizes are small (12 prompts in each script, 17 in the notebook). These are research-quality probes, not statistical studies.
- Pythia-410M is the largest model run. The next defensible scale is Pythia-1.4B–6.9B.
- The Margin Crystallization algorithm itself has not been run yet. This experiments tree is the empirical motivation; the v0 prototype is the next milestone.
