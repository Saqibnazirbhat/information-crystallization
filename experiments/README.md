# Pilots

Three preliminary tests cited in the paper (§5 and Appendix D) as motivating evidence for the four design choices in Margin Crystallization.

| Script | Model | Granularity | Reference | Metrics |
|---|---|---|---|---|
| [`test_strict.py`](test_strict.py)         | GPT-2 small (124M)    | per-scalar     | zero | argmax |
| [`test_charitable.py`](test_charitable.py) | Pythia-160M (162M)    | per-component  | mean | argmax |
| [`test_directeffect.py`](test_directeffect.py) | Pythia-410M (405M) | per-component | mean | argmax + top-3 + KL |

12 prompts × 4 query types per run. The numbers reported in the paper come from `test_directeffect.py`.

## Headline results (recap of paper §5)

Wins out of 12 on Pythia-410M:

| | argmax | top-3 | KL < 1.0 |
|---|--:|--:|--:|
| `\|∇ log P\|` (gradient saliency)            | 0 | 1\* | 0 |
| variance ranking                             | 3 | 4 | 5 |
| **direct effect (single-shot)**              | **2** | **4** | **8** |
| paper structure + DE seed/grow               | 3 | — | — |
| variance seed + DE grow                      | 2 | — | — |

\* the lone gradient-saliency win is a `" the"` default-token completion. See paper §5 for caveats.

## Requirements

```
torch
transformers
numpy
```

CPU-only is fine. Runtimes: `test_strict.py` ~50 min, `test_charitable.py` ~3 min, `test_directeffect.py` ~12 min.

## Reproducibility

- All scripts seed `torch.manual_seed(0)`.
- Models loaded in `eval()` mode, no dropout, pure forward + backward, no sampling.
- Outputs land in `experiments/outputs/` (gitignored).

## Limitations

- 12 prompts is a research-quality probe, not a statistical study. Paper §5 frames these as preliminary; the proposed FSCB benchmark (800 contrast pairs across 8 strata) is the test bench under which the algorithmic contribution is licensed or dissolved.
- Pythia-410M margins are 0.06–2.26 raw logits (median ~0.5); argmax preservation is noise-dominated in this regime, motivating the soft margin-coverage criterion adopted in the paper.
