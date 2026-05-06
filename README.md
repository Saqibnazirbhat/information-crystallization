# Information Crystallization

A bottom-up algorithm for **faithful, named, transferable** explanations of transformer predictions, via logit-margin budget allocation with online-faithfulness-gated growth.

**Author:** Saqib Nazir Bhat &nbsp;·&nbsp; **License:** [CC BY 4.0](LICENSE) &nbsp;·&nbsp; **Contact:** saqibnazirbhat3@gmail.com

---

## What this is

Existing circuit-discovery methods give partial answers — activation patching is faithful but unnamed; sparse feature circuits are named but post-hoc validate faithfulness; probes are fast but don't decompose the model. **Margin Crystallization** combines a margin-budget reformulation, an online faithfulness gate during growth, and first-class negative results into a single bottom-up procedure.

The full method is in **[`APPROACH.md`](APPROACH.md)**.

## The four design choices

| Choice | Why |
|---|---|
| **Logit margin as a budget**, not argmax preservation | Continuous measure; works on noisy small models; gracefully degrades to partial explanations |
| **Online faithfulness gate** during growth | Final S is faithful by construction; bidirectional necessity + sufficiency checked per accepted component |
| **Counterfactual contrast prompts** (X, X') | Removes default-token wins; defines what "explain x̂" means precisely |
| **Negative results as primary output** | When no sparse circuit exists, say so — the field underreports this |

## Repo layout

```
information-crystallization/
├── README.md                   this file
├── APPROACH.md                 method specification
├── LICENSE                     CC BY 4.0
└── experiments/
    ├── README.md               experiments overview
    ├── findings.md             preliminary results that motivated the design
    ├── results.md              early notebook results on GPT-2
    ├── seed_phase.ipynb        early seed-phase notebook
    ├── test_strict.py          per-scalar test on GPT-2 small
    ├── test_charitable.py      per-component test on Pythia-160M
    └── test_directeffect.py    direct-effect test on Pythia-410M
```

## Running the preliminary tests

```bash
pip install torch transformers numpy
cd experiments
python test_directeffect.py     # ~12 min, Pythia-410M, three metrics, direct-effect score
```

These motivated the design; they are not the algorithm. The algorithm is in `APPROACH.md` and is the next implementation milestone.

## Status

Conceptual specification complete. Implementation roadmap (`APPROACH.md` §Implementation roadmap) starts at v0 on Pythia-410M with a small custom SAE; v1 targets Pythia-1.4B with Gemma Scope; v2 is a public leaderboard with baselines.

## Citation

```bibtex
@misc{bhat2026margin,
  author       = {Bhat, Saqib Nazir},
  title        = {Information Crystallization: Faithful Bottom-up Circuit Discovery via Logit-Margin Allocation},
  year         = {2026},
  howpublished = {\url{https://github.com/Saqibnazirbhat/information-crystallization}},
  note         = {Independent research}
}
```
