# Margin Crystallization

**Faithful Bottom-up Circuit Discovery via Online-Gated Logit-Margin Allocation**

Saqib Nazir Bhat &nbsp;·&nbsp; Independent research &nbsp;·&nbsp; [CC BY 4.0](LICENSE)

---

## Paper

The full paper is in [`paper.tex`](paper.tex) with bibliography in [`references.bib`](references.bib). Build with:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Abstract

Mechanistic circuit discovery in transformer language models lacks a shared faithfulness bar: methods disagree on stopping criteria, validate faithfulness post hoc, and inflate reported circuits on prompts where no sparse explanation exists. Existing tools each cover part of what an interpretability claim requires — ACDC produces faithful circuits but unnamed units; Sparse Feature Circuits produces named SAE-feature circuits but validates faithfulness only on the final set; AtP\* produces fast rankings but no subnetwork — and none reports distributed predictions as a typed return.

We propose **Margin Crystallization** (MC), a procedure that grows an explanation set bottom-up over SAE features in direct-effect order, admits each candidate only when a bidirectional β-margin gate passes on a same-type holdout, stops at fractional margin coverage `α · δ_full`, and otherwise returns the literal value `"distributed"` with diagnostics. Four contributions: (i) margin-coverage stopping; (ii) the online faithfulness gate; (iii) typed `"distributed"` return; (iv) thresholded co-occurrence-and-magnitude type-level fingerprinting.

As preliminary motivating evidence, three pilots on Pythia-410M show direct-effect ranking dominates gradient-magnitude 8/12 vs. 0/12 at KL < 1 with 50% kept, and no setting yields a sparse argmax-preserving subset off default-token completions. We propose the **Faithful Sparse Circuit Benchmark** (FSCB) — 800 contrast pairs across eight strata with three views and a pre-registered protocol — as the test bench under which the algorithmic contribution is licensed or dissolved.

## Repo layout

```
information-crystallization/
├── paper.tex                      paper source
├── references.bib                 46 entries
├── README.md                      this file
├── LICENSE                        CC BY 4.0
└── experiments/
    ├── README.md                  pilot overview
    ├── test_strict.py             pilot 1: per-scalar / GPT-2 small / argmax
    ├── test_charitable.py         pilot 2: per-component / Pythia-160M / argmax
    └── test_directeffect.py       pilot 3: per-component / Pythia-410M / argmax + top-3 + KL
```

The three pilot scripts produced the preliminary evidence cited in the paper §5 and Appendix D.

## Running the pilots

```bash
pip install torch transformers numpy
cd experiments
python test_directeffect.py     # ~12 min on CPU; produces the numbers in the paper's Figure 2 / Table 2
```

Outputs land in `experiments/outputs/` (gitignored).

## Citation

```bibtex
@misc{bhat2026margin,
  author       = {Bhat, Saqib Nazir},
  title        = {Margin Crystallization: Faithful Bottom-up Circuit Discovery via Online-Gated Logit-Margin Allocation},
  year         = {2026},
  howpublished = {\url{https://github.com/Saqibnazirbhat/information-crystallization}},
  note         = {Independent research}
}
```

## Contact

saqibnazirbhat3@gmail.com
