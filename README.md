# A Joint Minimization Framework for Transformer Interpretability

**Information Crystallization** — a bottom-up algorithm for discovering minimal circuits in transformer language models.

**Author:** Saqib Nazir Bhat
**Date:** March 2026
**Affiliation:** None. Solo, independent research.
**License:** [CC BY 4.0](LICENSE)

---

## Abstract

Existing interpretability methods (activation patching, structured pruning, circuit discovery) work top-down: they start with the full model and attempt to remove components until a minimal sufficient set remains. The search space is exponential in the model size. This paper reformulates the problem bottom-up.

Given an observed output token $\hat{x}$, find the minimal parameter subset $S \subseteq [d]$ and input-position subset $T \subseteq [n]$ such that the argmax over the restricted model still equals $\hat{x}$. The central observation is that **softmax does not affect argmax**: preserving the predicted token requires only preserving the sign of the logit margin $\delta > 0$, not the full output distribution. This relaxation replaces a density-matching condition with a margin inequality, admitting substantially sparser solutions.

Building on this relaxation, I introduce **Information Crystallization**: an $O(k \cdot d)$ algorithm that seeds with the single most gradient-sensitive parameter and input position, greedily grows $S$ and $T$ via restricted-subnetwork gradients, and iteratively prunes redundancies until a fixed point. I decompose $S$ into a final-block component $S_L$ and a residual-stream support $S_{\text{supp}}$, connecting the sparsity conjecture to the Lottery Ticket Hypothesis and to Elhage et al.'s residual-stream framework. I identify the $T$-dependence of $S^*$ as the primary open problem and specify a GPT-2 ablation protocol for empirical verification.

Empirical validation on large-scale models has not yet been conducted.

---

## Artifacts

| File | Description |
|------|-------------|
| [`paper.tex`](paper.tex) | LaTeX source — full paper |
| [`references.bib`](references.bib) | BibTeX bibliography |
| [`docs/paper.pdf`](docs/paper.pdf) | Compiled PDF |
| [`docs/index.html`](docs/index.html) | Web-rendered version with MathJax equations |
| [`LICENSE`](LICENSE) | CC BY 4.0 |

## Live page

Once GitHub Pages is enabled, the paper is viewable at:
**https://saqibnazirbhat.github.io/information-crystallization/**

## Building the PDF

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or use [Overleaf](https://overleaf.com) — upload `paper.tex` and `references.bib`.

---

## Citation (BibTeX)

```bibtex
@misc{bhat2026information,
  author       = {Bhat, Saqib Nazir},
  title        = {A Joint Minimization Framework for Transformer Interpretability},
  year         = {2026},
  month        = {March},
  howpublished = {\url{https://github.com/Saqibnazirbhat/information-crystallization}},
  note         = {Independent research}
}
```

---

## Authorship and provenance

All mathematical formulations, algorithmic ideas, and theoretical claims in this repository are original work by Saqib Nazir Bhat. AI tools were used for English-language editing and LaTeX typesetting; the research content is the author's own. Git commit timestamps in this repository serve as evidence of authorship priority.

Contact: saqibnazirbhat3@gmail.com
