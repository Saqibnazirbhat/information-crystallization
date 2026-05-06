# Margin Crystallization

A bottom-up algorithm for finding **faithful, named, transferable** explanations of transformer predictions. The unit of explanation is a feature subset; the criterion is logit-margin allocation; the procedure is online-faithfulness-gated growth; negative results are first-class.

## The problem

Given a transformer M, a prompt X, and its prediction x̂, find a subnetwork S that *explains* the prediction in a precise sense:

| Property | What it requires |
|---|---|
| **Faithful** | M actually uses S to compute x̂ — validated by both *necessity* (ablate S, behavior shifts away from x̂) and *sufficiency* (patch S, behavior shifts toward x̂) |
| **Named** | every component in S has a semantic label (via a feature decomposition like SAEs) |
| **Transferable** | S explains x̂ on held-out prompts of the same type, not just this one |
| **Quantified** | each component's contribution is a continuous number, not a binary in/out |

Existing approaches give partial answers. Activation patching (Heimersheim & Nanda 2024) is faithful but unnamed. Sparse feature circuits (Marks et al. 2024) are named but post-hoc validate faithfulness. ACDC (Conmy et al. 2023) is top-down and slow at scale. Probes are fast but don't name circuits. None do all four together.

## The reframing — from minimum-set to margin-budget

Most circuit-discovery asks "find the smallest `|S|` that preserves argmax." This formulation is brittle: under almost any ablation, low-confidence small-model predictions flip argmax due to noise; sparse argmax-preserving subsets exist mostly for default-token completions; and "preserved or not" throws away most of the signal.

**The reframing:** treat the logit margin

```
δ  =  logit(x̂)  −  logit(x̂')
```

against a counterfactual prediction x̂' as a *budget*. Each component spends some fraction of it. The explanation is the smallest set capturing α of the budget (typically α = 0.9), not the smallest set preserving a binary flag.

To first order (Syed, Rager, Conmy 2024), per component k with activation `a_k`:

```
contribution_k(X, X')  =  (a_k(X)  −  a_k(X'))  ·  ∂δ / ∂a_k
```

These contributions are signed and roughly sum to δ. The algorithm becomes resource allocation: rank components by `|contribution|`, accept until α · δ is captured, gate every acceptance by an explicit faithfulness check.

The contrast prompt X' is load-bearing. "Why x̂ on X?" is underspecified and admits unigram-bias wins. "Why x̂ instead of x̂' on (X, X')?" is precise: the explanation is what differs along the dimension you cared about.

## The algorithm — online-faithfulness-gated growth

```
algorithm  margin_crystallize(M, (X, x̂), (X', x̂'), F, holdout, α=0.9, β=0.7):
    """
    M         transformer
    (X, x̂)    anchor prompt and its prediction
    (X', x̂')  counterfactual prompt and prediction (x̂ ≠ x̂')
    F         feature decomposition over residual stream (SAE, components, etc.)
    holdout   held-out prompts of the same type as X
    α         margin-coverage target (default 0.9)
    β         faithfulness rate threshold on holdout (default 0.7)

    Returns   either (S, role_labels, faithfulness_summary) or
              ("distributed", failure_reason)
    """
    δ_full  =  margin(M, X, x̂, x̂')
    contribs  =  direct_effect_per_feature(M, F, X, X', x̂, x̂')
    candidates  =  sort_by_abs(contribs, descending)

    S  =  empty
    captured  =  0
    rejected  =  []

    for f in candidates:
        S_new  =  S ∪ {f}
        Δ  =  margin(M, X, x̂, x̂' | restricted to S_new)  −  captured
        if Δ ≤ 0:
            continue                          # adding this didn't move the margin

        nec  =  necessity_rate(M, S_new, holdout, X')
        suf  =  sufficiency_rate(M, S_new, holdout, X)
        if min(nec, suf) < β:
            rejected.append((f, nec, suf))    # faithfulness gate
            continue

        S        =  S_new
        captured =  margin(M, X, x̂, x̂' | restricted to S)
        if captured ≥ α · δ_full:
            return  S, role_labels(S, F), faithfulness_summary(S, holdout)

    return  "distributed", {"captured_at_exhaustion": captured / δ_full,
                             "rejected_features": rejected}
```

Three things this does that other circuit-discovery algorithms do not:

1. **Online faithfulness gate.** Components are admitted only if they preserve bidirectional faithfulness on held-out same-type prompts *at the moment they are added*. The final `S` is faithful by construction, not by post-hoc validation. Faithfulness rate is therefore a property of `S`, not a separate report.

2. **Margin coverage instead of argmax preservation.** The stopping criterion is "captured ≥ α · δ_full," a continuous quantity. This works on small/low-confidence models where argmax flips are noise-dominated. It also produces *partial* explanations gracefully: an `S` capturing 60% of δ is informative even if it doesn't reach α.

3. **First-class negative results.** When no faithfulness-gated `S` reaches α, the algorithm returns `"distributed"` with diagnostics. The field underreports cases where predictions have no sparse explanation; treating them as a primary output rather than a failure mode is a contribution to the faithfulness-benchmarks literature.

## Type-level fingerprinting (extension)

Run the algorithm on N prompts of the same type. For each prompt i let `S_i` be its explanation. The **type circuit**

```
S_T  =  { f : f appears in ≥ K out of N per-prompt sets,
              with median margin contribution ≥ τ across those sets }
```

is the components that consistently spend the budget across the type. `S_T` is a *generalizable* explanation; per-prompt `S_i` may be prompt-specific compression. Reporting `S_T` plus per-prompt deviations is the unit of an interpretability claim about a query type.

## What is novel here, what is borrowed, and what is the bet

**Borrowed (cited, not claimed):**
- SAE feature units (Bricken et al. 2023; Templeton et al. 2024; Lieberum et al. 2024).
- Direct-effect / attribution-patching score (Syed, Rager, Conmy 2024; Hanna et al. 2024).
- Bidirectional faithfulness as a quality criterion (Hanna et al. 2024; Marks et al. 2024).
- Counterfactual contrast prompts (Geiger et al. causal abstraction line).

**Novel here:**
- The *margin-budget* formulation, replacing minimum-set with smallest-α-coverage.
- The *online faithfulness gate during growth*, not post-hoc.
- *Type-level fingerprinting via thresholded intersection across same-type prompts*.
- *Negative results as a first-class output*, not a failure mode.

**The bet:** these four design choices are individually small upgrades over current practice, but their *combination* yields a circuit-discovery method whose output (a) is faithful by construction, (b) generalizes by construction, (c) gracefully degrades to partial or null explanations, and (d) is comparable across runs because margin coverage is continuous.

## What would falsify it

1. **The growth + gate contributes nothing.** A simpler baseline (single-shot direct-effect ranking + post-hoc faithfulness pruning) matches Margin Crystallization on a held-out benchmark. If true, the algorithmic novelty dissolves and the contribution is the framing only.
2. **The online gate is too strict.** Faithfulness rate β = 0.7 rejects components that should be included, and the algorithm returns `"distributed"` on cases where other methods produce sparse circuits (e.g., the IOI task).
3. **Margin coverage doesn't transfer.** `S_T` (type fingerprint) captures < α on >25% of held-out same-type prompts, indicating per-prompt overfitting.
4. **Direct-effect linearization fails at scale.** First-order contributions don't sum to δ within tolerance on Pythia-1.4B+, breaking the budget framing.

## Implementation roadmap

1. **v0 (CPU, Pythia-410M).** Implement on the existing test scaffolding in [`experiments/`](experiments/). Margin coverage and online gate first. SAE features second (use a small custom top-k SAE on residual stream layer 12).
2. **v1 (GPU, Pythia-1.4B + Gemma Scope).** Use a frontier-class SAE (Gemma Scope). Build a contrast-pair benchmark — 10 query types × 100 pairs each, including IOI-class tasks where ground-truth circuits are known.
3. **v2 (leaderboard).** Compare against ACDC, sparse feature circuits, single-shot direct effect, magnitude pruning. Report margin-coverage curves, faithfulness rates, transfer rates, and `"distributed"` fractions per query type.
4. **Public release.** Publish the contrast-pair benchmark (CC-BY) and the algorithm (MIT/CC-BY). The benchmark contribution may matter more than the method — the field is methods-rich and benchmarks-poor.

## Citations

- Bricken et al. 2023, *Towards Monosemanticity* (Anthropic)
- Templeton et al. 2024, *Scaling Monosemanticity to Claude 3 Sonnet* (Anthropic)
- Lieberum et al. 2024, *Gemma Scope* (DeepMind)
- Wang et al. 2022, *Interpretability in the Wild* (IOI)
- Conmy et al. 2023, *Towards Automated Circuit Discovery* (ACDC)
- Heimersheim & Nanda 2024, *How to use and interpret activation patching*
- Syed, Rager, Conmy 2024, *Attribution Patching Outperforms Automated Circuit Discovery*
- Hanna et al. 2024, *Have Faith in Faithfulness*
- Marks et al. 2024, *Sparse Feature Circuits*
- Geiger et al., causal abstraction / interchange interventions line
- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*
