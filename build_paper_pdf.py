"""Generate a dash-free PDF of the paper directly from the paper content.

This is a reportlab best-effort renderer. It will not match pdflatex output
quality (no fine kerning, no ligatures, algorithm environment is styled
text rather than a floating algorithm box), but it produces a complete,
readable, dash-free PDF of the current paper.tex content. For a publication-
quality PDF, compile paper.tex on Overleaf.
"""
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem,
    Table, TableStyle, PageBreak,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "docs", "paper.pdf")

FONTS = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Body",      os.path.join(FONTS, "calibri.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", os.path.join(FONTS, "calibrib.ttf")))
pdfmetrics.registerFont(TTFont("Body-It",   os.path.join(FONTS, "calibrii.ttf")))
pdfmetrics.registerFont(TTFont("Body-BI",   os.path.join(FONTS, "calibriz.ttf")))
registerFontFamily("Body", normal="Body", bold="Body-Bold",
                   italic="Body-It", boldItalic="Body-BI")

pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONTS, "consola.ttf")))
pdfmetrics.registerFont(TTFont("Sym",  os.path.join(FONTS, "seguisym.ttf")))


def m(s): return f'<font name="Sym">{s}</font>'


ACCENT = HexColor("#1a4a7a")
MUTED  = HexColor("#444")

S = {
    "title":    ParagraphStyle("title", fontName="Body-Bold", fontSize=18,
                               leading=22, alignment=TA_CENTER, textColor=black),
    "subtitle": ParagraphStyle("subtitle", fontName="Body-It", fontSize=13,
                               leading=16, alignment=TA_CENTER, textColor=MUTED,
                               spaceAfter=10),
    "author":   ParagraphStyle("author", fontName="Body", fontSize=11, leading=14,
                               alignment=TA_CENTER, textColor=black, spaceAfter=2),
    "affil":    ParagraphStyle("affil", fontName="Body-It", fontSize=9.5,
                               leading=12, alignment=TA_CENTER, textColor=MUTED,
                               spaceAfter=14),
    "abstract_h":ParagraphStyle("abs_h", fontName="Body-Bold", fontSize=10.5,
                                leading=13, alignment=TA_CENTER, textColor=black,
                                spaceBefore=4, spaceAfter=4),
    "abstract": ParagraphStyle("abstract", fontName="Body", fontSize=10,
                               leading=13, alignment=TA_JUSTIFY, textColor=black,
                               leftIndent=22, rightIndent=22, spaceAfter=12),
    "section":  ParagraphStyle("section", fontName="Body-Bold", fontSize=13,
                               leading=16, alignment=TA_LEFT, textColor=ACCENT,
                               spaceBefore=14, spaceAfter=4),
    "subsec":   ParagraphStyle("subsec", fontName="Body-Bold", fontSize=11,
                               leading=14, alignment=TA_LEFT, textColor=black,
                               spaceBefore=8, spaceAfter=3),
    "para":     ParagraphStyle("para", fontName="Body", fontSize=10.5,
                               leading=14, alignment=TA_JUSTIFY, textColor=black,
                               spaceAfter=6),
    "eq":       ParagraphStyle("eq", fontName="Body-It", fontSize=10.5,
                               leading=14, alignment=TA_CENTER, textColor=black,
                               spaceBefore=4, spaceAfter=6),
    "algo_head":ParagraphStyle("algo_head", fontName="Body-Bold", fontSize=10,
                               leading=13, textColor=ACCENT, spaceBefore=6,
                               spaceAfter=3),
    "algo":     ParagraphStyle("algo", fontName="Mono", fontSize=9, leading=12,
                               textColor=black, leftIndent=16, rightIndent=16,
                               spaceAfter=10),
    "ref":      ParagraphStyle("ref", fontName="Body", fontSize=9.5, leading=12,
                               textColor=black, leftIndent=22, bulletIndent=0,
                               spaceAfter=3),
}


def section(title):
    return Paragraph(title, S["section"])


def subsec(title):
    return Paragraph(title, S["subsec"])


def para(text):
    return Paragraph(text, S["para"])


def eq(text):
    return Paragraph(m(text), S["eq"])


def build():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="A Joint Minimization Framework for Transformer Interpretability",
        author="Saqib Nazir Bhat",
    )
    Q = []

    # ===== FRONT MATTER =====
    Q.append(Paragraph("A Joint Minimization Framework<br/>for Transformer Interpretability", S["title"]))
    Q.append(Paragraph("Information Crystallization", S["subtitle"]))
    Q.append(Paragraph("Saqib Nazir Bhat", S["author"]))
    Q.append(Paragraph(
        "Independent research, March 2026 &nbsp; &bull; &nbsp; "
        '<link href="mailto:saqibnazirbhat3@gmail.com" color="#1a4a7a">saqibnazirbhat3@gmail.com</link> &nbsp; &bull; &nbsp; '
        '<link href="https://github.com/Saqibnazirbhat/information-crystallization" color="#1a4a7a">github.com/Saqibnazirbhat/information-crystallization</link>',
        S["affil"]))

    # ===== ABSTRACT =====
    Q.append(Paragraph("ABSTRACT", S["abstract_h"]))
    Q.append(Paragraph(
        "Transformer interpretability is usually posed as follows: identify the "
        "parameters and inputs responsible for a model's behaviour. Existing "
        "methods work top-down, searching the combinatorial space by ablation "
        "from the full model. I propose a different formulation. Given an observed "
        f"output token x{m('&#x302;')}, find the minimal {m('S &#8838; [d]')} and "
        f"{m('T &#8838; [n]')} such that the argmax over the restricted model still "
        f"equals x{m('&#x302;')}. A single observation simplifies the problem: "
        f"softmax does not affect argmax. Preserving the predicted token requires "
        f"only that the logit margin {m('&#948;')} remains positive, which is "
        f"strictly weaker than matching the full distribution. Under this relaxation "
        "I introduce <i>Information Crystallization</i>, a bottom-up O(k&middot;d) "
        "algorithm (seed, greedy growth, prune) that finds local minima of (S, T) "
        "jointly. I decompose S into a final-block component S<sub>L</sub> and a "
        "residual-stream support S<sub>supp</sub>, connect the sparsity conjecture "
        "to the Lottery Ticket Hypothesis, and identify the T-dependence of S* as "
        "the central open problem. Preliminary results from the seed phase on GPT-2 "
        "small (N = 17, stratified by query type) refute the suffix-only heuristic "
        "for input-position seeding: only 11.8% of seed positions land on the last "
        "token, and long-range sequences cluster near the subject noun (mean "
        "distance 7.67 tokens from the end). The parameter-seed result is "
        "inconclusive on GPT-2 because its input embedding and unembedding "
        "projection share one parameter tensor; decisive evaluation of the "
        "S<sub>L</sub> &cup; S<sub>supp</sub> decomposition requires untied "
        "architectures. Growth and prune phases are unimplemented.",
        S["abstract"]))

    # ===== 1. INTRODUCTION =====
    Q.append(section("1. Introduction"))
    Q.append(para(
        "Mechanistic interpretability of large language models has become one of "
        "the central open problems in AI safety. Prior work treats parameter "
        "attribution and input attribution as separate questions: circuit discovery "
        "[Elhage et al. 2021], activation patching, causal tracing [Meng et al. 2022], "
        "and input rationales [Lei et al. 2016; Bau et al. 2020] all prune components "
        "starting from the full model. The search space is O(2<sup>d</sup>) for "
        "parameters (d ~ 10<sup>11</sup>) and O(2<sup>n</sup>) for inputs. Practical "
        "methods rely on heuristics to make this tractable."))
    Q.append(para(
        "This paper takes a different route. I observe that the practical "
        "interpretability question, \"why did the model produce <i>this</i> token?\", "
        "does not require reproducing the model's full output distribution. It "
        "requires only that the restricted model still predicts the same token. "
        "That is a weaker condition. It is the key lever for the rest of the paper."))

    Q.append(subsec("Contributions"))
    Q.append(ListFlowable(
        [ListItem(para(t), leftIndent=12) for t in [
            "A joint formulation of parameter and input minimization with a single preservation condition (Section 2).",
            "The <i>softmax relaxation</i>: replacing distribution preservation with a margin inequality (Section 3).",
            "<i>Information Crystallization</i>, a bottom-up algorithm with seed, growth, and prune phases (Section 4).",
            "A residual-stream decomposition S = S<sub>L</sub> &cup; S<sub>supp</sub> (Section 6) and a formal statement of the coupling problem (Section 7).",
            "An empirical protocol on GPT-2 together with preliminary seed-phase results on N = 17 stratified sequences (Section 8).",
        ]],
        bulletType="1", leftIndent=22, spaceAfter=8))

    Q.append(subsec("Relation to prior work"))
    Q.append(para(
        "Information Crystallization is most closely related to attribution "
        "patching [Syed et al. 2023], which also uses gradient approximations to "
        "circumvent the combinatorial cost of full ablation. The critical "
        "difference is direction: attribution patching scores every parameter "
        "once by its linearized ablation effect; Information Crystallization "
        "iteratively grows a set from a single seed. The two methods are "
        "complementary. Attribution patching produces a dense importance map; "
        "Information Crystallization produces a minimal sufficient subset. "
        "Automated Circuit Discovery (ACDC) [Conmy et al. 2023] is the canonical "
        "top-down baseline we contrast with. The bottom-up construction also "
        "echoes inference-time pruning methods such as SparseGPT [Frantar and "
        "Alistarh 2023] and Wanda [Sun et al. 2024], though those operate "
        "globally on the weight distribution rather than per-prediction. The "
        "gradient-attribution step builds on Integrated Gradients "
        "[Sundararajan et al. 2017]."))

    # ===== 2. PROBLEM =====
    Q.append(section("2. Problem Formulation"))
    Q.append(para(
        "Let V be a finite vocabulary and X = (x<sub>1</sub>, ..., x<sub>n</sub>) "
        f"{m('&isin;')} V<sup>n</sup> an input sequence. A language model "
        f"parameterized by {m('&theta;')} {m('&isin;')} &real;<sup>d</sup> defines"))
    Q.append(eq("P<sub>&theta;</sub>(x<sub>n+1</sub> | x<sub>1</sub>, ..., x<sub>n</sub>) = softmax(f<sub>&theta;</sub>(X)),"))
    Q.append(para(
        f"where f<sub>&theta;</sub> : V<sup>n</sup> {m('&rarr;')} &real;<sup>|V|</sup> "
        "is a composition of L non-linear transformations "
        "f<sub>&theta;</sub> = f<sub>&theta;</sub><sup>(L)</sup> "
        f"{m('&#8728;')} ... {m('&#8728;')} f<sub>&theta;</sub><sup>(1)</sup>. "
        "The predicted token is"))
    Q.append(eq(f"x&#x302; = argmax<sub>v {m('&isin;')} V</sub> P<sub>&theta;</sub>(v | X)."))

    Q.append(subsec("Problem"))
    Q.append(para(
        "Given x&#x302;, find minimal "
        f"S {m('&#8838;')} [d] and T {m('&#8838;')} [n] such that"))
    Q.append(eq(f"argmax<sub>v {m('&isin;')} V</sub> P<sub>&theta;<sub>S</sub></sub>(v | X<sub>T</sub>) = x&#x302;,"))
    Q.append(para(
        "where &theta;<sub>S</sub> denotes &theta; restricted to indices S and "
        "X<sub>T</sub> denotes X restricted to positions T."))

    Q.append(subsec("Operationalization of &theta;<sub>S</sub>"))
    Q.append(para(
        "Throughout this paper I take zero-ablation: entries outside S are set "
        "to zero. This is the simplest commitment and is what Algorithm 1 "
        "evaluates. Alternative operationalizations (mean-ablation over a "
        "reference distribution, or activation-patching from a paired input) "
        "are compatible with the framework and in general yield different "
        "minimal (S, T). The choice matters empirically: zero-ablation is "
        "conservative (it over-attributes because the network sees degenerate "
        "inputs it was never trained on), while mean-ablation is more permissive. "
        "Future work will compare all three."))

    Q.append(subsec("Tokenization"))
    Q.append(para(
        "T is defined at the token level of the model's tokenizer (BPE for GPT-2). "
        "A BPE-split word therefore occupies multiple positions and may be partially "
        "selected. Claims about word-level salience must be recovered post-hoc from "
        "token-level T."))

    Q.append(para(
        "In general, this problem is computationally intractable: f<sub>&theta;</sub> "
        "is non-linear and non-invertible, and d ~ 10<sup>11</sup> for current "
        "frontier-scale models. Section 3 shows that a natural weakening of the "
        "preservation condition makes the problem substantially more tractable."))

    # ===== 3. SOFTMAX RELAXATION =====
    Q.append(section("3. The Softmax Relaxation"))
    Q.append(subsec("Margin definition"))
    Q.append(para(
        "The logit margin is the gap between the winning logit and its nearest "
        "competitor:"))
    Q.append(eq(f"&delta; = f<sub>&theta;</sub>(X)[x&#x302;] {m('&minus;')} max<sub>v &ne; x&#x302;</sub> f<sub>&theta;</sub>(X)[v]."))
    Q.append(para(
        f"The argmax condition argmax<sub>v</sub> f<sub>&theta;</sub>(X)[v] = "
        f"x&#x302; is equivalent to &delta; &gt; 0. Since softmax is a strictly "
        f"monotone transformation, preserving the argmax requires only preserving "
        f"the <i>sign</i> of &delta;. No particular probability value needs to match."))

    Q.append(subsec("The &delta;-buffer argument"))
    Q.append(para(
        "Most parameters in a language model do not determine the identity of the "
        "winning token. They refine probabilities of losing tokens, smooth the "
        "output distribution, or encode features relevant to positions outside T. "
        "A parameter &theta;<sub>i</sub> is irrelevant to x&#x302; if perturbing "
        "it changes &delta; by less than &delta; itself. The set of such parameters "
        "is large precisely because &delta; &gt; 0 creates a buffer around the "
        "decision boundary. This motivates the central <i>Sparse Explanation "
        "Hypothesis</i>: the true minimal (S, T) is small relative to (d, n)."))

    Q.append(subsec("Sufficient Condition"))
    Q.append(para(
        "For all &theta;<sub>i</sub> "
        f"{m('&notin;')} S,"))
    Q.append(eq(
        "| logit<sub>x&#x302;</sub>(&theta;<sub>S</sub>, X<sub>T</sub>) "
        f"{m('&minus;')} logit<sub>x&#x302;</sub>(&theta;, X) | &lt; &delta;."))
    Q.append(para(
        "A proof via Lipschitz continuity of f<sub>&theta;</sub> or a first-order "
        "perturbation bound would suffice. The difficulty is the non-linearity of "
        "f<sub>&theta;</sub>, which resists tight analytical bounds."))

    # ===== 4. ALGORITHM =====
    Q.append(section("4. Information Crystallization"))
    Q.append(para(
        "Top-down methods search the space O(2<sup>d</sup>) by ablating from "
        "|S| = d. I invert the direction of search: start with |S| = 1 and grow. "
        "If the true minimal (S, T) is small, growing finds it in time proportional "
        "to its size rather than the model's size."))

    Q.append(Paragraph("Algorithm 1: Information Crystallization", S["algo_head"]))
    algo = (
        "1.  g &larr; &nabla;<sub>&theta;</sub> log P<sub>&theta;</sub>(x&#x302; | X)<br/>"
        "2.  S &larr; {argmax<sub>j</sub> |g<sub>j</sub>|},  T &larr; {argmax<sub>t</sub> ||&part;f<sub>&theta;</sub>/&part;x<sub>t</sub>||}<br/>"
        "3.  <b>repeat</b><br/>"
        "4.      forward pass on restricted subnetwork: evaluate argmax<sub>v</sub> P<sub>&theta;<sub>S</sub></sub>(v | X<sub>T</sub>)<br/>"
        "5.      <b>if</b> it equals x&#x302; <b>then break</b><br/>"
        "6.      g&#x302; &larr; &nabla;<sub>&theta;</sub> log P<sub>&theta;<sub>S</sub></sub>(x&#x302; | X<sub>T</sub>)<br/>"
        "7.      S &larr; S &cup; {argmax<sub>j &notin; S</sub> |g&#x302;<sub>j</sub>|}<br/>"
        "8.  <b>until</b> argmax equals x&#x302;<br/>"
        "9.  <b>for each</b> s in S:<br/>"
        "10.     <b>if</b> argmax is preserved without s, drop s<br/>"
        "11. <b>if</b> argmax no longer preserved after pruning, <b>goto</b> 3<br/>"
        "12. <b>return</b> (S, T)"
    )
    Q.append(Paragraph(algo, S["algo"]))

    Q.append(para(
        "<b>Note on line 6.</b> The gradient g&#x302; is taken with respect to the "
        "full parameter vector &theta;, not &theta;<sub>S</sub>. Only the forward "
        "pass uses the restricted subnetwork. This is essential: to rank candidates "
        "j &notin; S, the gradient component g&#x302;<sub>j</sub> must be defined "
        "for all j in [d]."))

    Q.append(subsec("Complexity"))
    Q.append(para(
        "The seed phase costs one full backward pass, O(d). Each growth step costs "
        "a forward-backward pass through a restricted subnetwork of current size "
        "|S| plus a global ranking of candidates at cost O(d). Summing over k growth "
        "steps gives O(k &middot; d + k<sup>2</sup>), which simplifies to O(k &middot; d) "
        "when k &lt;&lt; &radic;d (the regime of practical interest under the "
        "Sparse Explanation Hypothesis). Compared to exhaustive search O(2<sup>d</sup>) "
        "or top-down pruning O(d<sup>2</sup>), this is tractable."))

    # ===== 5. MINIMIZING T =====
    Q.append(section("5. Minimizing T: Input Position Selection"))
    Q.append(para(
        "Let &alpha;<sup>h,&ell;</sup>(t) denote the attention weight on position t "
        "from head h at layer &ell;. Define the cumulative salience"))
    Q.append(eq(
        "&sigma;(t) = &sum;<sub>h,&ell;</sub> &alpha;<sup>h,&ell;</sup>(t)."))
    Q.append(para(
        "For a threshold &epsilon; chosen so that positions with &sigma;(t) &lt; "
        "&epsilon; contribute less than &delta;/2 to the winning logit, define "
        "T<sub>sal</sub> = { t in [n] : &sigma;(t) &gt; &epsilon; }."))

    Q.append(para(
        "<b>Conjecture 1 (suffix, deprecated).</b> T = {x<sub>n&minus;k</sub>, ..., "
        "x<sub>n</sub>} &cup; T<sub>sal</sub> with k &lt;&lt; n."))

    Q.append(para(
        "<b>Failure modes.</b> Transformers use global attention, so a position-1 "
        "token with high &sigma; can dominate the logit landscape. The suffix "
        "approximation breaks in three concrete cases: sentence-initial negation, "
        "long-range syntactic dependencies, and named entities. I formalize the "
        "corrected claim."))

    Q.append(para(
        "<b>Conjecture 2 (attention-concentration identification).</b> For a "
        "minimal T that preserves x&#x302; under the restricted model, every t in "
        "T satisfies &sigma;(t) &gt; &epsilon; for a threshold &epsilon; determined "
        "by the decision boundary. In particular, Conjecture 1 is superseded: the "
        "suffix is not structurally guaranteed to lie in T."))

    Q.append(para(
        "Preliminary seed-phase data on GPT-2 small (Section 8) is consistent with "
        "Conjecture 2: the seed input position t* lies on the final token in only "
        "11.8% of sequences, and tracks the subject noun on long-range dependency "
        "queries."))

    # ===== 6. MINIMIZING S =====
    Q.append(section("6. Minimizing S: Parameter Selection"))
    Q.append(subsec("Residual stream decomposition"))
    Q.append(para(
        "I conjecture S = S<sub>L</sub> &cup; S<sub>supp</sub>, where S<sub>L</sub> "
        "consists of parameters in the final transformer block that directly govern "
        "&delta; (the unembedding projection weights for x&#x302; and its nearest "
        "competitor, the attention heads in the last one or two layers that "
        "activate most strongly on T, and a small set of MLP neurons in the final "
        "block), and S<sub>supp</sub> consists of earlier-layer parameters whose "
        "representations are consumed by S<sub>L</sub> via the residual stream "
        "[Elhage et al. 2021]. The contribution of S<sub>supp</sub> must satisfy "
        "| &delta;(S<sub>L</sub>) &minus; &delta;(S<sub>L</sub> &cup; "
        "S<sub>supp</sub>) | &lt; &delta;/2."))

    Q.append(subsec("Weight-tying caveat"))
    Q.append(para(
        "The above decomposition presumes that the unembedding projection and the "
        "input token-embedding matrix are distinct parameter tensors. This "
        "presumption fails on GPT-2 and many other pretrained models, which tie "
        "lm_head.weight to transformer.wte.weight. On weight-tied models the "
        "S<sub>L</sub> &cup; S<sub>supp</sub> decomposition is ill-posed as stated: "
        "gradient flow through the unembedding projection and through the input "
        "embedding is indistinguishable at the level of the parameter tensor. The "
        "seed-phase experiment in Section 8 exposes exactly this. A clean empirical "
        "test therefore requires either an architecture with untied embeddings "
        "(for example Pythia, OPT), or an experimental design that separately "
        "attributes gradient mass to the embedding-path and unembedding-path "
        "contributions to a shared tensor. Both are future work."))

    Q.append(subsec("Lottery Ticket connection"))
    Q.append(para(
        "Frankle and Carbin [2019] show that dense networks contain sparse "
        "subnetworks, so-called \"winning tickets\", that match the full network's "
        "performance when trained in isolation. The conjecture here is the "
        "inference-time analogue: for a fixed input X and prediction x&#x302;, there "
        "exists a sparse subnetwork that preserves the argmax without retraining."))

    Q.append(subsec("Middle-layer localization"))
    Q.append(para(
        "Meng et al. [2022] show via causal tracing that factual associations in "
        "GPT-class models localize in middle-layer MLP modules. This challenges "
        "the assumption that S<sub>L</sub> alone suffices for factual recall "
        "queries. For \"The capital of France is...\" the decisive parameters may "
        "live in middle layers, meaning S<sub>supp</sub> could be substantially "
        "larger for factual than for syntactic continuation tasks. Any empirical "
        "protocol must stratify by query type."))

    # ===== 7. COUPLING =====
    Q.append(section("7. Joint Minimization: The Coupling Problem"))
    Q.append(para(
        "S and T are not independently minimizable. When T changes, the embeddings "
        "fed to layer 1 change. That perturbation propagates through all L layers "
        "via the residual stream, producing a different hidden state at layer L. "
        "The set of parameters in S<sub>L</sub> that are load-bearing for &delta; "
        "therefore depends on T:"))
    Q.append(eq("S*(T) = argmin<sub>S</sub> |S|   s.t.   argmax<sub>v</sub> P<sub>&theta;<sub>S</sub></sub>(v | X<sub>T</sub>) = x&#x302;."))
    Q.append(para(
        "Treating S and T as independent is a useful approximation but its error "
        "is only small when S<sub>supp</sub> is small relative to &delta;."))

    Q.append(para(
        "<b>Open problem.</b> Characterize the function T &rarr; S*(T). Two concrete "
        "sub-questions: (i) bound |S*(T) &minus; S*(&theta;, X)| as a function of "
        "|T|; (ii) determine whether the greedy decomposition in Algorithm 1 is "
        "within a constant factor of the true joint minimum."))

    Q.append(para(
        "<b>Greedy decomposition as upper bound.</b> Despite the coupling, "
        "Algorithm 1 yields a computable upper bound on |S| + |T|: identify T "
        "from cumulative attention weights (Section 5), then minimize S conditional "
        "on T via gradient-sensitivity scoring, then check whether the coupling "
        "error is small. This is suboptimal in general but empirically verifiable."))

    # ===== 8. PROTOCOL + RESULTS =====
    Q.append(section("8. Empirical Protocol and Preliminary Results"))
    Q.append(subsec("Protocol"))
    Q.append(para(
        "GPT-2 [Radford et al. 2019] is suitable for initial verification because "
        "its full parameter set and attention weights are publicly accessible. "
        "I propose the following protocol."))
    Q.append(ListFlowable(
        [ListItem(para(t), leftIndent=12) for t in [
            "<b>Sampling.</b> Draw 1,000 input sequences stratified by query type: syntactic continuation, factual recall, negation-heavy, and long-range syntactic dependency.",
            "<b>Baseline.</b> Record x&#x302; for each sequence using the full model &theta;.",
            "<b>T identification.</b> Greedily remove input positions in ascending order of &sigma;(t). Record the minimal T at which x&#x302; first flips.",
            "<b>S identification.</b> Greedily ablate parameters in ascending order of gradient-sensitivity score. Record the minimal S at which x&#x302; first flips.",
            "<b>Size measurement.</b> Report |S|/d and |T|/n distributions per query type (means, medians, 90th percentiles).",
            "<b>Coupling test.</b> For each T from Step 3, recompute S*(T) and compare with S*(&theta;, X). Small discrepancy empirically validates the greedy decomposition.",
        ]], bulletType="1", leftIndent=22, spaceAfter=8))

    Q.append(subsec("Preliminary seed-phase results"))
    Q.append(para(
        "I have run the seed phase (Step 1 of Algorithm 1) on N = 17 hand-curated "
        "sequences stratified into four query types (syntactic continuation 5, "
        "factual recall 5, negation-heavy 4, long-range syntactic dependency 3). "
        "Sample size is a sanity check, not a statistical study."))

    Q.append(para("<b>Input seed t*.</b> Across the 17 sequences, only 11.8% (2/17) "
                  "had t* on the final input token. Mean distance of t* from the "
                  "sequence end was 3.88 tokens. Stratified:"))

    tbl = Table([
        ["Query type", "Mean distance of t* from end"],
        ["factual (n = 5)", "2.00"],
        ["syntactic (n = 5)", "2.80"],
        ["negation (n = 4)", "4.75"],
        ["long-range (n = 3)", "7.67"],
    ], colWidths=[2.8 * inch, 2.4 * inch], hAlign="CENTER")
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Body-Bold", 10),
        ("FONT", (0, 1), (-1, -1), "Body", 10),
        ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#888")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, HexColor("#888")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    Q.append(tbl)
    Q.append(Spacer(1, 8))

    Q.append(para(
        "This is evidence, subject to the obvious caveat of small n (especially "
        "n = 3 in the long-range bucket), consistent with Conjecture 2 (attention "
        "concentration, not suffix recency) and inconsistent with Conjecture 1 "
        "(suffix only). On long-range sequences, t* lands on the subject noun "
        "rather than on any suffix position."))

    Q.append(para(
        "<b>Parameter seed j* (inconclusive due to weight-tying).</b> In 17/17 "
        "sequences the seed parameter tensor was transformer.wte.weight. This is "
        "uninformative for testing the S<sub>L</sub> &cup; S<sub>supp</sub> "
        "decomposition: on GPT-2 this single tensor serves as both the input "
        "embedding and the unembedding projection. The result is consistent with "
        "the S<sub>L</sub> conjecture (unembedding in S<sub>L</sub>) but does not "
        "adjudicate between S<sub>L</sub> and the input embedding as the dominant "
        "direct-gradient path."))

    Q.append(para(
        "<b>A stronger reading.</b> The concentration of the direct gradient of "
        "log P(x&#x302; | X) on a single tensor suggests that the direct "
        "(unembedding-path) contribution to the winning logit dominates the "
        "residual-stream-mediated contribution in the seed step. If this holds "
        "more generally, S<sub>L</sub> is the empirically dominant component and "
        "S<sub>supp</sub> functions as a refinement, consistent with the paper's "
        "framing. An untied-architecture experiment is required to confirm."))

    Q.append(para(
        "<b>Implementation.</b> Available in the repository at "
        "experiments/seed_phase.ipynb with full results in experiments/results.md."))

    # ===== 9. LIMITATIONS =====
    Q.append(section("9. Limitations"))
    Q.append(ListFlowable(
        [ListItem(para(t), leftIndent=12) for t in [
            "<b>The suffix approximation is not a theorem.</b> T is identified by attention analysis per input. No structural guarantee holds.",
            "<b>|S<sub>supp</sub>| is unbounded.</b> The inequality on &delta; is a requirement, not a proved result. For factual queries it may be violated.",
            "<b>Greedy decomposition is suboptimal.</b> The true joint minimum of (S, T) may be strictly smaller than the greedy bound.",
            "<b>The Sufficient Condition is unproved.</b> All conjectures depend on a Lipschitz or first-order perturbation bound on f<sub>&theta;</sub> that has not been derived.",
            "<b>Scale.</b> Verification on GPT-2 does not guarantee generalization to models with d ~ 10<sup>11</sup>.",
            "<b>Only the seed phase has been executed.</b> Growth, prune, and verification phases of Algorithm 1 remain unimplemented.",
            "<b>Weight-tying confound on GPT-2.</b> The preliminary j* result cannot cleanly test the decomposition; an untied-architecture replication is required.",
            "<b>Sample size is not statistically significant.</b> N = 17 is a feasibility check.",
        ]],
        bulletType="bullet", start="\u2022", leftIndent=22,
        bulletFontName="Body", bulletFontSize=9, spaceAfter=8))

    # ===== 10. CONCLUSION =====
    Q.append(section("10. Conclusion"))
    Q.append(para(
        "The standard interpretability problem asks for the parameters and inputs "
        "that explain a model's behaviour. Posed as distribution preservation, the "
        "problem is intractable. Posed as margin preservation it admits substantially "
        "sparser solutions, reachable via a bottom-up O(k &middot; d) algorithm."))
    Q.append(para(
        "The central open problem is the coupling between S and T through the "
        "residual stream. A formal perturbation bound on f<sub>&theta;</sub> would "
        "turn the conjectures here into theorems. Preliminary seed-phase results on "
        "GPT-2 small are reported in Section 8 and experiments/results.md; growth "
        "and prune phases, and replication on an untied architecture, are the "
        "immediate next steps."))

    # ===== REFERENCES =====
    Q.append(section("References"))
    refs = [
        "Elhage, N., Nanda, N., Olsson, C., et al. (2021). <i>A Mathematical Framework for Transformer Circuits.</i> Anthropic. transformer-circuits.pub/2021/framework/",
        "Lei, T., Barzilay, R., and Jaakkola, T. (2016). Rationalizing Neural Predictions. In Proc. EMNLP.",
        "Bau, A., Belinkov, Y., Sajjad, H., et al. (2020). Identifying and Controlling Important Neurons in Neural Machine Translation. In Proc. ICLR.",
        "Frankle, J. and Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. In Proc. ICLR.",
        "Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. In Proc. NeurIPS.",
        "Radford, A., Wu, J., Child, R., et al. (2019). <i>Language Models are Unsupervised Multitask Learners.</i> OpenAI technical report.",
        "Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic Attribution for Deep Networks. In Proc. ICML.",
        "Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. In Proc. NeurIPS.",
        "Syed, A., Rager, C., and Conmy, A. (2023). Attribution Patching Outperforms Automated Circuit Discovery. NeurIPS Workshop.",
        "Frantar, E. and Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. In Proc. ICML.",
        "Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. (2024). A Simple and Effective Pruning Approach for Large Language Models (Wanda). In Proc. ICLR.",
    ]
    Q.append(ListFlowable(
        [ListItem(Paragraph(r, S["ref"]), leftIndent=16) for r in refs],
        bulletType="1", leftIndent=22))

    doc.build(Q)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
