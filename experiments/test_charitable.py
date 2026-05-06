"""
CHARITABLE re-test of Information Crystallization (Bhat, March 2026).

Three changes from the strict run:
  - Pythia-160M instead of GPT-2 (UNTIED embeddings; removes wte/lm_head confound).
  - Per-COMPONENT granularity (attention heads + MLP intermediate neurons).
    NOT per-scalar; this is the activation-patching native unit.
  - MEAN-activation reference (computed over a small baseline corpus),
    NOT zero. This is the standard "mean ablation" used in circuit work.

Components on Pythia-160M (12L, 12H, hidden=768, MLP=3072):
  attention: 12 layers × 12 heads = 144
  MLP:       12 layers × 3072 neurons = 36,864
  total:     37,008

Three conditions:
  A  rank components by L2 norm of full-model activation-gradient (SNIP at component level)
  B  paper's restricted-network bottom-up (seed = top-1% by A; grow by restricted activation-grad)
  C  rank components by output VARIANCE on baseline corpus (low variance = cheap to ablate)

Find the smallest |S|/n_components keeping argmax of x̂ stable.
"""
import time, json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
MODEL = "EleutherAI/pythia-160m"

PROMPTS = {
    "syntactic": [
        "The cat sat on the",
        "She walked into the",
        "He picked up his",
    ],
    "factual": [
        "The capital of France is",
        "The largest planet in our solar system is",
        "Two plus two equals",
    ],
    "negation": [
        "Despite the rain, the picnic was not",
        "The medicine did not make him feel",
        "The food was not at all",
    ],
    "long_range": [
        "Mary went to the store. She bought apples. Then Mary",
        "John gave the book to Sarah. Later, Sarah",
        "Alice met Bob at the park. Together, they",
    ],
}

BASELINE_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "She opened the door and walked inside slowly.",
    "Time is the most valuable resource we have.",
    "Programming requires patience and a lot of practice.",
    "The book was lying on the wooden table.",
    "He drove to the airport early in the morning.",
    "Music can change a person's mood quite quickly.",
    "Children love to play outside in the park.",
    "Mathematics is the language of natural science.",
    "The river flowed gently through the green valley.",
    "Coffee in the morning helps me think clearly.",
    "She wrote a long letter to her friend yesterday.",
    "The mountain peak was completely covered in snow.",
    "Dogs are often considered loyal and friendly companions.",
    "He read the newspaper every Sunday morning at breakfast.",
    "The garden was full of bright colorful flowers.",
]

# Sweep aggressively into the sparse regime — that's where a circuit lives
FRACTIONS = [0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001]


def predict(model, ids):
    with torch.no_grad():
        return int(model(ids).logits[0, -1].argmax().item())


# ---- Hook plumbing ---- #
# We intervene at TWO points per layer:
#   - input to attention.dense  (shape (B, S, H*HD)) → reshape to (B, S, H, HD), per-head
#   - input to mlp.dense_4h_to_h (shape (B, S, intermediate)) → per-neuron

class HookManager:
    """Manages forward pre-hooks for either ablation OR gradient capture."""

    def __init__(self, model):
        self.layers = model.gpt_neox.layers
        self.cfg = model.config
        self.n_layers = len(self.layers)
        self.n_heads = self.cfg.num_attention_heads
        self.head_dim = self.cfg.hidden_size // self.n_heads
        self.intermediate = self.cfg.intermediate_size
        self.handles = []
        self.captured = {}  # for grad mode

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.captured = {}

    def install_grad_capture(self):
        """Forward pre-hooks that retain_grad on inputs to attn.dense and mlp.dense_4h_to_h."""
        self.captured = {}
        for L, layer in enumerate(self.layers):
            def make_hook(name):
                def hook(module, args):
                    x = args[0]
                    if x.requires_grad:
                        x.retain_grad()
                        self.captured[name] = x
                return hook
            self.handles.append(layer.attention.dense.register_forward_pre_hook(
                make_hook(f"attn_{L}")))
            self.handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                make_hook(f"mlp_{L}")))

    def install_ablation(self, attn_means, mlp_means, S_attn, S_mlp):
        """
        attn_means[L]: tensor (n_heads, head_dim)  mean over corpus
        mlp_means[L]:  tensor (intermediate,)
        S_attn[L]:     bool (n_heads,)            True = keep, False = mean-ablate
        S_mlp[L]:      bool (intermediate,)
        """
        for L, layer in enumerate(self.layers):
            am = attn_means[L]   # (H, HD)
            mm = mlp_means[L]    # (I,)
            sa = S_attn[L]       # (H,)
            sm = S_mlp[L]        # (I,)

            def make_attn_hook(am_, sa_, H_, HD_):
                def hook(module, args):
                    x = args[0]                              # (B, S, H*HD)
                    B, S, _ = x.shape
                    x = x.view(B, S, H_, HD_)
                    mask = sa_.view(1, 1, H_, 1).to(x.dtype)
                    am_typed = am_.view(1, 1, H_, HD_).to(x.dtype)
                    out = x * mask + am_typed * (1 - mask)
                    return (out.reshape(B, S, H_ * HD_).to(x.dtype),)
                return hook

            def make_mlp_hook(mm_, sm_, I_):
                def hook(module, args):
                    x = args[0]                              # (B, S, I)
                    mask = sm_.view(1, 1, I_).to(x.dtype)
                    mm_typed = mm_.view(1, 1, I_).to(x.dtype)
                    out = x * mask + mm_typed * (1 - mask)
                    return (out.to(x.dtype),)
                return hook

            self.handles.append(layer.attention.dense.register_forward_pre_hook(
                make_attn_hook(am, sa, self.n_heads, self.head_dim)))
            self.handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                make_mlp_hook(mm, sm, self.intermediate)))


def compute_baseline_stats(model, tok, hm, corpus):
    """Return mean and variance per component, averaged over (batch=1, seq, corpus)."""
    print("  computing baseline activations...", flush=True)
    # per-layer accumulators of shape (H, HD) and (I,)
    n_layers, n_heads, head_dim, intermediate = (
        hm.n_layers, hm.n_heads, hm.head_dim, hm.intermediate)
    attn_sum = [torch.zeros(n_heads, head_dim) for _ in range(n_layers)]
    attn_sumsq = [torch.zeros(n_heads, head_dim) for _ in range(n_layers)]
    mlp_sum = [torch.zeros(intermediate) for _ in range(n_layers)]
    mlp_sumsq = [torch.zeros(intermediate) for _ in range(n_layers)]
    n_pos = 0

    # Hooks just to capture pre-dense activations (no requires_grad needed)
    captured = {}
    handles = []
    for L, layer in enumerate(hm.layers):
        def make_h(name):
            def hook(module, args):
                captured[name] = args[0].detach()
            return hook
        handles.append(layer.attention.dense.register_forward_pre_hook(
            make_h(f"attn_{L}")))
        handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(
            make_h(f"mlp_{L}")))

    with torch.no_grad():
        for s in corpus:
            ids = tok(s, return_tensors="pt").input_ids
            captured.clear()
            model(ids)
            B, S = ids.shape
            n_pos += B * S
            for L in range(n_layers):
                a = captured[f"attn_{L}"].view(B, S, n_heads, head_dim)
                attn_sum[L] += a.sum(dim=(0, 1))
                attn_sumsq[L] += (a ** 2).sum(dim=(0, 1))
                m = captured[f"mlp_{L}"]
                mlp_sum[L] += m.sum(dim=(0, 1))
                mlp_sumsq[L] += (m ** 2).sum(dim=(0, 1))
    for h in handles:
        h.remove()

    attn_means = [s / n_pos for s in attn_sum]
    mlp_means = [s / n_pos for s in mlp_sum]
    attn_var = [(sq / n_pos - m ** 2).clamp_min(0)
                for sq, m in zip(attn_sumsq, attn_means)]
    mlp_var = [(sq / n_pos - m ** 2).clamp_min(0)
               for sq, m in zip(mlp_sumsq, mlp_means)]
    print(f"  baseline n_positions={n_pos}", flush=True)
    return attn_means, mlp_means, attn_var, mlp_var


def per_component_grad_score(model, hm, ids, tgt, S_attn=None, S_mlp=None,
                              attn_means=None, mlp_means=None):
    """
    Compute per-component score = L2 norm of activation gradient.
    If S_attn/S_mlp + means provided, run forward through ablated model
    (the "restricted gradient" of the paper).
    Returns: per-layer (H,) attn scores, per-layer (I,) mlp scores.
    """
    hm.clear()
    if S_attn is not None:
        hm.install_ablation(attn_means, mlp_means, S_attn, S_mlp)
    hm.install_grad_capture()

    model.zero_grad(set_to_none=True)
    logits = model(ids).logits[0, -1]
    logp = F.log_softmax(logits, dim=-1)[tgt]
    logp.backward()

    attn_scores = []  # per layer (H,)
    mlp_scores = []   # per layer (I,)
    for L in range(hm.n_layers):
        a = hm.captured[f"attn_{L}"]
        # a.grad: (B, S, H*HD)
        g = a.grad.view(a.shape[0], a.shape[1], hm.n_heads, hm.head_dim)
        attn_scores.append(g.pow(2).sum(dim=(0, 1, 3)).sqrt())  # (H,)
        m = hm.captured[f"mlp_{L}"]
        gm = m.grad
        mlp_scores.append(gm.pow(2).sum(dim=(0, 1)).sqrt())     # (I,)
    hm.clear()
    model.zero_grad(set_to_none=True)
    return attn_scores, mlp_scores


def flatten_scores(attn_scores, mlp_scores):
    """Concatenate all component scores into one flat (n_components,) tensor + index map."""
    parts = []
    layout = []  # list of (kind, layer, slice_start, slice_end)
    cursor = 0
    for L, s in enumerate(attn_scores):
        parts.append(s)
        layout.append(("attn", L, cursor, cursor + s.numel()))
        cursor += s.numel()
    for L, s in enumerate(mlp_scores):
        parts.append(s)
        layout.append(("mlp", L, cursor, cursor + s.numel()))
        cursor += s.numel()
    return torch.cat(parts), layout


def keep_mask_from_flat(flat_mask, layout, n_layers, n_heads, intermediate):
    """flat_mask: (n_components,) bool. Returns S_attn list[(H,)], S_mlp list[(I,)]."""
    S_attn = [None] * n_layers
    S_mlp = [None] * n_layers
    for kind, L, a, b in layout:
        slice_ = flat_mask[a:b]
        if kind == "attn":
            S_attn[L] = slice_
        else:
            S_mlp[L] = slice_
    return S_attn, S_mlp


def sweep_keep_fractions(model, hm, attn_means, mlp_means, ids, tgt,
                          flat_scores, layout, fractions, n_components):
    """For each fraction, keep top-frac by score, mean-ablate the rest, check argmax."""
    n_layers = hm.n_layers
    n_heads = hm.n_heads
    intermediate = hm.intermediate

    sorted_scores, _ = torch.sort(flat_scores)
    out = []
    min_pres = None
    for f in fractions:
        keep = max(1, int(round(f * n_components)))
        idx = max(0, n_components - keep)
        tau = float(sorted_scores[idx].item())
        flat_mask = (flat_scores >= tau)
        S_attn, S_mlp = keep_mask_from_flat(flat_mask, layout, n_layers,
                                             n_heads, intermediate)
        hm.clear()
        hm.install_ablation(attn_means, mlp_means, S_attn, S_mlp)
        pred = predict(model, ids)
        hm.clear()
        kept = int(flat_mask.sum().item())
        preserved = (pred == tgt)
        out.append({"frac": f, "kept": kept, "kept_frac": kept / n_components,
                    "pred": pred, "preserved": preserved})
        if preserved and (min_pres is None or f < min_pres):
            min_pres = f
    return min_pres, out


def cond_B(model, hm, attn_means, mlp_means, ids, tgt, n_components,
           seed_frac=0.01, step_frac=0.02, max_iters=20):
    """Paper's bottom-up: seed by full-model grad, grow by restricted grad."""
    n_layers, n_heads, intermediate = hm.n_layers, hm.n_heads, hm.intermediate
    # Seed score from full model
    a_scores, m_scores = per_component_grad_score(model, hm, ids, tgt)
    flat, layout = flatten_scores(a_scores, m_scores)
    sorted_full, _ = torch.sort(flat)
    keep = max(1, int(round(seed_frac * n_components)))
    tau = float(sorted_full[n_components - keep].item())
    flat_mask = (flat >= tau)

    traj = []
    for it in range(max_iters + 1):
        S_attn, S_mlp = keep_mask_from_flat(flat_mask, layout, n_layers,
                                             n_heads, intermediate)
        hm.clear()
        hm.install_ablation(attn_means, mlp_means, S_attn, S_mlp)
        pred = predict(model, ids)
        kept = int(flat_mask.sum().item())
        traj.append({"iter": it, "kept": kept, "kept_frac": kept / n_components,
                     "pred": pred, "preserved": pred == tgt})
        hm.clear()
        if pred == tgt or it == max_iters:
            break
        # Restricted gradient
        a_scores, m_scores = per_component_grad_score(
            model, hm, ids, tgt, S_attn, S_mlp, attn_means, mlp_means)
        rflat, _ = flatten_scores(a_scores, m_scores)
        # Mask out already-kept
        rflat = rflat.clone()
        rflat[flat_mask] = -1.0
        # Add top step_frac new
        k = max(1, int(round(step_frac * n_components)))
        topk_vals, topk_idx = torch.topk(rflat, k, largest=True, sorted=False)
        new_mask = torch.zeros_like(flat_mask)
        new_mask[topk_idx] = True
        flat_mask = flat_mask | new_mask
    flipped = (kept / n_components) if pred == tgt else None
    return flipped, traj, flat_mask, layout


def main():
    print(f"loading {MODEL}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL,
                                                  torch_dtype=torch.float32).eval()
    model = model.to(torch.float32)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"  model dtype: {next(model.parameters()).dtype}", flush=True)
    print(f"  loaded in {time.time()-t0:.1f}s, params={n_total_params:,}", flush=True)

    hm = HookManager(model)
    n_components = hm.n_layers * hm.n_heads + hm.n_layers * hm.intermediate
    print(f"  layers={hm.n_layers}, heads={hm.n_heads}, head_dim={hm.head_dim}, "
          f"intermediate={hm.intermediate}", flush=True)
    print(f"  n_components={n_components} "
          f"({hm.n_layers*hm.n_heads} attn + {hm.n_layers*hm.intermediate} mlp)",
          flush=True)

    # Baseline stats
    print("computing baseline activations...", flush=True)
    t0 = time.time()
    attn_means, mlp_means, attn_var, mlp_var = compute_baseline_stats(
        model, tok, hm, BASELINE_CORPUS)
    print(f"  baseline stats in {time.time()-t0:.1f}s", flush=True)

    # For Cond C: per-component score = mean of variance over the channel
    var_attn_scores = [v.mean(dim=1) for v in attn_var]   # (H,) per layer
    var_mlp_scores = mlp_var                                # (I,) per layer
    var_flat, var_layout = flatten_scores(var_attn_scores, var_mlp_scores)

    results = {"meta": {"model": MODEL, "n_components": n_components,
                        "fractions": FRACTIONS}, "per_type": []}

    overall_t0 = time.time()
    for qtype, prompts in PROMPTS.items():
        print(f"\n=== {qtype} ===", flush=True)
        baselines = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids
            tgt = predict(model, ids)
            baselines.append((p, ids, tgt))
            print(f"  '{p}' -> '{tok.decode([tgt])}' (id={tgt})", flush=True)

        type_block = {"qtype": qtype, "results": []}

        for i, (p, ids, tgt) in enumerate(baselines):
            print(f"\n  -- prompt {i}: '{p}' --", flush=True)
            t_p = time.time()

            # Cond A: full-model gradient saliency at component level
            print("    [A] full-model grad saliency...", flush=True)
            a_sc, m_sc = per_component_grad_score(model, hm, ids, tgt)
            grad_flat, grad_layout = flatten_scores(a_sc, m_sc)
            min_a, sweep_a = sweep_keep_fractions(
                model, hm, attn_means, mlp_means, ids, tgt,
                grad_flat, grad_layout, FRACTIONS, n_components)
            print(f"    [A] min preserving frac = {min_a}", flush=True)

            # Cond C: variance ranking (low var = ablate first)
            # Score for "keep" = variance (high variance = keep)
            print("    [C] variance ranking...", flush=True)
            min_c, sweep_c = sweep_keep_fractions(
                model, hm, attn_means, mlp_means, ids, tgt,
                var_flat, var_layout, FRACTIONS, n_components)
            print(f"    [C] min preserving frac = {min_c}", flush=True)

            # Cond B: paper bottom-up
            print("    [B] paper bottom-up (restricted-grad growth)...", flush=True)
            min_b, traj_b, mask_b, layout_b = cond_B(
                model, hm, attn_means, mlp_means, ids, tgt, n_components,
                seed_frac=0.01, step_frac=0.02, max_iters=25)
            print(f"    [B] flipped at frac = {min_b}", flush=True)
            print("        traj: " +
                  ", ".join(f"{t['kept_frac']:.3f}{'!' if t['preserved'] else ''}"
                            for t in traj_b), flush=True)

            entry = {
                "prompt": p, "target_id": tgt, "target_tok": tok.decode([tgt]),
                "A_min_frac": min_a, "A_sweep": sweep_a,
                "C_min_frac": min_c, "C_sweep": sweep_c,
                "B_flip_frac": min_b, "B_traj": traj_b,
                "elapsed_s": time.time() - t_p,
            }

            # Transfer test from prompt 0 only (using mask_b)
            if i == 0 and min_b is not None:
                others = [(bp, bids, btgt) for j, (bp, bids, btgt)
                          in enumerate(baselines) if j != 0]
                S_attn_b, S_mlp_b = keep_mask_from_flat(
                    mask_b, layout_b, hm.n_layers, hm.n_heads, hm.intermediate)
                hm.clear()
                hm.install_ablation(attn_means, mlp_means, S_attn_b, S_mlp_b)
                tr = []
                for bp, bids, btgt in others:
                    pr = predict(model, bids)
                    tr.append({"prompt": bp, "tgt": btgt, "pred": pr,
                               "tgt_tok": tok.decode([btgt]),
                               "pred_tok": tok.decode([pr]),
                               "preserved": pr == btgt})
                hm.clear()
                entry["transfer_from_prompt0"] = tr
                print("        transfer of B-mask:", flush=True)
                for r in tr:
                    print(f"          '{r['prompt']}' tgt='{r['tgt_tok']}' "
                          f"got='{r['pred_tok']}' ok={r['preserved']}", flush=True)

            type_block["results"].append(entry)

        results["per_type"].append(type_block)

    results["meta"]["total_elapsed_s"] = time.time() - overall_t0
    out_path = r"./outputs/paper_test_charitable_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time()-overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
