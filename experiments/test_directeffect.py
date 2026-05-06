"""
Information Crystallization — v3 with all four recommendations.

Recommendations applied:
  1. Bigger model: Pythia-410M (4x larger; better logit margins)
  2. Soft metrics: argmax preservation + top-3 containment + KL(p_full || p_rest)
  3. Direct-effect score (Syed/Nanda AtP):  DE_k = (a_k - mean_k) · ∂L/∂a_k
  4. Variance-seeded variant: drop the gradient seed, keep bottom-up structure

Conditions:
  A   single-shot |grad| ranking            (paper's seed criterion, baseline)
  C   single-shot variance ranking          (prompt-agnostic baseline)
  D   single-shot |direct effect| ranking   (recommendation 3, single-shot)
  E   paper bottom-up with DE seed + DE-on-restricted growth (paper's structure
       with the better score)
  F   variance seed + DE-on-restricted growth (recommendation 4)

Components on Pythia-410M (24L, 16H, hidden=1024, MLP=4096):
  24 * 16 = 384 attn  +  24 * 4096 = 98304 mlp  =  98688 components
"""
import time, json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
MODEL = "EleutherAI/pythia-410m"

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

FRACTIONS = [0.50, 0.30, 0.20, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001]
KL_THRESHOLD = 1.0  # soft preservation threshold


# ---- Hook plumbing ---- #

class HookManager:
    def __init__(self, model):
        self.layers = model.gpt_neox.layers
        cfg = model.config
        self.n_layers = len(self.layers)
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // self.n_heads
        self.intermediate = cfg.intermediate_size
        self.handles = []
        self.captured = {}

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.captured = {}

    def install_capture(self, want_grad=False):
        """Capture inputs to attn.dense and mlp.dense_4h_to_h. retain_grad if needed."""
        self.captured = {}
        for L, layer in enumerate(self.layers):
            def make_hook(name, retain):
                def hook(module, args):
                    x = args[0]
                    self.captured[name] = x
                    if retain and x.requires_grad:
                        x.retain_grad()
                return hook
            self.handles.append(layer.attention.dense.register_forward_pre_hook(
                make_hook(f"attn_{L}", want_grad)))
            self.handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                make_hook(f"mlp_{L}", want_grad)))

    def install_ablation(self, attn_means, mlp_means, S_attn, S_mlp):
        for L, layer in enumerate(self.layers):
            am, mm = attn_means[L], mlp_means[L]
            sa, sm = S_attn[L], S_mlp[L]
            H, HD, I = self.n_heads, self.head_dim, self.intermediate

            def make_attn(am_, sa_):
                def hook(module, args):
                    x = args[0]
                    B, S, _ = x.shape
                    x4 = x.view(B, S, H, HD)
                    mask = sa_.view(1, 1, H, 1).to(x.dtype)
                    am_t = am_.view(1, 1, H, HD).to(x.dtype)
                    out = (x4 * mask + am_t * (1 - mask)).reshape(B, S, H * HD)
                    return (out,)
                return hook

            def make_mlp(mm_, sm_):
                def hook(module, args):
                    x = args[0]
                    mask = sm_.view(1, 1, I).to(x.dtype)
                    mm_t = mm_.view(1, 1, I).to(x.dtype)
                    return (x * mask + mm_t * (1 - mask),)
                return hook

            self.handles.append(layer.attention.dense.register_forward_pre_hook(
                make_attn(am, sa)))
            self.handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                make_mlp(mm, sm)))


# ---- Stats helpers ---- #

def predict_logits(model, ids):
    with torch.no_grad():
        return model(ids).logits[0, -1]


def metrics(full_logits, restricted_logits, tgt):
    """Returns dict: argmax_preserved, top3, kl, margin_full, margin_rest."""
    rl = restricted_logits
    fl = full_logits
    pred = int(rl.argmax().item())
    top3 = list(rl.topk(3).indices.tolist())
    margin_rest = float(rl[tgt].item() - rl.masked_select(
        torch.arange(rl.numel()) != tgt).max().item()) if tgt in top3 else float(
        rl[tgt].item() - rl.max().item())
    margin_full = float(fl[tgt].item() - fl.masked_select(
        torch.arange(fl.numel()) != tgt).max().item())
    log_p_full = F.log_softmax(fl, dim=-1)
    log_p_rest = F.log_softmax(rl, dim=-1)
    p_full = log_p_full.exp()
    kl = float((p_full * (log_p_full - log_p_rest)).sum().item())
    return {
        "argmax_preserved": pred == tgt,
        "top3_contains_tgt": tgt in top3,
        "kl_full_to_rest": kl,
        "kl_below_threshold": kl < KL_THRESHOLD,
        "margin_rest": margin_rest,
        "margin_full": margin_full,
        "pred": pred,
    }


def compute_baseline_stats(model, tok, hm, corpus):
    n_layers, n_heads, head_dim, I = hm.n_layers, hm.n_heads, hm.head_dim, hm.intermediate
    attn_sum = [torch.zeros(n_heads, head_dim) for _ in range(n_layers)]
    attn_sumsq = [torch.zeros(n_heads, head_dim) for _ in range(n_layers)]
    mlp_sum = [torch.zeros(I) for _ in range(n_layers)]
    mlp_sumsq = [torch.zeros(I) for _ in range(n_layers)]
    n_pos = 0

    captured = {}
    handles = []
    for L, layer in enumerate(hm.layers):
        def mk(name):
            def hook(module, args):
                captured[name] = args[0].detach()
            return hook
        handles.append(layer.attention.dense.register_forward_pre_hook(mk(f"a{L}")))
        handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(mk(f"m{L}")))

    with torch.no_grad():
        for s in corpus:
            ids = tok(s, return_tensors="pt").input_ids
            captured.clear()
            model(ids)
            B, S = ids.shape
            n_pos += B * S
            for L in range(n_layers):
                a = captured[f"a{L}"].view(B, S, n_heads, head_dim)
                attn_sum[L] += a.sum(dim=(0, 1))
                attn_sumsq[L] += (a ** 2).sum(dim=(0, 1))
                m = captured[f"m{L}"]
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
    return attn_means, mlp_means, attn_var, mlp_var, n_pos


# ---- Score computations ---- #

def compute_grad_and_de(model, hm, ids, tgt, attn_means, mlp_means,
                        S_attn=None, S_mlp=None):
    """One forward+backward (optionally through ablated model). Returns:
       grad_attn[L]: (H,)        = L2 norm of grad over (B,S,HD)
       grad_mlp[L]: (I,)
       de_attn[L]:  (H,)         = |sum (a-mean) * grad|
       de_mlp[L]:   (I,)
    """
    hm.clear()
    if S_attn is not None:
        hm.install_ablation(attn_means, mlp_means, S_attn, S_mlp)
    hm.install_capture(want_grad=True)

    model.zero_grad(set_to_none=True)
    logits = model(ids).logits[0, -1]
    logp = F.log_softmax(logits, dim=-1)[tgt]
    logp.backward()

    n_heads, HD, I = hm.n_heads, hm.head_dim, hm.intermediate
    grad_attn, grad_mlp, de_attn, de_mlp = [], [], [], []
    for L in range(hm.n_layers):
        a = hm.captured[f"attn_{L}"]
        a4 = a.view(a.shape[0], a.shape[1], n_heads, HD)
        g4 = a.grad.view(a.shape[0], a.shape[1], n_heads, HD)
        # grad norm per head
        grad_attn.append(g4.pow(2).sum(dim=(0, 1, 3)).sqrt())
        # direct effect per head: sum_(B,S,HD) (a - mean) * grad,  abs
        am = attn_means[L].view(1, 1, n_heads, HD).to(a4.dtype)
        de = ((a4 - am) * g4).sum(dim=(0, 1, 3))
        de_attn.append(de.abs())

        m = hm.captured[f"mlp_{L}"]
        gm = m.grad
        grad_mlp.append(gm.pow(2).sum(dim=(0, 1)).sqrt())
        mm = mlp_means[L].view(1, 1, I).to(m.dtype)
        de = ((m - mm) * gm).sum(dim=(0, 1))
        de_mlp.append(de.abs())
    hm.clear()
    model.zero_grad(set_to_none=True)
    return grad_attn, grad_mlp, de_attn, de_mlp


def flatten(attn_scores, mlp_scores):
    parts = []
    layout = []
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


def mask_from_flat(flat_mask, layout, n_layers, n_heads, intermediate):
    S_attn = [None] * n_layers
    S_mlp = [None] * n_layers
    for kind, L, a, b in layout:
        if kind == "attn":
            S_attn[L] = flat_mask[a:b]
        else:
            S_mlp[L] = flat_mask[a:b]
    return S_attn, S_mlp


def sweep_singleshot(model, hm, attn_means, mlp_means, ids, tgt, full_logits,
                     flat_scores, layout, fractions, n_components):
    """For each fraction, keep top-frac by score, mean-ablate rest, record metrics."""
    sorted_, _ = torch.sort(flat_scores)
    out = []
    min_argmax = None
    min_top3 = None
    min_kl = None
    for f in fractions:
        keep = max(1, int(round(f * n_components)))
        idx = max(0, n_components - keep)
        tau = float(sorted_[idx].item())
        flat_mask = (flat_scores >= tau)
        S_attn, S_mlp = mask_from_flat(flat_mask, layout, hm.n_layers,
                                        hm.n_heads, hm.intermediate)
        hm.clear()
        hm.install_ablation(attn_means, mlp_means, S_attn, S_mlp)
        rl = predict_logits(model, ids)
        hm.clear()
        m = metrics(full_logits, rl, tgt)
        kept = int(flat_mask.sum().item())
        m.update({"frac": f, "kept": kept, "kept_frac": kept / n_components})
        out.append(m)
        if m["argmax_preserved"] and (min_argmax is None or f < min_argmax):
            min_argmax = f
        if m["top3_contains_tgt"] and (min_top3 is None or f < min_top3):
            min_top3 = f
        if m["kl_below_threshold"] and (min_kl is None or f < min_kl):
            min_kl = f
    return {"min_argmax_frac": min_argmax, "min_top3_frac": min_top3,
            "min_kl_frac": min_kl, "sweep": out}


def bottom_up(model, hm, attn_means, mlp_means, ids, tgt, full_logits,
              n_components, seed_flat, growth_score_fn,
              seed_frac=0.01, step_frac=0.05, max_iters=12):
    """Generic bottom-up: seed by `seed_flat` (precomputed), grow by `growth_score_fn`
       which takes (model, hm, ids, tgt, S_attn, S_mlp) and returns a flat score."""
    sorted_seed, _ = torch.sort(seed_flat)
    keep = max(1, int(round(seed_frac * n_components)))
    tau = float(sorted_seed[n_components - keep].item())
    flat_mask = (seed_flat >= tau)

    traj = []
    for it in range(max_iters + 1):
        S_attn, S_mlp = mask_from_flat(flat_mask, _layout, hm.n_layers,
                                        hm.n_heads, hm.intermediate)
        hm.clear()
        hm.install_ablation(attn_means, mlp_means, S_attn, S_mlp)
        rl = predict_logits(model, ids)
        hm.clear()
        m = metrics(full_logits, rl, tgt)
        kept = int(flat_mask.sum().item())
        m.update({"iter": it, "kept": kept, "kept_frac": kept / n_components})
        traj.append(m)
        if m["argmax_preserved"] or it == max_iters:
            break
        # restricted score
        rscore = growth_score_fn(model, hm, ids, tgt, S_attn, S_mlp)
        rscore = rscore.clone()
        rscore[flat_mask] = -1.0
        k = max(1, int(round(step_frac * n_components)))
        topk_vals, topk_idx = torch.topk(rscore, k, largest=True, sorted=False)
        new = torch.zeros_like(flat_mask)
        new[topk_idx] = True
        flat_mask = flat_mask | new
    return traj, flat_mask


# global layout reused across calls
_layout = None


def main():
    print(f"loading {MODEL}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    print(f"  loaded in {time.time()-t0:.1f}s, "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    hm = HookManager(model)
    n_components = hm.n_layers * hm.n_heads + hm.n_layers * hm.intermediate
    print(f"  layers={hm.n_layers}, heads={hm.n_heads}, head_dim={hm.head_dim}, "
          f"intermediate={hm.intermediate}", flush=True)
    print(f"  n_components={n_components}", flush=True)

    print("computing baseline stats...", flush=True)
    t0 = time.time()
    attn_means, mlp_means, attn_var, mlp_var, n_pos = compute_baseline_stats(
        model, tok, hm, BASELINE_CORPUS)
    print(f"  baseline done in {time.time()-t0:.1f}s, n_pos={n_pos}", flush=True)

    # variance scores (per-component)
    var_attn = [v.mean(dim=1) for v in attn_var]
    var_mlp = mlp_var
    var_flat, var_layout = flatten(var_attn, var_mlp)
    global _layout
    _layout = var_layout

    results = {"meta": {"model": MODEL, "n_components": n_components,
                        "fractions": FRACTIONS, "kl_threshold": KL_THRESHOLD},
               "per_type": []}
    overall_t0 = time.time()

    for qtype, prompts in PROMPTS.items():
        print(f"\n=== {qtype} ===", flush=True)
        type_block = {"qtype": qtype, "results": []}
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids
            full_logits = predict_logits(model, ids).detach()
            tgt = int(full_logits.argmax().item())
            tgt_tok = tok.decode([tgt])
            margin = float(full_logits[tgt].item() -
                          full_logits.masked_select(
                              torch.arange(full_logits.numel()) != tgt).max().item())
            print(f"\n  -- prompt {i}: '{p}' -> '{tgt_tok}' (margin={margin:.3f}) --", flush=True)
            t_p = time.time()

            # Single full-model gradient + DE
            ga, gm, da, dm = compute_grad_and_de(model, hm, ids, tgt,
                                                  attn_means, mlp_means)
            grad_flat, _ = flatten(ga, gm)
            de_flat, _ = flatten(da, dm)

            # Cond A: |grad|
            res_A = sweep_singleshot(model, hm, attn_means, mlp_means, ids, tgt,
                                      full_logits, grad_flat, _layout,
                                      FRACTIONS, n_components)
            print(f"    A (grad)     argmax={res_A['min_argmax_frac']} "
                  f"top3={res_A['min_top3_frac']} kl={res_A['min_kl_frac']}",
                  flush=True)

            # Cond C: variance
            res_C = sweep_singleshot(model, hm, attn_means, mlp_means, ids, tgt,
                                      full_logits, var_flat, _layout,
                                      FRACTIONS, n_components)
            print(f"    C (var)      argmax={res_C['min_argmax_frac']} "
                  f"top3={res_C['min_top3_frac']} kl={res_C['min_kl_frac']}",
                  flush=True)

            # Cond D: direct effect
            res_D = sweep_singleshot(model, hm, attn_means, mlp_means, ids, tgt,
                                      full_logits, de_flat, _layout,
                                      FRACTIONS, n_components)
            print(f"    D (DE)       argmax={res_D['min_argmax_frac']} "
                  f"top3={res_D['min_top3_frac']} kl={res_D['min_kl_frac']}",
                  flush=True)

            # Growth-score function: restricted DE
            def grow_de(model, hm, ids, tgt, S_attn, S_mlp):
                ga2, gm2, da2, dm2 = compute_grad_and_de(
                    model, hm, ids, tgt, attn_means, mlp_means, S_attn, S_mlp)
                df, _ = flatten(da2, dm2)
                return df

            # Cond E: paper structure with DE seed + DE grow
            traj_E, mask_E = bottom_up(model, hm, attn_means, mlp_means, ids, tgt,
                                        full_logits, n_components,
                                        seed_flat=de_flat, growth_score_fn=grow_de,
                                        seed_frac=0.01, step_frac=0.05, max_iters=12)
            min_E = next((t["kept_frac"] for t in traj_E if t["argmax_preserved"]), None)
            print(f"    E (DE+DE)    argmax_at={min_E}", flush=True)

            # Cond F: variance seed + DE grow (recommendation 4)
            traj_F, mask_F = bottom_up(model, hm, attn_means, mlp_means, ids, tgt,
                                        full_logits, n_components,
                                        seed_flat=var_flat, growth_score_fn=grow_de,
                                        seed_frac=0.01, step_frac=0.05, max_iters=12)
            min_F = next((t["kept_frac"] for t in traj_F if t["argmax_preserved"]), None)
            print(f"    F (var+DE)   argmax_at={min_F}", flush=True)

            entry = {
                "prompt": p, "target_id": tgt, "target_tok": tgt_tok,
                "margin_full": margin,
                "A": res_A, "C": res_C, "D": res_D,
                "E_traj": traj_E, "E_argmax_frac": min_E,
                "F_traj": traj_F, "F_argmax_frac": min_F,
                "elapsed_s": time.time() - t_p,
            }
            type_block["results"].append(entry)
            print(f"    elapsed: {time.time()-t_p:.1f}s", flush=True)

        results["per_type"].append(type_block)

    results["meta"]["total_elapsed_s"] = time.time() - overall_t0
    out_path = r"./outputs/paper_test_v3_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time()-overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
