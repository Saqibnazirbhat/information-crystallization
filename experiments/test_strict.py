"""
Test of the Information Crystallization framework (Bhat, March 2026).
Falsifiable experiment on GPT-2 small (124M params), CPU.
"""
import time, json
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(0)
DEVICE = "cpu"
MODEL_NAME = "gpt2"

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

# Sweep keep-fractions DESCENDING (less aggressive first).
# We expect |S|/d to live in 0.5-0.95 regime per smoke test.
FRACTIONS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10]


def predict(model, ids):
    with torch.no_grad():
        return int(model(ids).logits[0, -1].argmax().item())


def fullmodel_grad_abs(model, ids, target_id):
    """|∇_θ log P(target | X)|, on whatever the params currently are."""
    model.zero_grad(set_to_none=True)
    logits = model(ids).logits[0, -1]
    logp = F.log_softmax(logits, dim=-1)[target_id]
    logp.backward()
    out = {n: p.grad.detach().abs().clone() for n, p in model.named_parameters()
           if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return out


def thresholds_from_scores(score_dict, fractions):
    """Sort once, then index by fraction. Returns dict frac->tau, plus n."""
    flats = torch.cat([s.flatten() for s in score_dict.values()])
    n = flats.numel()
    print(f"      sorting {n:,} scores...", flush=True)
    t0 = time.time()
    sorted_flat, _ = torch.sort(flats)  # ascending
    print(f"      sort took {time.time()-t0:.1f}s", flush=True)
    thresholds = {}
    for f in fractions:
        keep = max(1, int(round(f * n)))
        idx = max(0, min(n - 1, n - keep))
        thresholds[f] = float(sorted_flat[idx].item())
    del sorted_flat, flats
    return thresholds, n


def apply_threshold(model, originals, score_dict, tau):
    with torch.no_grad():
        for nm, p in model.named_parameters():
            if nm in score_dict:
                mask = (score_dict[nm] >= tau).to(p.dtype)
                p.data = originals[nm] * mask
            else:
                p.data = originals[nm].clone()


def restore(model, originals):
    with torch.no_grad():
        for nm, p in model.named_parameters():
            p.data = originals[nm].clone()


def kept_count(score_dict, tau):
    return sum(int((s >= tau).sum().item()) for s in score_dict.values())


def sweep_condition(model, originals, score_dict, ids, target_id, fractions):
    """Sweep keep-fractions descending, find min preserving fraction."""
    taus, n = thresholds_from_scores(score_dict, fractions)
    sweep = []
    min_pres = None
    for f in fractions:
        tau = taus[f]
        apply_threshold(model, originals, score_dict, tau)
        pred = predict(model, ids)
        kept = kept_count(score_dict, tau)
        sweep.append({"frac": f, "tau": tau, "kept": kept,
                      "kept_frac": kept / n, "pred": pred,
                      "preserved": pred == target_id})
        if pred == target_id and (min_pres is None or f < min_pres):
            min_pres = f
        # short-circuit: once we hit a non-preserving fraction below a
        # preserving one, we can stop (heuristic; not strict monotone)
    restore(model, originals)
    return min_pres, sweep, n


def cond_B_restricted(model, originals, ids, target_id, n_total,
                      seed_frac=0.005, step_frac=0.05, max_iters=15):
    """
    Paper Alg. 1, chunked. Seed by full-model gradient (top seed_frac),
    grow by restricted-model gradient (add step_frac per iter).
    """
    # Seed
    grad_abs = fullmodel_grad_abs(model, ids, target_id)
    taus, n = thresholds_from_scores(grad_abs, [seed_frac])
    tau = taus[seed_frac]
    mask = {nm: (g >= tau) for nm, g in grad_abs.items()}
    apply_threshold_from_mask(model, originals, mask)
    del grad_abs

    traj = []
    pred = predict(model, ids)
    cur_kept = sum(int(m.sum().item()) for m in mask.values())
    traj.append({"iter": 0, "kept": cur_kept, "kept_frac": cur_kept / n_total,
                 "pred": pred, "preserved": pred == target_id})

    for it in range(1, max_iters + 1):
        if pred == target_id:
            break
        # restricted gradient at current S
        rgrad = fullmodel_grad_abs(model, ids, target_id)
        # exclude already-kept entries from ranking
        for nm in rgrad:
            rgrad[nm] = rgrad[nm].clone()
            rgrad[nm][mask[nm]] = -1.0
        # add top step_frac of remaining
        flats = torch.cat([s.flatten() for s in rgrad.values()])
        k = max(1, int(round(step_frac * n_total)))
        topk_vals, _ = torch.topk(flats, k, largest=True, sorted=False)
        tau_new = float(topk_vals.min().item())
        del flats, topk_vals
        for nm in mask:
            mask[nm] = mask[nm] | (rgrad[nm] >= tau_new)
        del rgrad
        apply_threshold_from_mask(model, originals, mask)
        cur_kept = sum(int(m.sum().item()) for m in mask.values())
        pred = predict(model, ids)
        traj.append({"iter": it, "kept": cur_kept,
                     "kept_frac": cur_kept / n_total,
                     "pred": pred, "preserved": pred == target_id})

    flipped_frac = (cur_kept / n_total) if pred == target_id else None
    restore(model, originals)
    return flipped_frac, traj, mask


def apply_threshold_from_mask(model, originals, mask):
    with torch.no_grad():
        for nm, p in model.named_parameters():
            if nm in mask:
                p.data = originals[nm] * mask[nm].to(p.dtype)
            else:
                p.data = originals[nm].clone()


def transfer_test(model, originals, mask, prompts_with_targets, tok):
    apply_threshold_from_mask(model, originals, mask)
    out = []
    for prompt, ids, tgt in prompts_with_targets:
        pred = predict(model, ids)
        out.append({"prompt": prompt, "tgt": tgt, "pred": pred,
                    "preserved": pred == tgt,
                    "tgt_tok": tok.decode([tgt]),
                    "pred_tok": tok.decode([pred])})
    restore(model, originals)
    return out


def main():
    print(f"loading {MODEL_NAME}...", flush=True)
    t0 = time.time()
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time()-t0:.1f}s, params={n_total:,}", flush=True)

    print("snapshotting originals...", flush=True)
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}

    results = {"meta": {"model": MODEL_NAME, "n_params": n_total,
                        "fractions": FRACTIONS}, "per_type": []}
    overall_t0 = time.time()

    for qtype, prompts in PROMPTS.items():
        print(f"\n=== {qtype} ===", flush=True)
        baselines = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
            tgt = predict(model, ids)
            baselines.append((p, ids, tgt))
            print(f"  '{p}' -> '{tok.decode([tgt])}' (id={tgt})", flush=True)

        type_block = {"qtype": qtype, "results": []}
        for i, (p, ids, tgt) in enumerate(baselines):
            print(f"\n  -- prompt {i}: '{p}' --", flush=True)
            t_prompt = time.time()

            # --- Cond A: full-model gradient saliency ---
            print("    [A] computing full-model gradient...", flush=True)
            grad_abs = fullmodel_grad_abs(model, ids, tgt)
            min_a, sweep_a, _ = sweep_condition(
                model, originals, grad_abs, ids, tgt, FRACTIONS)
            del grad_abs
            print(f"    [A] min preserving frac = {min_a}", flush=True)
            for s in sweep_a:
                if s["preserved"]:
                    print(f"        f={s['frac']} kept={s['kept_frac']:.4f} OK", flush=True)
                    break
            else:
                print(f"        NEVER preserved in sweep", flush=True)

            # --- Cond C: weight magnitude ---
            print("    [C] computing magnitudes...", flush=True)
            mags = {n: p.detach().abs().clone()
                    for n, p in model.named_parameters()}
            min_c, sweep_c, _ = sweep_condition(
                model, originals, mags, ids, tgt, FRACTIONS)
            del mags
            print(f"    [C] min preserving frac = {min_c}", flush=True)

            # --- Cond B: paper's restricted-gradient bottom-up ---
            print("    [B] running restricted-gradient bottom-up...", flush=True)
            min_b, traj_b, mask_b = cond_B_restricted(
                model, originals, ids, tgt, n_total,
                seed_frac=0.005, step_frac=0.05, max_iters=15)
            print(f"    [B] flipped at frac = {min_b}", flush=True)
            print("        traj: " +
                  ", ".join(f"{t['kept_frac']:.3f}{'!' if t['preserved'] else ''}"
                            for t in traj_b),
                  flush=True)

            entry = {
                "prompt": p, "target_id": tgt, "target_tok": tok.decode([tgt]),
                "A_min_frac": min_a, "A_sweep": sweep_a,
                "C_min_frac": min_c, "C_sweep": sweep_c,
                "B_flip_frac": min_b, "B_traj": traj_b,
                "elapsed_s": time.time() - t_prompt,
            }

            # transfer test from prompt 0 only
            if i == 0 and min_b is not None:
                others = [(bp, bids, btgt) for j, (bp, bids, btgt)
                          in enumerate(baselines) if j != 0]
                tr = transfer_test(model, originals, mask_b, others, tok)
                entry["transfer_from_prompt0"] = tr
                print("        transfer of B-mask to siblings:", flush=True)
                for r in tr:
                    print(f"          '{r['prompt']}' tgt='{r['tgt_tok']}' "
                          f"got='{r['pred_tok']}' ok={r['preserved']}",
                          flush=True)

            type_block["results"].append(entry)

        results["per_type"].append(type_block)

    results["meta"]["total_elapsed_s"] = time.time() - overall_t0
    out = r"./outputs/paper_test_raw.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {out}", flush=True)
    print(f"total elapsed: {time.time()-overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
