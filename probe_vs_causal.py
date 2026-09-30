"""
Decodable vs. used (pre-registered PR3) on the trained world models.

PR3: "the location identified by linear probing differs from the causal
circuit; ablating the probe-identified location produces <= 0.5x the drop of
ablating the causal circuit" for at least one invariant.

For each d = 64 world model we:
  1. Fit linear probes on the fast predictor's readout units, over held-out
     training-distribution contexts, for two permanence-relevant variables:
       - occlusion state: is some object currently hidden (logistic probe);
       - hidden position: the (x, y) of the hidden object, on occluded steps
         only (ridge regression).
     Decodability is reported on a held-out split (accuracy / R^2).
  2. Rank units by standardised probe weight (|w_c| * std(h_c)): the
     probe-located units.
  3. Rank units by the causal attribution a_perm[c] (measure_epsilon, Lemma 1)
     on the selection split of the permanence matched pairs: the causal units.
  4. Report the drop in the permanence score from resample-ablating the top-m
     units of each ranking on the evaluation split (exact by Lemma 1, checked
     against Monte-Carlo ablation), and random m-unit sets.

Run: python probe_vs_causal.py       -> results/probe_vs_causal.json
"""

from __future__ import annotations

import glob
import json
import re

import numpy as np
import torch

from environment import ControlledSampler, CLIP_LEN
from measure_epsilon import (load_model, pairs_cache, run_pairs, reference_readouts,
                             attributions, mc_resample_effect, pair_metric)


@torch.no_grad()
def probe_dataset(model, n_clips=600, seed=77):
    """Readout activations with occlusion labels on fresh clips (disjoint seeds)."""
    sampler = ControlledSampler(seed=seed)
    H, occ, pos, occ_mask = [], [], [], []
    for i in range(n_clips):
        c = sampler.generate_clip(5_000_000 + i)
        fr = torch.from_numpy(c.frames.astype(np.float32) / 255.0)[None, :, None]
        ac = torch.from_numpy(c.actions)[None]
        _, h, _, _ = model.fast.run(model.encode_seq(fr), ac)
        n = int(np.sum(~np.isnan(c.object_positions[0, :, 0])))
        for t in range(6, CLIP_LEN - 1):          # h[t] predicts frame t+1
            hidden = [j for j in range(n) if not c.object_visible[t, j]]
            H.append(h[0, t].numpy())
            occ.append(float(len(hidden) > 0))
            if hidden:
                pos.append(c.object_positions[t, hidden[0]] / 64.0)
                occ_mask.append(True)
            else:
                pos.append(np.zeros(2)); occ_mask.append(False)
    return np.array(H), np.array(occ), np.array(pos), np.array(occ_mask)


def fit_logistic(X, y, l2=1e-3, steps=2000):
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], 5e-2)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().numpy(), float(b.item())


def fit_ridge(X, Y, l2=1e-2):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    W = np.linalg.solve(Xa.T @ Xa + l2 * np.eye(Xa.shape[1]), Xa.T @ Y)
    return W[:-1], W[-1]


def analyse(ckpt, m=6, n_random=300, seed=0):
    model, _ = load_model(ckpt)
    rng = np.random.default_rng(seed)
    # --- probes -----------------------------------------------------------
    H, occ, pos, mask = probe_dataset(model)
    mu, sd = H.mean(0), H.std(0) + 1e-8
    Z = (H - mu) / sd
    idx = rng.permutation(len(Z)); tr, te = idx[: len(Z) // 2], idx[len(Z) // 2:]
    w_occ, b_occ = fit_logistic(Z[tr], occ[tr])
    acc = float((((Z[te] @ w_occ + b_occ) > 0) == (occ[te] > 0.5)).mean())
    base_acc = float(max(occ[te].mean(), 1 - occ[te].mean()))
    trm, tem = tr[mask[tr]], te[mask[te]]
    W_pos, b_pos = fit_ridge(Z[trm], pos[trm])
    pred = Z[tem] @ W_pos + b_pos
    r2 = float(1 - ((pred - pos[tem]) ** 2).sum() / ((pos[tem] - pos[tem].mean(0)) ** 2).sum())
    # standardised weights: probes were fit on z-scored units, so |w| is already standardised
    probe_occ_units = list(np.argsort(-np.abs(w_occ))[:m])
    probe_pos_units = list(np.argsort(-np.linalg.norm(W_pos, axis=1))[:m])
    # --- causal attributions on permanence pairs --------------------------
    pr = pairs_cache("permanence", 450)
    h, zh, zp, zm, _ = run_pairs(model, pr)
    perm = np.random.default_rng(0).permutation(len(pr))
    thirds = np.array_split(perm, 3)
    norms = (zp - zm).norm(dim=-1)
    kappa = 0.1 * norms[torch.as_tensor(thirds[0])].median()
    keep = (norms > kappa).numpy()
    met, sel, ev = [torch.as_tensor([i for i in ix if keep[i]]) for ix in thirds]
    denom = float(((zp[met] - zm[met]) ** 2).sum(-1).mean())
    ref = reference_readouts(model, seed=seed)
    W = model.fast.out.weight
    A_sel = attributions(W, h[sel], ref.mean(0), zp[sel], zm[sel], denom).mean(0).numpy()
    P_ev = attributions(W, h[ev], ref.mean(0), zp[ev], zm[ev], denom).numpy()
    A_ev = P_ev.mean(0)
    causal_units = list(np.argsort(-A_sel)[:m])
    whole = float(A_ev.sum())

    def drop(units):
        return float(A_ev[list(units)].sum())

    def boot(units, n=1000):
        v = P_ev[:, list(units)].sum(1)
        return [float(np.percentile([rng.choice(v, len(v)).mean() for _ in range(n)], q)) for q in (2.5, 97.5)]

    rand = [drop(rng.choice(len(A_ev), m, replace=False)) for _ in range(n_random)]
    mc_check = mc_resample_effect(W, h[ev], zh[ev], zp[ev], zm[ev], causal_units, ref, n_draws=50, denom=denom)
    out = dict(
        ckpt=ckpt, m=m, n_units=len(A_ev),
        probe_occlusion_acc=acc, probe_occlusion_majority=base_acc, probe_position_r2=r2,
        whole_layer_effect=whole,
        causal_drop=drop(causal_units), causal_ci=boot(causal_units),
        probe_occ_drop=drop(probe_occ_units), probe_occ_ci=boot(probe_occ_units),
        probe_pos_drop=drop(probe_pos_units), probe_pos_ci=boot(probe_pos_units),
        random_mean=float(np.mean(rand)), random_p95=float(np.percentile(rand, 95)),
        overlap_causal_probe_occ=len(set(causal_units) & set(probe_occ_units)),
        overlap_causal_probe_pos=len(set(causal_units) & set(probe_pos_units)),
        additivity_check=dict(analytic=drop(causal_units), monte_carlo=mc_check),
    )
    ratio = lambda x: x / out["causal_drop"] if out["causal_drop"] > 0 else float("nan")
    out["ratio_probe_occ"] = ratio(out["probe_occ_drop"]); out["ratio_probe_pos"] = ratio(out["probe_pos_drop"])
    out["PR3_pass"] = bool(out["causal_drop"] > 0 and min(out["ratio_probe_occ"], out["ratio_probe_pos"]) <= 0.5)
    return out


def main():
    res = []
    for ck in sorted(glob.glob("checkpoints/hjepa_r64_s*.pt")):
        torch.manual_seed(0)
        r = analyse(ck)
        res.append(r)
        print(f"{ck}: probe acc {r['probe_occlusion_acc']:.3f} (majority {r['probe_occlusion_majority']:.3f}), "
              f"position R2 {r['probe_position_r2']:.3f} | drop: causal {r['causal_drop']:+.3f} "
              f"[{r['causal_ci'][0]:+.3f},{r['causal_ci'][1]:+.3f}], probe-occ {r['probe_occ_drop']:+.3f}, "
              f"probe-pos {r['probe_pos_drop']:+.3f}, random {r['random_mean']:+.3f} (p95 {r['random_p95']:+.3f}) | "
              f"ratios {r['ratio_probe_occ']:.2f}/{r['ratio_probe_pos']:.2f} | overlap {r['overlap_causal_probe_occ']}/"
              f"{r['overlap_causal_probe_pos']} | MC {r['additivity_check']['monte_carlo']:+.3f} | PR3 {r['PR3_pass']}",
              flush=True)
    json.dump(res, open("results/probe_vs_causal.json", "w"), indent=1)
    print("wrote results/probe_vs_causal.json")


if __name__ == "__main__":
    main()
