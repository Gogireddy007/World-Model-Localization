"""
Temporal-Rollout Patching on a trained H-JEPA: does the time course matter?

TRP's claim (Sec. 4) is that where a component's intervention effect lands in
TIME carries information a single-step score throws away. We test the claim
that matters in practice: do components selected by their TRP effect at a
later, invariant-relevant moment differ from those selected by their
immediate effect, and are they more causally necessary for the invariant?

Setup (permanence). TRP pairs: two physically valid clips of the same scene
whose occluded object had slightly different velocities BEFORE occlusion, so
they re-emerge at different places. Components: the fast predictor's GRU
state units (the model's memory). For each unit c we patch s_c at the last
visible step t_o from the x- run into the x+ run, continue the x+ rollout
through the occlusion, and record

    immediate(c) = || z_hat_patched(t_o) - z_hat_clean(t_o) ||           (single step)
    TRP(c)       = < z_hat_clean(t_e-1) - z_hat_patched(t_e-1), u_e > / ||z-_e - z+_e||

where t_e is the re-emergence frame and u_e the axis between the two actual
re-emergence latents: how much of the counterfactual re-emergence the unit
carried through the occlusion.

Necessity test (held-out, standard permanence matched pairs of Sec. 5): we
resample-ablate the top-m units under each ranking over the last W context
steps and measure the drop in the permanence score L_perm, and, as a
specificity check, in L_collision. Random same-size unit sets give the baseline.

Run: python trp_experiment.py --ckpt checkpoints/hjepa_r64_s0.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math

import numpy as np
import torch

from environment import ControlledSampler, InvariantFlags, build_clip
from measure_epsilon import load_model, pairs_cache, pair_metric
from train_hjepa import build_cache


def trp_pairs(n, seed0=3_000_000):
    """(frames+, frames-, actions, t_o, t_e) for the occluded object of x+."""
    sampler = ControlledSampler(seed=seed0)
    rng = np.random.default_rng(seed0)
    out = []
    s = seed0
    while len(out) < n:
        s += 1
        f = sampler.sample_flags(); f.permanence = True
        base = sampler.generate_clip(s, flags=f)
        i = base.meta["roles"].get("occluded")
        if i is None or not base.events["permanence"]:
            continue
        objs = copy.deepcopy(base.meta["objects"])
        ang = rng.uniform(0.25, 0.45) * rng.choice([-1, 1])
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        objs[i].vel = R @ objs[i].vel
        foil = build_clip(objs, base.meta["occluders"], base.actions, base.flags, s, base.meta["roles"])
        vis = base.object_visible[:, i].astype(int)
        d = np.diff(vis)
        downs = np.where(d == -1)[0]
        if not len(downs):
            continue
        t_o = int(downs[0])                     # last visible frame
        ups = np.where((d == 1) & (np.arange(len(d)) > t_o))[0]
        if not len(ups):
            continue
        t_e = int(ups[0]) + 1                   # first visible frame after occlusion
        if t_e - t_o < 4 or t_o < 4:
            continue
        if not np.any(base.frames[t_e] != foil.frames[t_e]):
            continue
        out.append((base.frames, foil.frames, base.actions, t_o, t_e))
    return out


def _t(fr):
    return torch.from_numpy(fr.astype(np.float32) / 255.0)[None, :, None]


@torch.no_grad()
def trp_scores(model, pairs):
    fast = model.fast
    S = fast.state
    imm = np.zeros((len(pairs), S)); trp = np.zeros((len(pairs), S))
    for n, (fp, fm, ac, t_o, t_e) in enumerate(pairs):
        zp = model.encode_seq(_t(fp)); zm = model.encode_seq(_t(fm))
        a = torch.from_numpy(ac)[None]
        st_p, _, _, _ = fast.run(zp[:, :t_o + 1], a[:, :t_o + 1])
        st_m, _, _, _ = fast.run(zm[:, :t_o + 1], a[:, :t_o + 1])
        s_p = st_p[:, -1]; s_m = st_m[:, -1]
        # clean continuation through the occlusion
        _, _, zh_clean, _ = fast.run(zp[:, t_o + 1:t_e], a[:, t_o + 1:t_e], s0=s_p[None])
        zh0_clean = zp[:, t_o] + fast.out(fast.readout(s_p))
        ze_p = model.target_encoder(_t(fp)[:, t_e]); ze_m = model.target_encoder(_t(fm)[:, t_e])
        u = (ze_m - ze_p); scale = u.norm().clamp_min(1e-6); u = u / scale
        # batch all units: state copies with unit c patched
        s_batch = s_p.repeat(S, 1)
        idx = torch.arange(S)
        s_batch[idx, idx] = s_m[0, idx]
        zh0 = zp[:, t_o] + fast.out(fast.readout(s_batch))
        imm[n] = (zh0 - zh0_clean).norm(dim=-1).numpy()
        _, _, zh, _ = fast.run(zp[:, t_o + 1:t_e].repeat(S, 1, 1), a[:, t_o + 1:t_e].repeat(S, 1, 1),
                               s0=s_batch[None].contiguous())
        trp[n] = (((zh_clean[:, -1] - zh[:, -1]) @ u[0]) / scale).numpy()
    return imm, trp


@torch.no_grad()
def reference_states(model, n=2000, seed=0):
    frames, actions = build_cache(4000, seed=0)
    rng = np.random.default_rng(seed + 11)
    st = []
    for i in rng.integers(0, len(frames), size=n // 16 + 1):
        s, _, _, _ = model.fast.run(model.encode_seq(_t(frames[i])), torch.from_numpy(actions[i])[None])
        st.append(s[0, rng.integers(8, s.shape[1], size=16)])
    return torch.cat(st)[:n]


@torch.no_grad()
def ablated_scores(model, pairs, units, ref, window, g, metric_split):
    """Pair metric with `units` of the GRU state resampled over the last `window` steps."""
    fast = model.fast
    idx_eval = metric_split
    ms = []
    for n in idx_eval:
        fp, fm, ac, t = pairs[n]
        z = model.encode_seq(_t(fp)[:, :t]); a = torch.from_numpy(ac)[None, :t]
        t0 = max(0, t - window)
        st, _, _, sN = fast.run(z[:, :t0], a[:, :t0]) if t0 > 0 else (None, None, None, None)
        s = sN
        for tt in range(t0, t):
            _, _, _, s = fast.run(z[:, tt:tt + 1], a[:, tt:tt + 1], s0=s)
            if len(units):
                r = ref[torch.randint(0, len(ref), (1,), generator=g)]
                s = s.clone(); s[0, 0, units] = r[0, units]
        zh = z[:, -1] + fast.out(fast.readout(s[0]))
        ms.append(zh[0])
    return torch.stack(ms)


def necessity(model, invariant, rankings, m, window, ref, n_pairs=450, seed=0, n_random=20):
    pr = pairs_cache(invariant, n_pairs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pr)); metric_ix, eval_ix = perm[:150], perm[150:]
    ts = sorted(set(int(pr[i][3]) for i in eval_ix))
    with torch.no_grad():
        zpe = torch.stack([model.target_encoder(_t(pr[i][0])[:, pr[i][3]])[0] for i in eval_ix])
        zme = torch.stack([model.target_encoder(_t(pr[i][1])[:, pr[i][3]])[0] for i in eval_ix])
    denom = float(((zpe - zme) ** 2).sum(-1).mean())
    g = torch.Generator().manual_seed(seed)
    base = pair_metric(ablated_scores(model, pr, [], ref, window, g, eval_ix), zpe, zme, denom).mean().item()
    out = {"L_clean": base}
    for name, units in rankings.items():
        L = pair_metric(ablated_scores(model, pr, units[:m], ref, window, g, eval_ix), zpe, zme, denom).mean().item()
        out[name] = base - L
    S = model.fast.state
    rand = []
    for r in range(n_random):
        units = list(rng.choice(S, size=m, replace=False))
        rand.append(base - pair_metric(ablated_scores(model, pr, units, ref, window, g, eval_ix),
                                       zpe, zme, denom).mean().item())
    out["random_mean"] = float(np.mean(rand)); out["random_p95"] = float(np.percentile(rand, 95))
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-trp", type=int, default=150)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", default="results/trp_experiment.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    model, cfg = load_model(a.ckpt)
    pairs = trp_pairs(a.n_trp)
    imm, trp = trp_scores(model, pairs)
    imm_m, trp_m = imm.mean(0), trp.mean(0)
    rank_imm = list(np.argsort(-imm_m)); rank_trp = list(np.argsort(-np.abs(trp_m)))
    overlap = len(set(rank_imm[:a.m]) & set(rank_trp[:a.m]))
    rho = spearman(imm_m, np.abs(trp_m))
    print(f"TRP pairs: {len(pairs)}; occlusion length median {np.median([p[4] - p[3] for p in pairs]):.0f} steps")
    print(f"Spearman(immediate, |TRP|) over {len(imm_m)} state units = {rho:.3f}; "
          f"top-{a.m} overlap = {overlap}/{a.m}")
    ref = reference_states(model)
    rankings = {"trp": [int(x) for x in rank_trp], "immediate": [int(x) for x in rank_imm]}
    res = {"spearman_immediate_vs_trp": rho, "top_overlap": overlap, "m": a.m, "window": a.window,
           "n_trp_pairs": len(pairs)}
    for inv in ["permanence", "collision"]:
        r = necessity(model, inv, rankings, a.m, a.window, ref)
        res[inv] = r
        print(f"{inv:10s}: L_clean {r['L_clean']:+.3f} | drop from ablating top-{a.m} TRP units {r['trp']:+.3f}, "
              f"immediate units {r['immediate']:+.3f}, random {r['random_mean']:+.3f} (p95 {r['random_p95']:+.3f})")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
