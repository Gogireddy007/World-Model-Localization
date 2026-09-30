"""
Measure epsilon-modularity on a trained H-JEPA (v2 pipeline).

For each invariant k we build matched pairs (x+, x-) that are pixel-identical up
to a divergence frame t_div, after which x+ follows physics and x- violates the
invariant. The predictor sees the shared context (frames < t_div) and predicts
the latent of frame t_div. With target-encoder latents z+, z- of the two
candidate futures, the per-pair preference

    m_k(pair) = ( ||z_hat - z-||^2 - ||z_hat - z+||^2 ) / ||z+ - z-||^2
              = 2 < z_hat - (z+ + z-)/2 , z+ - z- > / ||z+ - z-||^2

is +1 when the prediction sits at the correct future, -1 at the foil, 0 when
equidistant. Because a per-pair denominator is heavy-tailed when a pair's two
futures nearly coincide, measure() uses the pooled form with the denominator
replaced by E||z+ - z-||^2 fixed on the metric split; L_k = E[m_k] over pairs
with ||z+ - z-|| above a floor kappa (0.1 x median). m_k is linear in z_hat. (The v1 /
pre-registered form <z_hat - z+,u> - <z_hat - z-,u> does not depend on z_hat at
all; see preregistration.md, Amendment 1.)

Components are the units of the fast predictor's readout layer, which feeds a
linear head, so the effect of resample-ablating a set C on L_j is exactly
additive in expectation:

    Delta_j(C) = sum_{c in C} a_j[c],
    a_j[c] = E_pairs[ 2 (h_c - E_ref h_c) <W_out[:, c], z+ - z-> / ||z+ - z-||^2 ].

We (i) verify this against Monte-Carlo resample ablation, (ii) compute the
Theorem-1 certificate on the attribution matrix A = [a_1; ...; a_K], (iii) find
circuits on a selection split with a selectivity-aware search and with the v1
"top-|CII|" rule, and (iv) report epsilon, on-target effect, sparsity and the
random-subset concentration ratio on a held-out evaluation split, with
bootstrap confidence intervals.

Run:  python measure_epsilon.py --ckpt checkpoints/hjepa_r64_s0.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from environment import ControlledSampler, MATCHED_PAIRS, INVARIANT_NAMES, CLIP_LEN
from rank_obstruction import eps_star_all, leverage_certificate, joint_bound
from train_hjepa import HJEPA, build_cache

SPLITS = ("metric", "select", "eval")


# ----------------------------------------------------------------------------
# Loading and pairs
# ----------------------------------------------------------------------------

def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = HJEPA(latent=cfg["latent"], readout=cfg["readout"], state=cfg["state"], hires=cfg.get("hires", False))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, cfg


def build_pairs(invariant, n_pairs, seed0=1_000_000):
    """List of (frames+, frames-, actions, t_div) with seeds disjoint from training."""
    sampler = ControlledSampler(seed=seed0 + INVARIANT_NAMES.index(invariant))
    out = []
    s = seed0 + 100_000 * INVARIANT_NAMES.index(invariant)
    while len(out) < n_pairs:
        try:
            a, b = MATCHED_PAIRS[invariant](sampler, seed=s)
            out.append((a.frames, b.frames, a.actions, a.meta["t_div"]))
        except RuntimeError:
            pass
        s += 1
    return out


def pairs_cache(invariant, n_pairs, cache_dir="data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"pairs_{invariant}_n{n_pairs}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return list(zip(d["fp"], d["fm"], d["ac"], d["t"]))
    pr = build_pairs(invariant, n_pairs)
    np.savez_compressed(p, fp=np.stack([x[0] for x in pr]), fm=np.stack([x[1] for x in pr]),
                        ac=np.stack([x[2] for x in pr]), t=np.array([x[3] for x in pr]))
    return pr


@torch.no_grad()
def run_pairs(model, pairs):
    """For each pair: readout h (at the step predicting frame t_div), z_hat, z+, z-,
    and the copy-last-frame baseline prediction (target latent of frame t_div - 1)."""
    H, ZH, ZP, ZM, ZC = [], [], [], [], []
    ts = np.array([p[3] for p in pairs])
    order = np.argsort(ts)
    res = [None] * len(pairs)
    for t in np.unique(ts):
        idx = [i for i in order if ts[i] == t]
        fp = torch.from_numpy(np.stack([pairs[i][0] for i in idx]).astype(np.float32) / 255.0)[:, :, None]
        fm = torch.from_numpy(np.stack([pairs[i][1] for i in idx]).astype(np.float32) / 255.0)[:, :, None]
        ac = torch.from_numpy(np.stack([pairs[i][2] for i in idx]))
        z = model.encode_seq(fp[:, :t])                     # shared context
        _, h, zh, _ = model.fast.run(z, ac[:, :t])
        zp = model.target_encoder(fp[:, t]); zm = model.target_encoder(fm[:, t])
        zc = model.target_encoder(fp[:, t - 1])
        for n, i in enumerate(idx):
            res[i] = (h[n, -1], zh[n, -1], zp[n], zm[n], zc[n])
    for r in res:
        H.append(r[0]); ZH.append(r[1]); ZP.append(r[2]); ZM.append(r[3]); ZC.append(r[4])
    return torch.stack(H), torch.stack(ZH), torch.stack(ZP), torch.stack(ZM), torch.stack(ZC)


@torch.no_grad()
def reference_readouts(model, n=2000, seed=0):
    """Readout activations on random training contexts (the resample reference)."""
    frames, actions = build_cache(4000, seed=0)
    rng = np.random.default_rng(seed + 5)
    idx = rng.integers(0, len(frames), size=n // 16 + 1)
    hs = []
    for i in idx:
        fr = torch.from_numpy(frames[i].astype(np.float32) / 255.0)[None, :, None]
        ac = torch.from_numpy(actions[i])[None]
        _, h, _, _ = model.fast.run(model.encode_seq(fr), ac)
        t = rng.integers(8, CLIP_LEN - 1, size=16)
        hs.append(h[0, t])
    return torch.cat(hs)[:n]


# ----------------------------------------------------------------------------
# Metric, attributions, ablation
# ----------------------------------------------------------------------------

def contrast_axis(zp, zm):
    """Kept for reference: pooled axis u_k and half-contrast (not used by the metric)."""
    u = (zp - zm).mean(0)
    u = u / u.norm().clamp_min(1e-8)
    s = ((zp - zm) @ u).mean() / 2
    return u, s


def pair_metric(zh, zp, zm, denom=None):
    """Preference for the correct future; linear in z_hat.

    denom=None: per-pair normalisation ||z+ - z-||^2 (exactly +1/-1 at the two
    futures, but heavy-tailed when a pair's futures nearly coincide).
    denom=float: a pooled normalisation E||z+ - z-||^2 fixed on the metric split
    (the default in measure(); same sign and additivity, far lower variance).
    """
    diff = zp - zm
    num = 2 * ((zh - (zp + zm) / 2) * diff).sum(-1)
    if denom is None:
        return num / (diff * diff).sum(-1).clamp_min(1e-12)
    return num / denom


class NeuronBasis:
    """Components = readout units; W maps component deviations into z_hat."""
    name = "neuron"

    def __init__(self, model):
        self.W = model.fast.out.weight                    # (latent, R)

    def encode(self, h):
        return h


class SAEBasis:
    """Components = features of a sparse autoencoder on the readout layer.

    Ablating features changes h by D (f' - f); the reconstruction error is left
    in place, so effects stay exactly additive over features (Lemma 1).
    """
    name = "sae"

    def __init__(self, model, ref_h, n_features, l1=3e-3, steps=4000, seed=0):
        torch.manual_seed(seed)
        R = ref_h.shape[1]
        self.enc = torch.nn.Linear(R, n_features)
        self.dec = torch.nn.Linear(n_features, R, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(R))
        params = list(self.enc.parameters()) + list(self.dec.parameters()) + [self.b]
        opt = torch.optim.Adam(params, 2e-3)
        X = ref_h.detach()
        scale = X.pow(2).sum(1).mean().sqrt().item() + 1e-8
        for _ in range(steps):
            x = X[torch.randint(0, len(X), (512,))]
            f = torch.relu(self.enc(x - self.b))
            loss = ((self.dec(f) + self.b - x) ** 2).sum(1).mean() / scale ** 2 + l1 * f.abs().sum(1).mean() / scale
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        with torch.no_grad():
            f = self.encode(X)
            self.fvu = float(((self.dec(f) + self.b - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
            self.l0 = float((f > 0).float().sum(1).mean())
            self.alive = int((f > 0).any(0).sum())
            self.W = model.fast.out.weight @ self.dec.weight   # (latent, n_features)

    @torch.no_grad()
    def encode(self, h):
        return torch.relu(self.enc(h - self.b))


@torch.no_grad()
def attributions(W, comps, ref_mean, zp, zm, denom=None):
    """Per-pair per-component attribution (N, n), exact for resample ablation in expectation."""
    diff = zp - zm
    wd = diff @ W                                             # (N, n) = <W[:, c], z+ - z->
    d = (diff * diff).sum(-1, keepdim=True).clamp_min(1e-12) if denom is None else denom
    return 2 * (comps - ref_mean) * wd / d


@torch.no_grad()
def mc_resample_effect(W, comps, zh, zp, zm, circuit, ref, n_draws=20, seed=0, denom=None):
    """Monte-Carlo resample ablation of `circuit` (independent reference draws)."""
    g = torch.Generator().manual_seed(seed)
    base = pair_metric(zh, zp, zm, denom).mean()
    effs = []
    for _ in range(n_draws):
        r = ref[torch.randint(0, len(ref), (len(comps),), generator=g)]
        c2 = comps.clone(); c2[:, circuit] = r[:, circuit]
        zh2 = zh + (c2 - comps) @ W.T
        effs.append((base - pair_metric(zh2, zp, zm, denom).mean()).item())
    return float(np.mean(effs))


# ----------------------------------------------------------------------------
# Circuit search on attribution matrices (K x R)
# ----------------------------------------------------------------------------

def circuit_eps(A, C, k):
    S = A[:, C].sum(1) if len(C) else np.zeros(A.shape[0])
    if S[k] <= 1e-12:
        return math.inf, S
    off = np.abs(np.delete(S, k)).max()
    return off / S[k], S


def search_circuit(A, k, max_size, min_on, beam=8):
    """Find a small component set minimising eps_k subject to Delta_k >= min_on.

    Two heuristics, best result kept (binary circuit selection is NP-hard in
    general; the certificate of Theorem 1 lower-bounds what any method finds):
      1. Lagrangian prefix sweep: rank components by a_k[c] - lam * sum_j |a_j[c]|
         for a grid of lam, and scan prefixes up to max_size.
      2. Beam search on eps, where a state that has not yet made proportional
         progress toward min_on is penalised, so the beam accumulates on-target
         mass instead of collecting near-zero components.
    """
    K, R = A.shape
    others = np.abs(np.delete(A, k, axis=0))
    best = (math.inf, [])

    def consider(C, S):
        nonlocal best
        if S[k] >= min_on and S[k] > 0:
            e = np.abs(np.delete(S, k)).max() / S[k]
            if e < best[0]:
                best = (e, list(C))

    for lam in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        score = A[k] - lam * others.sum(0)
        S = np.zeros(K); C = []
        for c in np.argsort(-score)[:max_size]:
            if score[c] <= 0 and C:
                break
            C.append(int(c)); S = S + A[:, c]
            consider(C, S)

    beams = [((), np.zeros(K))]
    for size in range(1, max_size + 1):
        target = min_on * size / max_size
        cand = {}
        for C, S in beams:
            for c in range(R):
                if c in C:
                    continue
                C2 = tuple(sorted(C + (c,)))
                if C2 in cand:
                    continue
                S2 = S + A[:, c]
                if S2[k] <= 0:
                    continue
                e = np.abs(np.delete(S2, k)).max() / S2[k]
                cand[C2] = (e + (10.0 if S2[k] < target else 0.0), S2)
        if not cand:
            break
        ranked = sorted(cand.items(), key=lambda kv: kv[1][0])[:beam]
        beams = [(C, v[1]) for C, v in ranked]
        for C, S in beams:
            consider(C, S)
    return best[1]


def top_cii_circuit(A, k, size):
    return list(np.argsort(-np.abs(A[k]))[:size])


# ----------------------------------------------------------------------------
# Main measurement
# ----------------------------------------------------------------------------

def measure(model, n_pairs=450, sparsity=0.1, seed=0, invariants=INVARIANT_NAMES,
            n_boot=300, verbose=True, basis="neuron", sae_mult=4):
    rng = np.random.default_rng(seed)
    K = len(invariants)
    ref_h = reference_readouts(model, n=8000 if basis == "sae" else 2000, seed=seed)
    B = NeuronBasis(model) if basis == "neuron" else SAEBasis(model, ref_h, sae_mult * ref_h.shape[1], seed=seed)
    ref = B.encode(ref_h)
    ref_mean = ref.mean(0)
    R = ref.shape[1]
    data = {}
    for inv in invariants:
        pr = pairs_cache(inv, n_pairs)
        h, zh, zp, zm, zc = run_pairs(model, pr)
        perm = rng.permutation(len(pr))
        thirds = np.array_split(perm, 3)
        norms = (zp - zm).norm(dim=-1)
        kappa = 0.1 * norms[torch.as_tensor(thirds[0])].median()
        keep = (norms > kappa).numpy()
        split = {n: torch.as_tensor([i for i in ix if keep[i]]) for n, ix in zip(SPLITS, thirds)}
        mi = split["metric"]
        denom = float(((zp[mi] - zm[mi]) ** 2).sum(-1).mean())
        data[inv] = dict(h=h, zh=zh, zp=zp, zm=zm, zc=zc, split=split, denom=denom,
                         dropped=float(1 - keep.mean()))

    # Behaviour (GATE 0): L_k on the eval split, with bootstrap CI.
    behaviour = {}
    for inv in invariants:
        d = data[inv]; e = d["split"]["eval"]
        m = pair_metric(d["zh"][e], d["zp"][e], d["zm"][e], d["denom"]).numpy()
        mc = pair_metric(d["zc"][e], d["zp"][e], d["zm"][e], d["denom"]).numpy()
        bs = [rng.choice(m, len(m)).mean() for _ in range(n_boot)]
        ix = [rng.integers(0, len(m), len(m)) for _ in range(n_boot)]
        bd = [(m[i] - mc[i]).mean() for i in ix]
        behaviour[inv] = dict(L=float(m.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              n_eval=int(len(m)), dropped_frac=d["dropped"],
                              L_copy=float(mc.mean()), gain_over_copy=float((m - mc).mean()),
                              gain_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))])

    # Attribution matrices: rows = invariant j, cols = units.
    def A_of(split_name, idx_override=None):
        A = np.zeros((K, R)); per_pair = {}
        for j, inv in enumerate(invariants):
            d = data[inv]
            ix = d["split"][split_name] if idx_override is None else idx_override[inv]
            P = attributions(B.W, B.encode(d["h"][ix]), ref_mean, d["zp"][ix], d["zm"][ix], d["denom"]).numpy()
            per_pair[inv] = P
            A[j] = P.mean(0)
        return A, per_pair

    A_sel, _ = A_of("select")
    A_eval, P_eval = A_of("eval")

    # Additivity check: analytic sum vs Monte-Carlo resample ablation.
    add_err = []
    for j, inv in enumerate(invariants):
        d = data[inv]; e = d["split"]["eval"]
        C = list(rng.choice(R, size=max(1, R // 8), replace=False))
        mc = mc_resample_effect(B.W, B.encode(d["h"][e]), d["zh"][e], d["zp"][e], d["zm"][e], C, ref,
                                denom=d["denom"])
        add_err.append(abs(mc - A_eval[j, C].sum()) / (abs(behaviour[inv]["L"]) + 1e-9))

    # Theorem-1 certificate on the eval attribution matrix.
    r = int(np.linalg.matrix_rank(A_eval, tol=1e-6 * np.abs(A_eval).max()))
    cert = leverage_certificate(A_eval)
    eps_opt = eps_star_all(A_eval)

    size = max(1, int(round(sparsity * R)))
    results = {}
    for k, inv in enumerate(invariants):
        Lk = float(A_eval[k].sum())                 # effect of ablating the whole readout layer
        behaviour[inv]["readout_mediated"] = Lk
        min_on = 0.2 * max(float(A_sel[k].sum()), 1e-6)
        circ = {"selective": search_circuit(A_sel, k, size, min_on),
                "top_cii": top_cii_circuit(A_sel, k, size)}
        results[inv] = {}
        for name, C in circ.items():
            e, S = circuit_eps(A_eval, C, k)
            # random same-size subsets for (M3)
            rand = [A_eval[k, rng.choice(R, size=max(1, len(C)), replace=False)].sum() for _ in range(300)]
            # bootstrap eps over eval pairs
            bs = []
            for _ in range(n_boot):
                Sb = np.array([P_eval[inv_j][rng.integers(0, len(P_eval[inv_j]), len(P_eval[inv_j]))][:, C].sum(1).mean()
                               if len(C) else 0.0 for inv_j in invariants])
                bs.append(np.abs(np.delete(Sb, k)).max() / Sb[k] if Sb[k] > 0 else np.inf)
            bs = np.array(bs)
            results[inv][name] = dict(
                size=len(C), sparsity=len(C) / R, on_target=float(S[k]),
                on_target_frac=float(S[k] / Lk) if Lk > 0 else float("nan"),
                off_target=[float(x) for x in S], eps=float(e),
                eps_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                random_mean=float(np.mean(rand)),
                concentration=float(S[k] / np.mean(rand)) if np.mean(rand) > 1e-6 else float("nan"))
    joint = {name: max(results[inv][name]["eps"] for inv in invariants) for name in ("selective", "top_cii")}
    out = dict(basis=B.name, R=R, K=K, rank=r, floor=joint_bound(K, r), certificate=[float(c) for c in cert],
               eps_weighted_opt=[float(x) for x in eps_opt], behaviour=behaviour,
               additivity_rel_err=[float(x) for x in add_err], circuits=results, joint_eps=joint)
    if basis == "sae":
        out["sae"] = dict(fvu=B.fvu, l0=B.l0, alive=B.alive, n_features=R)
    if verbose:
        report(out, invariants)
    return out


def report(out, invariants):
    print(f"\n[{out['basis']} basis] components={out['R']}, invariants K={out['K']}, rank(A)={out['rank']}, "
          f"Theorem-1 floor on joint eps = {out['floor']:.3f}")
    print("Behaviour (eval split; +1 = predicts correct future, 0 = indifferent):")
    for inv in invariants:
        b = out["behaviour"][inv]
        print(f"  {inv:11s} L = {b['L']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]  copy-baseline {b['L_copy']:+.3f}  "
              f"gain {b['gain_over_copy']:+.3f} [{b['gain_ci'][0]:+.3f}, {b['gain_ci'][1]:+.3f}]  "
              f"readout-mediated {b['readout_mediated']:+.3f}  (n={b['n_eval']}, dropped {100 * b['dropped_frac']:.0f}%)")
    print(f"additivity check (|MC resample - analytic| / L): max {max(out['additivity_rel_err']):.3f}")
    print("per-invariant certificate / weighted optimum eps*:",
          " ".join(f"{c:.3f}/{e:.3f}" for c, e in zip(out["certificate"], out["eps_weighted_opt"])))
    for name in ("selective", "top_cii"):
        print(f"\ncircuits: {name}")
        for inv in invariants:
            r = out["circuits"][inv][name]
            print(f"  {inv:11s} |C|={r['size']:3d} (s={r['sparsity']:.2f})  on-target {r['on_target']:+.3f} "
                  f"({100 * r['on_target_frac']:.0f}% of readout effect)  eps {r['eps']:.3f} "
                  f"[{r['eps_ci'][0]:.2f}, {r['eps_ci'][1]:.2f}]  random-subset on-target {r['random_mean']:+.3f}")
        print(f"  joint eps = {out['joint_eps'][name]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", type=int, default=450)
    ap.add_argument("--sparsity", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--basis", choices=["neuron", "sae"], default="neuron")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    model, cfg = load_model(a.ckpt)
    res = measure(model, n_pairs=a.pairs, sparsity=a.sparsity, seed=a.seed, basis=a.basis)
    res["config"] = cfg
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.out}")
