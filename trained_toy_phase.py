"""
The phase boundary in TRAINED networks (controlled setting, Sec. 5.1).

rank_obstruction.py checks Theorem 1 on arbitrary attribution matrices; this
script asks what trained networks actually do. A network must transmit K
sparse input features through a readout layer of d units:

    x in R^K,  x_k = Bernoulli(p) * Uniform(0.5, 1)      (feature density p)
    h = ReLU(W1 x + b1) in R^d                            (circuit components)
    y_hat = W2 h + b2          ("linear" head: Theorem 1 applies exactly)
    y_hat = ReLU(W2 h + b2)    ("relu" head: additivity fails; outside the theorem)

trained to reconstruct x. Invariant k's matched pairs toggle feature k
(x+ has it on, x- off, everything else shared); the metric and resample
ablation are the same as in measure_epsilon.py. For each trained network we
report, over the K_eff features it actually represents (L_k > 0.5):

  - the Theorem-1 floor for (K_eff, rank A),
  - the joint eps of the best circuit found by search (true ablation effects,
    measured on a fresh evaluation sample, not the attribution sum),
  - for the linear head, the additivity error (should be ~0), and for the
    ReLU head, how far measured joint eps falls below the linear floor.

Run: python trained_toy_phase.py          (~10 min on a laptop CPU)
Writes results/trained_toy_phase.json and figures/fig_trained_toy_phase.png
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from rank_obstruction import leverage_certificate, joint_bound, eps_star_all
from measure_epsilon import search_circuit

torch.set_num_threads(1)


class Toy(nn.Module):
    def __init__(self, K, d, head):
        super().__init__()
        self.l1 = nn.Linear(K, d); self.l2 = nn.Linear(d, K); self.head = head
        nn.init.constant_(self.l1.bias, 0.1)   # avoid dead readout units at init

    def hidden(self, x):
        return torch.relu(self.l1(x))

    def out_from_hidden(self, h):
        y = self.l2(h)
        return torch.relu(y) if self.head == "relu" else y

    def forward(self, x):
        return self.out_from_hidden(self.hidden(x))


def sample_x(n, K, p, g):
    on = (torch.rand(n, K, generator=g) < p).float()
    return on * (0.5 + 0.5 * torch.rand(n, K, generator=g))


def train_toy(K, d, p, head, seed, steps=2500):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = Toy(K, d, head)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for s in range(steps):
        x = sample_x(512, K, p, g)
        loss = ((m(x) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m, g


@torch.no_grad()
def pairs(K, k, p, n, g):
    x = sample_x(n, K, p, g)
    xp = x.clone(); xm = x.clone()
    xp[:, k] = 0.5 + 0.5 * torch.rand(n, generator=g); xm[:, k] = 0.0
    return xp, xm


@torch.no_grad()
def effects_true(m, K, p, C, ref, n, g):
    """Measured Delta_j(C) for all j by actual resample ablation (fresh sample)."""
    out = np.zeros(K); L = np.zeros(K)
    for j in range(K):
        xp, xm = pairs(K, j, p, n, g)
        h = m.hidden(xp)
        s = (xp[:, j] - xm[:, j]) / 2
        mid = (xp[:, j] + xm[:, j]) / 2
        base = ((m.out_from_hidden(h)[:, j] - mid) / s).mean()
        h2 = h.clone()
        if len(C):
            h2[:, C] = ref[torch.randint(0, len(ref), (n,), generator=g)][:, C]
        abl = ((m.out_from_hidden(h2)[:, j] - mid) / s).mean()
        out[j] = (base - abl).item(); L[j] = base.item()
    return out, L


@torch.no_grad()
def attribution_matrix(m, K, p, ref_mean, n, g):
    A = np.zeros((K, m.l1.out_features)); L = np.zeros(K)
    for j in range(K):
        xp, xm = pairs(K, j, p, n, g)
        h = m.hidden(xp)
        s = ((xp[:, j] - xm[:, j]) / 2)[:, None]
        # linear-head attribution (exact for "linear"; first-order proxy for "relu")
        A[j] = (((h - ref_mean) * m.l2.weight[j]) / s).mean(0).numpy()
        mid = (xp[:, j] + xm[:, j]) / 2
        L[j] = ((m(xp)[:, j] - mid) / s[:, 0]).mean().item()
    return A, L


class SAE(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.enc = nn.Linear(d, n); self.dec = nn.Linear(n, d, bias=False)
        self.b = nn.Parameter(torch.zeros(d))

    def encode(self, h):
        return torch.relu(self.enc(h - self.b))

    def decode(self, f):
        return self.dec(f) + self.b


def train_sae(H, n_feat, l1=3e-3, steps=3000, seed=0):
    torch.manual_seed(seed)
    sae = SAE(H.shape[1], n_feat)
    opt = torch.optim.Adam(sae.parameters(), 3e-3)
    scale = H.pow(2).sum(1).mean().sqrt().item() + 1e-8
    for _ in range(steps):
        x = H[torch.randint(0, len(H), (512,))]
        f = sae.encode(x)
        loss = ((sae.decode(f) - x) ** 2).sum(1).mean() / scale ** 2 + l1 * f.abs().sum(1).mean() / scale
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sae.dec.weight /= sae.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        f = sae.encode(H)
        fvu = ((sae.decode(f) - H) ** 2).sum() / ((H - H.mean(0)) ** 2).sum()
    return sae, float(fvu), float((f > 0).float().sum(1).mean())


@torch.no_grad()
def sae_effects_true(m, sae, K, p, C, fref, n, g):
    out = np.zeros(K)
    for j in range(K):
        xp, xm = pairs(K, j, p, n, g)
        h = m.hidden(xp); f = sae.encode(h)
        s = (xp[:, j] - xm[:, j]) / 2; mid = (xp[:, j] + xm[:, j]) / 2
        base = ((m.out_from_hidden(h)[:, j] - mid) / s).mean()
        f2 = f.clone()
        f2[:, C] = fref[torch.randint(0, len(fref), (n,), generator=g)][:, C]
        h2 = h + sae.dec(f2 - f)                       # reconstruction error left in place
        out[j] = (base - ((m.out_from_hidden(h2)[:, j] - mid) / s).mean()).item()
    return out


@torch.no_grad()
def sae_attribution_matrix(m, sae, K, p, fref_mean, n, g):
    A = np.zeros((K, sae.enc.out_features))
    Wd = sae.dec.weight                                # (d, n_feat)
    for j in range(K):
        xp, xm = pairs(K, j, p, n, g)
        f = sae.encode(m.hidden(xp))
        s = ((xp[:, j] - xm[:, j]) / 2)[:, None]
        A[j] = (((f - fref_mean) * (m.l2.weight[j] @ Wd)) / s).mean(0).numpy()
    return A


def sae_joint_eps(m, K, p, d, rep, L, ref, n, g, seed):
    sae, fvu, l0 = train_sae(ref, 4 * d, seed=seed)
    fref = sae.encode(ref)
    A = sae_attribution_matrix(m, sae, K, p, fref.mean(0), n, g)[rep]
    r_sae = int(np.linalg.matrix_rank(A, tol=1e-3 * np.abs(A).max()))
    eps_k = []
    for kk, k in enumerate(rep):
        C = search_circuit(A, kk, max_size=2 * d, min_on=0.2 * L[k])
        if not C:
            eps_k.append(math.inf); continue
        eff = sae_effects_true(m, sae, K, p, C, fref, n, g)[rep]
        eps_k.append(float(np.abs(np.delete(eff, kk)).max() / eff[kk]) if eff[kk] > 1e-6 else math.inf)
    return float(max(eps_k)), fvu, l0, r_sae, joint_bound(len(rep), r_sae)


def analyse(K, d, p, head, seed, n=4000, with_sae=True):
    m, g = train_toy(K, d, p, head, seed)
    ref = m.hidden(sample_x(20000, K, p, g)).detach()
    A, L = attribution_matrix(m, K, p, ref.mean(0), n, g)
    rep = np.where(L > 0.5)[0]
    Ke = len(rep)
    res = dict(K=K, d=d, p=p, head=head, seed=seed, K_eff=int(Ke), L=[float(x) for x in L])
    if Ke < 2:
        res.update(joint_eps=float("nan"), floor=float("nan"), rank=0)
        return res
    Ar = A[rep]
    r = int(np.linalg.matrix_rank(Ar, tol=1e-3 * np.abs(Ar).max()))
    res["rank"] = r
    res["floor"] = joint_bound(Ke, r)
    res["certificate_max"] = float(leverage_certificate(Ar).max())
    # best circuit per represented invariant: search on attributions (selection),
    # then evaluate TRUE ablation effects on a fresh sample (evaluation).
    eps_k, add_err = [], []
    for kk, k in enumerate(rep):
        C = search_circuit(Ar, kk, max_size=max(1, d // 2), min_on=0.2 * L[k])
        if not C:
            eps_k.append(math.inf); continue
        eff, _ = effects_true(m, K, p, C, ref, n, g)
        eff_r = eff[rep]
        add_err.append(float(np.abs(eff_r - Ar[:, C].sum(1)).max() / max(L[k], 1e-9)))
        eps_k.append(float(np.abs(np.delete(eff_r, kk)).max() / eff_r[kk]) if eff_r[kk] > 1e-6 else math.inf)
    res["eps_per_invariant"] = eps_k
    res["joint_eps"] = float(max(eps_k))
    res["additivity_err"] = float(max(add_err)) if add_err else float("nan")
    if Ke <= 10:
        res["weighted_opt_joint"] = float(eps_star_all(Ar).max())
    if with_sae:
        (res["joint_eps_sae"], res["sae_fvu"], res["sae_l0"],
         res["rank_sae"], res["floor_sae"]) = sae_joint_eps(m, K, p, d, rep, L, ref, n, g, seed)
    return res


def main():
    t0 = time.time()
    rows = []
    for head in ["linear", "relu"]:
        for p in [0.5, 0.1]:
            for d in [4, 8]:
                for K in sorted({2, d // 2, d, d + 1, d + 2, 2 * d, 3 * d}):
                    if K < 2:
                        continue
                    for seed in range(3):
                        r = analyse(K, d, p, head, seed)
                        rows.append(r)
                        print(f"{head:6s} p={p:.1f} d={d} K={K:2d} s={seed}: K_eff={r['K_eff']:2d} "
                              f"rank={r.get('rank', 0):2d} floor={r['floor']:.3f} "
                              f"joint eps={r['joint_eps']:.3f} SAE-basis={r.get('joint_eps_sae', float('nan')):.3f} add.err={r.get('additivity_err', float('nan')):.3f} "
                              f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/trained_toy_phase.json", "w") as f:
        json.dump(rows, f, indent=1)
    summarize(rows)
    make_figure(rows)


def summarize(rows):
    for head in ["linear", "relu"]:
        v = [r for r in rows if r["head"] == head and r["K_eff"] >= 2 and np.isfinite(r["joint_eps"])]
        below = [r for r in v if r["joint_eps"] < r["floor"] - 0.02]
        below_sae = [r for r in v if r.get("joint_eps_sae", 9) < r.get("floor_sae", 0) - 0.02]
        print(f"[{head}] {len(v)} trained nets; neuron-basis joint eps below its floor (by >0.02) in {len(below)}; "
              f"SAE-basis below its floor in {len(below_sae)}")
        sub = [r for r in v if r["K_eff"] <= r["d"]]
        sup = [r for r in v if r["K_eff"] > r["d"]]
        if sub and sup:
            for nm, key in [("neuron", "joint_eps"), ("SAE", "joint_eps_sae")]:
                print(f"   {nm:6s} basis: K_eff <= d median joint eps {np.median([r[key] for r in sub]):.3f} (n={len(sub)});"
                      f"  K_eff > d: {np.median([r[key] for r in sup]):.3f} (n={len(sup)})")


def make_figure(rows):
    import figstyle
    figstyle.apply()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained", sharey=True)
    for ax, head in zip(axes, ["linear", "relu"]):
        v = [r for r in rows if r["head"] == head and r["K_eff"] >= 2 and np.isfinite(r["joint_eps"])]
        x = np.array([r["K_eff"] / r["d"] for r in v])
        jit = np.random.default_rng(0).uniform(-0.03, 0.03, len(v))
        ax.scatter(x + jit, [min(r["joint_eps"], 1.6) for r in v], marker="o", color="#d62728", s=18,
                   alpha=0.7, edgecolor="k", lw=0.3, label="neuron basis")
        ax.scatter(x - jit, [min(r.get("joint_eps_sae", np.nan), 1.6) for r in v], marker="s", color="#1f77b4",
                   s=18, alpha=0.7, edgecolor="k", lw=0.3, label="SAE basis (4d features)")
        for d, ls in [(4, "-"), (8, "--")]:
            Ks = np.arange(d + 1, 3 * d + 1)
            ax.plot(Ks / d, [joint_bound(K, d) for K in Ks], ls, color="k", lw=1.1,
                    label=f"Theorem 1 floor, neuron basis (d={d})")
        ax.axvline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel(r"represented invariants per readout unit  $K_{\rm eff}/d$")
        ax.set_title("Linear head (Theorem 1 applies exactly)" if head == "linear"
                     else "ReLU head (additivity fails; outside Theorem 1)", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"joint $\varepsilon$ of best circuit (true ablations)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig("figures/fig_trained_toy_phase.png", dpi=170)
    print("wrote figures/fig_trained_toy_phase.png")


if __name__ == "__main__":
    main()
