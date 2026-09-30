"""
Rank obstruction for circuit modularity: exact computations behind Theorem 1.

Setting (paper Sec. 3). Invariant j has an attribution vector a_j in R^D: the
effect on invariant j's metric of intervening on component c is a_j[c], and
effects are additive over components (exact at a layer read out linearly, e.g.
the last hidden layer before a linear head, or SAE features decoding into it).
A (possibly weighted) circuit psi in R^D has effect Delta_j(psi) = <a_j, psi>,
and selectivity eps_k(psi) = max_{j!=k} |Delta_j(psi)| / |Delta_k(psi)|.

The best selectivity any intervention can reach for invariant k is

    eps*_k = min_psi eps_k(psi) = 1 / min{ ||lambda||_1 : sum_{j!=k} lambda_j a_j = a_k }

(LP duality; eps*_k = 0 if a_k is not in the span of the others). This file:

  1. computes eps*_k exactly (basic-feasible-solution enumeration, no LP
     library needed) and cross-checks it against direct numerical
     minimisation over psi;
  2. checks Theorem 1: with r = rank{a_j}, every circuit satisfies
         max_k eps_k >= sqrt((K - r) / (r (K - 1)))    (the Welch value),
     and the per-invariant leverage certificate
         eps*_k >= sqrt(p_k / ((K - 1)(1 - p_k))),  p_k = P_kk,
     where P projects onto the null space of the K x D attribution matrix.
     Equality for equiangular tight frames (e.g. the simplex, K = r + 1);
  3. reproduces the aligned-vs-unconstrained gap reported in Appendix A
     (the aligned family gets eps = mu_k; unconstrained circuits can beat the
     Welch value for a single invariant, but never jointly);
  4. searches over frames for the smallest achievable joint eps at each
     (K, d): the floor is attained at K = d + 1 and approached from above.

Run:  python rank_obstruction.py          (about a minute on one CPU core)
Writes: figures/fig_rank_obstruction.png and results/rank_obstruction.json
"""

from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np


# ----------------------------------------------------------------------------
# Exact eps*_k.
# ----------------------------------------------------------------------------

def min_l1_representation(A_others: np.ndarray, target: np.ndarray,
                          tol: float = 1e-9) -> float:
    """min ||lambda||_1 s.t. A_others.T @ lambda = target, or inf if infeasible.

    A_others: (m, D) rows are the other attribution vectors. The optimum of
    this LP is attained at a basic feasible solution, i.e. with support on a
    set of linearly independent rows of size rank(A_others); we enumerate them.
    """
    m, D = A_others.shape
    r = np.linalg.matrix_rank(A_others, tol=1e-8)
    # Feasibility: target must lie in the row span.
    coef, *_ = np.linalg.lstsq(A_others.T, target, rcond=None)
    if np.linalg.norm(A_others.T @ coef - target) > 1e-6 * max(1.0, np.linalg.norm(target)):
        return math.inf
    if r == 0:
        return math.inf
    # Project onto an orthonormal basis of the row span so each basis is r x r.
    U, S, Vt = np.linalg.svd(A_others, full_matrices=False)
    B = Vt[:r]                                # (r, D) basis of the span
    Ared = A_others @ B.T                     # (m, r)
    tred = B @ target                         # (r,)
    best = math.inf
    subsets = np.array(list(itertools.combinations(range(m), r)))
    M = Ared[subsets]                         # (n_sub, r, r) rows = chosen vectors
    Mt = np.transpose(M, (0, 2, 1))           # solve Mt @ lam = tred
    det = np.linalg.det(Mt)
    ok = np.abs(det) > tol
    if not ok.any():
        return math.inf
    lam = np.linalg.solve(Mt[ok], np.broadcast_to(tred, (ok.sum(), r))[..., None])[..., 0]
    best = float(np.abs(lam).sum(axis=1).min())
    return best


def eps_star(A: np.ndarray, k: int) -> float:
    """Best achievable selectivity for invariant k over all weighted circuits."""
    others = np.delete(A, k, axis=0)
    l1 = min_l1_representation(others, A[k])
    return 0.0 if math.isinf(l1) else 1.0 / l1


def eps_star_all(A: np.ndarray) -> np.ndarray:
    return np.array([eps_star(A, k) for k in range(A.shape[0])])


def leverage_certificate(A: np.ndarray) -> np.ndarray:
    """Per-invariant lower bound on eps*_k from the null-space projector.

    alpha = P e_k is a linear dependency sum_j alpha_j a_j = 0 with alpha_k =
    p_k; Cauchy-Schwarz on the induced representation of a_k gives
    ||lambda||_1 <= sqrt((K-1)(1-p_k)/p_k), hence the bound. Cheap (one SVD).
    """
    K = A.shape[0]
    r = np.linalg.matrix_rank(A, tol=1e-8)
    if r >= K:
        return np.zeros(K)
    _, _, Vt = np.linalg.svd(A.T, full_matrices=True)
    N = Vt[r:].T                               # (K, K - r) orthonormal null basis
    p = np.clip(np.einsum("ij,ij->i", N, N), 0.0, 1.0 - 1e-12)
    return np.sqrt(p / ((K - 1) * (1.0 - p)))


def joint_bound(K: int, r: int) -> float:
    """Theorem 1 floor on joint eps: sqrt((K - r) / (r (K - 1))) for K > r."""
    return 0.0 if K <= r else math.sqrt((K - r) / (r * (K - 1)))


def eps_star_numeric(A: np.ndarray, k: int, iters: int = 4000, seed: int = 0) -> float:
    """Direct minimisation of max_j |<a_j,psi>| s.t. <a_k,psi> = 1 (cross-check).

    Projected subgradient on the affine set, with a decaying step. Returns an
    upper estimate of eps*_k (it is a minimisation), so it should be >= the
    exact value up to optimiser error.
    """
    rng = np.random.default_rng(seed)
    ak = A[k]
    others = np.delete(A, k, axis=0)
    best = math.inf
    for restart in range(8):
        psi = rng.standard_normal(A.shape[1])
        psi = psi + (1 - ak @ psi) / (ak @ ak) * ak
        for t in range(iters):
            vals = others @ psi
            j = int(np.argmax(np.abs(vals)))
            best = min(best, float(np.abs(vals).max()))
            g = np.sign(vals[j]) * others[j]
            g = g - (g @ ak) / (ak @ ak) * ak      # stay on <a_k, psi> = 1
            psi = psi - (0.5 / math.sqrt(t + 1)) * g / (np.linalg.norm(g) + 1e-12)
            psi = psi + (1 - ak @ psi) / (ak @ ak) * ak
    return best


# ----------------------------------------------------------------------------
# Frames.
# ----------------------------------------------------------------------------

def simplex_frame(d: int) -> np.ndarray:
    """d+1 unit vectors in R^d summing to zero (regular simplex, an ETF)."""
    E = np.eye(d + 1) - 1.0 / (d + 1)
    U, S, Vt = np.linalg.svd(E)
    V = E @ Vt[:d].T                           # (d+1, d)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def welch(K: int, d: int) -> float:
    return 0.0 if K <= d else math.sqrt((K - d) / (d * (K - 1)))


def coherence_per_row(V: np.ndarray) -> np.ndarray:
    G = np.abs(V @ V.T)
    np.fill_diagonal(G, 0.0)
    return G.max(axis=1)


def normalize(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def search_min_joint_eps(K: int, d: int, rng: np.random.Generator,
                         restarts: int = 6, steps: int = 150) -> float:
    """Upper estimate of inf over frames of max_k eps*_k (hill climbing)."""
    best_overall = math.inf
    for _ in range(restarts):
        V = normalize(rng.standard_normal((K, d)))
        cur = eps_star_all(V).max()
        scale = 0.3
        for s in range(steps):
            Vn = normalize(V + scale * rng.standard_normal(V.shape))
            val = eps_star_all(Vn).max()
            if val < cur:
                V, cur = Vn, val
            else:
                scale *= 0.985
        best_overall = min(best_overall, cur)
    return float(best_overall)


# ----------------------------------------------------------------------------
# Experiments.
# ----------------------------------------------------------------------------

def main(seed: int = 0):
    rng = np.random.default_rng(seed)
    out = {}

    # (1) Exact solver vs direct numerical minimisation.
    diffs = []
    for trial in range(12):
        d = int(rng.integers(2, 5)); K = int(rng.integers(d + 1, d + 5))
        V = normalize(rng.standard_normal((K, d)))
        k = int(rng.integers(K))
        ex = eps_star(V, k); nu = eps_star_numeric(V, k, iters=1500, seed=trial)
        diffs.append(nu - ex)
    out["solver_check_max_abs_gap"] = float(np.max(np.abs(diffs)))
    out["solver_check_min_gap"] = float(np.min(diffs))
    print(f"[1] exact vs numeric eps*: max |gap| = {np.max(np.abs(diffs)):.4f} "
          f"(numeric is an upper estimate; min gap = {np.min(diffs):.4f})")

    # (2) Theorem 1 on random (also unnormalised) attribution matrices.
    viol = cert_viol = n = 0; ratios = []
    for d in [2, 3, 4, 6, 8]:
        for K in range(d + 1, d + 8):
            for _ in range(15):
                A = rng.standard_normal((K, d)) * rng.uniform(0.2, 5.0, size=(K, 1))
                e = eps_star_all(A); c = leverage_certificate(A)
                n += 1
                viol += int(e.max() < joint_bound(K, d) - 1e-9)
                cert_viol += int(np.any(e < c - 1e-9))
                ratios.append(e.max() / joint_bound(K, d))
    out["thm1_random_frames_checked"] = n
    out["thm1_violations"] = viol
    out["certificate_violations"] = cert_viol
    out["thm1_min_ratio_to_bound"] = float(np.min(ratios))
    print(f"[2] Theorem 1 (max_k eps* >= Welch value) on {n} random attribution matrices: "
          f"{viol} violations; leverage certificate violations: {cert_viol}; "
          f"min ratio to bound {np.min(ratios):.3f}")
    simplex = {}
    for d in [2, 3, 4, 6, 8]:
        e = eps_star_all(simplex_frame(d))
        simplex[d] = [float(e.min()), float(e.max())]
        print(f"    simplex d={d}: eps* in [{e.min():.4f}, {e.max():.4f}], bound = {joint_bound(d + 1, d):.4f}")
    out["simplex_eps_star"] = simplex
    # A non-simplex ETF: the 6 icosahedral lines in R^3 (K = 2d), Welch = 1/sqrt(5).
    phi = (1 + 5 ** 0.5) / 2
    ico = normalize(np.array([[0, 1, phi], [0, 1, -phi], [1, phi, 0], [1, -phi, 0],
                              [phi, 0, 1], [-phi, 0, 1]], float))
    e_ico, c_ico = eps_star_all(ico), leverage_certificate(ico)
    out["icosahedral_etf"] = dict(eps_star=[float(x) for x in e_ico], certificate=[float(x) for x in c_ico],
                                  bound=joint_bound(6, 3))
    print(f"    icosahedral ETF (d=3, K=6): eps* = {e_ico.max():.4f}, certificate = {c_ico.min():.4f}, "
          f"bound = {joint_bound(6, 3):.4f}")

    # K <= d: generic frames admit eps* = 0 for every k, whatever their coherence.
    zero_ok = all(eps_star_all(normalize(rng.standard_normal((K, d)))).max() < 1e-9
                  for d in [4, 8] for K in range(2, d + 1))
    out["K_le_d_all_zero"] = bool(zero_ok)
    print(f"[2b] K <= d, generic frames: eps* = 0 for all k: {zero_ok}")

    # (3) Aligned vs unconstrained on (near) worst-case-coherent frames.
    gap_rows = []
    for d, K in [(4, 5), (4, 6), (4, 8), (8, 9), (8, 10), (8, 16)]:
        # low-coherence frame: best of many random draws (as in the original search)
        best, bestmu = None, math.inf
        for _ in range(300):
            V = normalize(rng.standard_normal((K, d)))
            mu = coherence_per_row(V).max()
            if mu < bestmu:
                best, bestmu = V, mu
        mu_k = coherence_per_row(best)
        es = eps_star_all(best)
        row = dict(d=d, K=K, welch=welch(K, d), aligned_joint=float(mu_k.max()),
                   unconstrained_joint=float(es.max()),
                   unconstrained_min_over_k=float(es.min()))
        gap_rows.append(row)
        print(f"[3] d={d:2d} K={K:2d}: aligned eps (=max mu_k) {row['aligned_joint']:.3f} | "
              f"unconstrained joint {row['unconstrained_joint']:.3f} "
              f"(best single k {row['unconstrained_min_over_k']:.3f}) | "
              f"Theorem-1 floor {row['welch']:.3f}")
    out["aligned_vs_unconstrained"] = gap_rows

    # (4) How small can joint eps be made at each (K, d)? (upper estimates)
    curve = {}
    for d in [2, 3, 4]:
        pts = []
        for K in range(d + 1, min(3 * d + 3, 12) + 1):
            m = search_min_joint_eps(K, d, rng, restarts=3 if d >= 4 else 4, steps=120)
            pts.append((K, m, welch(K, d)))
            print(f"[4] d={d} K={K:2d}: min_frames max_k eps* <= {m:.3f}   "
                  f"(proved floor = {welch(K, d):.3f})")
        curve[d] = pts
    out["frame_search"] = {str(d): v for d, v in curve.items()}

    os.makedirs("results", exist_ok=True)
    with open("results/rank_obstruction.json", "w") as f:
        json.dump(out, f, indent=1)
    make_figure(curve, gap_rows)
    return out


def make_figure(curve, gap_rows):
    import figstyle
    figstyle.apply()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ax = axes[0]
    colors = {2: "#1f77b4", 3: "#d62728", 4: "#2ca02c"}
    for d, pts in curve.items():
        Ks = [d] + [p[0] for p in pts]
        ys = [0.0] + [p[1] for p in pts]
        ax.plot([K / d for K in Ks], ys, "o-", color=colors[d], label=f"d={d}: best frame found", ms=4)
        ax.plot([K / d for K in Ks], [0.0] + [p[2] for p in pts], "--", color=colors[d], lw=1.0, alpha=0.8)
    ax.axvline(1.0, color="grey", lw=0.8)
    ax.set_xlabel("K / d")
    ax.set_ylabel(r"smallest achievable joint $\varepsilon$")
    ax.set_title("(a) Smallest joint $\\varepsilon$ over frames (markers)\nvs. proved floor, Theorem 1 (dashed)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    ax = axes[1]
    labels = [f"({r['d']},{r['K']})" for r in gap_rows]
    x = np.arange(len(gap_rows)); w = 0.2
    ax.bar(x - 1.5 * w, [r["aligned_joint"] for r in gap_rows], w, label="aligned circuits, joint")
    ax.bar(x - 0.5 * w, [r["unconstrained_joint"] for r in gap_rows], w, label="any circuit, joint (exact)")
    ax.bar(x + 0.5 * w, [r["unconstrained_min_over_k"] for r in gap_rows], w, label="any circuit, best single invariant")
    ax.bar(x + 1.5 * w, [r["welch"] for r in gap_rows], w, label="Theorem 1 floor (joint)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("(d, K)")
    ax.set_ylabel(r"joint $\varepsilon$")
    ax.set_title("(b) Single invariants can beat the floor; the joint problem cannot", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, axis="y")
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_rank_obstruction.png", dpi=170)
    print("wrote figures/fig_rank_obstruction.png")


if __name__ == "__main__":
    main()
