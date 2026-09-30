"""
Readout-width sweep on the physics world model (Sec. 6).

Measures every checkpoint checkpoints/hjepa_r{d}_s{seed}.pt with
measure_epsilon.measure (K = 4 invariants) and aggregates, per readout width d:
behaviour vs. the copy-last-frame baseline, the rank of the attribution matrix,
the Theorem-1 floor, the exact weighted optimum, and the held-out joint epsilon
of circuits found by selectivity-aware search and by the v1 top-|CII| rule.

Run:  python jepa_width_sweep.py            (measures any checkpoints not yet measured)
Also evaluates open-loop rollout behaviour (horizons 1, 3, 5) against the copy
baseline. Writes results/jepa_r{d}_s{seed}[_sae].json, results/rollout_*.json,
results/jepa_width_sweep.json, results/jepa_rollout_behaviour.json,
figures/fig_jepa_width.png
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import torch

from environment import INVARIANT_NAMES
from measure_epsilon import load_model, measure
from rank_obstruction import joint_bound


def measure_all(force=False, bases=("neuron", "sae")):
    rows = []
    for ck in sorted(glob.glob("checkpoints/hjepa_r*_s*.pt")):
        m = re.search(r"hjepa_r(\d+)_s(\d+)\.pt", ck)
        d, seed = int(m.group(1)), int(m.group(2))
        for basis in bases:
            out = f"results/jepa_r{d}_s{seed}" + ("" if basis == "neuron" else "_sae") + ".json"
            if os.path.exists(out) and not force:
                res = json.load(open(out))
            else:
                print(f"\n=== measuring {ck} ({basis} basis) ===", flush=True)
                torch.manual_seed(0)
                model, cfg = load_model(ck)
                res = measure(model, n_pairs=450, sparsity=0.1, seed=0, basis=basis)
                res["config"] = cfg
                with open(out, "w") as f:
                    json.dump(res, f, indent=1)
            res["d"], res["seed"], res["basis"] = d, seed, basis
            rows.append(res)
    return rows


@torch.no_grad()
def rollout_behaviour(model, invariant, horizons=(1, 3, 5), n=200, n_boot=400):
    """Open-loop rollout from the shared context; preference at t_div + H - 1 vs copy-last-frame."""
    from measure_epsilon import pairs_cache
    from environment import CLIP_LEN
    pr = pairs_cache(invariant, 450)[:n]
    out = {}
    for H in horizons:
        zs = []
        for fp, fm, ac, t in pr:
            if t + H - 1 >= CLIP_LEN:
                continue
            fpt = torch.from_numpy(fp.astype(np.float32) / 255)[None, :, None]
            fmt_ = torch.from_numpy(fm.astype(np.float32) / 255)[None, :, None]
            a = torch.from_numpy(ac)[None]
            _, _, zh, s = model.fast.run(model.encode_seq(fpt[:, :t]), a[:, :t])
            cur = zh[:, -1]
            for k in range(1, H):
                _, _, zh2, s = model.fast.run(cur[:, None], a[:, t + k - 1:t + k], s0=s)
                cur = zh2[:, -1]
            zs.append((cur[0], model.target_encoder(fpt[:, t + H - 1])[0],
                       model.target_encoder(fmt_[:, t + H - 1])[0], model.target_encoder(fpt[:, t - 1])[0]))
        zh, zp, zm, zc = (torch.stack([z[i] for z in zs]) for i in range(4))
        d = zp - zm
        den = (d * d).sum(-1).mean()
        f = lambda x: (2 * ((x - (zp + zm) / 2) * d).sum(-1) / den).numpy()
        g = f(zh) - f(zc)
        rng = np.random.default_rng(0)
        bs = [rng.choice(g, len(g)).mean() for _ in range(n_boot)]
        out[H] = dict(L=float(f(zh).mean()), L_copy=float(f(zc).mean()), gain=float(g.mean()),
                      ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(g))
    return out


def rollout_all():
    res = {}
    for ck in sorted(glob.glob("checkpoints/hjepa_r*_s*.pt")):
        m = re.search(r"hjepa_r(\d+)_s(\d+)\.pt", ck)
        key = f"r{m.group(1)}_s{m.group(2)}"
        path = f"results/rollout_{key}.json"
        if os.path.exists(path):
            res[key] = json.load(open(path)); continue
        model, _ = load_model(ck)
        res[key] = json.loads(json.dumps({inv: rollout_behaviour(model, inv) for inv in INVARIANT_NAMES}))
        json.dump(res[key], open(path, "w"), indent=1)
        print(f"rollout behaviour {key}: " + "  ".join(
            f"{inv[:4]} H5 gain {res[key][inv]['5']['gain']:+.3f}" for inv in INVARIANT_NAMES), flush=True)
    return res


def summarize(rows, basis="neuron"):
    rows = [r for r in rows if r["basis"] == basis]
    widths = sorted({r["d"] for r in rows})
    table = []
    for d in widths:
        rs = [r for r in rows if r["d"] == d]
        row = {"d": d, "basis": basis, "n_seeds": len(rs), "floor": joint_bound(4, min(d, 4)),
               "floor_measured_rank": [r["floor"] for r in rs]}
        for inv in INVARIANT_NAMES:
            row[f"gain_{inv}"] = [r["behaviour"][inv]["gain_over_copy"] for r in rs]
            row[f"gain_ci_{inv}"] = [r["behaviour"][inv]["gain_ci"] for r in rs]
            row[f"readout_{inv}"] = [r["behaviour"][inv]["readout_mediated"] for r in rs]
        row["rank"] = [r["rank"] for r in rs]
        row["eps_weighted_opt_joint"] = [max(r["eps_weighted_opt"]) for r in rs]
        row["certificate_joint"] = [max(r["certificate"]) for r in rs]
        for name in ("selective", "top_cii"):
            row[f"joint_{name}"] = [r["joint_eps"][name] for r in rs]
            row[f"per_inv_{name}"] = [{inv: r["circuits"][inv][name]["eps"] for inv in INVARIANT_NAMES} for r in rs]
        row["additivity_err_max"] = [max(r["additivity_rel_err"]) for r in rs]
        table.append(row)
    return table


def fmt(xs):
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return "  inf "
    return f"{np.median(xs):6.3f}" + (f" ({min(xs):.2f}-{max(xs):.2f})" if len(xs) > 1 else "")


def report(table):
    print(f"\n[{table[0]['basis']} basis]")
    print("width | seeds | floor | weighted-opt joint | selective joint | top-|CII| joint | gains over copy (perm/coll/ident/cont)")
    for r in table:
        gains = " ".join(f"{np.median(r['gain_' + inv]):+.2f}" for inv in INVARIANT_NAMES)
        print(f"{r['d']:5d} | {r['n_seeds']:5d} | {r['floor']:.3f} | {fmt(r['eps_weighted_opt_joint'])} | "
              f"{fmt(r['joint_selective'])} | {fmt(r['joint_top_cii'])} | {gains}")


def make_figure(rows, roll):
    import figstyle
    figstyle.apply()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.35, 1]})
    # (a) behaviour gain over copy, by invariant, horizon and width (seed medians, CI of median seed)
    ax = axes[0]
    widths = sorted({int(k.split("_")[0][1:]) for k in roll})
    cols = {"permanence": "#1f77b4", "collision": "#d62728", "identity": "#2ca02c", "continuity": "#7f7f7f"}
    Hs = ["1", "3", "5"]
    x0 = 0
    ticks, labels = [], []
    for d in widths:
        keys = [k for k in roll if k.startswith(f"r{d}_")]
        for hi, H in enumerate(Hs):
            for j, inv in enumerate(INVARIANT_NAMES):
                g = [roll[k][inv][H]["gain"] for k in keys]
                lo = [roll[k][inv][H]["ci"][0] for k in keys]; hi_ = [roll[k][inv][H]["ci"][1] for k in keys]
                xx = x0 + hi * 1.0 + (j - 1.5) * 0.2
                ax.errorbar([xx], [np.median(g)], yerr=[[np.median(g) - np.median(lo)], [np.median(hi_) - np.median(g)]],
                            fmt="o", color=cols[inv], ms=3.5, capsize=1.5, lw=0.9,
                            label=inv if (d == widths[0] and hi == 0) else None)
            ticks.append(x0 + hi * 1.0); labels.append(f"H={H}")
        ax.text(x0 + 1.0, 0.04, f"d = {d}  ({len(keys)} seed{'s' if len(keys) > 1 else ''})",
                ha="center", va="bottom", fontsize=8, style="italic", transform=ax.get_xaxis_transform())
        x0 += 3.6
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("preference gain over copy-last-frame")
    ax.set_title("(a) What the world model learns beyond a trivial predictor", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    # (b) permanence circuit selectivity, held out, by basis and search method (width 64)
    ax = axes[1]
    cats = [("neuron", "selective"), ("neuron", "top_cii"), ("sae", "selective"), ("sae", "top_cii")]
    names = ["neurons,\nsearch", "neurons,\ntop-|CII|", "SAE,\nsearch", "SAE,\ntop-|CII|"]
    for i, (b, meth) in enumerate(cats):
        rs = [r for r in rows if r["d"] == 64 and r["basis"] == b]
        for n, r in enumerate(rs):
            c = r["circuits"]["permanence"][meth]
            e = c["eps"] if np.isfinite(c["eps"]) else np.nan
            lo, hi = c["eps_ci"]
            hi = hi if np.isfinite(hi) else 3.0
            xx = i + (n - (len(rs) - 1) / 2) * 0.15
            ax.errorbar([xx], [e], yerr=[[max(e - lo, 0) if np.isfinite(e) else 0], [max(hi - e, 0) if np.isfinite(e) else 0]],
                        fmt="s" if b == "sae" else "o", color="#1f77b4" if b == "sae" else "#d62728", ms=4, capsize=2)
    ax.axhline(0.30, color="k", ls="--", lw=0.9, label="pre-registered PR1 threshold (0.30)")
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 3.0)
    ax.set_ylabel("held-out $\\varepsilon$ of permanence circuit (95% CI)")
    ax.set_title("(b) Permanence circuits, d = 64 (one marker per seed)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig("figures/fig_jepa_width.png", dpi=170)
    print("wrote figures/fig_jepa_width.png")


if __name__ == "__main__":
    rows = measure_all()
    table = summarize(rows, "neuron")
    table_sae = summarize(rows, "sae")
    report(table)
    if table_sae:
        report(table_sae)
    with open("results/jepa_width_sweep.json", "w") as f:
        json.dump({"neuron": table, "sae": table_sae}, f, indent=1)
    roll = rollout_all()
    with open("results/jepa_rollout_behaviour.json", "w") as f:
        json.dump(roll, f, indent=1)
    make_figure(rows, roll)
