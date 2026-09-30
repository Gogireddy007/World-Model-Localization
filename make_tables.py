"""
Generate every results table in paper.tex directly from results/*.json.

The paper \\input{}s tables/*.tex, so the numbers in the tables cannot drift from
the measurements. verify.sh regenerates the tables and fails if they differ from
the files the paper was built with.

Run: python make_tables.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import re

import numpy as np

os.makedirs("tables", exist_ok=True)
INV = ["permanence", "collision", "identity", "continuity"]


def f2(x, nd=2):
    """Plain number; inf -> 'none' (no circuit with positive on-target effect)."""
    if x is None or not math.isfinite(x):
        return "none"
    return f"{x:.{nd}f}"


def sgn(x, nd=2):
    return f"${x:+.{nd}f}$".replace("-", "{-}") if x < 0 else f"${x:+.{nd}f}$"


def ci(c, nd=2):
    """Bootstrap interval; a non-finite bound means the interval is unbounded above."""
    lo, hi = c
    lo_s = f"{lo:.{nd}f}" if math.isfinite(lo) else r"\infty"
    hi_s = f"{hi:.{nd}f}" if math.isfinite(hi) else r"\infty"
    if not math.isfinite(lo):
        return ""
    return f"$[{lo_s}, {hi_s})$" if not math.isfinite(hi) else f"$[{lo_s}, {hi_s}]$"


def sign_test_p(k, n):
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def sci(p):
    e = int(math.floor(math.log10(p)))
    return rf"$<10^{{{e + 1}}}$"


def write(name, body):
    with open(f"tables/{name}.tex", "w") as f:
        f.write(body)
    print(f"wrote tables/{name}.tex")


# --------------------------------------------------------------------------
# Toy networks
# --------------------------------------------------------------------------
R = json.load(open("results/trained_toy_phase.json"))
V = [r for r in R if r["K_eff"] >= 2 and math.isfinite(r["joint_eps"])]
rows = []
for head, hname in [("linear", "linear"), ("relu", "ReLU")]:
    for reg, cond in [(r"$K_{\rm eff}\le d$", lambda r: r["K_eff"] <= r["d"]),
                      (r"$K_{\rm eff}> d$", lambda r: r["K_eff"] > r["d"])]:
        w = [r for r in V if r["head"] == head and cond(r)]
        ne = np.array([r["joint_eps"] for r in w]); sa = np.array([r["joint_eps_sae"] for r in w])
        ok = np.isfinite(sa)
        wins = int((sa[ok] < ne[ok]).sum()); n = int(ok.sum())
        below_n = sum(r["joint_eps"] < r["floor"] - 0.02 for r in w)
        below_s = sum(r["joint_eps_sae"] < r["floor_sae"] - 0.02 for r in w if math.isfinite(r["joint_eps_sae"]))
        q = lambda a: f"{np.median(a):.2f} [{np.percentile(a, 25):.2f}, {np.percentile(a, 75):.2f}]"
        rows.append(f"{hname} & {reg} & {len(w)} & {q(ne)} & {q(sa[ok])} & {np.median([r['floor'] for r in w]):.2f} & "
                    f"{wins}/{n} ({sci(sign_test_p(wins, n))}) & {below_n}\\,/\\,{below_s} \\\\")
write("toy_summary", "\n".join([
    r"\begin{tabular}{llcccccc}", r"\toprule",
    r"head & regime & $n$ & neuron basis & SAE basis & floor & SAE $<$ neuron ($p$) & below floor \\",
    r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))

# Full per-configuration toy table (medians over seeds with K_eff >= 2).
cfg_rows = []
for head in ["linear", "relu"]:
    for p in [0.5, 0.1]:
        for d in [4, 8]:
            Ks = sorted({r["K"] for r in R if r["head"] == head and r["p"] == p and r["d"] == d})
            for K in Ks:
                rs = [r for r in R if r["head"] == head and r["p"] == p and r["d"] == d and r["K"] == K]
                ok = [r for r in rs if r["K_eff"] >= 2 and math.isfinite(r["joint_eps"])]
                keff = "/".join(str(r["K_eff"]) for r in rs)
                if ok:
                    ne = np.median([r["joint_eps"] for r in ok])
                    sa = np.median([r["joint_eps_sae"] if math.isfinite(r["joint_eps_sae"]) else np.inf for r in ok])
                    fl = np.median([r["floor"] for r in ok])
                    cfg_rows.append(f"{'lin' if head == 'linear' else 'ReLU'} & {p} & {d} & {K} & {keff} & "
                                    f"{f2(fl)} & {f2(float(ne))} & {f2(float(sa))} \\\\")
                else:
                    cfg_rows.append(f"{'lin' if head == 'linear' else 'ReLU'} & {p} & {d} & {K} & {keff} & -- & -- & -- \\\\")
write("toy_full", "\n".join([
    r"\begin{tabular}{llllcccc}", r"\toprule",
    r"head & $p$ & $d$ & $K$ & $K_{\rm eff}$ (3 seeds) & floor & neuron $\eps$ & SAE $\eps$ \\",
    r"\midrule", *cfg_rows, r"\bottomrule", r"\end{tabular}"]))

# --------------------------------------------------------------------------
# World models
# --------------------------------------------------------------------------
def key(path):
    m = re.search(r"r(\d+)_s(\d+)", path)
    return int(m.group(1)), int(m.group(2))


roll = {key(f): json.load(open(f)) for f in glob.glob("results/rollout_r*_s*.json")}
beh_rows = []
for (d, s) in sorted(roll):
    r = roll[(d, s)]
    neu = json.load(open(f"results/jepa_r{d}_s{s}.json"))
    cells = []
    for inv in ["permanence", "collision"]:
        for H in ["1", "5"]:
            g = r[inv][H]
            star = "$^*$" if (g["ci"][0] > 0 or g["ci"][1] < 0) else ""
            cells.append(f"{sgn(g['gain'])}{star}")
    idc = max(abs(r["identity"][H]["gain"]) for H in ["1", "3", "5"])
    ctc = max(abs(r["continuity"][H]["gain"]) for H in ["1", "3", "5"])
    med = neu["behaviour"]["permanence"]["readout_mediated"]
    beh_rows.append(f"{d} & {s} & " + " & ".join(cells) + f" & {idc:.2f} & {ctc:.2f} & {sgn(med)} \\\\")
# The 4x-longer control run (12,000 steps, width 64, seed 3), same measurements.
if os.path.exists("results/long_r64_s3.json"):
    L = json.load(open("results/long_r64_s3.json"))
    r = L["rollout"]
    cells = []
    for inv in ["permanence", "collision"]:
        for H in ["1", "5"]:
            g = r[inv][H]
            star = "$^*$" if (g["ci"][0] > 0 or g["ci"][1] < 0) else ""
            cells.append(f"{sgn(g['gain'])}{star}")
    idc = max(abs(r["identity"][H]["gain"]) for H in ["1", "3", "5"])
    ctc = max(abs(r["continuity"][H]["gain"]) for H in ["1", "3", "5"])
    med = L["circuits"]["behaviour"]["permanence"]["readout_mediated"]
    beh_rows.append(r"\midrule")
    beh_rows.append(f"64 (12k steps) & 3 & " + " & ".join(cells) + f" & {idc:.2f} & {ctc:.2f} & {sgn(med)} \\\\")
write("jepa_behaviour", "\n".join([
    r"\begin{tabular}{cc cc cc cc c}", r"\toprule",
    r" & & \multicolumn{2}{c}{permanence gain} & \multicolumn{2}{c}{collision gain} & identity & continuity & readout \\",
    r"$d$ & seed & $H{=}1$ & $H{=}5$ & $H{=}1$ & $H{=}5$ & $\max_H|\text{gain}|$ & $\max_H|\text{gain}|$ & effect (perm.) \\",
    r"\midrule", *beh_rows, r"\bottomrule", r"\end{tabular}"]))

circ_rows = []
for f in sorted(glob.glob("results/jepa_r*_s*.json")):
    if f.endswith("_sae.json"):
        continue
    d, s = key(f)
    for basis, path in [("neuron", f), ("SAE", f.replace(".json", "_sae.json"))]:
        r = json.load(open(path))
        c = r["circuits"]["permanence"]
        sel, top = c["selective"], c["top_cii"]
        circ_rows.append(
            f"{d} & {s} & {basis} & {r['R']} & {r['rank']} & {f2(r['floor'])} & {f2(max(r['eps_weighted_opt']))} & "
            f"{f2(sel['eps'])} {ci(sel['eps_ci']) if math.isfinite(sel['eps']) else ''} & "
            f"{f2(top['eps'])} {ci(top['eps_ci']) if math.isfinite(top['eps']) else ''} & "
            f"{100 * max(r['additivity_rel_err']):.1f} \\\\")
write("jepa_circuits", "\n".join([
    r"\begin{tabular}{ccl cc cc cc c}", r"\toprule",
    r"$d$ & seed & basis & comps & rank & floor & weighted opt. & search $\eps_{\rm perm}$ [95\% CI] & top-$|\mathrm{CII}|$ $\eps_{\rm perm}$ [95\% CI] & add.\ err.\ (\%) \\",
    r"\midrule", *circ_rows, r"\bottomrule", r"\end{tabular}"]))

pv = json.load(open("results/probe_vs_causal.json"))
pv_rows = []
for r in pv:
    d, s = key(r["ckpt"])
    pv_rows.append(
        f"{s} & {r['probe_occlusion_acc']:.2f} ({r['probe_occlusion_majority']:.2f}) & {r['probe_position_r2']:.2f} & "
        f"{sgn(r['causal_drop'], 3)} {ci(r['causal_ci'], 3)} & {sgn(r['probe_occ_drop'], 3)} & {sgn(r['probe_pos_drop'], 3)} & "
        f"{sgn(r['random_mean'], 3)} ({r['random_p95']:.3f}) & {r['ratio_probe_occ']:.2f} / {r['ratio_probe_pos']:.2f} \\\\")
write("probe_vs_causal", "\n".join([
    r"\begin{tabular}{c cc c cc c c}", r"\toprule",
    r" & \multicolumn{2}{c}{decodability} & \multicolumn{4}{c}{drop in permanence score, 6 of 64 units ablated} & probe / causal \\",
    r"seed & occl.\ acc.\ (maj.) & position $R^2$ & causal units [95\% CI] & probe (occl.) & probe (pos.) & random (p95) & (occl.\ / pos.) \\",
    r"\midrule", *pv_rows, r"\bottomrule", r"\end{tabular}"]))

trp_rows = []
for s in [0, 1, 2]:
    r = json.load(open(f"results/trp_r64_s{s}.json"))
    pm, co = r["permanence"], r["collision"]
    trp_rows.append(
        f"{s} & {sgn(r['spearman_immediate_vs_trp'])} & {r['top_overlap']}/{r['m']} & {sgn(pm['trp'], 3)} & "
        f"{sgn(pm['immediate'], 3)} & {pm['random_p95']:.3f} & {sgn(co['trp'], 3)} & {co['random_p95']:.3f} \\\\")
write("trp", "\n".join([
    r"\begin{tabular}{ccc ccc cc}", r"\toprule",
    r" & & & \multicolumn{3}{c}{drop in permanence score} & \multicolumn{2}{c}{drop in collision score} \\",
    r"seed & Spearman & top-8 overlap & TRP units & single-step units & random p95 & TRP units & random p95 \\",
    r"\midrule", *trp_rows, r"\bottomrule", r"\end{tabular}"]))
