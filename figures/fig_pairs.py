"""
Figure 1: real matched pairs from the environment (no synthetic content).

For each invariant, a shared context frame and the first divergent frame of
the physically-correct clip x+ and of the violating foil x-, with the pixels
that differ outlined. Every causal measurement in the paper is made on pairs
like these.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from environment import ControlledSampler, MATCHED_PAIRS  # noqa: E402

LABELS = {
    "permanence": "permanence\nfoil: velocity changes while hidden",
    "collision": "collision\nfoil: objects pass through",
    "identity": "identity\nfoil: crossing objects swap appearance",
    "continuity": "continuity (control)\nfoil: object jumps",
}
SEEDS = {"permanence": 11, "collision": 4, "identity": 9, "continuity": 2}


def main():
    import figstyle
    figstyle.apply()
    sampler = ControlledSampler(seed=123)
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.6), layout="constrained")
    for col, (name, fn) in enumerate(MATCHED_PAIRS.items()):
        a, b = fn(sampler, seed=SEEDS[name])
        t = a.meta["t_div"]
        t_show = min(t + 2, a.frames.shape[0] - 1)
        ctx = a.frames[max(0, t - 4)]
        diff = a.frames[t_show] != b.frames[t_show]
        for row, (img, title) in enumerate([(ctx, f"shared context (t={max(0, t - 4)})"),
                                            (a.frames[t_show], f"x+ correct (t={t_show})"),
                                            (b.frames[t_show], f"x- foil (t={t_show})")]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if row > 0:
                ax.contour(diff, levels=[0.5], colors=["#ff5a36"], linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=7.5)
        axes[0, col].text(0.5, 1.28, LABELS[name], transform=axes[0, col].transAxes,
                          ha="center", va="bottom", fontsize=8, weight="bold")
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_pairs.png"), dpi=170)
    print("wrote figures/fig_pairs.png")


if __name__ == "__main__":
    main()
