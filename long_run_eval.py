"""
Does longer training make collision learnable? (Sec. 6, "Longer training")

Evaluates checkpoints/long_r64_s3.pt (12,000 steps, 4x the main models, same
architecture, width 64, a fresh seed) with exactly the measurements used for the
six main models: open-loop rollout behaviour against the copy-last-frame
baseline, and the neuron-basis circuit analysis.

Run: python long_run_eval.py   -> results/long_r64_s3.json
"""

from __future__ import annotations

import json

import torch

from environment import INVARIANT_NAMES
from jepa_width_sweep import rollout_behaviour
from measure_epsilon import load_model, measure


def main(ckpt="checkpoints/long_r64_s3.pt"):
    torch.manual_seed(0)
    model, cfg = load_model(ckpt)
    roll = json.loads(json.dumps({inv: rollout_behaviour(model, inv) for inv in INVARIANT_NAMES}))
    for inv in INVARIANT_NAMES:
        print(inv, "  ".join(f"H{H}: {roll[inv][H]['gain']:+.3f} [{roll[inv][H]['ci'][0]:+.3f}, "
                            f"{roll[inv][H]['ci'][1]:+.3f}]" for H in ["1", "3", "5"]))
    circ = measure(model, n_pairs=450, sparsity=0.1, seed=0, verbose=False)
    out = {"ckpt": ckpt, "steps": cfg["steps"], "rollout": roll,
           "circuits": circ, "final_loss": None}
    json.dump(out, open("results/long_r64_s3.json", "w"), indent=1)
    c = circ["circuits"]["permanence"]["selective"]
    print(f"permanence circuit eps {c['eps']:.3f} {c['eps_ci']}; joint (neuron) {circ['joint_eps']}")
    print("wrote results/long_r64_s3.json")


if __name__ == "__main__":
    main()
