# Measured results (2026-09-29)

Every number below comes from a JSON file in `results/`. Nothing in this file is
a target or a placeholder.

## Theory checks: `results/rank_obstruction.json`

- **Exact solver.** The exact ε* (basic-feasible-solution enumeration) matches
  direct minimisation to 5e-4.
- **Theorem 1.** Checked on 525 random attribution matrices, including
  unnormalised ones: 0 violations of the joint floor, and 0 of the per-invariant
  certificate.
- **Tightness.**

  | frame | ε* | floor |
  |---|---|---|
  | simplex, d = 2, 3, 4, 6, 8 | 1/d, exactly | 1/d |
  | icosahedral ETF, d = 3, K = 6 | 0.4472 for every invariant (certificate too) | 1/√5 = 0.4472 |

- **Single invariants can beat the floor; the joint problem cannot.**
  Example at d = 8, K = 10: the best single invariant reaches 0.013 against a
  floor of 0.167, while the joint optimum is 0.358.

## Trained toy networks: `results/trained_toy_phase.json`

SAE basis more selective than neuron basis in 131/137 networks (sign test p < 1e-30).

156 networks, 137 analysable (K_eff ≥ 2). Joint ε of the best circuit, measured
by true ablations on fresh samples:

| head | regime | n | neuron basis | SAE basis | SAE better than neuron |
|---|---|---|---|---|---|
| linear | K_eff ≤ d | 63 | 0.59 [0.33, 0.82] | 0.14 [0.01, 0.40] | 94% |
| linear | K_eff > d | 8 | 1.16 [1.09, 1.19] | 0.91 [0.64, 0.96] | 100% |
| ReLU | K_eff ≤ d | 35 | 0.74 [0.54, 0.88] | 0.19 [0.02, 0.31] | 100% |
| ReLU | K_eff > d | 31 | 1.33 [1.07, 1.48] | 0.81 [0.46, 1.08] | 94% |

Values are median [IQR].

- **Below the floor:** 0 of 274 basis-network combinations.
- **Additivity error:** median 0.03 with a linear head, 0.07 and 0.36 with a ReLU
  head (below / above the boundary).
- **SAE quality:** fraction of variance unexplained is at most 0.6%.

## Environment: `python environment.py`

- **Exact pairs:** every pair is pixel-identical before divergence.
- **Realised events (1,000 clips):** rates are 0.52 / 0.51 / 0.50; the largest
  pairwise mutual information is 0.0033 nats; the sampler needs 3.0 attempts per
  clip on average.

## World models: `results/jepa_*.json`, `results/rollout_*.json`

Six models: widths 2 (seeds 0–1), 4 (seed 0) and 64 (seeds 0–2).

- **Permanence is learned.** Preference gain over copy-last-frame (open-loop
  rollout evaluation, 200 pairs) has a bootstrap 95% CI above 0 in all 6 models:
  +0.06 to +0.17 at one step, +0.37 to +1.13 at 5 steps.
- **Collision is not learned.** The gain is significantly negative at one or
  more horizons in 5/6 models: the models continue motion through contact.
- **Identity and continuity:** no different from copying, which already satisfies
  them. The readout-mediated effect on them is below 0.05 in every model.
- **Floor on the trained models:**
  - width 2, neuron basis: rank 2, floor 0.577, weighted optimum 1.49 and 5.79;
  - width 2, 8-feature SAE: rank 4, floor 0.
- **Permanence circuits at width 64,** held-out ε per seed:

  | basis / rule | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | neuron, search | 0.52 | 0.62 | 2.36 |
  | SAE, search | 0.39 | 0.56 | 1.86 |
  | top-\|CII\| (either basis) | 0.40–1.21 across seeds | | |

  No confidence interval lies below the pre-registered 0.30.

## Decodable vs. used (PR3): `results/probe_vs_causal.json`

Width 64, seeds 0–2, 6 of 64 readout units ablated, effects on held-out
permanence pairs:

| seed | occlusion probe acc. (majority) | hidden-position R² | causal units [95% CI] | probe units (occl. / pos.) | ratio to causal |
|---|---|---|---|---|---|
| 0 | 0.83 (0.75) | 0.95 | +0.043 [0.034, 0.051] | +0.027 / +0.017 | 0.64 / 0.39 |
| 1 | 0.82 (0.75) | 0.95 | +0.086 [0.064, 0.107] | +0.029 / +0.029 | 0.34 / 0.34 |
| 2 | 0.83 (0.75) | 0.88 | +0.059 [0.044, 0.076] | +0.026 / +0.032 | 0.44 / 0.55 |

PR3 (at least one probe ≤ 0.5 × causal) passes in 3/3 seeds.

## TRP: `results/trp_r64_s{0,1,2}.json`

- **Different units from single-step patching:** top-8 overlap is 0/8 in every
  seed; Spearman correlation −0.23, −0.30, −0.02.
- **Drop in the permanence score** when ablating 8 units:

  | units ablated | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | top-8 TRP units | +0.013 | +0.049 | +0.032 |
  | top-8 single-step units | −0.025 | −0.035 | −0.018 |
  | random (95th percentile) | 0.016 | 0.022 | 0.015 |

  TRP units beat single-step units in 3/3 seeds, and exceed the random 95th
  percentile in 2/3.
- **Not specific:** the same TRP units also shift collision beyond its random
  95th percentile in 2/3 seeds.

## Longer-training control: `results/long_r64_s3.json`

Width 64, seed 3, 12,000 steps (4× the main models), measured identically.

- **Collision still not learned, and worse:** gain relative to copying is −0.48
  at H = 1 and −0.51 at H = 5, significant at every horizon. Ablating the readout
  layer raises the collision score by 0.70, i.e. the readout pushes predictions
  toward the pass-through foil.
- **Permanence** stays above copying, but by less: +0.01 at H = 1, +0.05 at H = 5
  (both significant).
- **Identity and continuity:** small gains at longer horizons, at most +0.10.
