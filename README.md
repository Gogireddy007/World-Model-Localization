# When Is World-Model Knowledge Localizable?

Code for **"When Is World-Model Knowledge Localizable? A Tight Floor on Circuit
Selectivity, and What Trained Predictors Actually Do"**
by Yugandhar Reddy Gogireddy\* and Jithendra Reddy Gogireddy\*
(\*equal contribution).

Everything runs on a laptop CPU.

## What is here

| file | what it does |
|---|---|
| `rank_obstruction.py` | Exact best selectivity ε\*, numerical checks of the selectivity floor (Theorem 1) and its per-regularity certificate, tightness on equiangular tight frames |
| `trained_toy_phase.py` | 156 trained toy networks: circuit selectivity against the floor, over neurons and over SAE features |
| `environment.py` | 2.5D physics environment: realised-event independence, pixel-identical matched pairs for permanence, collision, identity and continuity |
| `train_hjepa.py` | Recurrent two-level action-conditioned JEPA |
| `measure_epsilon.py` | Contrastive metric, attribution matrices, resample ablation, certificate, circuit search (neuron and SAE bases) |
| `jepa_width_sweep.py` | Runs the measurement on every checkpoint, plus open-loop rollout behaviour |
| `probe_vs_causal.py` | Decodable vs. used: linear probes against causally selected units |
| `trp_experiment.py` | Temporal-Rollout Patching vs. single-step patching |
| `long_run_eval.py` | Evaluation of the 4× longer training control |
| `cii_trp_reference.py` | Readable specification of the method, with a sanity test |
| `make_tables.py` | Result tables (LaTeX) generated from `results/` |
| `figures/fig_pairs.py`, `figstyle.py` | Matched-pair figure; shared figure style |
| `results/` | JSON outputs behind every reported number (summary in `RESULTS.md`) |
| `checkpoints/` | The seven trained world models (about 2 MB each) |
| `docs/` | Pre-registration (with its amendment) and the protocol for the larger study |

## Setup

```bash
pip install -r requirements.txt
bash verify.sh
```

`verify.sh` runs the checks in about two minutes.

## Reproduce

Theory checks (about 20 s) and trained toy networks (about 5 min):

```bash
python rank_obstruction.py
python trained_toy_phase.py
```

World models. The checkpoints are included, so training is optional (about
one hour per model on three CPU threads):

```bash
python train_hjepa.py --readout 64 --seed 0 --out checkpoints/hjepa_r64_s0.pt
```

Measurement, probes, TRP and the long-run control, from the checkpoints:

```bash
python jepa_width_sweep.py
python probe_vs_causal.py
python trp_experiment.py --ckpt checkpoints/hjepa_r64_s0.pt --out results/trp_r64_s0.json
python long_run_eval.py
python make_tables.py
```

The first run of any world-model script generates a training-clip cache in
`data_cache/` (about 20 s, 8 processes).

## Figures

![Theorem 1](figures/fig_rank_obstruction.png)
![Trained networks](figures/fig_trained_toy_phase.png)
![World models](figures/fig_jepa_width.png)
# World-Model-Localizable
