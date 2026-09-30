# Experimental Protocol

> **Status (2026-09-29).** This is the protocol for the *pre-registered,
> not-yet-run* GPU study. The CPU-scale study reported in `paper.tex` follows
> preregistration Amendment 1, which changes the following. Where this file
> conflicts with the amendment, the amendment governs.
> - **Metric:** the midpoint/preference metric replaces the degenerate one.
> - **Environment v2:** realised-event independence; exact pairs; an
>   appearance-swap identity foil; continuity never in training.
> - **Predictor:** recurrent predictors with a linear readout head.
> - **PR5:** replaced by the Theorem-1 floor check.
>
> The frozen metric in Phase 1 below must be read as the amended one.

Detailed protocol for executing the experiments described in `paper.tex` §5.3 ("Tier 2: the pre-registered study"). Designed to be handed to a collaborator (or future self) who can implement without further specification beyond reading code references.

---

## Phase 0, Environment and base training (weeks 1–4)

### 0.1 Environment

- **Engine.** 2.5D rigid-body sprite sim with parameterized object spawning, action conditioning (4-dim continuous action over a controlled effector), and configurable occluders. Frame rate 30 fps; clip length 64 frames.
- **Invariants and their generative controls.**
  - I₁ permanence: occluder presence p_occ, hidden trajectory r_hidden (sampled independently of other invariant flags).
  - I₂ collision: contact-event flag c, momentum-transfer coefficient k_mom.
  - I₃ identity: same-class crossing flag s_id (forces same-shape objects to cross paths).
  - I₄ continuity: teleport-perturbation flag t_p (control invariant; rare in training).
- **De-correlation enforcement.** Generate clips so that pairwise mutual information I(flag_i; flag_j) < 0.05 nats across the training distribution. Audit by sampling 100k clips and computing the 4×4 mutual-information matrix; if any entry exceeds 0.05, rebalance the sampler.
- **Matched-pair generation.** For each invariant I_k, generate matched pairs (x⁺, x⁻) by fixing the random seed and toggling only the invariant-k-relevant variable (e.g., for permanence: same scene, two hidden trajectories). At least 50k pairs per invariant.
- **Splits.** train / metric-construction / attribution / evaluation / OOD-test, 70/5/5/15/5 with no clip overlap.

### 0.2 Model

- **Architecture.** Two-level H-JEPA:
  - **Fast predictor:** 8-layer Transformer, width 384, 6 heads, action-conditioned, input span 8 frames.
  - **Slow predictor:** 6-layer Transformer, width 384, 6 heads, action-conditioned, input span 32 frames (temporally pooled latents from the encoder).
- **Encoder.** Shared ViT-style patch encoder, EMA target encoder with τ = 0.99.
- **Loss.** Smooth-L1 in latent space, summed across both predictors with weights 1.0 (fast) and 0.5 (slow).
- **Optim.** AdamW, lr 3e-4, warmup 2k steps, cosine decay, weight decay 0.05, batch size 256.
- **Seeds.** 20 for main; widths {192, 256, 384, 512} for scaling sweep with 5 seeds each.

### 0.3 Sanity (GATE 0)

For each invariant, train a probe on the *target encoder*'s latents and report sub-task accuracy. Train an additional model on a "physics-shuffled" control distribution (frames permuted within clips) and report the same. GATE 0 requires sub-task accuracy ≥ 0.85 on every invariant vs ≤ 0.55 for the shuffled control, on every seed.

If GATE 0 fails on any invariant: increase model capacity or training; if shuffled control matches, the invariant is reading off something other than physics, redesign the sub-task.

---

## Phase 1, Behavioral metrics (week 5)

### 1.1 Build u_k

For each invariant I_k, compute u_k on the metric-construction split as the normalized mean difference of target-encoder embeddings of (x⁺, x⁻) targets across 10k matched pairs.

### 1.2 Specificity matrix

Compute the 4×4 matrix M_{kj} = sensitivity of L_{I_k} to corruption of the I_j-relevant variable. Each entry is the change in L_{I_k} when only the j-th variable is perturbed in the input, averaged over 5k matched pairs.

GATE 1: diagonal dominance, max off-diagonal / diagonal ≤ 0.15. If fails: revisit u_k construction (use a larger basis, e.g., top-3 contrastive directions per invariant), or revisit the invariant design.

---

## Phase 2, Localization (weeks 6–10)

### 2.1 Probing map

Linear probe (logistic regression) at every layer, every spatial position, for each invariant. Record AUROC. Plot probing-AUROC maps (Figure 2, left).

### 2.2 Attribution-patched CII

For each invariant and each matched pair, compute the first-order CII via attribution patching:

  CII_k^attr(c) = ⟨∇_{a_c} ẑ · (a_c(x⁻) − a_c(x⁺)), u_k⟩ / ‖z⁺ − z⁻‖.

Aggregate across pairs (mean absolute value) to get a per-component importance map. Plot (Figure 2, right) and compare to probing map. Compute rank correlation.

### 2.3 Exact CII

Take top-K (K = 50) components per invariant by CII^attr. Run exact activation patching, computing real CII_k(c). Report concordance with attribution; use exact values henceforth.

### 2.4 SAE training

At the implicated layers (top-2 layers per invariant by §2.3), train sparse autoencoders. Dictionary size 8d, L1 sparsity penalty tuned on a reconstruction–sparsity Pareto curve on the metric-construction split. Train for 100M tokens; report dead-feature rate, mean L0, reconstruction MSE.

### 2.5 Feature-level CII

Recompute exact CII over SAE features at the implicated sites. Rank features per invariant.

### 2.6 Path patching

Take top-20 features per invariant; perform path patching to identify which downstream features they affect. Build the directed graph G_k of feature → feature edges.

### 2.7 ACDC

Run ACDC over G_k using CII_k as the behavioral metric. Sweep pruning thresholds {0.5, 0.6, 0.7, 0.8, 0.9} of attribution-patched CII preservation. Report the minimal circuit at each threshold; commit to the 0.8-preservation circuit for headline results (per pre-registration §4).

GATE 2: each invariant has a candidate circuit with |C_k| / |𝒱_SAE| ≤ 0.05 whose CII matches clean to ≥ 0.8. If no invariant achieves this, the project has discovered distributed implementation at the chosen (K, d), pivot to §7.6 framing.

---

## Phase 3. The double dissociation (weeks 11–13)

### 3.1 Run the table

For each invariant pair (k, j), measure L_{I_j}(g ∘ A_{C_k}). Fill the N×N table. Compute ε per invariant.

### 3.2 Controls

Run all controls listed in pre-registration §2: random-baseline (50 random subsets per |C|), whole-layer, probe-location. Report each row of the table.

### 3.3 Effect-size comparison

For each pair (k, j) with k ≠ j, test that |Δ_j(C_k)| < ε · |Δ_k(C_k)| at the pre-registered ε = 0.30, with Holm–Bonferroni correction across pairs.

### 3.4 The fork

Following pre-registration §3 PR1: classify outcome as positive / partial / distributed. Write the version of §8.2 corresponding to the outcome. Update the title per main paper §5.1 guidance.

---

## Phase 4, Verification (weeks 14–17)

### 4.1 Sufficiency / isolation

For each invariant, construct the *complement* ablation: ablate everything except C_k. Measure L_{I_k}. Compare to clean and to full ablation. Sufficiency is supported if complement-ablation preserves ≥ 0.7 of clean L_{I_k}.

### 4.2 Rollout causal scrubbing

State the hypothesis for each component in C_k (e.g., "feature f tracks the hidden-object position during occlusion"). Construct hypothesis-equivalence classes of inputs. At each rollout step, resample component activations within the equivalence class and measure preserved CII over horizons {1, 4, 8, 16}.

Per PR2: preservation ≥ 0.85 at horizon 16 supports the hypothesis. Document failure modes per component in §A.5.

### 4.3 Editing demonstration

- **Induce.** Set activations of C_perm to their values from a non-permanence-respecting foil input. Measure rate of permanence violation in subsequent rollout vs. baseline.
- **Repair.** On rollouts that naturally fail permanence, intervene to set C_perm activations to a "permanence-respecting" prototype (mean activation over respect-positive inputs). Measure improvement in failure rate.

Report both as Figure 5 panels.

---

## Phase 5, Pressure tests and universality (weeks 18–21)

### 5.1 Adversarial hijack

PGD attack on the input with the objective of *flipping* CII of C_perm. ℓ∞ budget {2/255, 4/255, 8/255}. Measure (i) the achieved CII shift, (ii) the rate of subsequent permanence violation in the prediction, (iii) baseline rate without attack.

### 5.2 OOD persistence

OOD test split varies object shape (held-out shape categories), occluder geometry (held-out geometries), and motion regime (held-out velocity ranges). Repeat the dissociation table on each axis; report the fraction of in-distribution effect size retained.

### 5.3 Compositional stress

Generate scenes requiring simultaneous permanence + collision (occluded object collides upon emergence). Measure (i) individual circuit effects, (ii) joint ablation effect, (iii) deviation from additivity.

### 5.4 Scaling-law sweep (the headline phase-boundary figure)

Train predictors at widths {192, 256, 384, 512} × invariant counts {2, 3, 4, 6} (24 configurations × 5 seeds = 120 runs). For each, run the full localization pipeline, compute ε per invariant. Plot ε vs. K/d and overlay Welch bound. Per PR5: Pearson r ≥ 0.7 supports the theory.

### 5.5 Lesion-and-retrain

Permanently zero C_perm in 5 trained models; fine-tune for 25k steps with the standard objective. Measure permanence recovery; rerun localization to identify any C_perm'. Compute SAE-feature overlap with original C_perm.

### 5.6 Cross-architecture universality

Train MLP-Mixer-predictor and SSM-predictor variants (same encoder, same data, 5 seeds each). Localize C_perm in each. Compute CCA and Procrustes alignment of SAE features across architectures and seeds (per PR6).

### 5.7 Diffusion contrast

Train a small latent diffusion world model on the same data. Run localization. Predict (and test): low alignment with JEPA C_perm despite comparable behavioral performance, indicating architecture-specific mechanism.

### 5.8 Pretrained transfer

Apply CII-TRP to a publicly available action-conditioned V-JEPA-2-style predictor on one curated permanence scene set. Report the candidate circuit, ablation effect, and concordance with the synthetic-environment findings. Scope: one invariant, single-model, qualitative.

---

## Phase 6, Planning auditor (weeks 22–23)

### 6.1 Planning setup

MPC with the H-JEPA as world model: receding-horizon 16-step rollout, cross-entropy method, 200 trajectories per step. Goal-reaching tasks: navigate effector to target through scenes including occluders. Generate 5k rollouts, label each as success / failure based on terminal goal achievement.

### 6.2 Auditor signals

For each rollout, log per-step:
- Behavioral signal: change in predicted-vs-actual latent alignment.
- Probe signal: linear-probe confidence in permanence.
- CII signal: real-time CII of C_perm computed against a permanence prototype.

### 6.3 Precision/recall

For each signal, sweep detection thresholds and compute precision/recall of failure prediction as a function of *lead time* (steps before terminal failure). Plot Figure 5 right panel: P/R curves for each signal, and lead-time-vs-recall curves.

### 6.4 Plan repair

For rollouts where the CII signal flags impending failure, intervene at the flagged step by *amplifying* C_perm activations toward the prototype. Measure success-rate improvement vs. matched control rollouts without intervention.

Per PR4: ≥ 0.70 precision at ≥ 0.50 recall with ≥ 3-step lead time supports the planning-auditor claim.

---

## Phase 7, Writing (overlap with 5 and 6, weeks 18–24)

- Lock all figures in their final form by week 22.
- Internal review and reviewer-style adversarial reading by week 23.
- Submission by week 24.

---

## Compute and reproducibility

- Estimated total compute: ~[X] GPU-days on A100-equivalent hardware. Largest line items: 120-run scaling sweep (~60%), main 20-seed training (~20%), SAE training across sites (~10%), planning rollouts (~5%), pretrained-transfer (~5%).
- All seeds, splits, hyperparameters, and circuit specifications versioned in code.
- Public release: environment generator, model checkpoints, SAE weights, discovered circuits as feature-index sets with CII-TRP code, and pre-registration timestamps.
