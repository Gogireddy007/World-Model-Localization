# Pre-Registration

**Project.** ε-Modularity in JEPA World Models.
**Frozen on.** 2026-06-24.
**Status.** Committed before any Phase 3 result is examined. Modifications to this file after that date are tracked as amendments with dates and reasons.

This document fixes the criteria by which empirical results in `paper.tex` (the canonical manuscript, §5.3 "Tier 2: the pre-registered study") will be judged. The intent is to foreclose post-hoc framing of the headline claims. Section references below (§5.3, §6, Appendix E) refer to `paper.tex`'s numbering; if you are cross-checking against `paper.md`, note that file is superseded and uses a different, incompatible section structure (see `README.md`).

---

## 1. Frozen behavioral metrics

Each invariant has a single contrastive latent metric, defined in §4 ("Method: CII-TRP") of `paper.tex`:

  L_{I_k}(g) = 𝔼[⟨ẑ − z⁺, u_k⟩ − ⟨ẑ − z⁻, u_k⟩],

with u_k = (z⁺ − z⁻) / ‖z⁺ − z⁻‖ computed on a held-out *metric-construction* split that is disjoint from training, attribution, and evaluation splits. The metric and split are fixed and will not be retuned after results are examined.

The specificity matrix (4×4) is computed on the metric-construction split. GATE 1 requires max off-diagonal / diagonal ≤ 0.15. If GATE 1 fails, the environment is redesigned (this is a Phase 0 / 1 problem, not a place to relax the metric).

## 2. Frozen ablation protocol

- **Operator:** resample ablation. Reference distribution: in-distribution activation marginal at each component, computed over a fixed activation-reference split.
- **Random-baseline control:** for each reported ablation, the random-ablation effect at matched |C| is computed over 50 random subsets, with the random-baseline median and 95% percentile reported.
- **Whole-layer control:** entire layer ablation at the modal layer of C_k.
- **Probe-location control:** ablation of the components highest-loading on the linear probe at the probe's peak layer.
- All ablations are repeated across 20 training seeds; effect sizes are reported as seed-pooled means with 95% CIs.

## 3. Pre-registered predictions and decision rules

| # | Prediction | Decision rule |
|---|------------|----------------|
| PR1 | For at least two of {I₁, I₂, I₃}, ε_k ≤ 0.30 at sparsity s ≤ 0.05 with Δ_k ≥ 3× seed-pooled noise. | If zero invariants meet this: modularity hypothesis fails in our regime. Paper reports the distributed-representation finding instead (§6, "What we are and are not claiming"). If exactly one invariant meets it: report as partial modularity. If two or three: positive headline. |
| PR2 | Rollout-level causal scrubbing preservation ≥ 0.85 of clean CII at horizon 16. | Below 0.85: the *hypothesis* about what each component computes is wrong, even if localization is correct. Report explicitly. |
| PR3 | The probe-located region for at least one invariant differs from the causal circuit, with probe-location ablation producing ≤ 0.5× the CII drop of causal-circuit ablation. | If probe-location ablation matches or exceeds causal ablation everywhere: there is no novel mechanistic content beyond probing, and that becomes a (still publishable) negative methodological finding. |
| PR4 | C_perm CII flags impending planning failures with ≥ 0.70 precision at ≥ 0.50 recall, leading behavioral detection by ≥ 3 rollout steps. | Below threshold: the planning-auditor application is not supported; remove the capability claim from the abstract. |
| PR5 | Across the (K, d) sweep, empirical ε per invariant correlates with the Welch lower bound at Pearson r ≥ 0.7. | Below 0.7: the phase boundary fits qualitatively but not quantitatively, report and discuss as falsifier F3 candidate. |
| PR6 | Within-architecture cross-seed SAE-feature alignment for C_perm ≥ 0.75 (CCA); cross-architecture ≥ 0.55. | Below thresholds: report as non-universality, which is an interesting finding rather than a failure. |
| PR7 | ≥ 0.7 of localization effect for low-temporal-frequency invariants concentrates in the slow predictor; for high-temporal-frequency invariants, in the fast predictor. | Below threshold: the H-JEPA temporal-abstraction story is not supported by this experiment; report null. |

## 4. What is *not* pre-registered (and why)

- Specific SAE hyperparameters (sparsity level, dictionary size). These are tuned on the metric-construction split using reconstruction–sparsity Pareto criteria and explicitly *not* tuned on the dissociation metrics. SAE diagnostics will be reported alongside the Tier-2 study once executed.
- Exact ACDC pruning thresholds. Threshold sweeps are reported; we commit to reporting the dissociation matrix at the threshold that maximizes circuit sparsity subject to preserving ≥ 0.8 of attribution-patched CII.
- The choice of invariants beyond the four named. If exploratory analysis suggests additional natural invariants in the environment, they are reported separately as exploratory and clearly marked.

## 5. Negative-result publication commitment

We commit to publishing the project regardless of which of the three outcomes (positive / partial / distributed) the empirical fork produces, with the framing in `paper.tex` §6 ("What we are and are not claiming" / "Why a null pilot is still worth reporting"). Specifically, we will *not* drop invariants from the report because they failed to localize; the per-invariant ε table (Appendix E of `paper.tex` once Tier 2 executes) will include every invariant we tested.

## 6. Statistical protocol

- Effect-size primary statistic: standardized mean difference (Cohen's d) between targeted-ablation and random-ablation distributions per invariant, with bootstrap 95% CIs over seeds.
- Multiple-comparison correction: Holm–Bonferroni across the K invariants for each headline test.
- Power: with 20 seeds and the observed seed-variance from GATE 0, we have ≥ 0.8 power to detect d = 0.6 at α = 0.05 across the 4-invariant Holm correction. Per-test power will be reported alongside the Tier-2 results once executed.

## 7. Amendment log

(Empty at freeze date 2026-06-24.)

### Amendment 1 (2026-09-28). Metric, environment, predictor; what is and is not pre-registered

Made before any Tier-2 study was run. The Tier-1.5 pilot of June 2026 did not use
the frozen metric (it used a CII score), so no reported number was selected
under the old definitions.

**A1.1 The frozen behavioural metric was degenerate.** As written in §1,
L_{I_k}(g) = E[<z_hat - z+, u_k> - <z_hat - z-, u_k>] = E[<z- - z+, u_k>], which does
not depend on the prediction z_hat at all, so every ablation effect under it is
identically zero. It is replaced by the midpoint form

  L_{I_k}(g) = E[ <z_hat - (z+ + z-)/2, u_k> / (<z+ - z-, u_k>/2) ],

which is +1 when the prediction sits at the correct future along u_k, -1 at the
foil, and 0 when indifferent. u_k is still estimated once on the
metric-construction split. GATE 0 becomes: L_k > 0 with a bootstrap 95% CI
excluding 0, for every invariant, on the evaluation split.

**A1.2 Environment v2.** (i) The de-correlation audit is computed on realised
events (occlusion-and-reemergence, contact, near-crossing), not on the sampled
flags, over generated clips; clips are rejection-sampled until realised events
match independently drawn flags. (ii) Matched pairs are generated from a
stored exact initial state by one simulator, so x+ and x- are pixel-identical
before divergence (v1 had a one-step time shift in every foil). (iii) The
identity foil is an appearance swap of two objects at a near-crossing: a pure
identity-label swap between identical-looking objects is pixel-identical and
leaves u_k undefined. (iv) Continuity violations no longer appear in training;
the continuity invariant is evaluated only through its foil (a visible jump).

**A1.3 Predictor.** The CPU-scale study uses a recurrent (GRU) two-level
predictor whose circuit components are the units of a readout layer feeding a
linear head (so ablation effects are exactly additive). The 8-layer
transformer, width-384, 20-seed study of §2 remains the pre-registered
large-scale study and has not been run.

**A1.4 PR5.** The joint-selectivity floor sqrt((K - r)/(r(K - 1))) is now a
theorem for additive component bases (paper, Theorem 1), so "empirical eps tracks
the Welch curve at r >= 0.7" is no longer a test of the theory. PR5 is replaced by:
(a) no measured joint eps below the floor for the neuron basis (a check of the
measurement pipeline, since it cannot happen if additivity holds), and (b) the
gap between measured joint eps and the floor, reported per width, as the
quantity of scientific interest.

**A1.5 Status of the new experiments.** The readout-width sweep, the trained
toy-network sweep, the SAE-basis comparison and the TRP necessity test were
designed and run on 2026-09-28 and are exploratory, not pre-registered. The
paper labels them as such. PR1-PR4, PR6 and PR7 are unchanged and remain
untested at the pre-registered scale.
