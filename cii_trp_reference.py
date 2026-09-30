"""
CII-TRP reference implementation.

Contrastive Interchange Interventions (CII) with Temporal-Rollout Patching (TRP)
for decoder-free latent predictors (the JEPA family).

This file is the reference specification of the method, written to be read as
a specification. The implementation used for every number in the paper is
`measure_epsilon.py` (CII, attribution matrices, resample ablation, circuit
search, neuron and SAE bases) and `trp_experiment.py` (TRP); this file it depends only on
PyTorch and assumes a predictor with a hookable forward pass. All functions are
documented with their mathematical definition next to the implementation so that
the paper, the appendix, and the code are word-for-word consistent.

Conventions
-----------
- Batch dimension B leading. Latents are shape (B, T, d) where T is the rollout
  horizon and d is the residual width.
- "Pair" inputs come in (x_plus, x_minus, target_plus, target_minus) tuples
  drawn from the matched-pair generator described in experimental_protocol.md.
- A "site" is a (layer, kind) tuple where kind is one of {"resid_pre",
  "resid_post", "mlp_out", "attn_out", "sae_feature"}. The predictor exposes
  forward hooks keyed by site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# 1. Invariant contrast axis u_k
# ----------------------------------------------------------------------------

def invariant_contrast_axis(
    target_encoder: Callable[[Tensor], Tensor],
    matched_pair_loader: Iterable[tuple[Tensor, Tensor]],
    num_pairs: int = 10_000,
) -> Tensor:
    """
    Compute u_k for one invariant.

        u_k = E[ z_plus - z_minus ] / || E[ z_plus - z_minus ] ||

    where z_plus, z_minus are EMA-target-encoder embeddings of the physically
    correct and incorrect continuations for the invariant.

    The expectation is taken on the metric-construction split, exactly once,
    and then frozen. Re-tuning u_k after seeing dissociation results would
    break the pre-registration.
    """
    accum = None
    n = 0
    for plus_batch, minus_batch in matched_pair_loader:
        with torch.no_grad():
            z_plus = target_encoder(plus_batch)
            z_minus = target_encoder(minus_batch)
        delta = (z_plus - z_minus).mean(dim=0)
        accum = delta if accum is None else accum + delta
        n += 1
        if n * plus_batch.shape[0] >= num_pairs:
            break
    u_k = accum / accum.norm()
    return u_k.detach()


# ----------------------------------------------------------------------------
# 2. Contrastive Interchange Score (CII)
# ----------------------------------------------------------------------------

@dataclass
class Site:
    """Identifies a patchable activation site in the predictor."""
    layer: int
    kind: str  # one of {"resid_pre", "resid_post", "mlp_out", "attn_out", "sae_feature"}
    feature_index: int | None = None  # populated for SAE features

    def key(self) -> str:
        if self.feature_index is None:
            return f"L{self.layer}/{self.kind}"
        return f"L{self.layer}/{self.kind}/f{self.feature_index}"


def cii_exact(
    predictor: Callable,
    x_plus: Tensor,
    x_minus: Tensor,
    u_k: Tensor,
    z_plus_norm: Tensor,
    site: Site,
    hook_cache: dict[str, Tensor],
) -> Tensor:
    """
    Exact CII at one site for a batch of matched pairs.

        CII_k(c) = < z_clean - z_patched, u_k > / || z_plus - z_minus ||

    where z_patched is the predictor's output when the activation at `site` is
    replaced by its value on x_minus.

    The hook_cache argument is the predictor's stored activations from a prior
    forward pass on x_minus, keyed by site. Two forward passes total:
        (1) cache_run = predictor(x_minus, store=True)
        (2) z_patched = predictor(x_plus, patch={site: cache_run[site.key()]})
        (3) z_clean   = predictor(x_plus)
    Steps (2) and (3) can run in parallel as a single double-batched pass with
    distinct hook configurations on each half.
    """
    z_clean = predictor(x_plus)
    z_patched = predictor(x_plus, patch={site.key(): hook_cache[site.key()]})
    direction = z_clean - z_patched  # (B, T, d) or (B, d) depending on metric horizon
    # Project onto u_k, average over time if needed, normalize by pair scale.
    if direction.dim() == 3:
        direction = direction.mean(dim=1)  # average over rollout for the scalar CII
    return (direction @ u_k) / z_plus_norm


def cii_attribution(
    predictor: Callable,
    x_plus: Tensor,
    x_minus: Tensor,
    u_k: Tensor,
    z_plus_norm: Tensor,
    sites: list[Site],
) -> dict[str, Tensor]:
    """
    Attribution-patched CII for every site in `sites`, in one forward + one
    backward pass per matched pair (vs. one forward per site for exact CII).

    First-order Taylor approximation of CII_k(c):

        CII_k^attr(c) = < grad_{a_c} z_pred . ( a_c(x_minus) - a_c(x_plus) ), u_k >
                        / || z_plus - z_minus ||

    This is the predictor-side analogue of "attribution patching" (Nanda 2023;
    Syed et al. 2023) generalized from logit difference to the latent-contrast
    metric. Equivalence to exact CII holds in the limit of small differences and
    is verified empirically; large differences are confirmed via exact CII on
    the top-K hits.
    """
    # 1. Cached forward on x_minus, store a_c(x_minus) for every site.
    with torch.no_grad():
        _ = predictor(x_minus, store_sites=[s.key() for s in sites])
    a_minus = {s.key(): predictor.activation_cache[s.key()].detach() for s in sites}

    # 2. Forward on x_plus with gradient tracking, scalar = < z_pred(x_plus), u_k >.
    predictor.zero_grad()
    z_pred = predictor(x_plus, store_sites=[s.key() for s in sites])
    if z_pred.dim() == 3:
        z_pred_scalar = (z_pred.mean(dim=1) @ u_k).sum()
    else:
        z_pred_scalar = (z_pred @ u_k).sum()
    z_pred_scalar.backward()

    # 3. For each site, dot product of gradient with (a_c(x_minus) - a_c(x_plus)).
    a_plus = {s.key(): predictor.activation_cache[s.key()].detach() for s in sites}
    grads = {s.key(): predictor.activation_grad[s.key()] for s in sites}
    out = {}
    for s in sites:
        k = s.key()
        diff = a_minus[k] - a_plus[k]
        out[k] = ((grads[k] * diff).sum(dim=tuple(range(1, grads[k].dim())))) / z_plus_norm
    return out


# ----------------------------------------------------------------------------
# 3. Temporal-Rollout Patching (TRP)
# ----------------------------------------------------------------------------

def trp_response_curve(
    predictor: Callable,
    x_plus: Tensor,
    x_minus: Tensor,
    u_k: Tensor,
    z_plus_norm: Tensor,
    site: Site,
    patch_step: int,
    horizons: list[int],
    hook_cache: dict[str, Tensor],
) -> dict[int, Tensor]:
    """
    Patch at rollout step `patch_step`, measure CII at each step in `horizons`
    (relative to patch_step). Returns {horizon: CII tensor of shape (B,)}.

    Definition:

        CII_k(c, t_patch, t_eval)
            = < z_clean[t_eval] - z_patched[t_eval], u_k > / || z_plus - z_minus ||

    where the patch is applied only at t_patch in the rollout, and propagation
    is through the predictor's own autoregressive rollout (latents predicted at
    step s become inputs at step s+1).
    """
    z_clean_rollout = predictor.rollout(x_plus, horizon=max(horizons))
    z_patched_rollout = predictor.rollout(
        x_plus,
        horizon=max(horizons),
        patch_schedule={(patch_step, site.key()): hook_cache[site.key()]},
    )
    out = {}
    for h in horizons:
        t = patch_step + h
        diff = z_clean_rollout[:, t] - z_patched_rollout[:, t]
        out[h] = (diff @ u_k) / z_plus_norm
    return out


# ----------------------------------------------------------------------------
# 4. SAE feature CII
# ----------------------------------------------------------------------------

def cii_over_sae_features(
    predictor: Callable,
    sae: Callable,  # encodes residual to sparse features and decodes back
    x_plus: Tensor,
    x_minus: Tensor,
    u_k: Tensor,
    z_plus_norm: Tensor,
    layer: int,
    feature_indices: list[int],
) -> dict[int, Tensor]:
    """
    Compute CII for individual SAE features at one layer.

    Patching a feature: encode the residual at `layer` on x_minus, take the
    feature's activation, replace just that coordinate in x_plus's encoded
    feature vector, decode back to a residual increment, and add it during the
    forward pass on x_plus.

    Decomposing patching into SAE-feature granularity is the key step from
    "important parts of the network" to "interpretable components". Polysemantic
    neurons make raw-neuron circuits hard to interpret; SAE-feature circuits
    are the level at which we report mechanism in §7.
    """
    out = {}
    for fi in feature_indices:
        site = Site(layer=layer, kind="sae_feature", feature_index=fi)
        # Build the patch by encoding x_minus with the SAE and isolating feature fi.
        with torch.no_grad():
            resid_minus = predictor.get_residual(x_minus, layer=layer)
            feat_minus = sae.encode(resid_minus)  # (B, T, n_features)
            patch_increment = sae.decode_single_feature(feat_minus[..., fi], fi)
        hook_cache = {site.key(): patch_increment}
        out[fi] = cii_exact(predictor, x_plus, x_minus, u_k, z_plus_norm, site, hook_cache)
    return out


# ----------------------------------------------------------------------------
# 5. Resample-ablation primitive
# ----------------------------------------------------------------------------

def resample_ablation(
    predictor: Callable,
    x: Tensor,
    sites: list[Site],
    reference_distribution: Callable[[Site], Tensor],
) -> Tensor:
    """
    Predict on x with the activations at `sites` replaced by samples from
    `reference_distribution`. Sites can be raw activations or SAE features.

    Reference distributions used in the paper:
      - global: marginal of the activation across training-distribution inputs
      - layer-matched: marginal restricted to the same layer position
      - context-matched: marginal restricted to inputs whose other features
        approximately match x's (a strong but expensive control)

    The sensitivity table in §A.6 reports results under all three.
    """
    patch = {}
    for s in sites:
        sample = reference_distribution(s)
        patch[s.key()] = sample
    return predictor(x, patch=patch)


# ----------------------------------------------------------------------------
# 6. Behavioral metric L_{I_k}
# ----------------------------------------------------------------------------

def behavioral_metric(
    predictor: Callable,
    matched_pair_loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    target_encoder: Callable,
    n_batches: int = 200,
) -> float:
    """
    L_{I_k}(g) = E[ ( ||z_hat - z_minus||^2 - ||z_hat - z_plus||^2 ) / ||z_plus - z_minus||^2 ]
               = E[ 2 < z_hat - (z_plus + z_minus)/2, z_plus - z_minus > / ||z_plus - z_minus||^2 ]

    The loader yields (context, future_plus, future_minus): the predictor sees
    the shared context; z_plus / z_minus are target-encoder latents of the
    physically-correct and violating futures. +1 = prediction at the correct
    future, -1 = at the foil, 0 = equidistant. Linear in z_hat, so ablation
    effects at a linearly-read layer are additive (paper, Lemma 1).

    NOTE: the earlier form E[<z_hat - z_plus, u> - <z_hat - z_minus, u>] is
    identically E[<z_minus - z_plus, u>] and does not depend on z_hat.
    """
    total, count = 0.0, 0
    for i, (ctx, fut_plus, fut_minus) in enumerate(matched_pair_loader):
        if i >= n_batches:
            break
        with torch.no_grad():
            z_hat = predictor(ctx)
            z_plus = target_encoder(fut_plus)
            z_minus = target_encoder(fut_minus)
            diff = z_plus - z_minus
            m = 2 * ((z_hat - (z_plus + z_minus) / 2) * diff).sum(-1) / (diff * diff).sum(-1).clamp_min(1e-12)
            total += m.sum().item()
            count += m.numel()
    return total / max(count, 1)


# ----------------------------------------------------------------------------
# 7. Pipeline orchestration
# ----------------------------------------------------------------------------

def localize_invariant(
    predictor,
    sae_per_layer,
    target_encoder,
    matched_pair_loader,
    invariant_name: str,
    config: dict,
):
    """
    The full localization pipeline for one invariant, as described in §4.4.

    Steps:
      1. Build u_k (frozen once on metric-construction split).
      2. Attribution-patched CII over all raw sites; rank.
      3. Exact CII on top-K candidates.
      4. SAE-feature CII at implicated layers.
      5. Path patching to derive wiring.
      6. ACDC over CII graph with threshold sweep.

    Returns: a Circuit object with feature indices, wiring edges, and per-step
    CII scores under the committed threshold (0.8 attribution preservation).
    """
    # See experimental_protocol.md §Phase 2 for hyperparameters at each step.
    raise NotImplementedError("orchestration scaffold; implement against the real predictor")


# ----------------------------------------------------------------------------
# Sanity tests on a tiny linear model, runnable.
# ----------------------------------------------------------------------------

def _sanity_test_linear() -> None:
    """
    Minimal linear-model sanity test: with a known invariant direction v and a
    known "circuit" of components aligned with v, CII should recover them with
    high score and assign near-zero score to off-axis components.

    This is the unit test we run before deploying CII on a real predictor.
    """
    torch.manual_seed(0)
    d = 32
    n_components = 64
    # Two invariant directions, near-orthogonal.
    v1 = torch.randn(d); v1 /= v1.norm()
    v2 = torch.randn(d); v2 -= (v2 @ v1) * v1; v2 /= v2.norm()
    # Components: half aligned with v1, half with v2, with noise.
    W = torch.zeros(n_components, d)
    for i in range(n_components):
        target = v1 if i < n_components // 2 else v2
        W[i] = target + 0.1 * torch.randn(d)
        W[i] /= W[i].norm()
    # "Predictor": z = sum_c x_c * W[c]
    def predictor(x):
        return x @ W  # (B, d)

    # Matched pair: x_plus = uniform, x_minus = x_plus with sign flip on c_target.
    B = 1024
    x_plus = torch.randn(B, n_components)
    x_minus = x_plus.clone()
    flipped = 7  # one of the v1-aligned components
    x_minus[:, flipped] *= -1
    z_plus = predictor(x_plus)
    z_minus = predictor(x_minus)
    u_k = (z_plus - z_minus).mean(0)
    u_k = u_k / u_k.norm()

    # Brute-force CII per component: ablate component c (zero its contribution).
    cii_scores = {}
    for c in range(n_components):
        x_ablated = x_plus.clone()
        x_ablated[:, c] = 0.0
        z_clean = predictor(x_plus)
        z_patched = predictor(x_ablated)
        cii_scores[c] = ((z_clean - z_patched) @ u_k).mean().item()

    # Assert: v1-aligned components get higher CII than v2-aligned ones for
    # this v1-flavored u_k.
    v1_mean = sum(cii_scores[c] for c in range(0, n_components // 2)) / (n_components // 2)
    v2_mean = sum(cii_scores[c] for c in range(n_components // 2, n_components)) / (n_components // 2)
    print(f"[sanity] mean CII for v1-aligned components: {v1_mean:.4f}")
    print(f"[sanity] mean CII for v2-aligned components: {v2_mean:.4f}")
    assert v1_mean > 5 * abs(v2_mean), "CII failed sanity: v1 vs v2 separation too small"
    print("[sanity] PASS")


if __name__ == "__main__":
    _sanity_test_linear()
