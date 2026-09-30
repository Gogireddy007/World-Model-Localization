"""
Hierarchical action-conditioned JEPA with a recurrent predictor (v2).

v1 used a single-frame MLP predictor: it could not infer velocities or
remember an occluded object, so permanence and collision were unlearnable by
construction. v2 gives both levels a GRU state:

  encoder f_theta          : CNN, frame -> latent z_t (dim `latent`)
  target encoder f_thetabar: EMA copy, stop-gradient (standard JEPA anti-collapse)
  fast predictor           : GRU over [z_t, a_t]  -> readout layer h_t (width
                             `readout`, the layer whose units are the circuit
                             components) -> linear head W_out;
                             z_hat_{t+1} = z_t + W_out h_t
  slow predictor           : same, over latents/actions pooled over `slow_pool` steps

Because the head is linear, the effect of intervening on any set of readout
units on a linear read-out of z_hat is exactly additive over units, which is
the additivity assumption of Theorem 1. The readout width is the `d` of the
phase-boundary experiments.

Clips come from a pre-generated cache (multiprocessing) because environment
generation, not the network, dominates CPU time.

Smoke test (CPU, < 1 min):   python train_hjepa.py --smoke
Train one model:             python train_hjepa.py --steps 3000 --readout 64 --seed 0 --out ckpt.pt
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import ControlledSampler, CLIP_LEN, FRAME_SIZE, ACTION_DIM

CACHE_DIR = os.environ.get("HJEPA_CACHE", "data_cache")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def _gen(args):
    seed, n, base = args
    s = ControlledSampler(seed=seed)
    fr = np.zeros((n, CLIP_LEN, FRAME_SIZE, FRAME_SIZE), np.uint8)
    ac = np.zeros((n, CLIP_LEN, ACTION_DIM), np.float32)
    for i in range(n):
        c = s.generate_clip(base + i)
        fr[i] = c.frames; ac[i] = c.actions
    return fr, ac


def build_cache(n_clips: int, seed: int = 0, workers: int = 8, name: str | None = None):
    """Generate (or load) n_clips training clips. Returns (frames uint8, actions)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name or f"clips_n{n_clips}_s{seed}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["frames"], d["actions"]
    per = int(np.ceil(n_clips / workers))
    jobs = [(seed * 1000 + w, per, seed * 10_000_000 + w * per) for w in range(workers)]
    with Pool(workers) as p:
        parts = p.map(_gen, jobs)
    frames = np.concatenate([a for a, _ in parts])[:n_clips]
    actions = np.concatenate([b for _, b in parts])[:n_clips]
    np.savez_compressed(path, frames=frames, actions=actions)
    return frames, actions


def sample_batch(frames, actions, batch, seq, rng):
    idx = rng.integers(0, len(frames), size=batch)
    t0 = rng.integers(0, CLIP_LEN - seq + 1, size=batch)
    fr = np.stack([frames[i, t:t + seq] for i, t in zip(idx, t0)]).astype(np.float32) / 255.0
    ac = np.stack([actions[i, t:t + seq] for i, t in zip(idx, t0)])
    return torch.from_numpy(fr)[:, :, None], torch.from_numpy(ac)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    """Small CNN. Default: 2x-downsampled input (fast). hires=True: full 64x64
    input with one more conv layer (needed to resolve sub-pixel contact events)."""

    def __init__(self, latent: int, hires: bool = False):
        super().__init__()
        head = [nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU()] if hires else [nn.AvgPool2d(2)]
        c0 = 32 if hires else 1
        self.net = nn.Sequential(
            *head,
            nn.Conv2d(c0, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, latent),
        )

    def forward(self, x):
        return self.net(x)


class RecurrentPredictor(nn.Module):
    """GRU state -> readout layer (circuit components) -> linear head."""

    def __init__(self, latent, action_dim, state, readout):
        super().__init__()
        self.gru = nn.GRU(latent + action_dim, state, batch_first=True)
        self.to_readout = nn.Linear(state, readout)
        self.out = nn.Linear(readout, latent)
        self.state = state

    def readout(self, states):
        return F.silu(self.to_readout(states))

    def run(self, zs, acts, s0=None, readout_hook=None):
        """zs (B,T,D), acts (B,T,A). Returns (states (B,T,S), readout h (B,T,R),
        z_hat (B,T,D) where z_hat[:, t] predicts z[:, t+1], final GRU state)."""
        states, sN = self.gru(torch.cat([zs, acts], -1), s0)
        h = self.readout(states)
        if readout_hook is not None:
            h = readout_hook(h)
        return states, h, zs + self.out(h), sN

    def forward(self, zs, acts):
        return self.run(zs[:, :-1], acts[:, :-1])[2]


class HJEPA(nn.Module):
    def __init__(self, latent=64, action_dim=ACTION_DIM, state=128, readout=64,
                 slow_readout=64, slow_pool=4, hires=False):
        super().__init__()
        self.latent, self.slow_pool = latent, slow_pool
        self.encoder = Encoder(latent, hires)
        self.target_encoder = Encoder(latent, hires)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.fast = RecurrentPredictor(latent, action_dim, state, readout)
        self.slow = RecurrentPredictor(latent, action_dim, state, slow_readout)

    @torch.no_grad()
    def ema_update(self, tau=0.99):
        for pt, ps in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(tau).add_(ps, alpha=1 - tau)

    def encode_seq(self, frames, target=False):
        B, T = frames.shape[:2]
        enc = self.target_encoder if target else self.encoder
        return enc(frames.reshape(B * T, *frames.shape[2:])).reshape(B, T, self.latent)

    def forward(self, frames, actions):
        z_ctx = self.encode_seq(frames)
        with torch.no_grad():
            z_tgt = self.encode_seq(frames, target=True)
        fast_pred = self.fast(z_ctx, actions)
        P = self.slow_pool
        usable = (frames.shape[1] // P) * P
        slow = None
        if usable >= 2 * P:
            B = frames.shape[0]
            zc = z_ctx[:, :usable].reshape(B, -1, P, self.latent).mean(2)
            zt = z_tgt[:, :usable].reshape(B, -1, P, self.latent).mean(2)
            ac = actions[:, :usable].reshape(B, -1, P, actions.shape[-1]).mean(2)
            slow = (self.slow(zc, ac), zt[:, 1:])
        return fast_pred, z_tgt[:, 1:], slow, z_ctx


def jepa_loss(pred, tgt):
    tgt = tgt.detach()
    return (1 - F.cosine_similarity(pred, tgt, dim=-1).mean()) + 0.1 * F.smooth_l1_loss(pred, tgt)


def variance_term(z, eps=1e-4):
    std = torch.sqrt(z.reshape(-1, z.shape[-1]).var(0) + eps)
    return torch.relu(1.0 - std).mean()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train(steps=3000, batch=32, seq=24, latent=64, readout=64, state=128, lr=1e-3,
          seed=0, n_clips=4000, log_every=250, data=None, device="cpu", verbose=True, hires=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frames, actions = data if data is not None else build_cache(n_clips, seed=0)
    model = HJEPA(latent=latent, readout=readout, state=state, hires=hires).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    losses = []
    t0 = time.time()
    for step in range(steps):
        fr, ac = sample_batch(frames, actions, batch, seq, rng)
        fr, ac = fr.to(device), ac.to(device)
        fp, ft, slow, zc = model(fr, ac)
        loss = jepa_loss(fp, ft)
        if slow is not None:
            loss = loss + 0.5 * jepa_loss(*slow)
        loss = loss + 0.1 * variance_term(zc)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); model.ema_update()
        losses.append(loss.item())
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:5d}  loss {loss.item():.4f}  latent std {zc.std().item():.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return model, losses


def smoke_test():
    print("=== HJEPA smoke test (CPU) ===")
    data = build_cache(256, seed=0, workers=4, name="smoke_n256.npz")
    model, losses = train(steps=150, batch=16, seq=16, latent=32, readout=32, state=64,
                          data=data, log_every=50)
    early, late = float(np.mean(losses[:10])), float(np.mean(losses[-10:]))
    print(f"loss {early:.4f} -> {late:.4f} ({100 * (early - late) / early:.1f}% decrease)")
    assert late < early * 0.9, "pipeline not learning"
    fr, _ = sample_batch(*data, 8, 8, np.random.default_rng(1))
    with torch.no_grad():
        std = model.encode_seq(fr).std().item()
    assert std > 0.05, "latents collapsed"
    print(f"final latent std {std:.3f}\n[smoke] PASS: loss decreases and representations do not collapse.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=24)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--readout", type=int, default=64)
    ap.add_argument("--state", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clips", type=int, default=4000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--hires", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.smoke:
        smoke_test()
    else:
        m, l = train(steps=a.steps, batch=a.batch, seq=a.seq, latent=a.latent,
                     readout=a.readout, state=a.state, seed=a.seed, n_clips=a.n_clips, hires=a.hires)
        if a.out:
            torch.save({"state_dict": m.state_dict(), "config": vars(a), "losses": l}, a.out)
            print(f"saved {a.out}")
