"""
Controlled physics environment for epsilon-modularity experiments (v2).

Design priorities, in order:

  1. Statistical independence of the invariant-triggering EVENTS that actually
     occur in each clip (occlusion-and-reemergence, contact, near-crossing),
     not merely of the flags used to request them. Clips are rejection-sampled
     until the realised events match independently drawn flags, and the audit
     is computed on realised events.
  2. Exact matched pairs. Every clip stores its full initial state; the
     physically-correct clip x+ and the violating foil x- are produced by the
     same simulator from that state, so they are identical up to the moment
     the intervention takes effect (checked by `divergence_time`).
  3. Action conditioning on a designated effector object.
  4. Determinism given a seed.

Invariants (the four Spelke-style regularities of the paper):
  permanence  - an object keeps moving on its path while hidden by an occluder.
                Foil: its velocity changes while it is hidden.
  collision   - objects in contact exchange momentum (elastic).
                Foil: objects pass through each other ("ghost" contact).
  identity    - objects keep their features through a near-crossing.
                Foil: the two crossing objects swap appearance at closest approach.
  continuity  - objects trace connected paths (control; never violated in training).
                Foil: a visible object jumps 12-20 px.

v1 bugs fixed here (see pressure_test_log.md, C17): foils were re-simulated
from frame 0 of an already-stepped clip (a one-step time shift confounded every
pair); class-7 objects rendered at the occluder intensity and were detected as
occluders; the de-correlation audit measured the sampled flags only.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

FRAME_SIZE = 64
CLIP_LEN = 48
DT = 1.0 / 30.0
MAX_OBJECTS = 4
NUM_OBJECT_CLASSES = 8
OBJECT_RADIUS_RANGE = (3.5, 5.5)
SPEED_RANGE = (10.0, 22.0)          # pixels per second
ACTION_DIM = 4
ACTION_SCALE = 40.0

OCCLUDER_INTENSITY = 235            # never used by any object class
BACKGROUND = 0


def class_intensity(c: int) -> int:
    """Object shade per class: 50..162, disjoint from the occluder value."""
    return 50 + 16 * int(c)


INVARIANT_NAMES = ["permanence", "collision", "identity", "continuity"]
EVENT_NAMES = ["permanence", "collision", "identity"]   # events that occur in training


@dataclass
class InvariantFlags:
    permanence: bool = False
    collision: bool = False
    identity: bool = False


@dataclass
class Object:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    class_id: int
    is_effector: bool = False


@dataclass
class Occluder:
    bbox: np.ndarray   # (x0, y0, x1, y1)


@dataclass
class Clip:
    frames: np.ndarray             # (T, H, W) uint8
    actions: np.ndarray            # (T, ACTION_DIM) float32
    flags: InvariantFlags          # requested
    events: dict                   # realised: name -> bool
    seed: int
    object_positions: np.ndarray   # (T, MAX_OBJECTS, 2), NaN for absent
    object_visible: np.ndarray     # (T, MAX_OBJECTS)
    object_class: np.ndarray       # (T, MAX_OBJECTS), -1 absent
    contact_events: list
    meta: dict = field(default_factory=dict)   # exact initial state, roles


# ----------------------------------------------------------------------------
# Simulation and rendering
# ----------------------------------------------------------------------------

def _occluded(pos, occluders) -> bool:
    return any(o.bbox[0] <= pos[0] <= o.bbox[2] and o.bbox[1] <= pos[1] <= o.bbox[3]
               for o in occluders)


def simulate(objects: list[Object], occluders: list[Occluder], actions: np.ndarray,
             n_steps: int, intervention: dict | None = None):
    """Step the world. Frame t shows the state AFTER step t.

    intervention (optional), at most one of:
      {"kind": "hidden_velocity", "obj": i, "vel": v}  - first step obj i is
            occluded, its velocity is set to v (permanence foil)
      {"kind": "ghost"}                                - no contact response
      {"kind": "swap_appearance", "pair": (i, j), "t": t}  - classes swapped
            from step t on (identity foil; physics unchanged)
      {"kind": "teleport", "obj": i, "t": t, "offset": o}  - position jump at t
    Returns positions (T, MAX_OBJECTS, 2), classes (T, MAX_OBJECTS), contacts.
    """
    objects = copy.deepcopy(objects)
    n = len(objects)
    positions = np.full((n_steps, MAX_OBJECTS, 2), np.nan)
    classes = np.full((n_steps, MAX_OBJECTS), -1, dtype=int)
    contacts = []
    iv = intervention or {}
    hidden_done = False
    for t in range(n_steps):
        if iv.get("kind") == "hidden_velocity" and not hidden_done:
            o = objects[iv["obj"]]
            if _occluded(o.pos, occluders):
                o.vel = np.array(iv["vel"], dtype=float)
                hidden_done = True
        for o in objects:
            if o.is_effector:
                o.vel = o.vel + actions[t, :2] * ACTION_SCALE * DT
        for o in objects:
            o.pos = o.pos + o.vel * DT
        if iv.get("kind") == "teleport" and t == iv["t"]:
            objects[iv["obj"]].pos = objects[iv["obj"]].pos + np.asarray(iv["offset"], float)
        for o in objects:
            for ax in (0, 1):
                if o.pos[ax] - o.radius < 0:
                    o.pos[ax] = o.radius; o.vel[ax] = -o.vel[ax] * 0.9
                if o.pos[ax] + o.radius > FRAME_SIZE:
                    o.pos[ax] = FRAME_SIZE - o.radius; o.vel[ax] = -o.vel[ax] * 0.9
        for i in range(n):
            for j in range(i + 1, n):
                a, b = objects[i], objects[j]
                dvec = b.pos - a.pos
                dist = float(np.linalg.norm(dvec))
                if dist < a.radius + b.radius and dist > 1e-6:
                    nh = dvec / dist
                    rel = float(np.dot(b.vel - a.vel, nh))
                    if rel < 0:
                        contacts.append((t, i, j))
                        if iv.get("kind") != "ghost":
                            a.vel = a.vel + rel * nh
                            b.vel = b.vel - rel * nh
                    if iv.get("kind") != "ghost":
                        ov = a.radius + b.radius - dist
                        a.pos = a.pos - 0.5 * ov * nh
                        b.pos = b.pos + 0.5 * ov * nh
        for i, o in enumerate(objects):
            positions[t, i] = o.pos
            classes[t, i] = o.class_id
        if iv.get("kind") == "swap_appearance" and t >= iv["t"]:
            i, j = iv["pair"]
            classes[t, i], classes[t, j] = classes[t, j], classes[t, i]
    return positions, classes, contacts


def render(positions, classes, radii, occluders):
    T = positions.shape[0]
    frames = np.zeros((T, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    visible = np.zeros((T, MAX_OBJECTS), dtype=bool)
    yy, xx = np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE]
    for t in range(T):
        frame = frames[t]
        for i in range(MAX_OBJECTS):
            if np.isnan(positions[t, i, 0]):
                continue
            cx, cy = positions[t, i]
            vis = not _occluded((cx, cy), occluders)
            visible[t, i] = vis
            if vis:
                frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= radii[i] ** 2] = class_intensity(classes[t, i])
        for occ in occluders:
            x0, y0, x1, y1 = occ.bbox
            frame[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))] = OCCLUDER_INTENSITY
    return frames, visible


# ----------------------------------------------------------------------------
# Realised events
# ----------------------------------------------------------------------------

def realised_events(positions, visible, contacts, radii, n_objects) -> dict:
    occl = False
    for i in range(n_objects):
        v = visible[:, i].astype(int)
        d = np.diff(v)
        downs = np.where(d == -1)[0]
        ups = np.where(d == 1)[0]
        if len(downs) and len(ups) and ups.max() > downs.min():
            occl = True
            break
    near = False
    for i in range(n_objects):
        for j in range(i + 1, n_objects):
            dist = np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
            if np.nanmin(dist) < radii[i] + radii[j] + 8.0 and \
               not any((ci, cj) == (i, j) for _, ci, cj in contacts):
                near = True
    return {"permanence": occl, "collision": len(contacts) > 0, "identity": near}


# ----------------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------------

class ControlledSampler:
    """Clips whose realised invariant events are independent by construction.

    Each clip draws flags independently from per-invariant marginals, then
    rejection-samples scenes until the realised events equal the flags.
    """

    def __init__(self, p_permanence=0.5, p_collision=0.5, p_identity=0.5,
                 seed: int = 0, max_tries: int = 400):
        self.p = {"permanence": p_permanence, "collision": p_collision, "identity": p_identity}
        self.rng = np.random.default_rng(seed)
        self.max_tries = max_tries

    def sample_flags(self) -> InvariantFlags:
        return InvariantFlags(**{k: bool(self.rng.random() < p) for k, p in self.p.items()})

    def generate_clip(self, seed: int, flags: InvariantFlags | None = None) -> Clip:
        flags = flags or self.sample_flags()
        local = np.random.default_rng(seed)
        want = {"permanence": flags.permanence, "collision": flags.collision,
                "identity": flags.identity}
        clip = None
        for attempt in range(self.max_tries):
            objects, occluders, roles = _spawn_scene(local, flags)
            actions = np.clip(local.normal(0, 0.3, size=(CLIP_LEN, ACTION_DIM)), -1, 1).astype(np.float32)
            clip = build_clip(objects, occluders, actions, flags, seed, roles)
            if clip.events == want:
                clip.meta["rejections"] = attempt
                return clip
        clip.meta["rejections"] = self.max_tries
        clip.meta["matched"] = False
        return clip


def build_clip(objects, occluders, actions, flags, seed, roles, intervention=None) -> Clip:
    positions, classes, contacts = simulate(objects, occluders, actions, CLIP_LEN, intervention)
    radii = np.zeros(MAX_OBJECTS)
    for i, o in enumerate(objects):
        radii[i] = o.radius
    frames, visible = render(positions, classes, radii, occluders)
    events = realised_events(positions, visible, contacts, radii, len(objects))
    return Clip(frames=frames, actions=actions, flags=flags, events=events, seed=seed,
                object_positions=positions, object_visible=visible, object_class=classes,
                contact_events=contacts,
                meta={"objects": copy.deepcopy(objects), "occluders": copy.deepcopy(occluders),
                      "roles": roles, "radii": radii, "intervention": intervention,
                      "matched": True})


def _spawn_scene(rng, flags: InvariantFlags):
    n = int(rng.integers(3 if flags.identity else 2, MAX_OBJECTS + 1))
    classes = list(rng.choice(NUM_OBJECT_CLASSES, size=n, replace=False))
    objects = []
    for i in range(n):
        pos = rng.uniform(10, FRAME_SIZE - 10, size=2)
        ang = rng.uniform(0, 2 * math.pi)
        spd = rng.uniform(*SPEED_RANGE)
        objects.append(Object(pos=pos, vel=spd * np.array([math.cos(ang), math.sin(ang)]),
                              radius=float(rng.uniform(*OBJECT_RADIUS_RANGE)),
                              class_id=int(classes[i]), is_effector=(i == 0)))
    roles = {}
    occluders: list[Occluder] = []
    if flags.permanence:
        # Put an occluder on object 1's path so it hides and re-emerges.
        o = objects[1]
        spd = rng.uniform(16, 22)
        o.vel = o.vel / np.linalg.norm(o.vel) * spd
        tau = rng.uniform(0.35, 0.6)                   # seconds until centred behind it
        c = o.pos + o.vel * tau
        w, h = rng.uniform(9, 13, size=2)
        c = np.clip(c, [w / 2 + 1, h / 2 + 1], [FRAME_SIZE - w / 2 - 1, FRAME_SIZE - h / 2 - 1])
        occluders.append(Occluder(bbox=np.array([c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2])))
        roles["occluded"] = 1
    if flags.collision:
        a, b = objects[0], objects[-1] if n > 2 else objects[1]
        d = b.pos - a.pos
        a.vel = d / (np.linalg.norm(d) + 1e-6) * rng.uniform(18, 24)
        roles["collide"] = (0, n - 1 if n > 2 else 1)
    if flags.identity:
        i, j = 1, 2
        a, b = objects[i], objects[j]
        # b passes a with a lateral miss distance just above contact.
        t_meet = rng.uniform(0.5, 0.9)
        meet = a.pos + a.vel * t_meet
        miss = a.radius + b.radius + rng.uniform(1.5, 4.0)
        dir_ = rng.standard_normal(2); dir_ /= np.linalg.norm(dir_)
        perp = np.array([-dir_[1], dir_[0]])
        target = meet + perp * miss
        spd = rng.uniform(*SPEED_RANGE)
        b.pos = np.clip(target - dir_ * spd * t_meet, 6, FRAME_SIZE - 6)
        b.vel = (target - b.pos) / t_meet
        roles["cross"] = (i, j)
    return objects, occluders, roles


# ----------------------------------------------------------------------------
# Matched pairs
# ----------------------------------------------------------------------------

def divergence_time(a: Clip, b: Clip) -> int:
    """First frame index at which the two clips' frames differ (CLIP_LEN if never)."""
    diff = np.any(a.frames != b.frames, axis=(1, 2))
    idx = np.where(diff)[0]
    return int(idx[0]) if len(idx) else CLIP_LEN


def _pair(sampler: ControlledSampler, seed: int, need: str, make_iv, tries: int = 50):
    rng = np.random.default_rng(seed + 7_919)
    for k in range(tries):
        f = sampler.sample_flags()
        setattr(f, need, True)
        base = sampler.generate_clip(seed * 100 + k, flags=f)
        if not base.meta.get("matched", True):
            continue
        iv = make_iv(base, rng)
        if iv is None:
            continue
        m = base.meta
        foil = build_clip(m["objects"], m["occluders"], base.actions, base.flags,
                          base.seed, m["roles"], intervention=iv)
        t = divergence_time(base, foil)
        if 6 <= t < CLIP_LEN - 1:
            base.meta["t_div"] = t; foil.meta["t_div"] = t
            return base, foil
    raise RuntimeError(f"could not build a {need} pair for seed {seed}")


def matched_pair_permanence(sampler, seed):
    def iv(base, rng):
        i = base.meta["roles"].get("occluded")
        if i is None:
            return None
        v = base.meta["objects"][i].vel
        ang = rng.uniform(math.pi / 3, 2 * math.pi / 3) * rng.choice([-1, 1])
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        return {"kind": "hidden_velocity", "obj": i, "vel": R @ v * rng.uniform(0.8, 1.2)}
    return _pair(sampler, seed, "permanence", iv)


def matched_pair_collision(sampler, seed):
    return _pair(sampler, seed, "collision", lambda b, r: {"kind": "ghost"} if b.contact_events else None)


def matched_pair_identity(sampler, seed):
    def iv(base, rng):
        pair = base.meta["roles"].get("cross")
        if pair is None:
            return None
        i, j = pair
        dist = np.linalg.norm(base.object_positions[:, i] - base.object_positions[:, j], axis=1)
        return {"kind": "swap_appearance", "pair": (i, j), "t": int(np.nanargmin(dist))}
    return _pair(sampler, seed, "identity", iv)


def matched_pair_continuity(sampler, seed):
    def iv(base, rng):
        n = len(base.meta["objects"])
        for _ in range(20):
            i = int(rng.integers(0, n)); t = int(rng.integers(10, CLIP_LEN - 6))
            if base.object_visible[t, i]:
                ang = rng.uniform(0, 2 * math.pi)
                return {"kind": "teleport", "obj": i, "t": t,
                        "offset": rng.uniform(12, 20) * np.array([math.cos(ang), math.sin(ang)])}
        return None
    return _pair_any(sampler, seed, iv)


def _pair_any(sampler, seed, make_iv):
    rng = np.random.default_rng(seed + 104_729)
    for k in range(50):
        base = sampler.generate_clip(seed * 100 + 50 + k)
        iv = make_iv(base, rng)
        if iv is None:
            continue
        m = base.meta
        foil = build_clip(m["objects"], m["occluders"], base.actions, base.flags,
                          base.seed, m["roles"], intervention=iv)
        t = divergence_time(base, foil)
        if 6 <= t < CLIP_LEN - 1:
            base.meta["t_div"] = t; foil.meta["t_div"] = t
            return base, foil
    raise RuntimeError("could not build a continuity pair")


MATCHED_PAIRS = {
    "permanence": matched_pair_permanence,
    "collision": matched_pair_collision,
    "identity": matched_pair_identity,
    "continuity": matched_pair_continuity,
}


# ----------------------------------------------------------------------------
# Audit on realised events
# ----------------------------------------------------------------------------

def mutual_information_binary(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, int); y = np.asarray(y, int)
    pxy = np.zeros((2, 2))
    np.add.at(pxy, (x, y), 1)
    pxy /= len(x)
    px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = pxy * np.log(pxy / (px @ py))
    return max(float(np.nansum(terms)), 0.0)


def audit_decorrelation(sampler: ControlledSampler, n_clips: int = 2000,
                        tau: float = 0.05, seed0: int = 0) -> dict:
    """Pairwise MI between REALISED events over n_clips generated clips."""
    ev = {k: [] for k in EVENT_NAMES}
    rej = []
    unmatched = 0
    for s in range(n_clips):
        c = sampler.generate_clip(seed0 + s)
        rej.append(c.meta.get("rejections", 0))
        unmatched += int(not c.meta.get("matched", True))
        for k in EVENT_NAMES:
            ev[k].append(c.events[k])
    M = np.zeros((3, 3))
    for i, a in enumerate(EVENT_NAMES):
        for j, b in enumerate(EVENT_NAMES):
            if i != j:
                M[i, j] = mutual_information_binary(ev[a], ev[b])
    rates = {k: float(np.mean(v)) for k, v in ev.items()}
    return {"mi_matrix": M, "event_names": EVENT_NAMES, "tau": tau,
            "passed": bool(np.all(M < tau)), "event_rates": rates,
            "mean_rejections": float(np.mean(rej)), "unmatched": unmatched,
            "n_clips": n_clips}


def _smoke_test() -> None:
    sampler = ControlledSampler(seed=0)
    clip = sampler.generate_clip(seed=42)
    print(f"[smoke] clip seed=42 flags={clip.flags} events={clip.events}")
    print(f"[smoke] frames {clip.frames.shape} {clip.frames.dtype}, actions {clip.actions.shape}")
    for name, fn in MATCHED_PAIRS.items():
        a, b = fn(sampler, seed=3)
        pre_equal = bool(np.array_equal(a.frames[:a.meta['t_div']], b.frames[:a.meta['t_div']]))
        print(f"[smoke] {name:11s} pair: t_div={a.meta['t_div']:2d}, identical before divergence={pre_equal}")
        assert pre_equal
    audit = audit_decorrelation(sampler, n_clips=1000)
    print(f"[smoke] realised-event audit over {audit['n_clips']} clips: rates={audit['event_rates']}, "
          f"mean rejections={audit['mean_rejections']:.1f}, unmatched={audit['unmatched']}")
    for row, name in zip(audit["mi_matrix"], audit["event_names"]):
        print(f"  {name:>11}: " + " ".join(f"{v:.4f}" for v in row))
    if audit["passed"]:
        print("[smoke] GATE 0 prerequisite (de-correlation) holds.")
    else:
        print("[smoke] FAIL: de-correlation audit; rebalance sampler.")


if __name__ == "__main__":
    _smoke_test()
