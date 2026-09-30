#!/usr/bin/env bash
# Verification of the code and results in this repository.
#
# Runs every executable artifact in the project and reports pass/fail. Acts as
# a CI-style sanity check: an independent reviewer or replicator can run this
# and confirm the codebase actually behaves as the paper claims.
#
# Usage:
#   bash verify.sh        # run all checks
#   bash verify.sh -q     # quiet mode (only report failures)
#
# Exit status: 0 if every check passes, 1 otherwise.

set -u
cd "$(dirname "$0")"

PASS=0
FAIL=0
RESULTS=()
QUIET=0
if [ "${1:-}" = "-q" ]; then QUIET=1; fi

run_check() {
    local label="$1"; shift
    local cmd="$*"
    local out
    out=$(eval "$cmd" 2>&1)
    local rc=$?
    if [ $rc -eq 0 ]; then
        PASS=$((PASS + 1))
        RESULTS+=("PASS  ${label}")
        [ $QUIET -eq 0 ] && echo "PASS  ${label}"
    else
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  ${label}")
        echo "FAIL  ${label}"
        echo "      --- output:"
        echo "$out" | sed 's/^/      /' | head -10
    fi
}

echo "=== Repository verification ==="
echo "Working dir: $(pwd)"
echo ""

# Pick an interpreter that has the FULL scientific stack the checks need
# (numpy, matplotlib, torch), not merely numpy. Multiple installed Pythons
# commonly resolve `python3` to one without these packages. Override with
# PYTHON=/path/to/python bash verify.sh.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for cand in python3 python3.11 python3.12 python3.13 python3.14 python3.10 \
                /opt/homebrew/bin/python3 /opt/homebrew/bin/python3.11 \
                /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
                /usr/local/bin/python3; do
        if command -v "$cand" >/dev/null 2>&1 && \
           "$cand" -c "import numpy, matplotlib, torch" >/dev/null 2>&1; then
            PYTHON="$cand"
            break
        fi
    done
fi
if [ -z "$PYTHON" ]; then
    PYTHON="python3"
fi
export PYTHON
echo "Using interpreter: $PYTHON ($(command -v "$PYTHON" 2>/dev/null))"
echo ""
run_check "interpreter has numpy, matplotlib, torch" "$PYTHON -c 'import numpy, matplotlib, torch'"

# --- Theory (Sec. 4): exact eps*, Theorem 1, certificate, tightness. ---
run_check "rank_obstruction.py: Theorem 1 and certificate hold, tight on simplex and icosahedral ETF" \
    "$PYTHON rank_obstruction.py >/dev/null && $PYTHON -c '
import json
d = json.load(open(\"results/rank_obstruction.json\"))
assert d[\"thm1_violations\"] == 0 and d[\"certificate_violations\"] == 0, d
assert d[\"solver_check_max_abs_gap\"] < 5e-3
assert d[\"K_le_d_all_zero\"]
for k, (lo, hi) in d[\"simplex_eps_star\"].items():
    assert abs(lo - 1/int(k)) < 1e-9 and abs(hi - 1/int(k)) < 1e-9
e = d[\"icosahedral_etf\"]
assert max(abs(x - e[\"bound\"]) for x in e[\"eps_star\"] + e[\"certificate\"]) < 1e-9
print(\"ok\")'"

# --- Environment (Sec. 6): exact pairs, realised-event independence. ---
run_check "environment.py: pairs identical before divergence; realised-event audit passes" \
    "$PYTHON environment.py | grep -q 'GATE 0 prerequisite (de-correlation) holds'"

run_check "environment.py: no object class renders at the occluder intensity" \
    "$PYTHON -c '
from environment import class_intensity, NUM_OBJECT_CLASSES, OCCLUDER_INTENSITY
assert all(class_intensity(c) != OCCLUDER_INTENSITY for c in range(NUM_OBJECT_CLASSES))
print(\"ok\")'"

# --- Method: metric is not degenerate; additivity (Lemma 1) holds exactly. ---
run_check "metric depends on the prediction; resample-ablation effects are additive" \
    "$PYTHON -c '
import torch
from measure_epsilon import pair_metric, attributions, mc_resample_effect
torch.manual_seed(0)
N, D, R = 400, 16, 12
W = torch.randn(D, R); zp = torch.randn(N, D); zm = zp - W.sum(1)
assert torch.allclose(pair_metric(zp, zp, zm), torch.ones(N), atol=1e-5)
assert torch.allclose(pair_metric(zm, zp, zm), -torch.ones(N), atol=1e-5)
h = torch.randn(N, R) + 1.5; ref = torch.randn(5000, R)
zh = torch.randn(N, D)
A = attributions(W, h, ref.mean(0), zp, zm).mean(0)
C = [0, 3, 7]
mc = mc_resample_effect(W, h, zh, zp, zm, C, ref, n_draws=400)
an = A[C].sum().item(); assert abs(an) > 0.1 and abs(mc - an) < 0.03 * abs(an), (mc, an)
print(\"ok\")'"

run_check "cii_trp_reference.py sanity test" "$PYTHON cii_trp_reference.py | grep -q PASS"

# --- Trained toy networks (Sec. 5): floor never beaten on a quick subset. ---
run_check "trained_toy_phase.py: trained nets respect the floor (quick subset)" \
    "$PYTHON -c '
from trained_toy_phase import analyse
for args in [(6, 4, 0.5, \"linear\", 0), (12, 4, 0.1, \"relu\", 0), (12, 8, 0.5, \"linear\", 1)]:
    r = analyse(*args)
    if r[\"K_eff\"] >= 2:
        assert r[\"joint_eps\"] >= r[\"floor\"] - 0.02, r
        assert r[\"joint_eps_sae\"] >= r[\"floor_sae\"] - 0.02, r
print(\"ok\")'"

# --- World model training pipeline. ---
run_check "train_hjepa.py smoke test (loss decreases, no collapse)" \
    "$PYTHON train_hjepa.py --smoke | grep -q 'PASS'"

# --- Figures that are regenerated from code. ---
run_check "figures/fig_pairs.py regenerates" "$PYTHON figures/fig_pairs.py | grep -q '^wrote '"

# --- Result files referenced by the paper exist. ---
run_check "result files are present" \
    "test -f results/rank_obstruction.json && test -f results/trained_toy_phase.json && test -f results/jepa_width_sweep.json && test -f results/probe_vs_causal.json && test -f results/long_r64_s3.json && test -f results/trp_r64_s0.json && test -f results/trp_r64_s1.json && test -f results/trp_r64_s2.json"

# --- Result tables regenerate from results/. ---
run_check "make_tables.py generates all tables from results/" "$PYTHON make_tables.py | grep -c '^wrote ' | grep -qx 6"
run_check "PR3 (decodable vs used) result present and consistent" \
    "$PYTHON -c '
import json
r = json.load(open(\"results/probe_vs_causal.json\"))
assert len(r) == 3
for x in r:
    assert x[\"probe_position_r2\"] > 0.8 and x[\"causal_drop\"] > 0
    assert min(x[\"ratio_probe_occ\"], x[\"ratio_probe_pos\"]) <= 0.5 and x[\"PR3_pass\"]
    mc, an = x[\"additivity_check\"][\"monte_carlo\"], x[\"additivity_check\"][\"analytic\"]
    assert abs(mc - an) < 0.1 * abs(an)
print(\"ok\")'"


echo ""
echo "=== Summary ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo ""
if [ $FAIL -eq 0 ]; then
    echo "ALL CHECKS PASSED."
    exit 0
else
    echo "FAILURES PRESENT."
    exit 1
fi
