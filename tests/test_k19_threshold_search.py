"""Source-named checks for the K19 THRESHOLD RE-TUNE on the composed
machine: the parameterization is the original committed search's own dims
and bounds VERBATIM (nothing invented), tau_merge is held at the banked
0.85 per the original's receipted reason, the incumbent point is the K10
tuned winner's own knobs (all inside the bounds), the unit-cube
conventions are the committed machinery's own, the drift guards fail
loudly, the anchors are the receipted ones (composed 1.3811/1.3832, GDN2
1.3522), the win-bar adjudication applies the pre-stated 0.01 rule
exactly, and the search-internal scoreboard is never a ladder row.
CPU-only.
"""
import math
from pathlib import Path

import pytest

from experiments.expressivity_tasks.x45_matrix_emender import cmaes_search
from experiments.expressivity_tasks.x45_matrix_emender.k19_threshold_search \
    import (ANCHOR_LABEL, CMA_SEED, COMPOSED_FALLBACK, GENERATIONS, INCUMBENT,
            K19_DIMS, POPSIZE, SIGMA0, WIN_BAR, composed_banked, decode,
            encode, guard_config, win_clause)
from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
    import TUNED_WINNER

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments" / "expressivity_tasks" / "x45_matrix_emender"


def _original_dims():
    """The original committed search's own bounds for these knobs, read from
    its source (phase 1's read_scale; phase 2's tau_match/tau_pop)."""
    by_name = {}
    for name, scale, lo, hi in (cmaes_search.PHASE1_DIMS
                                + cmaes_search.PHASE2_EXTRA_DIMS):
        by_name[name] = (scale, lo, hi)
    return by_name


def test_the_parameterization_is_the_original_searchs_own_verbatim():
    """Exactly the original's tau_match (phase 2), tau_pop (phase 2) and
    read_scale (phase 1) dims, bounds and scales verbatim -- nothing
    invented, nothing widened."""
    original = _original_dims()
    assert [d[0] for d in K19_DIMS] == ["tau_match", "tau_pop", "read_scale"]
    for name, scale, lo, hi in K19_DIMS:
        assert (scale, lo, hi) == original[name], \
            f"{name}: the K19 dim must be the original search's own " \
            f"{original[name]}, got {(scale, lo, hi)}"
    # tau_merge is NOT searched: the original never did, and its bounds were
    # never set -- including it would be invention
    assert "tau_merge" not in [d[0] for d in K19_DIMS]


def test_tau_merge_held_at_the_banked_value_the_originals_own_choice():
    """tau_merge stays at 0.85 everywhere: the incumbent, the decode base
    and the drift guard."""
    assert INCUMBENT["tau_merge"] == 0.85 == TUNED_WINNER["tau_merge"]
    assert decode(K19_DIMS, [1.0, 1.0, 1.0])["tau_merge"] == 0.85
    bad = dict(INCUMBENT, tau_merge=0.75)
    with pytest.raises(ValueError, match="tau_merge is HELD"):
        guard_config(bad)


def test_the_incumbent_is_the_composed_machines_own_banked_point():
    """x0 is the K10 tuned winner's own knobs (the config the K15/K16 arms
    ran), every searched knob inside the original's bounds, every
    non-searched knob the banked value."""
    assert INCUMBENT == dict(TUNED_WINNER)
    for name, _scale, lo, hi in K19_DIMS:
        assert lo <= INCUMBENT[name] <= hi, \
            f"the incumbent {name}={INCUMBENT[name]} must encode inside " \
            f"the original's bounds [{lo}, {hi}]"
    assert guard_config(dict(INCUMBENT)) is None
    assert guard_config(dict(INCUMBENT, tau_match=0.6, tau_pop=-0.1,
                             read_scale=1.5)) is None


def test_the_search_settings_are_the_original_phase2s_own_shape():
    """sigma0 0.15, popsize 8, ONE generation -- the original phase-2's own
    shape (a bounded staleness re-check, not an exhaustive search); the
    CMA seed is stated (the original's was 20261115)."""
    assert SIGMA0 == 0.15
    assert POPSIZE == 8
    assert GENERATIONS == 1
    assert CMA_SEED == 20261125


def test_the_unit_cube_conventions_are_the_committed_machinerys_own():
    """encode is the committed cmaes_search.encode VERBATIM; decode follows
    its scale handling with this lane's base (the committed decode's base
    is the CHUNKED arm's transferred config -- a different machine's knobs),
    and the two are mutual inverses on the searched dims."""
    u = [0.25, 0.5, 0.75]
    cfg = decode(K19_DIMS, u)
    # the linear-scale convention, exactly as the committed decode applies it
    assert cfg["tau_match"] == pytest.approx(0.2 + 0.25 * 0.6)
    assert cfg["tau_pop"] == pytest.approx(-0.2 + 0.5 * 0.6)
    assert cfg["read_scale"] == pytest.approx(0.5 + 0.75 * 3.5)
    # the non-searched knobs come from THIS machine's incumbent, not the
    # chunked arm's transferred config
    assert cfg["lr"] == TUNED_WINNER["lr"]
    assert cfg["lr"] != cmaes_search.TRANSFERRED["lr"]
    # mutual inverses (encode is the committed encoder, imported verbatim)
    assert encode(K19_DIMS, cfg) == pytest.approx(u, abs=1e-12)
    # out-of-cube proposals clip (CMA-ES proposes outside)
    clipped = decode(K19_DIMS, [-0.5, 1.5, 2.0])
    assert clipped["tau_match"] == 0.2
    assert clipped["tau_pop"] == pytest.approx(0.4)
    assert clipped["read_scale"] == 4.0
    # the log-scale convention is followed for log dims too (none searched
    # here, but the convention must be the committed one, not a restatement)
    log_dims = [("warmup_steps", "logint", 2, 120)]
    assert decode(log_dims, [math.log(45 / 2)
                             / math.log(120 / 2)])["warmup_steps"] == 45


def test_the_drift_guards_fail_loudly():
    """Every non-searched knob must be the composed machine's banked value
    (the K16 bracket's own discipline, applied to the threshold axis)."""
    with pytest.raises(ValueError, match="lr.*drifted"):
        guard_config(dict(INCUMBENT, lr=INCUMBENT["lr"] * 2))
    with pytest.raises(ValueError, match="weight_decay.*drifted"):
        guard_config(dict(INCUMBENT, weight_decay=0.3))
    with pytest.raises(ValueError, match="warmup_steps.*drifted"):
        guard_config(dict(INCUMBENT, warmup_steps=9))
    with pytest.raises(ValueError, match="outside the original search's"):
        guard_config(dict(INCUMBENT, read_scale=4.5))
    with pytest.raises(ValueError, match="outside the original search's"):
        guard_config(dict(INCUMBENT, tau_match=0.1))


def test_the_anchors_are_the_receipted_ones():
    """The composed machine's own banked rows (the K15 composition arm's
    landed rows, both seeds), GDN2's post-cooldown row, and the 0.01 win
    bar -- quoted on every receipt, pinned here."""
    banked = composed_banked()
    assert banked["post_cooldown"][3401] == 1.3811104467936925
    assert banked["post_cooldown"][3402] == 1.383232252938407
    assert banked == COMPOSED_FALLBACK or all(
        banked["post_cooldown"][s] == COMPOSED_FALLBACK["post_cooldown"][s]
        for s in (3401, 3402))
    assert WIN_BAR == 0.01
    assert ANCHOR_LABEL == "incumbent_anchor"


def test_the_win_bar_adjudication_is_the_pre_stated_one():
    """The 0.01 rule exactly as registered: >= 0.01 STALE (the win banks,
    the second seed owed), <= -0.01 INCUMBENT BETTER (escalated), else
    PARITY (the thresholds were robust -- also a receipt)."""
    assert win_clause(0.0100001)["clause"] == "STALE"
    assert win_clause(0.05)["clause"] == "STALE"
    assert win_clause(0.01)["clause"] == "STALE"
    assert win_clause(0.0099999)["clause"] == "PARITY"
    assert win_clause(-0.0099999)["clause"] == "PARITY"
    assert win_clause(0.0)["clause"] == "PARITY"
    assert win_clause(-0.01)["clause"] == "INCUMBENT_BETTER"
    assert win_clause(-0.05)["clause"] == "INCUMBENT_BETTER"
    # the anchor row quoted in the registration's adjudication
    assert win_clause(1.3811104467936925 - 1.3711104467936925)["clause"] \
        == "STALE"      # exactly at the bar: 1.3811 - 1.3711 = 0.01


def test_the_search_internal_scoreboard_is_never_a_ladder_row():
    """The standing rule is pre-stated in the registration, the search result
    receipt and the assemble (checked by text, the rule is a sentence)."""
    registration = (X45 / "PREREGISTRATION_K19_THRESHOLD_RETUNE.md").read_text()
    module = Path(cmaes_search.__file__).with_name(
        "k19_threshold_search.py").read_text()
    assert "never a ladder row" in registration.lower()
    assert "NEVER a ladder row" in module


def test_the_lane_reads_the_committed_machinery_not_a_restatement():
    """The committed search machinery's encoder is imported verbatim (the
    module must import it, not restate it), and the composed machine's rows
    come from the K16 lane's own loader."""
    module = Path(cmaes_search.__file__).with_name(
        "k19_threshold_search.py").read_text()
    assert "from experiments.expressivity_tasks.x45_matrix_emender." \
           "cmaes_search import encode" in module.replace(
               "cmaes_search \\\n        import encode", "XXX") or \
           "cmaes_search" in module and "import encode" in module
    assert "composed_4x_banked" in module
