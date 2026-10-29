import math

import pytest

from experiments.expressivity_tasks.x42_pile_scale.velocity_score import score_metrics


def metrics(**changes):
    row = dict(nll_heldout=3., occ_frac=.5, ever_touched_fractions=[.5] * 24,
               r_cap_fire_rate=.01, occupancy_slope=.5)
    row.update(changes)
    return score_metrics(**row)


def test_occupancy_tiebreak_scale():
    assert metrics(occ_frac=.75)['score'] - metrics()['score'] == pytest.approx(.0025)


def test_exact_degeneracy_boundaries_and_additivity():
    boundary = metrics(ever_touched_fractions=[.01], r_cap_fire_rate=.05,
                       occupancy_slope=math.nextafter(.9, 0))
    assert not any(boundary['degeneracy_flags'].values())
    flagged = metrics(ever_touched_fractions=[.5, math.nextafter(.01, 0)],
                      r_cap_fire_rate=math.nextafter(.05, 1), occupancy_slope=.9)
    assert all(flagged['degeneracy_flags'].values())
    assert flagged['degeneracy_penalty'] == pytest.approx(.12)
    assert flagged['score'] == pytest.approx(3.125)


@pytest.mark.parametrize('field,value', [
    ('nll_heldout', float('nan')), ('nll_heldout', -1),
    ('occ_frac', float('inf')), ('occ_frac', -.001), ('occ_frac', 1.001),
    ('occ_frac', True), ('ever_touched_fractions', []),
    ('ever_touched_fractions', [float('nan')]),
    ('ever_touched_fractions', [-.1]), ('ever_touched_fractions', [1.1]),
    ('r_cap_fire_rate', -.1), ('occupancy_slope', float('nan')),
])
def test_invalid_metrics_cannot_be_scored(field, value):
    with pytest.raises(ValueError):
        metrics(**{field: value})


def test_multiple_head_fires_per_token_not_clipped():
    assert metrics(r_cap_fire_rate=2.)['degeneracy_flags']['pressure_runaway']
