"""Mixture encoding + the deterministic anchor set for redo-e99-1-3b.

Mixture variable = `head_type_logits`, a 6-vector over the canonical head types
    [gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell]
(softmax -> fractions -> largest-remainder integer head counts in
TypedHeadMixtureLayer.allocate_types). A target-fraction vector f is encoded to
logits as log(f) (softmax(log f) == f), with zero entries pinned to LOG0 = -30.

The anchors are built so the report can read three things off matched-fraction
triples at a "nonlinear-capacity fraction" f in {1/6, 1/3, 1/2}:
    (a) native GDN-2 (linear)      -> M0_dense (the f-slot stays linear gdn2_recall)
    (b) GDN-2-shell nonlinear      -> S{1,2,3}: gdn2_recall + gdn2_nonlin_shell
    (c) legacy UnifiedCell nonlin  -> C{1,2,3}: gdn2_recall + nonlin corner
(a)-vs-(b) isolates the nonlinearity itself; (b)-vs-(c) isolates external plumbing.
M1 is the EXACT prior-E99 5:1 winner mixture (required anchor / comparability).
"""
from __future__ import annotations
import math
from typing import Dict, List

TYPE_NAMES: List[str] = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
                         'gdn2_nonlin_shell']
LOG0 = -30.0  # softmax(-30) ~ 0 -> a type that gets 0 heads


def fracs_to_logits(fr: Dict[str, float]) -> List[float]:
    """Target per-type fraction dict -> 6 logits (log f, 0 -> LOG0)."""
    out = []
    for t in TYPE_NAMES:
        f = float(fr.get(t, 0.0))
        out.append(math.log(f) if f > 0 else LOG0)
    return out


# Prior-E99 5:1 winner logits (5-type legacy; the layer pads the 6th with -inf =>
# 0 shell heads, reproducing the prior allocation exactly).
PRIOR_E99_LOGITS_5 = [3.9995, -1.9008, -0.9211, -2.8866, 2.4146]


def build_anchors() -> Dict[str, dict]:
    """name -> {logits, role, f_nonlin_slot, kind, note}."""
    A: Dict[str, dict] = {}
    A['M0_dense_gdn2'] = dict(
        logits=fracs_to_logits({'gdn2_recall': 1.0}),
        role='(a) native GDN-2 linear — BASELINE/CONTROL (not assumed winner)',
        f_slot=0.0, kind='dense', note='all gdn2_recall')
    A['M1_priorE99_5to1'] = dict(
        logits=list(PRIOR_E99_LOGITS_5),
        role='prior-E99 5:1 GDN2:nonlinear winner (required comparability anchor)',
        f_slot=None, kind='prior', note='softmax(prior logits) ~ 82% gdn / 17% nonlin / 1% count')
    # (c) legacy UnifiedCell nonlin corner at matched fractions
    A['C1_gdn_nonlin_f17'] = dict(
        logits=fracs_to_logits({'gdn2_recall': 5/6, 'nonlin': 1/6}),
        role='(c) legacy nonlin corner, f=1/6', f_slot=1/6, kind='nonlin_corner',
        note='gdn2_recall + UnifiedCell nonlin')
    A['C2_gdn_nonlin_f33'] = dict(
        logits=fracs_to_logits({'gdn2_recall': 2/3, 'nonlin': 1/3}),
        role='(c) legacy nonlin corner, f=1/3 (higher-nonlinear #1)', f_slot=1/3,
        kind='nonlin_corner', note='gdn2_recall + UnifiedCell nonlin')
    A['C3_gdn_nonlin_f50'] = dict(
        logits=fracs_to_logits({'gdn2_recall': 1/2, 'nonlin': 1/2}),
        role='(c) legacy nonlin corner, f=1/2 (higher-nonlinear #2)', f_slot=1/2,
        kind='nonlin_corner', note='gdn2_recall + UnifiedCell nonlin')
    # (b) GDN-2-shell nonlinear at matched fractions (the fairness control)
    A['S1_gdn_shell_f17'] = dict(
        logits=fracs_to_logits({'gdn2_recall': 5/6, 'gdn2_nonlin_shell': 1/6}),
        role='(b) GDN-2-shell nonlinear, f=1/6', f_slot=1/6, kind='shell',
        note='gdn2_recall + gdn2_nonlin_shell (native plumbing + nonlinear-in-time)')
    A['S2_gdn_shell_f33'] = dict(
        logits=fracs_to_logits({'gdn2_recall': 2/3, 'gdn2_nonlin_shell': 1/3}),
        role='(b) GDN-2-shell nonlinear, f=1/3', f_slot=1/3, kind='shell',
        note='gdn2_recall + gdn2_nonlin_shell')
    A['S3_gdn_shell_f50'] = dict(
        logits=fracs_to_logits({'gdn2_recall': 1/2, 'gdn2_nonlin_shell': 1/2}),
        role='(b) GDN-2-shell nonlinear, f=1/2', f_slot=1/2, kind='shell',
        note='gdn2_recall + gdn2_nonlin_shell')
    return A


def head_counts(logits: List[float], n_heads: int) -> dict:
    """Deterministic largest-remainder head counts for logging (uses the real
    model allocator so the counts match the trained model exactly)."""
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from ndm.models.typed_head_mixture import allocate_types
    a = allocate_types(n_heads, logits)
    return a['counts']


if __name__ == '__main__':
    import json
    A = build_anchors()
    for name, spec in A.items():
        c = head_counts(spec['logits'], 102)
        print(f"{name:22s} f_slot={str(spec['f_slot']):5s} counts={c}")
