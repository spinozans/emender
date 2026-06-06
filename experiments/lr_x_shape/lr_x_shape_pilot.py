#!/usr/bin/env python3
"""LR x shape warm pilot for dense GDN-2 full-run shape selection.

This is a narrow wrapper around the E99 matched-controls harness.  It keeps the
same real-Pile training, held-out BPB measurement, throughput accounting, and
fresh-process checkpoint round-trip gate, while adding the two dense FLA-GDN
candidate shapes requested by task lr-x-shape.
"""
import os
import sys


THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..'))
sys.path.insert(0, ROOT)

from experiments.e99_1p3b_controls import e99_lm_controls as controls


PILOT_CONFIGS = {
    'fla-gdn-controls-shape': dict(
        level='fla-gdn',
        dim=2688,
        depth=21,
        n_heads=44,
        n_state=64,
        expansion=2.0,
        bf16=True,
        lr=8.63e-4,
        knob_lr_mult=1.0,
        batch_size=2,
        layer_kwargs=None,
        role=(
            'LR x shape candidate A: E99 controls dense GDN-2 shape '
            '(dim2688/depth21/44h/ns64)'
        ),
    ),
    'fla-gdn-handoff-shape': dict(
        level='fla-gdn',
        dim=3456,
        depth=12,
        n_heads=38,
        n_state=64,
        expansion=2.0,
        bf16=True,
        lr=8.627e-4,
        knob_lr_mult=1.0,
        batch_size=2,
        layer_kwargs=None,
        role=(
            'LR x shape candidate B: handoff FLA-GDN CMA optimum '
            '(dim3456/exp2/depth12/38h/ns64)'
        ),
    ),
}


controls.CONFIGS.update(PILOT_CONFIGS)


if __name__ == '__main__':
    controls.main()
