"""Analytical MAC estimate, deliberately separate from measured runtime."""
def estimate(vocab=32768, context=512):
    # Dense matrix multiplies, full causal SDPA triangle counted conservatively
    # as the full square (backend may evaluate masked tiles).
    d=256
    common=6*(4*d*d+2*d*1024+2*context*d)+d*vocab
    tape_projections=d*(4*64*2+4*128+4*3)+512*d
    lane_parameters=256*256*2+256*512+256*12+256*2+512+12+4+512*256+256
    width=round((lane_parameters-256)/513)
    baseline=common+2*d*width
    # Empty initial state: at most t live slots before token t. X31's tile
    # reach is context, so use all context slots for the conservative bound.
    tape=common+tape_projections+4*min(2944,context)*(64+128)
    return dict(context=context,vocab=vocab,base_dense_mac_per_token=baseline,
                tape_mac_per_token_upper_scoring_estimate=tape,
                ratio=tape/baseline,within_10_percent=tape/baseline<=1.1,
                exclusions=['softmax, elementwise operations, routing and journal bookkeeping',
                            'backward replay overhead and GPU launch overhead'],
                interpretation='approximate forward MAC match; not measured GPU compute equivalence')
