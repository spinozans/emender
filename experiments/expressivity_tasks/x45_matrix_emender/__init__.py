"""X45 — the registered matrix-state Emender (rank-1 ops on a chunked scan).

See ``PREREGISTRATION_MATRIX_EMENDER.md`` in ``x43_fork_resolved/``: the state is
``S in R^{r x d_v}`` per head on a fixed orthonormal key family, the five ops are
rank-1/diagonal updates consumed as a *fixed decision schedule*, and the scan is
chunked so nothing is checkpoint-replayed.
"""
