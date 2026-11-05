from experiments.expressivity_tasks.x43_fork_resolved.theory_forced_span import context_counts,labels,forced_runs


def test_exact_counts_never_cross_documents_or_window_reset():
    counts=context_counts([b'abx',b'aby',b'ccz'])
    assert counts[b'ab']=={120:1,121:1}
    assert b'xa' not in counts and b'yc' not in counts
    bins,h=labels(b'abx',1,1,counts)
    # Window resets at byte1: only b'b', not b'ab', is accessible.
    assert bins==[1] and h[0]>0


def test_forced_runs_do_not_include_forks_or_short_runs():
    assert list(forced_runs([0,0,1,0,0,0,0,2,0,0,0,0,0]))==[(3,7),(8,13)]
