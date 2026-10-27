"""Retrospective, conservative certificates; never a predictive dispatcher."""
from dataclasses import dataclass
from ..x30_tape_cell.routing_trace import NO_WRITE, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL


@dataclass(frozen=True)
class Classification:
    mutation_candidate: bool
    entry_pure: bool
    reasons: tuple[str, ...]
    intra_match: int
    intra_pop: int


def classify(trace, batch=0, head=0, start=0, stop=None):
    """Classify one store; copy diagnostics to CPU before calling repeatedly."""
    stop = trace.case.shape[1] if stop is None else stop
    if not 0 <= start < stop <= trace.case.shape[1]:
        raise ValueError('expected nonempty in-range segment')
    if trace.freed is None:
        raise ValueError('death trace is required (fail closed)')
    allocated = set()
    reasons = set()
    intra_match = intra_pop = 0
    for t in range(start, stop):
        case = int(trace.case[batch, t, head])
        matched = int(trace.matched[batch, t, head])
        pop = int(trace.pop_target[batch, t, head])
        dst = int(trace.alloc_target[batch, t, head])
        intra_match += matched >= 0 and matched in allocated
        intra_pop += pop >= 0 and pop in allocated
        if allocated and bool(trace.w_on[batch, t, head]):
            reasons.add('write_after_append')
        if case not in (NO_WRITE, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL):
            reasons.add('correction_or_cap')
        if bool(trace.pop_fired[batch, t, head]):
            reasons.add('pop')
        if bool(trace.freed[batch, t, head].any()):
            reasons.add('death')
        if dst >= 0:
            if dst in allocated:
                reasons.add('allocation_reuse')
            allocated.add(dst)
    candidate = not (reasons - {'write_after_append'})
    return Classification(candidate, not reasons, tuple(sorted(reasons)), intra_match, intra_pop)


def segments(trace, batch=0, head=0):
    """Greedy entry-pure subsegments; event tokens are singleton X31 work."""
    result = []
    start = 0
    total = trace.case.shape[1]
    while start < total:
        end = start + 1
        if not classify(trace, batch, head, start, end).entry_pure:
            result.append((start, end, 'event'))
        else:
            while end < total and classify(trace, batch, head, start, end + 1).entry_pure:
                end += 1
            result.append((start, end, 'entry_pure'))
        start = end
    return result
