"""Byte comparisons of the mask-only candidate across slot lifetimes.

The baseline cache is poisoned before every launch. Comparing with both a
finite sentinel and NaN detects accidental consumption of a not-yet-stored
cache value. This is differential evidence, not unchanged-X31 certification.
"""
from dataclasses import replace
from unittest.mock import patch
import torch
OLD = 'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live,0.)'
NEW = 'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live & (r_touch<Start+t-1),0.)'
BIRTH_OLD = 'born=tl.load(Birth+n*S+slots,valid,Clock)'
BIRTH_NEW = 'born=tl.load(Birth+n*S+slots,valid & live,Clock)'
from experiments.expressivity_tasks.x40_emender_lm import kernels, forward
from experiments.expressivity_tasks.x40_emender_lm.model import POLICIES
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import init_state


def bytes_equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            a.detach().cpu().numpy().tobytes() == b.detach().cpu().numpy().tobytes())


class Poison:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if 'ReadStrengthCache' in kernels.scan_kernel.arg_names:
                i = kernels.scan_kernel.arg_names.index('ReadStrengthCache')
                args[i].fill_(self.value)
            return kernels.scan_kernel[grid](*args, **kwargs)
        return launch


def fields(out, state, trace, pressure, evidence, errors):
    result = dict(output=out.clone(), pressure=pressure.clone(), evidence=evidence.clone(), errors=errors.clone(),
                  history=state.history.clone())
    for name in ('keys', 'values', 'strength', 'live', 'age', 'usage'):
        result['native.' + name] = getattr(state.native, name).clone()
    for i, value in enumerate(state.metadata):
        # due/terminal/births for never-live rows are undefined by the API.
        # Compare every defined metadata row; full bitmap/list/header arrays.
        if i in (2, 3, 4):
            value = value.reshape_as(state.native.live)[state.native.live]
        result[f'metadata.{i}'] = value.clone()
    for name in ('case', 'matched', 'alloc_target', 'pop_target', 'pop_fired', 'freed', 'w_on'):
        result['trace.' + name] = getattr(trace, name).clone()
    return result


def validate(target="x40"):
    global kernels, forward
    if target == "x31":
        from experiments.expressivity_tasks.x31_tape_cell_fused import kernels as kernels31, cell_fused
        kernels, forward = kernels31, cell_fused
    torch.set_num_threads(1)
    candidate = kernels.scan_kernel.src
    assert candidate.count(NEW) == (1 if target == "x40" else 0)
    birth_candidate = kernels.materialize_state.src
    assert birth_candidate.count(BIRTH_NEW) == 1
    birth_baseline = birth_candidate.replace(BIRTH_NEW, BIRTH_OLD)
    baseline = candidate.replace(NEW, OLD)
    def call_scan(*args, **kwargs):
        if target == "x31":
            for name in ("wire_logits", "match_coef", "error_coef", "evidence", "error_evidence"):
                kwargs.pop(name)
        return forward.tape_scan(*args, **kwargs)
    rows = []
    try:
        for policy, original_cfg in enumerate(POLICIES):
            for imported in (False, True):
                # Include both decaying and exactly-unit decay branches.
                for decay in (original_cfg.decay_rate, 1.):
                    cfg = replace(original_cfg, decay_rate=decay)
                    torch.manual_seed(4018)
                    b, t, h = 2, 70, 1
                    k = torch.randn(b, t, h, 64, device='cuda') * .3
                    q = torch.randn_like(k) * .3
                    v = torch.randn(b, t, h, 128, device='cuda')
                    pop = torch.zeros(b, t, h, device='cuda')
                    pop[:, 2::3] = 1.  # force POP death and lowest-free reuse
                    mask = torch.ones_like(pop)
                    append = torch.zeros_like(pop)
                    logits = torch.full_like(pop, 10.)
                    zeros = torch.zeros(h, device='cuda')
                    observed = []
                    coverage = dict(pops=0, allocations=0, freed=0, reused=0)
                    for variant, (source, poison) in enumerate(((baseline, 12345.), (baseline, float('nan')),
                                           (candidate, float('nan')))):
                        kernels.materialize_state._unsafe_update_src(birth_candidate if variant == 2 else birth_baseline)
                        kernels.materialize_state.hash = None
                        kernels.scan_kernel._unsafe_update_src(source)
                        kernels.scan_kernel.hash = None
                        state = None
                        if imported:
                            state = init_state(b, cfg, 'cuda')
                            state.live[..., :2] = True
                            state.strength[..., :2] = .5
                            state.age[..., :2] = 3
                            state.keys[..., :2, :] = .1
                            state.values[..., :2, :] = .2
                        snapshots = []
                        seen = [set() for _ in range(b)]
                        start = 0
                        for length in (1, 2, 31, 1, 35):
                            sl = slice(start, start + length)
                            e = torch.zeros(b, h, length, device='cuda')
                            err = torch.zeros_like(e)
                            with torch.no_grad(), patch.object(forward, 'scan_kernel', Poison(poison)):
                                out, state, trace, pressure = call_scan(
                                    k[:, sl], q[:, sl], v[:, sl], mask[:, sl].clone(),
                                    pop[:, sl], append[:, sl], state, cfg, zeros,
                                    wire_logits=logits[:, sl], match_coef=zeros,
                                    error_coef=zeros, evidence=e, error_evidence=err)
                            torch.cuda.synchronize()
                            snapshots.append(fields(out, state, trace, pressure, e, err))
                            if variant == 2:
                                coverage['pops'] += int(trace.pop_fired.sum())
                                coverage['freed'] += int(trace.freed.sum())
                                alloc = trace.alloc_target.cpu().reshape(b, length)
                                for n in range(b):
                                    for slot in alloc[n].tolist():
                                        if slot >= 0:
                                            coverage['allocations'] += 1
                                            coverage['reused'] += slot in seen[n]
                                            seen[n].add(slot)
                            start += length
                        observed.append(snapshots)
                    failures = []
                    for variant, snapshots in enumerate(observed[1:], 1):
                        for chunk, (a, z) in enumerate(zip(observed[0], snapshots)):
                            for name in a:
                                if not bytes_equal(a[name], z[name]):
                                    failures.append(dict(variant=variant, chunk=chunk, field=name))
                    row = dict(target=target, policy=policy, slots=cfg.slots, imported=imported, decay=decay,
                               chunks=[1, 2, 31, 1, 35], coverage=coverage, failures=failures)
                    rows.append(row)
                    print(row, flush=True)
                    assert not failures, row
                    assert all(coverage[name] > 0 for name in coverage), row
    finally:
        kernels.materialize_state._unsafe_update_src(birth_candidate)
        kernels.materialize_state.hash = None
        kernels.scan_kernel._unsafe_update_src(candidate)
        kernels.scan_kernel.hash = None
    return rows
