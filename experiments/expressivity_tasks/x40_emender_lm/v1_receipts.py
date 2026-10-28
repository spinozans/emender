"""Digest the completed ignition-v1 evaluation into receipt files.

Reads receipts/ignition_v1/evaluation.json (written by the training process
against the final in-memory model after step 3052) and emits two receipts:

  long_context.json       five-length held-out PPL + 128-token NLL bins +
                          per-block/head occupancy/pressure telemetry, compared
                          with the X34 base/tape long-context receipts.
  compression_slopes.json occupancy-vs-length log-log slopes per block/head,
                          capacity/pressure context, hierarchy ordering and the
                          X39 semantic single-lane reference.

CPU-only; no model or checkpoint is loaded. The raw telemetry stays in
evaluation.json, bound by sha256 here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REC = ROOT / 'receipts' / 'ignition_v1'
BUDGETS = (64, 256, 736, 1472)
HORIZONS = (128, 512, 2048, 8192)
LENGTHS = (512, 1024, 2048, 4096, 8192)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    path = REC / 'evaluation.json'
    d = json.loads(path.read_text())
    assert d['status'] == 'completed', d['status']
    return d, dict(path=str(path.relative_to(ROOT.parent)),
                  sha256=sha(path))


def head_digest(h):
    """One block/head telemetry summary; exact floats from the raw receipt."""
    return dict(
        peak_live=h['peak_live_per_head'][0],
        mean_live=h['mean_live_per_head'][0],
        final_live=h['final_live_per_head'][0],
        pressure_mean=h['pressure_mean_per_head'][0],
        pressure_max=h['pressure_max_per_head'][0],
        pressure_fraction=h['pressure_fraction_per_head'][0],
        rcap_fraction=h['rcap_fraction_per_head'][0],
        w64_saturation_fraction=h['w64_saturation_fraction_per_head'][0],
        write_enabled_fraction=h['write_enabled_fraction_per_head'][0],
        allocation_rate_per_token=h['allocation_rate_per_token_head'][0],
        freed_decay=h['freed_decay_per_head'][0],
        freed_pop=h['freed_pop_per_head'][0],
        pop_count=h['pop_count_per_head'][0],
        gate_mean=h['gate_mean'],
        match_mean=h['match_mean'],
        error_mean=h['error_mean'],
        routing_fractions=h['routing_fractions'],
    )


def aggregate(digests):
    """Peak/pressure take the max over the two windows; means take the mean."""
    out = {}
    for k in digests[0]:
        if k == 'routing_fractions':
            continue
        values = [d[k] for d in digests]
        if k in ('peak_live', 'pressure_max', 'pressure_fraction',
                 'rcap_fraction', 'w64_saturation_fraction'):
            out[k] = max(values)
        else:
            out[k] = float(np.mean(values))
    out['routing_fractions'] = {
        r: float(np.mean([d['routing_fractions'][r] for d in digests]))
        for r in digests[0]['routing_fractions']}
    return out


def long_context(ev, source):
    contexts = []
    for row in ev['contexts']:
        per_block = {}
        for b in range(6):
            per_block[str(b)] = [
                aggregate(
                    [head_digest(win[b][h]) for win in row['telemetry']])
                for h in range(4)]
        contexts.append(dict(
            context=row['context'], windows=row['windows'],
            nll=row['nll'], perplexity=row['perplexity'],
            nll_bins128=row['nll_bins128'],
            per_block_head=per_block))
    x34 = {}
    for name in ('base_lm', 'tape_lm'):
        path = ROOT.parent / 'x34_lm_pilot/receipts' / f'probe_pilot_{name}.json'
        receipt = json.loads(path.read_text())
        x34[name] = dict(
            path=str(path.relative_to(ROOT.parent)), sha256=sha(path),
            context_ppl={str(r['context']): r['perplexity'] for r in receipt['context']},
            protocol_limit='Historical X34 B8 receipts, not batch-matched X40 B64; '
                           'lengths beyond 4096 were never run and remain missing.')
    ratio = ev['long_context_ppl_ratio']
    return dict(
        status='completed',
        receipt='long_context_no_kv_cache_thesis',
        model='ignition-v1 final weights (in-memory, step 3052, 100007936 tokens)',
        source=source,
        held_out_validation=ev['validation'],
        protocol='X34 context_probe: two fixed held-out windows per length, store '
                 'reset per window, full-position NLL, 128-token NLL bins, all 24 '
                 'head occupancy/pressure profiles; peak/pressure aggregated as '
                 'max over the two windows, means as window means.',
        registered_prediction=dict(
            text='Expect graceful loss growth (8192/512 PPL < 2).',
            limit=2.0, measured=ratio,
            t512=ev['contexts'][0]['perplexity'],
            t8192=ev['contexts'][-1]['perplexity'],
            verdict='pass' if ratio < 2.0 else 'fail'),
        contexts=contexts,
        x34_comparison=x34,
        notes=[
            'Decay is graceful, not cliff-like: 8192/512 PPL ratio '
            f'{ratio:.4f} vs X34 base_lm 4096/512 ratio '
            f'{x34["base_lm"]["context_ppl"]["4096"]/x34["base_lm"]["context_ppl"]["512"]:.3f} '
            '(6.381 -> 29.738) and tape_lm 4.501 (6.515 -> 29.322); X34 arms were '
            'never measured at 8192.',
            'All six gates stayed open at sigmoid(2)=0.88080 throughout; every '
            'nonpadding token had a write opportunity in every block.',
            'Peak occupancy is block-invariant for heads 1-3 (identical integers '
            'in all six blocks at every length) and near-invariant for head 0 '
            '(62-64 of 64): the always-enable frozen gate plus shared head '
            'geometry, not learned per-block selectivity, sets occupancy.',
            'At 8x training length the store saturates: heads 2 and 3 are pinned '
            'at their budgets (736/1472) with R-CAP write refusal fractions 0.2812 '
            'and 0.6406 respectively; head 3 refuses 64% of write requests at T8192.',
        ])


def compression(ev, source):
    slopes = ev['slopes_per_block_head']
    defined = [s for row in slopes for s in row if s is not None]
    mean_slope = float(np.mean(defined))
    peaks = {}
    for row in ev['contexts']:
        peaks[str(row['context'])] = [
            max(win[b][h]['peak_live_per_head'][0]
                for win in row['telemetry'] for b in range(6))
            for h in range(4)]
    x39_path = ROOT.parent / 'x39_memory_usage/receipts/part9_semantic_full_evaluation.json'
    x39 = json.loads(x39_path.read_text())
    x39_slopes = x39['occupancy_log_log_slopes_per_head']
    return dict(
        status='completed',
        receipt='six_lane_compression_slopes',
        model='ignition-v1 final weights (in-memory, step 3052, 100007936 tokens)',
        source=source,
        registered_predictions=dict(
            mean_slope=dict(text='Predict mean slope <= .5 (undefined for zero '
                                  'occupancy, no pseudocount).',
                            measured=mean_slope, limit=0.5,
                            verdict='pass' if mean_slope <= 0.5 else 'fail'),
            hierarchy=dict(
                text='Hierarchy should produce unequal head peaks.',
                peaks_per_length=peaks,
                peaks_at_8192=peaks['8192'],
                ordered_by_horizon=peaks['8192'] == sorted(peaks['8192']),
                equal_four_head_peak_stories_per_block=
                    ev['equal_four_head_peak_stories_per_block'],
                verdict='unequal at and past 2048; at T512 heads 1-3 tie at 256')),
        budgets=BUDGETS, horizons=HORIZONS,
        slopes_per_block_head=slopes,
        slopes_block_invariance='heads 1-3 slopes are identical across all six '
            'blocks; head 0 slope is 8.6e-17 in blocks 0-3 and 0.0046/0.0045 in '
            'blocks 4-5 (peak 62-64 of 64 everywhere).',
        capacity_and_pressure=dict(
            peak_over_budget={str(t): [peaks[str(t)][h] / BUDGETS[h] for h in range(4)]
                              for t in LENGTHS},
            rcap_fraction_at_8192=[0.0, 0.0018, 0.2812, 0.6406],
            caveat='Registered caveat: budget-capped occupancy alone is not '
                   'evidence of semantic compression. Heads 2/3 reach their '
                   'budgets and refuse writes (R-CAP) by T2048/T4096; the '
                   'sublinear slopes are substantially capacity-forced.'),
        x39_semantic_reference=dict(
            path=str(x39_path.relative_to(ROOT.parent)), sha256=sha(x39_path),
            single_lane_slopes=x39_slopes,
            single_lane_mean=float(np.mean(x39_slopes)),
            note='X39 semantic arm: one four-head lane, slope mean 0.108. The v1 '
                 'pure six-block stack is deeper-sloped (0.254) but the deep '
                 'lanes are budget-capped, unlike the uncapped semantic arm.'),
        compression_answer='The six-lane stack compresses in the '
            'occupancy-vs-length sense: mean log-log slope 0.2539 (16x tokens -> '
            '5.75x peak on the deepest lane), every lane sublinear, fast lane '
            'flat at its 64-slot budget from T512 on. Ordering follows the '
            'horizon constants past training length: peaks at T8192 are '
            '[64, 256, 736, 1472] = strictly increasing with horizon. This is '
            'capacity+decay ordering, not learned semantic compression; the '
            'always-enable diagnostic removes selective-write evidence.')


def main():
    ev, source = load()
    (REC / 'long_context.json').write_text(
        json.dumps(long_context(ev, source), indent=2) + '\n')
    (REC / 'compression_slopes.json').write_text(
        json.dumps(compression(ev, source), indent=2) + '\n')
    print(json.dumps(dict(status='completed',
        long_context_ppl={str(r['context']): round(r['perplexity'], 4)
                          for r in ev['contexts']},
        mean_slope=round(float(np.mean([s for row in ev['slopes_per_block_head']
                                        for s in row if s is not None])), 4))))


if __name__ == '__main__':
    main()
