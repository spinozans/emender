#!/usr/bin/env python3
"""Verify the window-coverage floor rule directly on a planned schedule.

Independent of the fast key screen: reads the planner's own output
(schedule-proposal / schedule-segmentN-proposal json) and enforces the same
rule as scripts/audit_e97_pi_native_repair_generic.py — cohorts at or above
min_target_fraction of the scheduled assistant targets (majors) must appear
in every update; smaller cohorts (minors) must appear at least once in every
aligned window_updates consecutive updates. Fails closed on any violation.

--window-cohort (repeatable, strict opt-in) names cohorts verified at window
granularity instead of every update. INELIGIBILITY CONDITION (fail-closed,
machine-checked here): a cohort may be declared window-class ONLY if its
average whole record is structurally large relative to the pack sequence —
source_token_tokens / source_record_occurrences >= 10% of (context_size + 1)
— because no-splitting whole-record packing cannot place such a cohort in
most packs. Ordinary small-record cohorts (conversation, fillers, restored
pools, tool-talk) FAIL this check and must keep the every-update rule; this
prevents future preps from using the window class to weaken stratification.
Absent --window-cohort the verifier behaves exactly as before.
"""
import argparse, json
from pathlib import Path

WINDOW_COHORT_MIN_RECORD_FRACTION = 0.10


def verify(args):
    schedule = json.loads(Path(args.schedule).read_text())
    if schedule.get('training_eligible') or schedule.get('optimizer_updates_authorized'):
        raise SystemExit('schedule is not a planning-only proposal')
    totals = schedule['source_target_totals']
    total = sum(totals.values())
    steps = schedule['steps']
    if len(steps) != args.expected_steps:
        raise SystemExit(f'expected {args.expected_steps} updates, found {len(steps)}')
    unknown = [c for c in args.window_cohort if c not in totals]
    if unknown:
        raise SystemExit(f'--window-cohort not in schedule cohorts: {unknown}')
    context = schedule['context_size']
    for c in args.window_cohort:
        avg_record = schedule['source_token_totals'][c] / schedule['source_record_occurrences'][c]
        floor = WINDOW_COHORT_MIN_RECORD_FRACTION * (context + 1)
        if avg_record < floor:
            raise SystemExit(
                f'window-class cohort {c} is NOT structurally capped '
                f'(avg record {avg_record:.0f} tokens < 10% of the {context + 1}-token pack sequence); '
                'it must keep the every-update rule')
    majors = sorted(c for c, v in totals.items()
                    if v / total >= args.min_target_fraction and c not in args.window_cohort)
    minors = sorted(c for c in totals if c not in majors and c not in args.window_cohort)
    windowed = sorted(args.window_cohort)
    per_update = [set(s['source_targets']) for s in steps]
    for c in majors:
        missing = [i + 1 for i, u in enumerate(per_update) if c not in u]
        if missing:
            raise SystemExit(f'major cohort {c} missing from updates {missing[:5]}')
    for c in minors + windowed:
        for start in range(0, len(per_update), args.window_updates):
            if not any(c in u for u in per_update[start:start + args.window_updates]):
                raise SystemExit(f'cohort {c} absent from window at update {start + 1}')
    print('WINDOW_RULE_OK', Path(args.schedule).name,
          'updates', len(steps),
          'majors', ','.join(majors),
          'minors', ','.join(minors),
          'window_cohorts', ','.join(windowed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schedule', required=True)
    p.add_argument('--expected-steps', type=int, default=128)
    p.add_argument('--window-updates', type=int, default=32)
    p.add_argument('--min-target-fraction', type=float, default=0.02)
    p.add_argument('--window-cohort', action='append', default=[],
                   help='whole-record document cohort verified at window granularity instead of every update (opt-in; structurally capped cohorts only)')
    verify(p.parse_args())


if __name__ == '__main__':
    main()
