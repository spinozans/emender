#!/usr/bin/env python3
"""Verify the window-coverage floor rule directly on a planned schedule.

Independent of the fast key screen: reads the planner's own output
(schedule-proposal / schedule-segmentN-proposal json) and enforces the same
rule as scripts/audit_e97_pi_native_repair_generic.py — cohorts at or above
min_target_fraction of the scheduled assistant targets (majors) must appear
in every update; smaller cohorts (minors) must appear at least once in every
aligned window_updates consecutive updates. Fails closed on any violation.
"""
import argparse, json
from pathlib import Path


def verify(args):
    schedule = json.loads(Path(args.schedule).read_text())
    if schedule.get('training_eligible') or schedule.get('optimizer_updates_authorized'):
        raise SystemExit('schedule is not a planning-only proposal')
    totals = schedule['source_target_totals']
    total = sum(totals.values())
    steps = schedule['steps']
    if len(steps) != args.expected_steps:
        raise SystemExit(f'expected {args.expected_steps} updates, found {len(steps)}')
    majors = sorted(c for c, v in totals.items() if v / total >= args.min_target_fraction)
    minors = sorted(c for c in totals if c not in majors)
    per_update = [set(s['source_targets']) for s in steps]
    for c in majors:
        missing = [i + 1 for i, u in enumerate(per_update) if c not in u]
        if missing:
            raise SystemExit(f'major cohort {c} missing from updates {missing[:5]}')
    for c in minors:
        for start in range(0, len(per_update), args.window_updates):
            if not any(c in u for u in per_update[start:start + args.window_updates]):
                raise SystemExit(f'minor cohort {c} absent from window at update {start + 1}')
    print('WINDOW_RULE_OK', Path(args.schedule).name,
          'updates', len(steps),
          'majors', ','.join(majors),
          'minors', ','.join(minors))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schedule', required=True)
    p.add_argument('--expected-steps', type=int, default=128)
    p.add_argument('--window-updates', type=int, default=32)
    p.add_argument('--min-target-fraction', type=float, default=0.02)
    verify(p.parse_args())


if __name__ == '__main__':
    main()
