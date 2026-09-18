#!/usr/bin/env python3
"""T1 dir-listing drop adjudication (plan: e97-oh-translation...-v1.md, T1 step 3).

Re-verifies the DROPPED candidates from a completed replay pass with per-record
instrumentation. For every dir_listing_mismatch it captures the evidence needed
to distinguish a rendering-normalization artifact from a genuine path-set
divergence at the same trajectory point:

  - the mapped find command and both path sets (recorded vs executed)
  - for each recorded-missing-from-executed path: is it hidden ('/.' segment)?
    is it deeper than the find's maxdepth? does it exist in the replay tree?
  - the executed listing's own header/shape

Classification per record:
  normalization_artifact : every missing path is hidden or deeper than the
                           mapped find's semantics (OH view rendered a superset
                           the mapped find cannot emit)
  genuine_discrepancy    : at least one missing path is visible, within depth,
                           and exists in the replay tree at that trajectory
                           point — a real divergence; the drop stands.

Nothing is re-admitted here; this tool only produces evidence. Re-admission
happens by re-running the standard fail-closed verifier with the documented
normalization, so the final-state gate still applies to every re-admitted
record.
"""
import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    'verify_e97', _HERE / 'verify_e97_oh_translation_replay.py')
verify_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_mod)


def classify_missing(missing, find_maxdepth=2):
    kinds = []
    for rp in missing:
        rel = rp[len('/workspace/'):]
        hidden = any(part.startswith('.') for part in rel.split('/'))
        # depth relative to the instance dir (find root)
        depth = len(rel.strip('/').split('/')) - 1  # instance dir itself = depth 0
        too_deep = depth > find_maxdepth
        kinds.append({'path': rp, 'hidden': hidden, 'deeper_than_find': too_deep})
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--translation-dir', type=Path, required=True)
    ap.add_argument('--verified-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0,
                    help='adjudicate at most N dropped records (0 = all)')
    args = ap.parse_args()

    candidates = sorted((args.translation_dir / 'candidates').glob('*.json'))
    passed_ids = {p.stem for p in args.verified_dir.glob('*.json')}
    dropped = [c for c in candidates if c.stem not in passed_ids]
    if args.limit:
        dropped = dropped[:args.limit]
    print(f'candidates={len(candidates)} passed={len(passed_ids)} '
          f'dropped_to_adjudicate={len(dropped)}', flush=True)

    ctx = verify_mod.build_context(args) if hasattr(verify_mod, 'build_context') else None
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = collections.Counter()
    fh = out.open('w')

    def one(cpath):
        cand = json.loads(cpath.read_text())
        text, reason = verify_mod.replay_candidate_for_debug(
            cand, args.translation_dir)
        if text is not None:
            reason = None  # success: verify_record returns a stats dict here
        row = {'trajectory_id': cand['trajectory_id'],
               'instance_id': cand.get('instance_id'),
               'fail_reason': reason,
               'replayed_ok': text is not None}
        if reason and reason.startswith('dir_listing_mismatch'):
            detail = verify_mod.last_dir_listing_debug()
            row['dir_listing'] = detail
            missing = detail.get('missing_paths') or []
            kinds = classify_missing(missing)
            row['dir_listing']['missing_kinds'] = kinds
            if missing and all(k['hidden'] or k['deeper_than_find'] for k in kinds):
                row['classification'] = 'normalization_artifact'
            elif missing:
                row['classification'] = 'genuine_discrepancy'
            else:
                row['classification'] = 'no_missing_captured'
        else:
            row['classification'] = 'other_fail_reason'
        return row

    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, c) for c in dropped]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 50 == 0:
                print(f'adjudicated {i + 1}/{len(dropped)}', flush=True)
    for row in sorted(rows, key=lambda r: r['trajectory_id']):
        summary[row['classification']] += 1
        fh.write(json.dumps(row, sort_keys=True) + '\n')
    print('ADJUDICATION_SUMMARY', json.dumps(dict(summary), sort_keys=True), flush=True)
    fh.close()


if __name__ == '__main__':
    main()
