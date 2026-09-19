#!/usr/bin/env python3
"""Fast in-process segment-key screen for the repair-v3 (scrub-reversal) full arc.

Replicates the epoch-permutation pack selection of
scripts/plan_e97_pi_native_training_schedule.py (same sha256-derived affine
permutation over pack ids) and screens candidate sampler keys for 128-update
segment schedules under the window-coverage floor rule enforced by
scripts/audit_e97_pi_native_repair_generic.py: cohorts at or above
--min-target-fraction of the prep's assistant target tokens (majors) must
appear in EVERY update (each update = world_size consecutive packs); smaller
cohorts (minors) must appear at least once in every aligned --window-updates
window. Admits --keys keys, in screened order, one per 128-update segment.

Any key admitted here must be re-verified with the real planner before use
(precedent: the v2 arc and extension probe screens).
"""
import argparse, hashlib, json, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plan_e97_pi_native_training_schedule import PACK, permutation  # noqa: E402

PREP_DEFAULT = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-full-preparation-v3')
PACKS_DEFAULT = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-full-preparation-v3/packs/manifest.json')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def screen(args):
    prep_manifest_bytes = (args.preparation / 'manifest.json').read_bytes()
    prep = json.loads(prep_manifest_bytes)
    packs_manifest_bytes = args.packs.read_bytes()
    packs = json.loads(packs_manifest_bytes)
    totals = prep['source_target_totals']
    total = sum(totals.values())
    majors = [c for c, v in totals.items() if v / total >= args.min_target_fraction]
    minors = [c for c in totals if c not in majors]
    sources = [json.loads(line)['source'] for line in (args.preparation / 'records.jsonl').open()]
    pack_rows = (args.preparation / 'packs/train_packs.idx').read_bytes()
    members = (args.preparation / 'packs/pack_records.uint32.bin').read_bytes()
    pack_count = len(pack_rows) // PACK.size
    member_ids = struct.unpack('<%dI' % (len(members) // 4), members)
    flags = {c: [False] * pack_count for c in totals}
    for pid in range(pack_count):
        first, count = PACK.unpack_from(pack_rows, pid * PACK.size)[:2]
        for rid in member_ids[first:first + count]:
            flags[sources[rid]][pid] = True
    base_meta = {'authority_manifest_sha256': hashlib.sha256(prep_manifest_bytes).hexdigest(),
                 'pack_manifest_sha256': hashlib.sha256(packs_manifest_bytes).hexdigest(),
                 'data_world_size': args.world_size, 'context_size': 65536, 'split': 'train',
                 'schema': 'emender-record-pack-counter-v1', 'sampler_mode': 'epoch-permutation'}
    positions = list(range(args.steps * args.world_size))
    admitted = []
    tried = 0
    for key in range(args.start, args.start + args.count):
        if len(admitted) >= args.keys:
            break
        tried += 1
        meta = {**base_meta, 'sampler_key': key}
        multiplier, offset = permutation(meta, 'epoch-permutation', 0, pack_count)
        packs_hit = [(multiplier * p + offset) % pack_count for p in positions]
        ok = True
        for cursor in range(args.steps):
            window = packs_hit[cursor * args.world_size:(cursor + 1) * args.world_size]
            for c in majors:
                if not any(flags[c][pid] for pid in window):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for start in range(0, args.steps, args.window_updates):
                window = packs_hit[start * args.world_size:(start + args.window_updates) * args.world_size]
                for c in minors:
                    if not any(flags[c][pid] for pid in window):
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            admitted.append(key)
            print('SEGMENT_KEY_OK', key, 'admitted', len(admitted), 'after', tried, 'keys')
    print('majors', ','.join(sorted(majors)))
    print('minors', ','.join(sorted(minors)))
    print('SCREEN_DONE', 'admitted', len(admitted), 'of', args.keys, 'tried', tried)
    return 0 if len(admitted) == args.keys else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preparation', type=Path, default=PREP_DEFAULT)
    p.add_argument('--packs', type=Path, default=PACKS_DEFAULT)
    p.add_argument('--start', type=int, default=1401002)
    p.add_argument('--count', type=int, default=1000000)
    p.add_argument('--keys', type=int, default=8)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--world-size', type=int, default=8)
    p.add_argument('--window-updates', type=int, default=32)
    p.add_argument('--min-target-fraction', type=float, default=0.02)
    raise SystemExit(screen(p.parse_args()))


if __name__ == '__main__':
    main()
