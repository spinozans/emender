#!/usr/bin/env python3
"""Fast in-process probe-key screen (STAGED machinery for the post-v10 LR screen).

Replicates the epoch-permutation pack selection of
scripts/plan_e97_pi_native_training_schedule.py (same sha256-derived affine
permutation over pack ids) and screens candidate sampler keys against the
window-coverage floor rule for the exposure-matched 32-update probe schedule:
majors (>=2% of the prep's assistant target tokens) in EVERY update, minors at
least once per 32-update window. Any key found here is re-verified with the
real planner before use.
"""
import argparse, hashlib, json, math, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plan_e97_pi_native_training_schedule import PACK, permutation  # noqa: E402

PACKS_DEFAULT = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/'
                     'e97-extension-preparation-v1/packs/manifest.json')
PREP_DEFAULT = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-extension-preparation-v1')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def screen(args):
    prep = json.loads((args.preparation / 'manifest.json').read_text())
    packs = json.loads(args.packs.read_text())
    totals = prep['source_target_totals']
    total = sum(totals.values())
    majors = [c for c, v in totals.items() if v / total >= 0.02]
    minors = [c for c in totals if c not in majors]
    rows = (args.preparation / 'records.jsonl').open()
    sources = [json.loads(line)['source'] for line in rows]
    pack_rows = (args.preparation / 'packs/train_packs.idx').read_bytes()
    members = (args.preparation / 'packs/pack_records.uint32.bin').read_bytes()
    pack_count = len(pack_rows) // PACK.size
    member_ids = struct.unpack('<%dI' % (len(members) // 4), members)
    flags = {c: [False] * pack_count for c in totals}
    for pid in range(pack_count):
        first, count = PACK.unpack_from(pack_rows, pid * PACK.size)[:2]
        for rid in member_ids[first:first + count]:
            flags[sources[rid]][pid] = True
    base_meta = {'authority_manifest_sha256': hashlib.sha256((args.preparation / 'manifest.json').read_bytes()).hexdigest(),
                 'pack_manifest_sha256': hashlib.sha256(args.packs.read_bytes()).hexdigest(),
                 'data_world_size': 8, 'context_size': 65536, 'split': 'train',
                 'schema': 'emender-record-pack-counter-v1', 'sampler_mode': 'epoch-permutation'}
    positions = list(range(32 * 8))
    tried = 0
    for key in range(args.start, args.start + args.count):
        tried += 1
        meta = {**base_meta, 'sampler_key': key}
        multiplier, offset = permutation(meta, 'epoch-permutation', 0, pack_count)
        packs_hit = [(multiplier * p + offset) % pack_count for p in positions]
        ok = True
        for cursor in range(32):
            window = packs_hit[cursor * 8:(cursor + 1) * 8]
            for c in majors:
                if not any(flags[c][pid] for pid in window):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for c in minors:
                if not any(flags[c][pid] for pid in packs_hit):
                    ok = False
                    break
        if ok:
            print('PROBE_KEY_OK', key, 'after', tried, 'keys')
            return 0
    print('NO_KEY_FOUND', tried)
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preparation', type=Path, default=PREP_DEFAULT)
    p.add_argument('--packs', type=Path, default=PACKS_DEFAULT)
    p.add_argument('--start', type=int, default=1400301)
    p.add_argument('--count', type=int, default=1000000)
    raise SystemExit(screen(p.parse_args()))


if __name__ == '__main__':
    main()
