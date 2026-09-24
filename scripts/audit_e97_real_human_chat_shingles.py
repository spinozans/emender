#!/usr/bin/env python3
"""Shingle-exclusion audit of the real-human-chat authority against another
admitted masked-SFT authority (SmolTalk2 admitted, Tulu 3, or the pile-derived
long-document anchors).

Proves (or disproves and reports) that the admitted real-chat records share no
long token span with the comparison authority, so the real-human register adds
new signal instead of duplicating conversation-rehearsal or document-anchor
training content. The span bar follows the repository's established
identity+shingle exclusion policy (commapile document-causal precedent:
5 normalized lines >= 160 chars): WIDTH consecutive tokens at ~4 chars/token.
Windows never cross record boundaries on either side; hash hits are verified
byte-exact before counting, so reported collisions are content, not hash
accidents. Unlike the hybrid-collection variant this audit runs in report
mode: a content collision lists the candidate record identities so the
intake build can exclude exactly those records and rebuild, and exits
non-zero (fail closed) until a rebuild achieves zero collisions.
"""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np

RECORD_INDEX = struct.Struct('<QQQB7x')
WIDTH = 40                       # ~160-char span bar of the identity+shingle policy
MULT = np.uint64(0x9E3779B97F4A7C15)   # golden-ratio odd multiplier; uint64 wraparound
CHUNK = 1 << 25                  # 32M-token streaming windows over the large corpus
SCHEMA = 'emender-e97-real-human-chat-shingle-audit-v1'
MAX_REPORT = 100000              # collision identities reported in the receipt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_authority(root):
    """Verify the four payload outputs against the authority manifest and return
    (manifest, metadata rows, starts, lengths, tokens memmap)."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    paths = {}
    for key in ('tokens', 'mask', 'index', 'metadata'):
        descriptor = manifest['outputs'][key]
        raw = Path(descriptor['path'])
        path = raw if raw.is_absolute() else root / raw
        if path.stat().st_size != descriptor['bytes'] or sha(path) != descriptor['sha256']:
            raise ValueError(f'authority output identity: {key}')
        paths[key] = path
    index = Path(paths['index']).read_bytes()
    metadata = Path(paths['metadata']).read_text().splitlines()
    if len(index) != RECORD_INDEX.size * len(metadata):
        raise ValueError('authority shape')
    records = [RECORD_INDEX.unpack_from(index, i * RECORD_INDEX.size) for i in range(len(metadata))]
    starts = np.array([r[0] for r in records], dtype=np.uint64)
    lengths = np.array([r[1] for r in records], dtype=np.uint64)
    tokens = np.memmap(paths['tokens'], dtype='<u4', mode='r')
    if len(tokens) != sum(lengths):
        raise ValueError('authority token count')
    return manifest, metadata, starts, lengths, tokens



def audit(args):
    ours_manifest, ours_meta, ours_starts, ours_lengths, ours_tokens = load_authority(args.candidate)
    theirs_manifest, theirs_meta, theirs_starts, theirs_lengths, theirs_tokens = load_authority(args.against)
    if ours_manifest.get('training_eligible'):
        raise ValueError('candidate side must not claim training eligibility at audit time')
    # our windows: flat chunked hashing with boundary masking (in-record only);
    # in-corpus duplicate windows (real corpora repeat common phrasings)
    # deduplicate for membership; byte-exact verification resolves any hash
    # hit to real content. window -> record index comes from the ends search.
    ours_ends = np.cumsum(ours_lengths, dtype=np.uint64)
    our_total = len(ours_tokens)
    our_hashes = []
    our_starts_arr = []
    our_windows_total = 0
    for c0 in range(0, our_total, CHUNK):
        c1 = min(c0 + CHUNK, our_total)
        span = c1 - c0
        if span < WIDTH:
            break
        count = span - WIDTH + 1
        hashes = np.zeros(count, dtype=np.uint64)
        for k in range(WIDTH):
            hashes = hashes * MULT + np.asarray(ours_tokens[c0 + k:c0 + k + count], dtype=np.uint64)
        starts = np.arange(c0, c0 + count, dtype=np.uint64)
        before = np.searchsorted(ours_ends, starts, side='right')
        inside = np.searchsorted(ours_ends, starts + WIDTH, side='left')
        valid = before == inside
        our_windows_total += int(valid.sum())
        our_hashes.append(hashes[valid])
        our_starts_arr.append(starts[valid])
    our_hashes = np.concatenate(our_hashes) if our_hashes else np.zeros(0, dtype=np.uint64)
    our_starts_all = np.concatenate(our_starts_arr) if our_starts_arr else np.zeros(0, dtype=np.uint64)
    order = np.argsort(our_hashes)
    our_hashes = our_hashes[order]
    our_starts_all = our_starts_all[order]
    # keep ALL (hash, start) pairs: in-corpus duplicate windows (real corpora
    # repeat common phrasings) must all be attributable, so the report lists
    # EVERY record owning a colliding window, not just one representative
    # owner per unique hash — otherwise exclusion-based rebuilds leave the
    # duplicate-window siblings in the corpus and the re-audit resurfaces.
    ours_window_count = len(np.unique(our_hashes))
    collisions = {}
    hash_only = 0
    theirs_window_count = 0
    total = len(theirs_tokens)
    ends = np.cumsum(theirs_lengths, dtype=np.uint64)
    for c0 in range(0, total, CHUNK):
        c1 = min(c0 + CHUNK, total)
        span = c1 - c0
        if span < WIDTH:
            break
        count = span - WIDTH + 1
        hashes = np.zeros(count, dtype=np.uint64)
        for k in range(WIDTH):
            hashes = hashes * MULT + np.asarray(theirs_tokens[c0 + k:c0 + k + count], dtype=np.uint64)
        starts = np.arange(c0, c0 + count, dtype=np.uint64)
        before = np.searchsorted(ends, starts, side='right')
        inside = np.searchsorted(ends, starts + WIDTH, side='left')
        valid = before == inside
        hashes = hashes[valid]
        starts = starts[valid]
        theirs_window_count += len(hashes)
        if not len(hashes):
            continue
        where = np.searchsorted(our_hashes, hashes)
        where = np.clip(where, 0, len(our_hashes) - 1)
        hit = our_hashes[where] == hashes
        for h_index, t_start in zip(np.nonzero(hit)[0], starts[hit]):
            theirs_window = np.asarray(theirs_tokens[int(t_start):int(t_start) + WIDTH], dtype=np.uint64)
            # enumerate EVERY our-side window with this hash (duplicates kept)
            lo = int(np.searchsorted(our_hashes, hashes[h_index], side='left'))
            hi = int(np.searchsorted(our_hashes, hashes[h_index], side='right'))
            matched = False
            for ours_start in our_starts_all[lo:hi]:
                ours_start = int(ours_start)
                record = int(np.searchsorted(ours_ends, np.uint64(ours_start), side='right'))
                ours_window = np.asarray(ours_tokens[ours_start:ours_start + WIDTH], dtype=np.uint64)
                if np.array_equal(theirs_window, ours_window):
                    matched = True
                    collisions[record] = collisions.get(record, 0) + 1
            if not matched:
                hash_only += 1
    identities = sorted(collisions)
    receipt = {'schema': SCHEMA,
        'status': 'pass' if not collisions else 'collisions',
        'shingle_policy': 'width-40-token in-record windows (~160-char span bar of the commapile identity+shingle policy); hash hits verified byte-exact; windows never cross record boundaries',
        'shingle_width_tokens': WIDTH,
        'our_records': len(ours_meta), 'our_windows': ours_window_count, 'our_windows_all': our_windows_total,
        'their_records': len(theirs_meta), 'their_windows': theirs_window_count, 'their_tokens': total,
        'content_collisions': len(collisions),
        'collision_windows_total': int(sum(collisions.values())),
        'collision_candidate_records': [json.loads(ours_meta[i])['identity'] for i in identities[:MAX_REPORT]],
        'hash_only_collisions': hash_only,
        'our_authority_sha256': sha(Path(args.candidate) / 'manifest.json'),
        'their_manifest_sha256': sha(Path(args.against) / 'manifest.json'),
        'their_dataset_id': theirs_manifest.get('dataset_id'),
        'their_purpose': theirs_manifest.get('purpose'),
        'checker_sha256': sha(__file__), 'training_eligible': False,
        'packing_authorized': False, 'optimizer_updates_authorized': 0}
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('REAL_HUMAN_CHAT_SHINGLE_AUDIT', receipt['status'],
          len(collisions), 'records', int(sum(collisions.values())), 'windows', sha(args.output))
    if collisions:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True,
                   help='real-human-chat authority root (manifest.json + payloads)')
    p.add_argument('--against', type=Path, required=True,
                   help='comparison admitted authority root (manifest.json)')
    p.add_argument('--output', type=Path, required=True)
    audit(p.parse_args())


if __name__ == '__main__':
    main()
