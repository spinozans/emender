"""Bind selected document receipts to read-only pile.txt offsets; never copy text."""
import hashlib
import json
import mmap
import time
from pathlib import Path

from .c14_documents import sha

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT / 'receipts/c14_dense'
SOURCE = Path('/mnt/nvme2n1/erikg/pile.txt')
OFFSETS = Path('/home/erikg/c14_dense_bytes/pile_offsets.jsonl')


def main():
    binding = json.loads((ROOT / 'receipts/c14_byte_source.json').read_text())
    index = REPO / binding['documents_index']
    assert sha(index) == binding['documents_index_sha256']
    prior = json.loads((OUT / 'source_preflight.json').read_text())
    lengths_path = Path('/home/erikg/c14_dense_bytes/offsets.jsonl')
    assert sha(lengths_path) == prior['offsets_sha256']
    initial = SOURCE.stat()
    matched = skipped = excluded = 0
    samples = []
    first_skip = None
    started = time.monotonic()
    position = 0
    with index.open() as receipts, lengths_path.open() as lengths, SOURCE.open('rb') as source, OFFSETS.open('w') as offsets:
        wanted = json.loads(next(receipts))
        length_row = json.loads(next(lengths))
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            while position < len(raw):
                end = raw.find(b'\x1e', position)
                if end < 0:
                    end = len(raw)
                # Prior hash-bound metadata supplies only the capped byte length.
                # Text comes exclusively from pile.txt; its full digest must match.
                length = end-position if wanted['truncated'] else length_row['byte_length']
                if not wanted['truncated']:
                    assert length_row['sha256'] == wanted['sha256']
                digest = hashlib.sha256(raw[position:position+length]).hexdigest()
                if digest == wanted['sha256']:
                    entry = dict(**wanted, byte_offset=position, byte_length=length)
                    if matched in (0, 3915606, 7831212):
                        samples.append(dict(ordinal=matched, **entry))
                    matched += 1
                    if wanted['truncated']:
                        excluded += 1
                    else:
                        offsets.write(json.dumps(entry)+'\n')
                        length_row = json.loads(next(lengths, 'null'))
                    end = raw.find(b'\x1e', position+length)
                    if end < 0:
                        end = len(raw)
                    line = next(receipts, None)
                    if line is None:
                        wanted = None
                        position = end
                        break
                    wanted = json.loads(line)
                else:
                    skipped += 1
                    if first_skip is None:
                        first_skip = dict(byte_offset=position, byte_length=end-position,
                                          actual_sha256=hashlib.sha256(raw[position:end]).hexdigest(), next_expected=wanted)
                position = end + 1
                if (matched + skipped) % 500000 == 0:
                    print(json.dumps(dict(matched=matched, skipped=skipped, position=position)), flush=True)
    final = SOURCE.stat()
    unchanged = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns) == (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
    result = dict(status='passed' if wanted is None and unchanged else 'failed',
                  source=str(SOURCE), source_size=initial.st_size, source_mtime_ns=initial.st_mtime_ns,
                  source_inode=initial.st_ino, source_unchanged=unchanged,
                  documents_index=str(index.relative_to(REPO)), documents_index_sha256=binding['documents_index_sha256'],
                  matched_documents=matched, excluded_token_truncated=excluded,
                  skipped_raw_documents=skipped, first_skipped_document=first_skip,
                  span_start=0, span_end_exclusive=position, samples=samples,
                  selected_documents_are_contiguous=skipped == 0,
                  offsets_path=str(OFFSETS), offsets_sha256=sha(OFFSETS),
                  length_metadata_path=str(lengths_path), length_metadata_sha256=prior['offsets_sha256'],
                  next_missing_receipt=wanted, materialized_bytes=0, training_started=False,
                  delimiter='0x1e; delimiter excluded from document digest and windows',
                  elapsed_seconds=time.monotonic()-started)
    (OUT/'pile_span.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
