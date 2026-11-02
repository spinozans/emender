"""C14 raw-byte windows. A window and every shifted target belong to ONE document.

No concatenated-stream API is exposed. State must reset between returned windows.
The receipt's document digest is checked before a natural-text document is admitted.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_documents(path):
    import zstandard
    with Path(path).open('rb') as f:
        with zstandard.ZstdDecompressor().stream_reader(f) as stream:
            for line in io.TextIOWrapper(stream, encoding='utf8'):
                yield json.loads(line)['text'][:1_000_000].encode('utf8')


@dataclass(frozen=True)
class Window:
    document: int
    split: str
    start: int
    inputs: bytes
    targets: bytes


class DocumentWindows:
    """Nonoverlapping, source-ordered, truncated windows; never pad with another doc.

    Short final windows are separate batches (or collated with masked padding).
    Every returned target is real text. Empty/single-byte docs have no targets.
    Validation/calibration/test membership is by document, never by byte seam.
    """
    def __init__(self, context=128):
        if not isinstance(context, int) or context < 1:
            raise ValueError('context must be a positive integer')
        self.context = context

    def windows(self, raw, receipt, split=None):
        if not isinstance(raw, bytes):
            raise TypeError('natural text must be UTF-8 bytes')
        if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('raw document does not match documents.jsonl')
        if receipt.get('truncated'):
            # Token-budget truncation is not a raw-byte boundary. Never guess it.
            raise ValueError('token-truncated document requires an explicit byte receipt')
        if receipt['split'] not in ('train', 'valid'):
            raise ValueError('unknown split')
        if split is not None and split != receipt['split']:
            raise ValueError('split mismatch')
        for start in range(0, len(raw)-1, self.context):
            end = min(start+self.context, len(raw)-1)
            yield Window(receipt['source_index'], receipt['split'], start,
                         raw[start:end], raw[start+1:end+1])

    @staticmethod
    def causal_context(raw, target, reset_start, cap=1024):
        """Only preceding bytes in this window/document; no target enters lookup."""
        if not 0 <= reset_start < target < len(raw) or cap < 1:
            raise ValueError('invalid causal window')
        return raw[max(reset_start, target-cap):target]



def evaluation_halves(valid_document_ids):
    """Freeze a document-disjoint calibration/test assignment before inference."""
    ids = list(valid_document_ids)
    if len(ids) < 2 or any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError('need unique valid documents in strict source order')
    half = len(ids)//2
    return {i: 'calibration' if j < half else 'final_test'
            for j, i in enumerate(ids)}


def attest_prefix(source, receipts, limit=4096):
    """Locate natural source by SHA matching in order; never assume source row parity.

    Original Pile includes excluded documents; receipt source_index belongs to the
    uncopyrighted stream. Match document hashes, preserving the attested order.
    This bounded discovery receipt does NOT attest a complete corpus build.
    """
    sampler = DocumentWindows()
    matched = targets = windows = scanned = 0
    sample_rows = []
    with Path(receipts).open() as f:
        wanted = json.loads(next(f))
        for raw in raw_documents(source):
            scanned += 1
            if hashlib.sha256(raw).hexdigest() != wanted['sha256']:
                continue
            ws = list(sampler.windows(raw, wanted))
            for w in ws:
                assert w.inputs == raw[w.start:w.start+len(w.inputs)]
                assert w.targets == raw[w.start+1:w.start+1+len(w.targets)]
                assert w.start+len(w.targets) < len(raw)
            targets += sum(len(w.targets) for w in ws)
            windows += len(ws)
            matched += 1
            if len(sample_rows) < 8:
                sample_rows.append(dict(source_index=wanted['source_index'],
                                        sha256=wanted['sha256'], bytes=len(raw), split=wanted['split']))
            if matched == limit:
                break
            wanted = json.loads(next(f))
    if matched != limit:
        raise ValueError(f'only {matched}/{limit} documents located in source')
    return dict(status='passed', scope='bounded natural-source discovery; not full corpus attestation',
                raw_source=str(source), raw_source_sha256=sha(source),
                documents_index=str(receipts), documents_index_sha256=sha(receipts),
                matched_documents=matched, scanned_raw_documents=scanned,
                windows=windows, targets=targets, cross_document_windows=0,
                context=128, examples=sample_rows)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--documents', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=4096)
    a = p.parse_args()
    result = attest_prefix(a.source, a.documents, a.limit)
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
