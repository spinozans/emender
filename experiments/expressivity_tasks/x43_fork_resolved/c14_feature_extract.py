"""Fail-closed raw-byte C14 feature bridge; never reads a target for a lookup.

CLI transport is exact for records without byte 30, and argv without NUL.
Patterns requiring both are rejected (no lossy FASTA workaround). This is an
explicit partial transport, not the requested universal length-delimited API.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

XSA = Path.home() / 'sxgc/xsa/target/release/xsa'
INDEX = Path('/mnt/nvme2n1/erikg/wf-finish-as-100gb-20261102T044713Z/corpus.sxi2')
N = 100093934467


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


class CountBridge:
    def __init__(self, output, xsa=XSA, index=INDEX):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.xsa, self.index = Path(xsa), Path(index)
        self.timings = []

    def query(self, patterns, name):
        if not patterns:
            return []
        if any(not p or len(p) > 65536 for p in patterns):
            raise ValueError('query must contain 1..65536 raw bytes')
        bad = [p.hex() for p in patterns if b'\x1e' in p and b'\0' in p]
        if bad:
            (self.output / (name + '.transport_failure.json')).write_text(json.dumps(bad))
            raise ValueError('unencodable NUL+30 pattern; native length-delimited API required')
        result = [None] * len(patterns)
        groups = [(0, [(i,p) for i,p in enumerate(patterns) if b'\x1e' not in p]),
                  (1, [(i,p) for i,p in enumerate(patterns) if b'\x1e' in p and not p.startswith(b'-')]),
                  (2, [(i,p) for i,p in enumerate(patterns) if b'\x1e' in p and p.startswith(b'-')])]
        for mode, group in groups:
            positional = mode == 1
            # argv batches are bounded below Linux per-argument and total limits.
            batches = []
            batch = []; used = 0
            for item in group:
                cost = len(item[1])+16
                if positional and batch and used+cost > 60000:
                    batches.append(batch); batch = []; used = 0
                batch.append(item); used += cost
            if batch: batches.append(batch)
            for start, batch in enumerate(batches):
                tag = f'{name}_{mode}_{start}'
                cmd = [os.fsencode(self.xsa), b'query', os.fsencode(self.index), b'--count', b'-j', b'1']
                if positional:
                    # Prefix avoids CLI option interpretation for a pattern beginning '-'.
                    # Such a pattern cannot be escaped by this CLI; reject explicitly.
                    if any(p.startswith(b'-') for _, p in batch):
                        raise ValueError('byte-30 argv pattern begins with CLI option marker')
                    cmd += [p for _, p in batch]
                else:
                    path = self.output / (tag + '.patterns')
                    if mode == 2:
                        if any(b'\n' in p or p.endswith(b'\r') for _,p in batch):
                            raise ValueError('dash-leading byte-30 pattern cannot be represented exactly')
                        path.write_bytes(b''.join(b'>p\n'+p+b'\n' for _,p in batch))
                        cmd += [b'--patterns-fasta', os.fsencode(path)]
                    else:
                        path.write_bytes(b'\x1e'.join(p for _, p in batch) + b'\x1e')
                        cmd += [b'--patterns', os.fsencode(path)]
                # Exact inputs survive any assertion/subprocess failure.
                (self.output / (tag + '.requests.json')).write_text(json.dumps([p.hex() for _, p in batch]))
                begin = time.monotonic()
                with (self.output / (tag + '.stdout')).open('wb') as out, (self.output / (tag + '.stderr')).open('wb') as err:
                    proc = subprocess.run(cmd, stdout=out, stderr=err)
                wall = time.monotonic() - begin
                journal = (self.output / (tag + '.stderr')).read_text(errors='replace')
                timing = dict(name=tag, patterns=len(batch), wall_seconds=wall, returncode=proc.returncode,
                              query_ms=re.findall(r'\bquery_ms=([\d.]+)', journal))
                self.timings.append(timing)
                (self.output / 'timings.json').write_text(json.dumps(self.timings, indent=2))
                if proc.returncode or 'QUERY_COUNT_IMPOSSIBLE' in journal:
                    raise RuntimeError(f'{tag}: query failed; exact patterns and journal retained')
                lines = (self.output / (tag + '.stdout')).read_bytes().split(b'\n')
                if lines[-1] == b'': lines.pop()
                if len(lines) != len(batch): raise ValueError('count response cardinality mismatch')
                for pid, (line, (original, _)) in enumerate(zip(lines, batch)):
                    if int(line.split(b'\t', 1)[0]) != pid: raise ValueError('response ID mismatch')
                    count = int(line.rsplit(b'\t', 1)[1])
                    if not 0 <= count <= N: raise ValueError('impossible count; extraction rejected')
                    result[original] = count
        return result


def extract(lefts, query):
    """Repeated exact suffix searches: prepend one byte, preserving causal order.

    The CLI cannot preserve an FM interval. Batch all suffixes to amortize open;
    account for their full cost. No target/future bytes enter this function.
    """
    suffixes = list(dict.fromkeys(s[-k:] for s in lefts for k in range(1, len(s)+1)))
    counts = dict(zip(suffixes, query(suffixes, 'suffixes')))
    lengths = [max((k for k in range(1, len(s)+1) if counts[s[-k:]]), default=0) for s in lefts]
    contexts = list(dict.fromkeys(s[-k:] for s, k in zip(lefts, lengths) if k))
    branches = [s + bytes([b]) for s in contexts for b in range(256)]
    values = query(branches, 'branches')
    rows = {}
    for j, s in enumerate(contexts):
        row = values[j*256:(j+1)*256]
        raw_total = sum(row)
        if raw_total > counts[s]: raise ValueError('branch sum exceeds parent interval; stop')
        # Wikipedia byte 30 is the document delimiter: measure it, then
        # treat it as terminal mass, never a natural text continuation.
        total = raw_total-row[30]
        continuation = [v if b != 30 else 0 for b,v in enumerate(row)]
        probs = [v/total if total else 0.0 for v in continuation]
        entropy = -sum(p*math.log(p) for p in probs if p)
        rows[s] = dict(counts=row, context_count=counts[s], support_size=total, terminal_count=row[30],
                       distinct_branches=sum(v > 0 for v in continuation), entropy=entropy,
                       coverage=bool(total), context_bytes=len(s), censored=False,
                       features=probs+[math.log1p(total), entropy, float(sum(v > 0 for v in continuation)>1), math.log1p(len(s))] if total else [0.0]*260)
    result = []
    for s, k in zip(lefts, lengths):
        row = dict(rows[s[-k:]]) if k else dict(counts=[0]*256, context_count=0, support_size=0,
            distinct_branches=0, terminal_count=0, entropy=0., coverage=False, context_bytes=0, features=[0.]*260)
        row['censored'] = bool(k and k == len(s))
        result.append(row)
    return result, {str(k): dict(eligible=sum(len(s)>=k for s in lefts),
        supported=sum(len(s)>=k and counts[s[-k:]]>0 for s in lefts)) for k in (16,64,256,1024)}


def slice_contexts(span_receipt, documents, windows=16):
    """Reuse externally located pile.txt spans; perform no span discovery.

    Receipt schema: pile_path, documents_index_sha256, documents:[{source_index,
    byte_offset, byte_length, sha256, split}]. Offsets are absolute pile.txt bytes.
    Hash-check entire selected documents, then use first source-order train windows.
    """
    receipt = json.loads(Path(span_receipt).read_text())
    if digest(documents) != receipt['documents_index_sha256']: raise ValueError('index hash mismatch')
    spans = receipt['documents']
    if not spans: raise ValueError('empty span receipt')
    selected = []
    with Path(documents).open() as f:
        for line in f:
            d = json.loads(line)
            if d.get('truncated'): continue
            if d['split'] == 'train': selected.append(d)
            if len(selected) == len(spans): break
    if len(selected) != len(spans): raise ValueError('span receipt exceeds index')
    lefts, positions = [], []
    used_windows = 0
    with Path(receipt['pile_path']).open('rb') as f:
        for d, span in zip(selected, spans):
            if any(span[k] != d[k] for k in ('source_index','sha256','split')):
                raise ValueError('span is not the registered training prefix')
            if span['byte_offset'] < 0 or span['byte_length'] < 0: raise ValueError('negative span')
            f.seek(span['byte_offset']); raw = f.read(span['byte_length'])
            if len(raw)!=span['byte_length'] or hashlib.sha256(raw).hexdigest()!=d['sha256']:
                raise ValueError('pile.txt span differs from documents.jsonl')
            for start in range(0, len(raw)-1, 128):
                end = min(start+128, len(raw)-1)
                for target in range(start+1, end+1):
                    lefts.append(raw[max(start,target-1024):target])
                    positions.append(dict(document=d['source_index'], reset=start, target_offset=span['byte_offset']+target))
                used_windows += 1
                if used_windows == windows: return lefts, positions
    raise ValueError('receipt does not cover requested first training slice')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--span-receipt', required=True); p.add_argument('--documents', required=True)
    p.add_argument('--output', required=True); p.add_argument('--windows', type=int, default=16)
    a = p.parse_args()
    if a.windows < 1: raise ValueError('windows must be positive')
    begin = time.monotonic()
    lefts, positions = slice_contexts(a.span_receipt, a.documents, a.windows)
    bridge = CountBridge(a.output)
    rows, rates = extract(lefts, bridge.query)
    elapsed = time.monotonic()-begin
    with (Path(a.output)/'features.jsonl').open('w') as f:
        for position, row in zip(positions,rows): f.write(json.dumps(dict(**position,**row))+'\n')
    summary = dict(status='passed', positions=len(rows), seconds=elapsed, features_per_second=len(rows)/elapsed,
        support=rates, span_receipt_sha256=digest(a.span_receipt), documents_sha256=digest(a.documents),
        xsa_sha256=digest(XSA), feature_sha256=digest(Path(a.output)/'features.jsonl'),
        full_alphabet=True, gpu_used=False, seconds_per_40m=40000000*elapsed/len(rows),
        training_budget_seconds=2700, required_features_per_second_40m=40000000/2700,
        caveat='count range/branch consistency validated; no independent oracle for the entire Wikipedia index; Wikipedia separator byte 30 is measured in counts and terminal_count; excluded from nonterminal support, entropy and feature probabilities')
    (Path(a.output)/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__ == '__main__': main()
