"""Record initialized X31 F12g traces; distinguish mutation and routing purity."""
import hashlib
import json
import os
from pathlib import Path
import torch
from ..x31_tape_cell_fused.cell_fused import TapeCell, TapeHeadConfig
from ..x30_tape_cell.routing_trace import RoutingTrace
from ..x28e_memory_head.generators import stage1_mqar as mqar
from .classifier import classify

ROOT = Path(__file__).resolve().parent


def summarize(trace, ids, chunk=64):
    tr = RoutingTrace(**{k: v.cpu() for k, v in vars(trace).items()})
    ids = ids.cpu()
    buckets = {key: dict(chunks=0, mutation_candidate=0, entry_pure=0, intra_match=0, intra_pop=0, event_tokens=0, reasons={})
               for key in ('all', 'all_a', 'all_b', 'all_pad', 'other')}
    b, t, h = tr.case.shape
    for batch in range(b):
        for start in range(0, t - chunk + 1, chunk):
            tokens = ids[batch, start:start + chunk]
            phase = 'other'
            for token, name in ((1, 'all_a'), (2, 'all_b'), (0, 'all_pad')):
                if bool((tokens == token).all()):
                    phase = name
            for head in range(h):
                cert = classify(tr, batch, head, start, start + chunk)
                for name in ('all', phase):
                    row = buckets[name]
                    row['chunks'] += 1
                    row['mutation_candidate'] += cert.mutation_candidate
                    row['entry_pure'] += cert.entry_pure
                    row['intra_match'] += cert.intra_match
                    row['intra_pop'] += cert.intra_pop
                    cases = tr.case[batch, start:start + chunk, head]
                    events = (cases >= 4) | tr.pop_fired[batch, start:start + chunk, head] | tr.freed[batch, start:start + chunk, head].any(-1)
                    row['event_tokens'] += int(events.sum())
                    for reason in cert.reasons:
                        row['reasons'][reason] = row['reasons'].get(reason, 0) + 1
    for row in buckets.values():
        row['event_token_fraction'] = row['event_tokens'] / (row['chunks'] * chunk) if row['chunks'] else None
        for name in ('mutation_candidate', 'entry_pure'):
            row[name + '_fraction'] = row[name] / row['chunks'] if row['chunks'] else None
    return dict(buckets=buckets, counts=tr.counts(), excluded_tail_tokens_per_store=t % chunk,
                tokens=t, batch=b, heads=h, chunk=chunk,
                token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())


@torch.no_grad()
def main():
    held = os.environ.get('GPU_LEASE_HELD')
    if held not in tuple(map(str, range(7))) or os.environ.get('CUDA_VISIBLE_DEVICES') != held:
        raise RuntimeError('broker lease required; card 7 forbidden')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = TapeHeadConfig(n_heads=4, budget=2944)
    cell = TapeCell(256, cfg).cuda()
    embed = torch.nn.Embedding(65, 256).cuda()
    # Exact positive ladder words at k=128, with four full phase windows.
    ladder = torch.tensor([[3] + [1] * 128 + [2] * 128 + [4]] * 8, device='cuda')
    words = torch.from_numpy(mqar.generate_pool(128, 8, 28302)[0]).cuda()
    result = dict(seed=0, physical_gpu=held, torch=torch.__version__, slots=2944,
                  model='initialized X31 TapeCell, d_model=256, K64 V128 H4; no training',
                  scope='8 exact positive k128 ladder words and 8 MQAR validation-prefix words; store-chunks, not whole-batch certificates',
                  trained_checkpoint=False)
    for name, ids in (('ladder', ladder), ('mqar', words)):
        _, _, tr, _ = cell(embed(ids), ids)
        result[name] = summarize(tr, ids)
        if name == 'mqar':
            # Token IDs 1/2 are content classes, not ladder phases.
            result[name]['buckets'] = {'all': result[name]['buckets']['all'],
                                      'all_pad': result[name]['buckets']['all_pad'],
                                      'other': result[name]['buckets']['other']}
        print(name, json.dumps(result[name]), flush=True)
    paths = [Path(__file__), ROOT / 'classifier.py', ROOT / 'CHUNK_PARALLEL.md', ROOT / 'RECEIPTS.md',
             ROOT.parent / 'x31_tape_cell_fused/cell_fused.py',
             ROOT.parent / 'x31_tape_cell_fused/kernels.py']
    result['source_sha256'] = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (ROOT / 'density.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
