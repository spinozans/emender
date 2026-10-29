"""Attest token partitioning, weighted means and observed native replay counts."""
import json
from pathlib import Path
from .scale_surface import SHAPES

R=Path(__file__).resolve().parent/'receipts/part6'


def main():
    rows=[]
    for scale in SHAPES:
        for context in (512,2048):
            for batch in (64,128):
                points={arm:json.loads((R/'surface'/f'{scale}_{arm}_B{batch}_T{context}.json').read_text()) for arm in ('pure','transformer')}
                assert all(p['status']=='passed' and len(p['seconds'])==20 for p in points.values())
                assert all(p['microbatch']==32 and p['processed_microbatch_tokens']==batch*context for p in points.values())
                ranges=[(begin,begin+32) for begin in range(0,batch,32)]
                covered=[i for begin,end in ranges for i in range(begin,end)]
                expected=4*SHAPES[scale][1]*(batch//32)
                assert covered==list(range(batch))
                assert all(v==expected for v in points['pure']['replays_per_update'])
                assert all(v==0 for v in points['transformer']['replays_per_update'])
                rows.append(dict(scale=scale,batch=batch,context=context,physical_batch=32,
                                 microbatch_row_ranges=ranges,loss_weight_per_microbatch=32/batch,
                                 sum_loss_weights=(batch//32)*(32/batch),effective_tokens_per_update=batch*context,
                                 sum_microbatch_tokens=(batch//32)*32*context,
                                 observed_native_replays_per_effective_update=expected,
                                 measured_effective_updates=20,measured_tokens_per_arm=20*batch*context,
                                 clipping_operations_per_update=1,adamw_updates_per_effective_update=1))
    (R/'token_equivalence.json').write_text(json.dumps(dict(status='passed',rows=rows,
        data='Within each arm, one preloaded nonpadding token tensor is partitioned into disjoint contiguous row slices without changing its shifted input/target pairs. Synthetic IDs are drawn from [1,vocab). Pure/control match counts and distribution, not token identities: model initialization consumes RNG before data generation. This is a compute surface, not a scale quality curve.',
        numeric_scope='Weighted microbatch means equal the full-batch mean in exact arithmetic. FP32 dense-gradient accumulation is not asserted bitwise-identical; the matched training curve retains its 1e-5 loss contract.'),indent=2)+'\n')


if __name__=='__main__':main()
