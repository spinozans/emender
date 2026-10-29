"""Static allocation arithmetic and historical observed OOM evidence; no timings inferred."""
import json
import re
from pathlib import Path
from .model import BUDGETS
from .scale_surface import SHAPES
from ..x36_tape_perf.journal import AllFieldJournal

ROOT = Path(__file__).resolve().parent


def terms(layers, batch, context):
    native = layers * batch * sum(BUDGETS) * (64 + 128) * 4
    capacities = [AllFieldJournal.capacity_for(batch, context, s, 64, 128, 0) for s in BUDGETS]
    journals = layers * batch * (20 * sum(capacities) + len(BUDGETS) * (16 * context + 8))
    freed = layers * batch * context * sum(BUDGETS)
    return dict(native_payload_bytes=native, journal_reserved_bytes=journals,
                per_head_event_reservations=capacities, trace_freed_bytes=freed,
                dense_logits_bf16_bytes=batch*context*32768*2,
                dense_logits_fp32_bytes=batch*context*32768*4)


def main():
    rows=[]
    for path in sorted((ROOT/'receipts/part5/surface').glob('*.json')):
        old=json.loads(path.read_text())
        scale,b,t=old['scale'],old['batch'],old['context']
        counts=old.get('parameters',old.get('pure_parameters',{}))
        row=dict(source=str(path.relative_to(ROOT)),scale=scale,arm=old['arm'],batch=b,context=t,
                 historical_status=old['status'],error=old.get('error'),
                 weight_bytes_fp32=4*counts.get('total',0),
                 optimizer_state_bytes_after_initialization=8*counts.get('trainable',0),
                 gradient_bytes_fp32=4*counts.get('trainable',0),
                 measurement_scope='allocation formulas, not concurrent peaks; Adam states may not exist at failure')
        if old['arm']=='pure':
            row.update(terms(SHAPES[scale][1],b,t))
            row['dominant_static_term']=max(('native_payload_bytes','journal_reserved_bytes','trace_freed_bytes'),key=lambda k:row[k])
            row['checkpoint_one_block_microB32']=terms(1,min(32,b),t)
        for label,pattern in [('allocated_gib',r'Of the allocated memory ([\d.]+) GiB'),
                              ('reserved_unallocated_gib',r'and ([\d.]+) GiB is reserved')]:
            match=re.search(pattern,old.get('error',''))
            row[label]=float(match[1]) if match else None
        row.update(activation_peak_bytes=None,replay_scratch_peak_bytes=None,
                   fragmentation_bytes=None,
                   missing_evidence='No allocator snapshot in part5; reserved-unallocated is not a fragmentation measurement. Live activation/replay peaks require instrumented reruns.')
        rows.append(row)
    out=ROOT/'receipts/part6/memory_arithmetic.json'
    out.write_text(json.dumps(dict(status='partial',arms=rows,
        budget_policy=dict(applied='unchanged pilot-family per-head budgets',horizons=[128,512,2048,8192],budgets=list(BUDGETS),
                           total_slots_per_block=sum(BUDGETS),native_formula='layers * B * sum(per-head slots) * (64+128) * 4',
                           note='Actual heterogeneous budgets are 64/256/736/1472, not 2944 at every head. Journal retention, not native payload, dominates static scale memory.',
                           optional_unvalidated_scale_only=dict(budgets=[16,64,184,368],native_payload_fraction=.25,applied=False))),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
