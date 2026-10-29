"""Allocator history replay at the logical-allocation high water, including failed runs."""
from collections import Counter


def category(frames):
    names='\n'.join(f.get('filename','')+':'+f.get('name','') for f in frames)
    if 'all_field_journal.py' in names:return 'journals'
    if 'replay_setup.py' in names:return 'replay_scratch'
    if '/optim/' in names:return 'optimizer'
    if 'naive_reference.py:init_state' in names:return 'native_stores'
    if 'backward.py:backward' in names:return 'replay_scratch'
    if '/nn/modules/' in names and any(n in names for n in ('reset_parameters','__init__','_apply')):return 'weights'
    if 'forward.py:tape_scan' in names:return 'native_trace_and_forward_scratch'
    if frames:return 'activations_and_other'
    return 'unattributed'


def summarize(snapshot):
    rows=[]
    for device,events in enumerate(snapshot.get('device_traces',[])):
        live={};physical={};segments={};peak={};stacks={};size=0;maximum=0;ooms=[]
        for event in events:
            action=event['action'];addr=event.get('addr')
            if action=='alloc':
                entry=(event['size'],category(event.get('frames',[])))
                if addr in live:size-=live[addr][0]
                live[addr]=entry;physical[addr]=event['size'];stacks[addr]=event.get('frames',[]);size+=entry[0]
                if size>maximum:maximum=size;peak=live.copy();peak_stacks={a:stacks[a] for a in peak}
            elif action=='free_requested' and addr in live:
                size-=live.pop(addr)[0]
            elif action=='free_completed':physical.pop(addr,None)
            elif action=='segment_alloc':segments[addr]=event['size']
            elif action=='segment_free':segments.pop(addr,None)
            elif action=='oom':
                observation={k:v for k,v in event.items() if k!='frames'}
                gaps=[]
                for base,length in segments.items():
                    cursor=base
                    for pointer,span in sorted(physical.items()):
                        if base<=pointer<base+length:
                            gaps.append(max(0,pointer-cursor));cursor=pointer+span
                    gaps.append(max(0,base+length-cursor))
                logical_categories=Counter()
                physical_categories=Counter()
                for span,label in live.values():logical_categories[label]+=span
                for pointer,span in physical.items():physical_categories[category(stacks.get(pointer,[]))]+=span
                observation.update(logical_allocated_bytes=size,logical_categories=dict(logical_categories),
                                   physical_categories=dict(physical_categories),pending_free_bytes=sum(physical.values())-size,
                                   dominant_logical_term=max(logical_categories,key=logical_categories.get) if logical_categories else None,
                                   reconstructed_reserved_bytes=sum(segments.values()),
                                   reconstructed_physical_allocated_bytes=sum(physical.values()),
                                   reconstructed_inactive_bytes=sum(gaps),largest_reconstructed_inactive_block_bytes=max(gaps,default=0),
                                   reconstruction_scope='Default allocator segment_alloc/free and alloc/free_completed events; not valid for expandable segment_map traces.')
                ooms.append(observation)
        buckets=Counter()
        for n,c in peak.values():buckets[c]+=n
        if events:
            samples=sorted(peak.items(),key=lambda item:item[1][0],reverse=True)[:12]
            rows.append(dict(device=device,peak_allocated_bytes_from_history=maximum,categories=dict(buckets),oom_events=ooms,
                             largest_peak_allocations=[dict(bytes=entry[0],category=entry[1],frames=peak_stacks[addr]) for addr,entry in samples]))
    free=[b['size'] for seg in snapshot.get('segments',[]) for b in seg['blocks'] if b['state']=='inactive']
    return dict(devices=rows,terminal_inactive_bytes=sum(free),terminal_largest_inactive_block_bytes=max(free,default=0),
                scope='Allocated-byte logical-lifetime high water reconstructed from alloc/free_requested history; trace sizes may include allocator rounding. Physical pending frees are retained separately in the OOM reconstruction. Categories use Python allocation stacks; activations_and_other is not a pure activation attribution. Inactive blocks are allocator slack, not an asserted fragmentation cause.')
