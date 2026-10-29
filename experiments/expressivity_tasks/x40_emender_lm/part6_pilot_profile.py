"""Unchanged pilot timing; retain decision statistics outside the profiled update."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from . import backward, part5_pilot


def main():
    original=part5_pilot.instrumentation
    label=sys.argv[sys.argv.index('--label')+1]
    out=part5_pilot.ROOT/'receipts/part5'/label
    @contextmanager
    def instrument():
        rows=[]
        with original() as journals:
            native=backward.Replay.backward
            def replay(ctx,*args):
                # Tensor references only: no host transfer or extra device work
                # in the profiler. Owners and whole contexts are not retained.
                rows.append((ctx.cfg.slots,ctx.trace[0].detach(),ctx.trace[2].detach(),ctx.log[3].detach()))
                return native(ctx,*args)
            with patch.object(backward.Replay,'backward',staticmethod(replay)):
                yield journals
        records=[]
        for slots,cases,alloc,counts in rows:
            cases=cases.cpu();alloc=alloc.cpu();counts=counts.cpu()
            writers=(cases==4)|(cases==5)|(alloc>=0)
            tokens=cases.shape[-1];flat=writers.reshape(-1,tokens)
            positions=torch.arange(tokens).expand_as(flat)
            first=torch.where(flat,positions,tokens).min(1).values
            records.append(dict(slots=slots,tokens_per_lane=tokens,lanes=len(flat),
                                case_histogram={str(int(i)):int((cases==i).sum()) for i in cases.unique()},
                                value_consuming_events=int(writers.sum()),total_token_events=writers.numel(),
                                lanes_without_any_value_consumer=int((~flat.any(1)).sum()),
                                read_prefix_tokens_before_first_value_consumer=int(first.sum()),
                                measured_journal_events_per_lane=counts.tolist()))
        (out/'decision_sparsity.json').write_text(json.dumps(dict(rows=records,
            scope='Recorded decisions from the separate profile update; no performance timing changed. Current global dead-value-adjoint gate is eligible only on lanes without any consumer. Prefix counts bound a narrower temporal eligibility but are not a measured speedup; row-specific adjoints and journal undo dependencies remain.'),indent=2)+'\n')
    with patch.object(part5_pilot,'instrumentation',instrument):part5_pilot.main()


if __name__=='__main__':main()
