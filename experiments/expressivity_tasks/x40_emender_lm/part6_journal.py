"""Pack completed undo records to measured lane maxima, without re-deriving decisions.

The bounded writer reservation remains transient. Only the actual used prefix
is retained for trace-following replay; overflow assertions stay in the writer.
This adds one host maximum read per scan and does not claim a no-sync gate.
"""
from contextlib import contextmanager
from unittest.mock import patch
from . import model


@contextmanager
def packed_journals(records):
    original=model.wired_forward
    def forward(*args,**kwargs):
        result=original(*args,**kwargs)
        journal=result[1].all_field_journal
        if journal is not None:
            before=journal.capacity
            used=int(journal.counts.max().item())
            assert 0<=used<=before
            pitch=max(1,used)
            # clone, even for B=1: contiguous() can alias the full reservation.
            journal.addresses=journal.addresses[:,:pitch].clone()
            journal.before=journal.before[:,:pitch].clone()
            journal.widths=journal.widths[:,:pitch].clone()
            records.append(dict(stores=len(journal.counts),reserved_events_per_lane=before,
                                measured_max_events_per_lane=used,retained_events_per_lane=pitch,
                                before_bytes=20*len(journal.counts)*before,
                                after_bytes=20*len(journal.counts)*pitch))
        return result
    with patch.object(model,'wired_forward',forward):
        yield
