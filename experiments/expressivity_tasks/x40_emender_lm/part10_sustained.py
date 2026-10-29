"""Unchanged production surface at a fixed, shared checkpoint/B32 envelope."""
from contextlib import contextmanager
import sys
from unittest.mock import patch
from . import part8_surface as surface


def main():
    original=surface.OUT;gate=surface.validate_parity
    out=surface.ROOT/'receipts/part10/sustained';out.mkdir(exist_ok=True)
    def parity(context,spec):
        with patch.object(surface,'OUT',original):gate(context,spec)
    def specs(batch,arm):return [dict(name='matched_checkpoint_B32',microbatch=32,checkpoint=True,packed=arm=='pure',offload=False)]
    with patch.object(surface,'OUT',out),patch.object(surface,'validate_parity',parity),patch.object(surface,'envelope_specs',specs):surface.main()

if __name__=='__main__':main()
