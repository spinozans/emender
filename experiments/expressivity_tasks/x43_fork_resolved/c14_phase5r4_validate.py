"""Explicit phase-5 validation only. Never installed by production runners."""
import argparse
import os
from pathlib import Path
import sys
from . import c14_b_control_native as native
from .c14_phase2_gluon_rowkeys_kernel import kernel as reference
from .c14_phase5r4_stream_pages import kernel as chunked
from .c14_launch_shape import install as install_shape


def install():
    class Dispatch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                selected = reference if kwargs['S']*7 < 128 else chunked
                return selected[grid](*args, **kwargs)
            return launch
    native.kernel=Dispatch()
    return install_shape()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('gate',choices=('oracle','battery','shape','gradient'))
    args,rest=p.parse_known_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3','4','5','6')
    if args.gate in ('shape','gradient'):
        from . import c14_phase2_shape_parity as shape
        from . import c14_phase2_gradient_parity as gradient
        gate=shape if args.gate=='shape' else gradient
        gate.install=install
        sys.argv=[sys.argv[0],*rest]
        gate.main()
    else:
        import pytest
        install()
        root=Path(__file__).parent
        tests=([str(root.parent/'x31_tape_cell_fused/tests')] if args.gate=='oracle'
               else sorted(str(f) for pat in ('test_c14*.py','test_operator*.py') for f in root.glob(pat)))
        sys.exit(pytest.main(['--import-mode=importlib','-q',*tests,*rest]))

if __name__=='__main__':
    main()
