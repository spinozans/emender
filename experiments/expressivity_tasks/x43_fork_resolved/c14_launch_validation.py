"""Validate an experimental warp count without editing a production launcher."""
import argparse
import os
from pathlib import Path
import sys
from . import c14_b_control_native as native


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--warps', type=int, choices=(8,16,32), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    p.add_argument('--battery', action='store_true')
    a = p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3','5','6')
    original = native.kernel
    class Variant:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                kw['num_warps'] = a.warps if kw['S'] == 257 else 8
                return original[grid](*args, **kw)
            return launch
    native.kernel = Variant()
    try:
        if a.battery:
            import pytest
            root = Path(__file__).parent
            tests = sorted(str(f) for f in root.glob('test_c14*.py'))
            tests += sorted(str(f) for f in root.glob('test_operator*.py'))
            sys.exit(pytest.main(['--import-mode=importlib', '-q', *tests]))
        from . import c14_b_unbudgeted_profile as profile
        sys.argv = [sys.argv[0], '--output',str(a.output),'--steps','20']
        if a.compare:
            sys.argv += ['--candidate','--compare',str(a.compare)]
        profile.main()
    finally:
        native.kernel = original

if __name__ == '__main__':
    main()
