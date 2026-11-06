"""Diagnostic-only phase-2 entrypoint. Does not register or launch training."""
import argparse
import sys
from . import c14_phase2_staged as candidate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gluon', action='store_true')
    p.add_argument('--rowkeys', action='store_true')
    p.add_argument('gate', choices=('gradient', 'battery', 'model', 'oracle'))
    args, rest = p.parse_known_args()
    install = candidate.install_rowkeys if args.rowkeys else (candidate.install_gluon if args.gluon else candidate.install)
    if args.gate == 'gradient':
        from . import c14_phase2_gradient_parity as gate
        gate.install = install
    else:
        install()
        if args.gate == 'model':
            from . import c14_b_unbudgeted_profile as gate
        elif args.gate == 'oracle':
            import pytest
            from pathlib import Path
            sys.exit(pytest.main(['--import-mode=importlib', '-q', str(Path(__file__).resolve().parent.parent/'x31_tape_cell_fused/tests'), *rest]))
        else:
            from . import c14_launch_validation as gate
            rest = ['--warps', '8', '--battery', *rest]
    sys.argv = [sys.argv[0], *rest]
    gate.main()


if __name__ == '__main__':
    main()
