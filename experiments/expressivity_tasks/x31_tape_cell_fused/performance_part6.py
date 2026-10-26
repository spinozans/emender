"""Collect registered timing and storage receipts on one broker lease."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys
from . import cost_gates, append_cost, storage_cost
from .train_gates import require_parity

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity-receipt',required=True)
    args=parser.parse_args()
    receipt=str(Path(args.parity_receipt).resolve())
    require_parity(receipt)
    original=sys.argv
    try:
        for module,name,extra in (
            (cost_gates,'cost_part6',['--output','cost_part6.json']),
            (append_cost,'append_cost_part6',['--output','append_cost_part6.json']),
            (storage_cost,'storage_part6',[])):
            sys.argv=[name,'--parity-receipt',receipt,*extra]
            with (ROOT/(name+'.txt')).open('w') as out, redirect_stdout(out):
                module.main()
            print(name+' collection completed',flush=True)
    finally:
        sys.argv=original


if __name__=='__main__':
    main()
