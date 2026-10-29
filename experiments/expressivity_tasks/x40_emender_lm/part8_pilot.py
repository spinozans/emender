"""Run existing certified pilot helpers with outputs isolated from part7."""
import argparse
import shutil
import sys
from .part8_surface import ROOT,OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('parity','throughput','curve'))
    a,rest=p.parse_known_args()
    if a.action=='parity':
        from . import part6_pool_fusion as module
    elif a.action=='throughput':
        from . import part6_pilot_checkpoint as module
        (OUT/'curve').mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'receipts/part6/curve/summary.json',OUT/'curve/summary.json')
        shutil.copyfile(OUT/'production_T512.json',OUT/'production_T512_expandable.json')
    else:
        from . import part6_batch_curve as module
    module.ROOT=OUT
    sys.argv=[sys.argv[0]]+rest
    module.main()

if __name__=='__main__':main()
