"""Capture the promoted production specialization without changing its code."""
import os
from pathlib import Path
import runpy
import sys

assert sys.executable == '/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3', '4')
from capture import install

install()
mode = sys.argv[1:]
sys.argv = ['production_checks.py', *mode]
runpy.run_path(str(Path(__file__).resolve().parent.parent / 'part22/production_checks.py'),
              run_name='__main__')
