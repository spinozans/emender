#!/usr/bin/env python3
"""Uniform workspace repair, independently frozen from composite v1."""
import argparse
from pathlib import Path
from ndm.numerical_policy import UNIFORM_POLICY
from scripts.qualify_e97_fp32_linear import freeze,audit

RECIPE_SHA='3ab8a9237cd2102b34c83f125ed529758982f74a39aa6beef52c069c5811e364'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','audit'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a,policy=UNIFORM_POLICY,expected_recipe_sha=RECIPE_SHA,kernel_cases=14)
