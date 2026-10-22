"""X24 activation-only amendment; all data/training code is the X23E protocol."""
from pathlib import Path
import argparse

from experiments.expressivity_tasks import run_merrill26_separators as protocol

ROOT = Path('experiments/expressivity_tasks/results_stage24_softplus')
COMPARISON_ROOT = protocol.ROOT
STOCK_BUILD_MODEL = protocol.build_model


def build_model(arm, task):
    if (arm, task) != ('e97', 'graph'):
        raise ValueError('X24 authorizes only E97 graph')
    model = STOCK_BUILD_MODEL(arm, task)
    for layer in model.layers:
        assert layer.use_split_edit and layer.use_triton and not layer.linear_state
        assert layer.state_activation == 'tanh' and layer.recurrent_state_precision == 'legacy'
        layer.state_activation = 'softplus'
        layer._nonsat_state = True
    return model


def configure():
    protocol.ROOT = ROOT
    protocol.build_model = build_model


def link_datasets():
    """Use the actual X23E pools, locks, seeds and hash receipts, not a replica."""
    target = COMPARISON_ROOT.resolve() / 'datasets'
    if not target.is_dir():
        raise ValueError('X23E shared datasets unavailable')
    link = ROOT / 'datasets'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target:
        raise ValueError('X24 must use the X23E dataset directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    configure()
    if args.smoke:
        protocol.smoke('e97', 'graph', args.smoke)
    else:
        link_datasets()
        protocol.run('e97', 'graph')
