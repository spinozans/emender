"""X25 paper-recipe amendment; reuse the registered X23E tasks and training loop."""
from pathlib import Path
import argparse

from experiments.expressivity_tasks import run_merrill26_separators as protocol
from experiments.expressivity_tasks import run_e97_softplus_graph as softplus

ROOT = Path('experiments/expressivity_tasks/results_stage25_recipe_controlled')
COMPARISON_ROOT = softplus.COMPARISON_ROOT
CARDS = {'e97_graph_seed0': 0, 'e97softplus_graph_seed0': 7,
         'gdn2_graph_seed0': 1, 'e97_integer_seed0': 2, 'gdn2_integer_seed0': 3}


def build_model(arm, task):
    if f'{arm}_{task}_seed0' not in CARDS:
        raise ValueError('X25 authorizes only the five recipe-controlled rungs')
    if arm == 'e97softplus':
        return softplus.build_model('e97', task)
    return softplus.STOCK_BUILD_MODEL(arm, task)


def learning_rate(step):
    if not 0 <= step <= protocol.CEILING:
        raise ValueError('step outside registration')
    return 1e-4


def configure():
    protocol.ROOT = ROOT
    protocol.build_model = build_model
    protocol.learning_rate = learning_rate
    protocol.WEIGHT_DECAY = 1e-4


def link_datasets():
    """Reuse the actual frozen pools, receipts and locks; never regenerate a replica."""
    target = COMPARISON_ROOT.resolve() / 'datasets'
    if not target.is_dir():
        raise ValueError('X23E shared datasets unavailable')
    link = ROOT / 'datasets'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target:
        raise ValueError('X25 must use the X23E dataset directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['e97', 'e97softplus', 'gdn2'], required=True)
    parser.add_argument('--task', choices=['graph', 'integer'], required=True)
    args = parser.parse_args()
    if f'{args.arm}_{args.task}_seed0' not in CARDS:
        raise ValueError('unauthorized rung')
    configure()
    link_datasets()
    protocol.run(args.arm, args.task)
