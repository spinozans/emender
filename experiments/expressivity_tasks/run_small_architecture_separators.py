"""X27 size-only graph arms; unchanged X25 protocol and reviewed X23E pools."""
from pathlib import Path
import argparse

from experiments.expressivity_tasks import run_merrill26_separators as protocol
from experiments.expressivity_tasks import run_recipe_controlled_separators as comparison

ROOT = Path('experiments/expressivity_tasks/results_stage27_small_architectures')
COMPARISON_ROOT = comparison.COMPARISON_ROOT
# Queue behind the corresponding X25 physical cards; card6 is initially free.
CARDS = {'e97_graph_seed0': 6, 'e97softplus_graph_seed0': 0, 'gdn2_graph_seed0': 1}
CONFIGS = {
    'e97': {'dim': 192, 'depth': 1, 'n_heads': 16, 'n_state': 16, 'expansion': 1.0},
    'e97softplus': {'dim': 192, 'depth': 1, 'n_heads': 16, 'n_state': 16, 'expansion': 1.0},
    'gdn2': {'dim': 128, 'depth': 1, 'n_heads': 16, 'n_state': 16, 'expansion': 8.625},
}


def build_model(arm, task):
    if f'{arm}_{task}_seed0' not in CARDS:
        raise ValueError('X27 authorizes only three small graph rungs')
    from ndm.models.hybrid_ladder import HybridLadderLM
    e97 = arm != 'gdn2'
    model = HybridLadderLM(vocab_size=protocol.VOCABS[task], **CONFIGS[arm],
        layer_pattern=['E97' if e97 else 'fla-gdn'],
        layer_kwargs=[{'state_activation': 'tanh', 'raw_write': False} if e97
                      else {'allow_neg_eigval': True}], use_triton_e88=e97, mlp_ratio=2.)
    if e97:
        for layer in model.layers:
            layer.recurrent_state_precision = 'legacy'
            layer.fused_inference = True
            if arm == 'e97softplus':
                assert layer.use_split_edit and layer.use_triton and not layer.linear_state
                layer.state_activation = 'softplus'
                layer._nonsat_state = True
    return model


def configure():
    comparison.configure()
    protocol.ROOT = ROOT
    protocol.build_model = build_model


def link_datasets():
    target = COMPARISON_ROOT.resolve() / 'datasets'
    if not target.is_dir():
        raise ValueError('X23E shared datasets unavailable')
    link = ROOT / 'datasets'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target:
        raise ValueError('X27 must use the X23E dataset directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=list(CONFIGS), required=True)
    parser.add_argument('--task', choices=['graph'], required=True)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    configure()
    link_datasets()
    if args.smoke:
        protocol.smoke(args.arm, args.task, args.smoke)
    else:
        protocol.run(args.arm, args.task)
