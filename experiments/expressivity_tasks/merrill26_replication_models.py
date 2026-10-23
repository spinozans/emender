"""X26 task-specific upstream models, with explicit scalar-stream IMM adapters.

Upstream files are immutable receipts, not a claim that the public repository
uniquely identifies the models plotted in Figure 2. See the discrepancy ledger.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

SOURCES = Path(__file__).parent / 'results_stage26_replication/sources/LinearRNN'


@lru_cache(None)
def upstream(task, model):
    path = SOURCES / task / f'train_{model}.py'
    name = f'x26_upstream_{task}_{model}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ScalarEncoder(nn.Module):
    """Keep X23E's exact 9N scalar stream, without BOS or prefix-state inputs."""
    def __init__(self, dropout):
        super().__init__()
        self.embedding = nn.Embedding(4, 256, padding_idx=3)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens, unused):
        return self.drop(self.embedding(tokens))


class IntegerModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        source = upstream('imm_unbound', {'rnn': 'rnn', 'delta': 'deltanet', 'rwkv': 'rwkv7'}[arm])
        if arm == 'rnn':
            core = source.ModelRNNFinalBinary(2700, 256, 256, 2, .3, 'gru')
            dropout = .3
        elif arm == 'delta':
            core = source.ModelDeltaNetFinalBinary(2700, 256, 1, 4, .2,
                                                  deltanet_mode='chunk', allow_neg_eigval=False)
            dropout = .2
        else:
            core = source.ModelRWKVFinalBinary(2700, 256, 64, 2, .1, 'chunk')
            dropout = .1
        core.enc = ScalarEncoder(dropout)
        self.core, self.arm = core, arm

    def forward(self, tokens, lengths):
        if self.arm == 'rnn':
            return self.core(tokens, None, lengths)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        return self.core(tokens, None, mask)


def build_model(arm, task):
    if arm not in ('rnn', 'delta', 'rwkv') or task not in ('graph', 'integer'):
        raise ValueError('X26 authorizes only the upstream-model graph/integer arms')
    if task == 'integer':
        return IntegerModel(arm)
    source = upstream('dgc', {'rnn': 'rnn', 'delta': 'deltanet', 'rwkv': 'rwkv7'}[arm])
    if arm == 'rnn':
        return source.StableFastReLURNNClassifier(8, 0, 128, 256, 1, .1,
                                                 act_clip=6., hh_shrink=.3)
    if arm == 'delta':
        return source.DeltaNetClassifier(8, 0, 128, 256, 1, 4, .1, 'chunk', False, 4, False)
    return source.RWKV7Classifier(8, 0, 128, 256, 2, 64, 'chunk', .1)
