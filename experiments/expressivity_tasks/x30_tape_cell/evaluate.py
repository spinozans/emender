"""Evaluate a frozen checkpoint with the current X30 source (no updates)."""
import argparse
import hashlib
import json
import os
import torch
from .train_stage1 import ROOT, WordModel, artifact_path, device_for, save_json
from .train_stage1 import evaluate as evaluate_mqar
from .train_ladder import evaluate as evaluate_ladder
from experiments.expressivity_tasks.x28e_memory_head.generators import stage1_mqar, stage3_ladder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--task', choices=['stage1', 'ladder'], required=True)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    dev = device_for(args.device)
    torch.manual_seed(0)
    model = WordModel(args.heads)
    if args.task == 'ladder':
        model.attach_ladder()
    path = artifact_path(args.checkpoint)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.to(dev)
    result = dict(config=vars(args), physical_gpu=os.environ.get('GPU_LEASE_HELD'),
                  checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ('cell.py', 'float_decay.py', 'train_stage1.py')})
    if args.task == 'stage1':
        pool = stage1_mqar.generate_pool(128, 20000, stage1_mqar.STAGE1_SPLITS['validation'].seed)
        result['mean_per_class_recall'] = evaluate_mqar(model, pool, dev)
        result['validation_hash'] = stage1_mqar.stage1_hash('validation', *pool[:3])
    else:
        result['bins'], result['pool_hashes'] = {}, {}
        for name in stage3_ladder.BIN_ORDER:
            pool = stage3_ladder.ladder_bin_pool(name)
            result['bins'][name] = evaluate_ladder(model, pool, dev)
            result['pool_hashes'][name] = stage3_ladder.ladder_hash(name, *pool[:3])
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
