#!/usr/bin/env python3
"""Bind W3's current runtime, historical sampler and diagnostic retention chain."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
from importlib.metadata import distributions, version

import schedulefree
import tiktoken
import torch

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
from scripts.e97_open_swe_native_codec import vocabulary
from scripts.w3_matcheddose_preflight import ROOT, REPO, AUTHORITY_SHA, PACK_SHA, checked, file_sha, digest
from scripts.w3_adamw_viability import ARGS_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    historical = torch.load(ROOT / 'wash-v2/checkpoints/checkpoint_agent_sft_u000512_loss_1.0614.pt',
                            map_location='cpu', mmap=True, weights_only=False)
    sampler_key = historical['sampler_key']
    identity = SFTSamplerIdentity(AUTHORITY_SHA, PACK_SHA, sampler_key, 1, 12288)
    data = MaskedSFTPackedDataset(ROOT / 'wash-v2/authority', ROOT / 'wash-v2/packs',
                                 identity=identity, rank=0, sampler_mode='epoch-permutation')
    for index, row in enumerate(preflight['exposure']['sequence']):
        pack_id = data.pack_id_at(index)
        values = data.pack_at_with_boundaries(pack_id)
        tokens, loss, valid, reset, length, targets, pack_name = values
        if (pack_name != row['pack_id'] or [data.sample_id(index)] != row['sample_ids']
                or length != row['inputs'] or targets != row['targets']
                or digest(tokens.numpy().astype('<u4').tobytes()) != row['tokens_sha256']
                or digest(loss.numpy().tobytes()) != row['loss_mask_sha256']
                or digest(valid.numpy().tobytes()) != row['valid_mask_sha256']
                or digest(reset.numpy().tobytes()) != row['reset_mask_sha256']):
            raise ValueError(f'current sampler/materializer differs from independent historical replay: {index+1}')
    data.close()
    tokenizer_sha = vocabulary(tiktoken.get_encoding('p50k_base'))[1]
    if tokenizer_sha != preflight['instruments']['stage_b']['tokenizer_vocabulary_sha256']:
        raise ValueError('tokenizer differs')
    sources = sorted(REPO.glob('ndm/**/*.py'))
    sources += [REPO / p for p in ('scripts/train_e97_4b_pi_sft.py', 'scripts/rl-loop-v1/rl_train_step.py',
                                   'scripts/rl-loop-v1/wash2_verdict_receipt.py')]
    sources += sorted((REPO / 'scripts').glob('w3_*.py'))
    sources += sorted((REPO / 'scripts').glob('w3_*.sh'))
    sources += [REPO / 'scripts/watch_w3_arms.py']
    sources += sorted((REPO / 'configs').rglob('*autotune*'))
    sources += [Path(inspect.getfile(schedulefree.AdamWScheduleFree)),
                Path(inspect.getfile(torch.optim.AdamW))]
    bindings = [dict(path=str(p), sha256=file_sha(p)) for p in sources if p.is_file()]
    read_root = ROOT / 'bank-heldout-reads'
    doc_result_path = read_root / 'wash-verdict-read-v1/docnll-gpu/docnll-gpu.json'
    doc_result = json.loads(doc_result_path.read_text())
    doc_bindings = [dict(path=str(doc_result_path), sha256=file_sha(doc_result_path)),
                    checked(doc_result['panel'], doc_result['panel_sha256']),
                    checked(ARGS_PATH, doc_result['args_sha256'])]
    for p in [read_root / 'operator-ruling-docnll-20261002.json', read_root / 'HANDOFF.md',
              ROOT / 'scripts/score_docnll_gpu.py', *sorted((ROOT / 'scripts').glob('bank-heldout-read-phase*.sh'))]:
        doc_bindings.append(dict(path=str(p), sha256=file_sha(p)))
    substrate = doc_result['models']['e4-u256']
    doc_bindings.append(checked(substrate['checkpoint'], substrate['sha256']))
    inventory = Path(preflight['instruments']['stage_b']['pi_inventory'])
    inventory_check = subprocess.run(['sha256sum', '-c', str(inventory)], capture_output=True, text=True)
    if inventory_check.returncode:
        raise ValueError('Pi inventory payload verification failed: ' + inventory_check.stderr)
    sf_groups = [{k: v for k, v in g.items() if k != 'params'}
                 for g in historical['optimizer_state_dict']['param_groups']]
    result = dict(schema='emender-w3-runtime-freeze-v1', preflight_sha256=file_sha(args.preflight),
                  script_sha256=file_sha(Path(__file__)), source_bindings=bindings,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  python=sys.version, executable=sys.executable,
                  versions={name: version(name) for name in ('torch', 'triton', 'schedulefree', 'tiktoken', 'numpy')},
                  installed_distributions=sorted(f'{d.metadata["Name"]}=={d.version}' for d in distributions()),
                  distribution_inventory_method='importlib.metadata.distributions (this environment has no pip module)',
                  historical_optimizer_groups=sf_groups, sampler_identity=identity.to_metadata(),
                  sampler_replay='all 512 pack/sample IDs and token/loss/valid/reset tensor hashes agree with independent recount',
                  context_size=12288, data_world_size=1, sampler_mode='epoch-permutation',
                  precision=historical['sft_precision'], grad_clip=historical['grad_clip'],
                  tokenizer_vocabulary_sha256=tokenizer_sha,
                  pi_inventory=dict(path=str(inventory), sha256=file_sha(inventory),
                                    verified_payload_files=inventory_check.stdout.count(': OK\n')),
                  docnll=dict(bindings=doc_bindings, mode='saved', targets=198295, docs=8,
                              baseline=substrate, wash1_mean_nll=doc_result['models']['wash-u256']['totals']['mean_nll'],
                              criterion='trajectory only; never a stop signal; no numeric pass/fail bound',
                              authority='operator-ruling-docnll-20261002.json; supervisor confirmed prelaunch amendment'),
                  branch='WAIT_X9b_FINAL', launch_ready=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print('W3_RUNTIME_FREEZE: 512 replay tensors match; tokenizer/Pi verified; diagnostic docNLL bound')


if __name__ == '__main__':
    main()
