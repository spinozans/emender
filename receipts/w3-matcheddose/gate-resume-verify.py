"""Read-only W3G relaunch verification; does not alter frozen instruments."""
import json
from pathlib import Path
import subprocess
import sys
import time
from importlib.metadata import version
import inspect
import tiktoken
import torch
from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
from scripts.e97_open_swe_native_codec import vocabulary
from scripts.w3_matcheddose_preflight import ROOT, REPO, checked, file_sha, digest, recount, instruments, AUTHORITY_SHA, PACK_SHA
from scripts.w3_train_arm import verify_sources
from scripts.w3_fix_regime_branch import fix_branch, X9B

receipts = REPO / 'receipts/w3-matcheddose'
preflight_path = receipts / 'preflight.json'
preflight = json.loads(preflight_path.read_text())
freeze_path = receipts / 'runtime-freeze-v2.json'
freeze = json.loads(freeze_path.read_text())
prereg = json.loads((receipts / 'preregistration.json').read_text())
records = []
for path in (preflight_path, freeze_path, receipts / 'preregistration.json', receipts / 'fixed-branch.json'):
    relative = str(path.relative_to(REPO))
    committed = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=REPO)
    assert committed == path.read_bytes(), f'uncommitted frozen receipt {path}'
    records.append(checked(path, file_sha(path)))
for binding in prereg['bindings']:
    records.append(checked(binding['path'], binding['sha256']))
assert freeze['preflight_sha256'] == file_sha(preflight_path)
verify_sources(freeze)
for binding in freeze['source_bindings']:
    path = Path(binding['path'])
    if path.is_relative_to(REPO) and '.venv' not in path.parts:
        assert subprocess.check_output(['git', 'show', 'HEAD:' + str(path.relative_to(REPO))], cwd=REPO) == path.read_bytes(), path
for binding in freeze['docnll']['bindings']:
    records.append(checked(binding['path'], binding['sha256']))
assert recount(ROOT) == preflight['exposure'], 'independent 512-step mask/exposure recount changed'
assert instruments(ROOT) == preflight['instruments'], 'wash-2 instrument changed'
assert {name: version(name) for name in freeze['versions']} == freeze['versions'], 'runtime versions changed'
assert vocabulary(tiktoken.get_encoding('p50k_base'))[1] == freeze['tokenizer_vocabulary_sha256']
assert json.loads((receipts / 'fixed-branch.json').read_text()) == fix_branch(X9B)
# Verify the same sampler/materializer against every independently recounted batch.
historical = torch.load(ROOT / 'wash-v2/checkpoints/checkpoint_agent_sft_u000512_loss_1.0614.pt', map_location='cpu', mmap=True, weights_only=False)
identity = SFTSamplerIdentity(AUTHORITY_SHA, PACK_SHA, historical['sampler_key'], 1, 12288)
data = MaskedSFTPackedDataset(ROOT / 'wash-v2/authority', ROOT / 'wash-v2/packs', identity=identity, rank=0, sampler_mode='epoch-permutation')
for i, row in enumerate(preflight['exposure']['sequence']):
    tokens, loss, valid, reset, length, targets, name = data.pack_at_with_boundaries(data.pack_id_at(i))
    assert (name, [data.sample_id(i)], length, targets, digest(tokens.numpy().astype('<u4').tobytes()), digest(loss.numpy().tobytes()), digest(valid.numpy().tobytes()), digest(reset.numpy().tobytes())) == (row['pack_id'], row['sample_ids'], row['inputs'], row['targets'], row['tokens_sha256'], row['loss_mask_sha256'], row['valid_mask_sha256'], row['reset_mask_sha256']), i

data.close()
print('512 independent exposure recounts and current sampler/mask/tensor replays agree', flush=True)
with (Path(__file__).parent / 'pi-inventory.txt').open('x') as log:
    subprocess.run(['sha256sum', '-c', freeze['pi_inventory']['path']], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
assert (Path(__file__).parent / 'pi-inventory.txt').read_text().count(': OK\n') == 14070
launch_path = ROOT / 'w3-matcheddose/gate-supervision/launcher.json'
launch = json.loads(launch_path.read_text())
for binding in (launch['runner'], launch['source']):
    records.append(checked(binding['path'], binding['sha256']))
result = dict(schema='emender-w3g-resume-preflight-v1', verified_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(), verification_script=checked(__file__, file_sha(__file__)), frozen_sources_verified=len(freeze['source_bindings']), preregistration_bindings_verified=len(prereg['bindings']), mask_recount_and_sampler_replay_steps=512, pi_payload_files_verified=14070, runtime_versions=freeze['versions'], records=records, no_gpu_read=True, fresh_checkpoint_entry_and_final_sha_verification='performed by existing guard only after eight-free-GPU window, before prepare/read', original_guard_pid=launch['pid'], original_guard_starttime=launch['starttime'])
with (receipts / 'gate-resume-preflight.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print('W3G resume preflight passed; no serving/gate read invoked', flush=True)
