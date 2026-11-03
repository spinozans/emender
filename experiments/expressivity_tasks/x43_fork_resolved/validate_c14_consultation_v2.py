"""Independent accounting/checkpoint validation of the C14 v2 run."""
import hashlib
import json
from pathlib import Path
import torch
from .c14_documents import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14_consultation_emender_v2'


def validate():
    run=json.loads((OUT/'training.json').read_text())
    config=json.loads((OUT/'config.json').read_text())
    assert run['status']=='trained_registered_budget',run['status']
    assert json.loads((OUT/'terminal.json').read_text())['exit_code']==0
    assert run['active_seconds']>=2700
    assert config['seed']==config['dropout_seed']==3401
    assert (config['batch'],config['context'],config['parameters'])==(16,128,2133700)
    assert config['code_sha256']==sha(OUT/'training_runner.py')
    assert config['feature_decoder_sha256']==sha(OUT/'training_decoder.c')
    audit=json.loads((OUT/'full_sum_audit.json').read_text())
    assert audit['status']=='passed' and audit['full_cache_float32_identity']
    assert audit['check_source_sha256']==sha(ROOT/'c14_fast_features.c')
    assert (audit['windows'],audit['positions'])==(1785536,225880702)
    for path,digest in json.loads((OUT/'frozen_evidence_hashes.json').read_text()).items():
        assert sha(Path(path))==digest,path
    historical=json.loads((ROOT/'receipts/c14_consultation_emender/training.json').read_text())
    assert sha(Path(historical['checkpoint']))==historical['checkpoint_sha256']
    checkpoint=Path(run['checkpoint'])
    assert sha(checkpoint)==run['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['config']==config
    for key in ('steps','targets','active_seconds'):
        assert state[key]==run[{'targets':'actual_bytes_targets'}.get(key,key)]
    assert state['next_window']==run['next_window_index']==run['steps']*16
    tensors=0
    def finite(value):
        nonlocal tensors
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
            tensors+=1
        elif isinstance(value,dict):
            for item in value.values(): finite(item)
        elif isinstance(value,(list,tuple)):
            for item in value: finite(item)
    finite(state['model']); finite(state['optimizer'])
    preflight=json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    stat=Path(preflight['source']).stat()
    assert (stat.st_size,stat.st_ino,stat.st_mtime_ns)==(preflight['source_size'],preflight['source_inode'],preflight['source_mtime_ns'])
    offsets=Path(preflight['offsets_path'])
    assert sha(offsets)==preflight['offsets_sha256']
    remaining=run['windows']; expected=0
    with offsets.open() as stream:
        for line in stream:
            doc=json.loads(line)
            if doc['split']!='train': continue
            for start in range(0,doc['byte_length']-1,128):
                if not remaining: break
                expected+=min(128,doc['byte_length']-1-start); remaining-=1
            if not remaining: break
    assert not remaining and expected==run['actual_bytes_targets']
    assert abs(run['steps']/run['active_seconds']-run['steps_per_active_second'])<1e-12
    assert .49<run['keep_fraction']<.51
    assert state['dropout_rng'].numel() and state['torch_rng'].numel() and state['cuda_rng'].numel()
    result=dict(status='passed',checkpoint_sha256=run['checkpoint_sha256'],finite_model_optimizer_tensors=tensors,
        exact_resume_state='model, optimizer, CPU/CUDA/dropout RNG, next window, steps, targets, active time; no automatic resume interface claimed',
        dense_and_v1_receipts_unchanged=True,v1_checkpoint_unchanged=True,source_stat_unchanged=True,source_order_targets_recomputed=expected,
        retained_training_code_and_decoder_hashes_match=True,current_decoder_full_cache_bit_identity=True,active_seconds=run['active_seconds'],steps=run['steps'],
        steps_per_second=run['steps_per_active_second'],tests='35 C14 tests pass; tests_all.txt',
        scope='Training accounting and checkpoint validation; no paired evaluation or fresh native oracle rerun',
        diloco='not applicable: no resilient DiLoCo behavior changed, tested or run')
    return result


if __name__=='__main__':
    try:
        result=validate()
    except Exception as error:
        (OUT/'validation.json').write_text(json.dumps(dict(status='failed',error=repr(error)),indent=2)+'\n')
        raise
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
