"""Attest transformer cache, training accounting, checkpoint and paired evaluation."""
import argparse
import json
from pathlib import Path
import torch
from .c14_documents import sha
from .c11 import Boosted
from .c6 import count

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14_transformer'


def validate(evaluation=False):
    reconciliation=json.loads((OUT/'cache_reconciliation.json').read_text())
    assert reconciliation['status']=='passed' and reconciliation['training_authorized']
    assert (reconciliation['windows'],reconciliation['targets'])==(7685648,972375733)
    for path,digest in json.loads((OUT/'frozen_evidence_hashes.json').read_text()).items():
        assert sha(Path(path))==digest,path
    for phase in reconciliation['phases']:
        assert sha(Path('/home/erikg/c14_consultation_features')/('registered_'+phase['phase'])/'journal.jsonl')==phase['journal_sha256']
    config=json.loads((OUT/'config.json').read_text())
    run=json.loads((OUT/'training.json').read_text())
    assert json.loads((OUT/'terminal.json').read_text())['exit_code']==0
    assert run['status']=='trained_registered_budget' and run['active_seconds']>=2700
    assert config['code_sha256']==sha(ROOT/'c14_transformer_train.py')==sha(OUT/'training_runner.py')
    assert config['feature_decoder_sha256']==sha(OUT/'training_decoder.c')
    for path,digest in config['kernel_state_sha256'].items():
        assert sha(ROOT/path)==digest,path
    assert config['feature_decoder_sha256']==sha(ROOT/'c14_fast_features.c')==reconciliation['decoder_sha256']
    assert config['cache_reconciliation_sha256']==sha(OUT/'cache_reconciliation.json')
    assert (config['seed'],config['dropout_seed'],config['batch'],config['context'],config['parameters'])==(3401,3401,16,128,2133748)
    assert .49<run['keep_fraction']<.51
    checkpoint=Path(run['checkpoint'])
    assert sha(checkpoint)==run['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['config']==config
    for key in ('steps','targets','active_seconds'):
        assert state[key]==run[{'targets':'actual_bytes_targets'}.get(key,key)]
    assert state['next_window']==run['next_window_index']==run['windows']==run['steps']*16
    def finite(value):
        if torch.is_tensor(value):assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(list,tuple)):
            for v in value:finite(v)
    finite(state['model']);finite(state['optimizer'])
    model=Boosted('transformer',2000324)
    model.load_state_dict(state['model'],strict=True)
    assert count(model)==2133748
    for key in ('dropout_rng','torch_rng','cuda_rng'):assert state[key].numel()>0
    remaining=run['windows'];targets=0
    offsets=Path('/home/erikg/c14_dense_bytes/pile_offsets.jsonl')
    assert sha(offsets)==reconciliation['offsets_sha256']
    with offsets.open() as f:
        for line in f:
            d=json.loads(line)
            if d['split']!='train':continue
            n=min(remaining,(max(d['byte_length']-1,0)+127)//128)
            targets+=min(n*128,max(d['byte_length']-1,0))
            remaining-=n
            if not remaining:break
    assert not remaining and targets==run['actual_bytes_targets']
    result=dict(status='passed',steps=run['steps'],targets=targets,active_seconds=run['active_seconds'],
        checkpoint_sha256=run['checkpoint_sha256'],frozen_evidence_unchanged=True,finite_model_optimizer=True,
        source_order_target_accounting=True,diloco='not applicable; independent single GPU head training; no resilient training behavior')
    if evaluation:
        e=ROOT/'receipts/c14_transformer_eval'
        resume=json.loads((e/'resume.json').read_text())
        assert resume['status']=='complete' and (resume['windows'],resume['targets'])==(3431242,434204084)
        assert resume['config']['checkpoint_hashes']['v2']==run['checkpoint_sha256']
        table=json.loads((e/'table.json').read_text())
        for half,n in [('final_test',214097022),('calibration',220107062),('all',434204084)]:
            for arm in ('dense','v2_on','v2_off'):
                assert table[f'{arm}/{half}/total']['targets']==n
                for key in ('total','window_reset_1_16','window_interior_17_128','supported_target','zero_target','fork'):
                    row=table[f'{arm}/{half}/{key}']
                    assert row['targets']>0 and row['nll']==row['total_nll']/row['targets']
                    assert row['targets']==table[f'dense/{half}/{key}']['targets']
        emender=json.loads((ROOT/'receipts/c14_paired_eval/table.json').read_text())
        checks=0
        partitions=[('window_reset_1_16','window_interior_17_128'),('supported_target','zero_target'),
                    ('wikipedia_support','empty_support'),('supported_target','nonempty_zero_target','empty_support')]
        for arm in ('dense','v2_on','v2_off'):
            for half in ('all','calibration','final_test'):
                total=table[f'{arm}/{half}/total']
                for partition in partitions:
                    rows=[table[f'{arm}/{half}/{k}'] for k in partition]
                    assert sum(r['targets'] for r in rows)==total['targets']
                    assert abs(sum(r['total_nll'] for r in rows)-total['total_nll'])<1e-4
                    checks+=1
                for key in ('total','window_reset_1_16','window_interior_17_128','wikipedia_support','empty_support','supported_target','nonempty_zero_target','zero_target','agreement','argmax_agreement','fork'):
                    row=table[f'{arm}/{half}/{key}']
                    assert row['targets']==emender[f'dense/{half}/{key}']['targets']
                    if half=='all':
                        parts=[table[f'{arm}/{h}/{key}'] for h in ('calibration','final_test')]
                        assert sum(r['targets'] for r in parts)==row['targets']
                        assert abs(sum(r['total_nll'] for r in parts)-row['total_nll'])<1e-4
                        checks+=1
        assert resume['config']['code_sha256']==sha(ROOT/'c14_transformer_eval.py')
        dense=json.loads((ROOT/'receipts/c14_dense/training.json').read_text())['models'][1]
        assert abs(table['dense/final_test/total']['nll']-dense['heldout']['final_test']['nll'])<1e-4
        assert json.loads((e/'terminal.json').read_text())['exit_code']==0
        result['partition_sum_checks']=checks
        result['paired_evaluation']='passed; three modes, frozen document-disjoint halves, matching denominators'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--eval',action='store_true');args=p.parse_args()
    try:result=validate(args.eval)
    except Exception as error:
        result=dict(status='failed',error=repr(error))
        (OUT/('validation_eval.json' if args.eval else 'validation.json')).write_text(json.dumps(result,indent=2)+'\n')
        raise
    (OUT/('validation_eval.json' if args.eval else 'validation.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
