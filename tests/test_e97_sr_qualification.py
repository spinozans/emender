import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts.aggregate_e97_sr_precision import aggregate


def fixture(root, world=4):
    for rank in range(world):
        modes=[]
        for mode in ('ddp','diloco','hybrid'):
            data=f'{mode}/{rank}'.encode()
            (root/f'{mode}-rank-{rank:02d}.pt').write_bytes(data)
            digest=hashlib.sha256(data).hexdigest()
            (root/f'{mode}-rank-{rank:02d}-expected.json').write_text(json.dumps({'checkpoint_sha256':digest,'final_state_sha256':digest}))
            modes.append({'mode':mode,'status':'passed','island_size':{'ddp':world,'diloco':1,'hybrid':world//2}[mode],
                          'merge_count':0 if mode=='ddp' else 2,'final_state_sha256':digest})
        for phase in ('produce','resume'):
            (root/f'{phase}-rank-{rank:02d}.json').write_text(json.dumps({
                'schema':'emender-sr-cuda-distributed-v1','status':'passed','phase':phase,'rank':rank,
                'world_size':world,'source_sha256':'source','reports':modes,
                'device_identity':{'local_rank':rank,'current_cuda_device':rank,'visible_devices':'0,1,2,3'}}))
    (root/'cuda-arithmetic-memory.json').write_text(json.dumps({'status':'passed','arithmetic':{'counter_cpu_cuda_and_bucket_parity':True}}))


def test_aggregate_and_checkpoint_tamper(tmp_path):
    fixture(tmp_path)
    assert aggregate(tmp_path,4,'source')['rank_reports']==8
    (tmp_path/'ddp-rank-00.pt').write_bytes(b'corrupt')
    with pytest.raises(AssertionError):aggregate(tmp_path,4,'source')


def test_routing_is_not_inferred_from_rank(tmp_path):
    fixture(tmp_path)
    p=tmp_path/'resume-rank-03.json';r=json.loads(p.read_text())
    r['device_identity']['current_cuda_device']=0;p.write_text(json.dumps(r))
    with pytest.raises(AssertionError):aggregate(tmp_path,4,'source')


def test_partial_resume_cannot_pass(tmp_path):
    fixture(tmp_path)
    (tmp_path/'resume-rank-02.json').unlink()
    with pytest.raises(FileNotFoundError):aggregate(tmp_path,4,'source')


def test_qualifier_selects_local_device_before_process_group():
    p=Path(__file__).parents[1]/'scripts/qualify_e97_sr_precision.py'
    tree=ast.parse(p.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
    select=next(n for n in calls if ast.unparse(n.func)=='torch.cuda.set_device')
    init=next(n for n in calls if ast.unparse(n.func)=='dist.init_process_group')
    assert ast.unparse(select.args[0])=='local_rank' and select.lineno<init.lineno
