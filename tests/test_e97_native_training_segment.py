from copy import deepcopy
from pathlib import Path
import pytest
from scripts.prepare_e97_native_training_segment import check_steps,render

TEMPLATE='''ROOT=/baseline
DATA=/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-training-mix-v1
MIX=/mnt/nvme2n1/erikg/sft/e97-4b-native-training-smoke-mix-v1
eval "$(bash scripts/gpu_lease.sh acquire 8 --no-wait)"
nvidia-smi --query-gpu=index
--new-stage-from "$PARENT" --new-stage-weight-mode train
--authority-sha256 69f26726e32523dd840e363a5b9b7663b5e9ed37bd48a68e0bdbc9827d98a2c7
--pack-sha256 51c8dc8cc06a895ea39be780d211af6d645e48117fe1be9c70f93088b0b393a6
--output-root "$ROOT/checkpoints"
--steps 8 --save-every 4 --keep-checkpoints 3 --diloco-k 4 --island-size 8 --disable-diloco-merge
--sampler-key 974117 --lr 5e-5
--optimizer-precision bf16-sr-candidate --sr-seed 927413 --loss-chunk-size 128
CUDA_VISIBLE_DEVICES='' "$EMENDER_PYTHON" - "$ROOT" "$DATA" <<'PY'
old collector
PY
'''


def test_first_and_resume_render_preserve_numerical_flags_and_shared_output():
    args=[TEMPLATE,Path('/baseline'),Path('/phase'),Path('/run'),Path('/data'),Path('/mix'),'a'*64,'b'*64,128,Path('/controller')]
    first=render(*args)
    assert '--steps 128 --save-every 128 --keep-checkpoints 32 --diloco-k 4' in first
    assert '--output-root "$RUN/checkpoints"' in first
    assert '--optimizer-precision bf16-sr-candidate --sr-seed 927413 --loss-chunk-size 128' in first
    assert '--new-stage-from "$PARENT"' in first and '--lr 1e-5' in first
    assert 'old collector' not in first and ' collect --phase /phase' in first
    resumed=render(*args,start=128,anchor=Path('/run/checkpoints/committed.pt'))
    assert '--resume "$RUN/checkpoints/latest.pt"' in resumed
    assert '--new-stage-from' not in resumed and 'readlink -f' in resumed
    with pytest.raises(ValueError):render(*([TEMPLATE+TEMPLATE]+args[1:]))


def fixture():
    planned=[dict(update=i,global_tokens=10*i,global_targets=2*i,rank_sample_ids=[[str(i)] for _ in range(8)]) for i in (1,2,3)]
    steps=[dict(**planned[i-1],event='step',total_tokens=10*i*(i+1)//2,total_targets=i*(i+1),loss=1.,grad_norm=2.) for i in (2,3)]
    return planned,steps+[dict(event='complete')]


def test_resumed_clock_and_sample_coverage():
    planned,events=fixture()
    assert len(check_steps(events,planned,1,3))==2
    for key,value in [('total_tokens',0),('total_targets',0),('rank_sample_ids',[['wrong']]),('loss',float('nan')),('grad_norm',float('inf'))]:
        changed=deepcopy(events);changed[0][key]=value
        with pytest.raises(ValueError):check_steps(changed,planned,1,3)
    with pytest.raises(ValueError):check_steps(events[:-1],planned,1,3)
    with pytest.raises(ValueError):check_steps(events+[events[0]],planned,1,3)
