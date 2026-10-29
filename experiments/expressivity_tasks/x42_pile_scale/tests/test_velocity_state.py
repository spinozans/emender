import numpy as np
import pytest
from experiments.expressivity_tasks.x42_pile_scale.cmaes import CMAES
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import save_generation,restore_generation


def test_generation_restart_preserves_population_and_future(tmp_path):
    es=CMAES([.5]*4,population=12);population=es.ask()
    path=tmp_path/'generation.json';save_generation(path,es,'hash',{'0':{'score':1}})
    loaded,completed=restore_generation(path,'hash')
    np.testing.assert_array_equal(loaded.x,population)
    assert completed=={'0':{'score':1}}
    fitness=(population**2).sum(1);es.tell(fitness);loaded.tell(fitness)
    np.testing.assert_array_equal(es.C,loaded.C)
    np.testing.assert_array_equal(es.ask(),loaded.ask())
    with pytest.raises(ValueError):restore_generation(path,'wrong')


def test_sigkill_resume(tmp_path):
    import subprocess,sys,os,signal,time
    state=tmp_path/'generation.json'
    code='''import time,sys
from experiments.expressivity_tasks.x42_pile_scale.cmaes import CMAES
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import save_generation
es=CMAES([.5]*4,population=12);es.ask()
save_generation(sys.argv[1],es,'bound',{'2':{'score':4}})
print('durable',flush=True)
time.sleep(60)
'''
    child=subprocess.Popen([sys.executable,'-c',code,str(state)],stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='durable'
        os.kill(child.pid,signal.SIGKILL);assert child.wait(timeout=10)==-signal.SIGKILL
        es,completed=restore_generation(state,'bound')
        baseline=CMAES([.5]*4,population=12)
        np.testing.assert_array_equal(es.x,baseline.ask())
        assert completed=={'2':{'score':4}}
    finally:
        if child.poll() is None:child.kill();child.wait()
