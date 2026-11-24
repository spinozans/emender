from pathlib import Path
from .k15_horizon import serve,POLICIES


def test_resident_chosen_prefix_parity():
    raw=(b'apple apricot paper apple paper.'*70)[:1056]
    exe=Path.home()/'sxgc/xsa/target/debug/xsa'
    for mode in POLICIES:
        rows,stats=serve(exe,mode,raw,1024,True)
        assert stats['work']['seal']==1 and stats['work']['append']==64
        for lag in (0,5,31):
            assert rows[lag]==serve(exe,mode,raw[:1024+lag],1024,True)[0][-1]


def test_distribution_mix_including_count_prior():
    from collections import Counter
    import numpy as np
    from .k15_horizon_decode import interpolate
    base=np.full(256,1/256); ev=Counter({7:3,8:1})
    for weight,a in ((.05,.05),('count',.8)):
        z=interpolate(base,ev,weight); assert abs(z.sum()-1)<1e-12 and z[7]==(1-a)/256+a*.75
    assert np.array_equal(interpolate(base,Counter(),.5),base)


def test_precommitted_long_gain_interaction_and_cost_are_joint_requirements():
    from experiments.expressivity_tasks.x45_matrix_emender.k15_horizon_run import evaluate
    cfg={h:dict(pair=['newest','base'],weights={'newest':0,'base':0}) for h in ['1024','4096']}
    def gate(short,long,cost=0):
        row={}
        for h,gain in [('1024',short),('4096',long)]:
            row[h]=dict(scores={'base':[[1.]],'newest':[[1.-gain]]},neural=[1.],stats={'newest':{'total':cost}})
        return evaluate([row]*8,cfg)['passed']
    assert gate(0,.002) and not gate(0,.0005) and not gate(.002,.002) and not gate(0,.002,1.01)
