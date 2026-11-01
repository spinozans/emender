"""Independent substring-scan validation of the X43 part-2 receipt."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/part2'


def main():
    raw=(ROOT/'input.txt').read_bytes()
    report=json.loads((OUT/'fusion.json').read_text())
    counts=np.load(OUT/'counts.npz'); pos=counts['position']; split=report['split']
    half=report['calibration_targets']; test=np.arange(len(pos))>=half
    assert np.array_equal(pos,np.arange(split+1,len(raw)))
    assert report['calibration_position_range'][1]+1==report['final_test_position_range'][0]
    checks=0
    for k in np.random.default_rng(4302).choice(len(pos),100,replace=False):
        i=int(pos[k]); length=int(counts['context_length'][k]); context=raw[i-length:i]
        occurrences=[]
        if not context:
            occurrences=list(range(len(raw)))
        else:
            start=0
            while True:
                start=raw.find(context,start)
                if start<0: break
                j=start+length
                if j<len(raw): occurrences.append(j)
                start+=1
        assert len(occurrences)>=2
        for scope,boundary in [('train',split),('corpus',len(raw))]:
            eligible=[j for j in occurrences if j<boundary]
            c=np.bincount([raw[j] for j in eligible],minlength=256)
            q=c/c.sum() if c.sum() else np.zeros(256)
            assert np.isclose(q.sum(),1 if eligible else 0)
            assert counts[f'q_{scope}'][k]==q[raw[i]]
            assert counts[f'support_{scope}'][k]==np.count_nonzero(c)
            assert counts[f'total_{scope}'][k]==len(eligible)
            checks+=1
    for arm,m in report['models'].items():
        ev=np.load(OUT/f'{arm}_prediction.npz'); p=ev['p']
        retained=np.load(ROOT/f'{arm}_per_position.npz')
        assert np.allclose(-np.log(p),retained['nll'],atol=2e-4,rtol=0)
        for scope,r in m['scopes'].items():
            gates=np.load(OUT/f'{arm}_{scope}_gates.npz')
            empty=counts[f'support_{scope}']==0; forced=~counts['fork']
            assert np.all(gates['learned_index_weight'][empty]==0)
            assert np.array_equal(gates['learned_target_probability'][empty],p[empty])
            assert np.array_equal(gates['learned_target_probability'][forced],p[forced])
            covered_forced=forced&~empty
            assert np.all(gates['learned_walk_target_probability'][covered_forced]==1.)
            for key,name in [('learned_target_probability','learned'),('learned_walk_target_probability','learned_walk')]:
                assert np.isclose(-np.log(gates[key][test]).mean(),r['variants'][name]['overall'],rtol=0,atol=1e-12)
    part1=json.loads((ROOT/'receipts/part1c/decomposition.json').read_text())
    assert int(np.sum((counts['q_train']==0)&(counts['support_train']>0))) == sum(v['oracle_zero_probability_targets'] for v in part1['baselines'].values())
    result=dict(status='passed',independent_substring_count_checks=checks,
        checks=['held-out positions contiguous and calibration/test disjoint','200 independent overlapping substring count/distribution checks','frozen target probabilities replay retained NLL','empty-support abstention exact','covered forced walk probability one','final-test NLL independently recomputed from persisted per-target probabilities','zero-count totals agree with part1c'])
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
