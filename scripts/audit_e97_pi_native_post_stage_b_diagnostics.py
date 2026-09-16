#!/usr/bin/env python3
"""Reconstruct the matched post-Stage-B diagnostic comparison."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(a):
 root=a.root;terminal=json.loads((root/'terminal.json').read_text())
 if terminal!={'original_exit':0,'audited_exit':0,'optimizer_updates':0,'promotion':False}:raise ValueError('terminal')
 ep=json.loads((root/'execution/panel.json').read_text());es=json.loads((root/'execution/summary.json').read_text());lp=json.loads((root/'learning/panel.json').read_text());ls=json.loads((root/'learning/summary.json').read_text())
 expected={'pi-native-y','pi-native-x','bridge-y-control','bridge-x-control'}
 if set(es['models'])!=expected or set(ls['models'])!=expected or es['checkpoint_promotion'] or ls['capability_promotion']:raise ValueError('model/result coverage')
 if any(m['sha256']!=sha(m['checkpoint']) for p in (ep,lp) for m in p['models']):raise ValueError('checkpoint identity')
 cohorts={c['id']:c['cohort'] for c in ep['cases']};execution={}
 for model,result in es['models'].items():
  if result['episodes']!=96 or len(result['outcomes'])!=96:raise ValueError('execution coverage')
  totals=Counter();success=Counter()
  for row in result['outcomes']:
   cohort=cohorts[row['id']];totals[cohort]+=1;success[cohort]+=bool(row['grade']['success'])
  execution[model]={'successes':result['successes'],'episodes':result['episodes'],'cohorts':{k:{'successes':success[k],'episodes':totals[k]} for k in sorted(totals)}}
 learning={m:{k:v for k,v in rows.items()} for m,rows in ls['models'].items()}
 if execution['pi-native-x']['successes'] or execution['pi-native-y']['successes'] or execution['bridge-x-control']['successes']!=67 or execution['bridge-y-control']['successes']!=67:raise ValueError('comparison accounting')
 report={'schema':'emender-e97-pi-native-post-stage-b-diagnostic-audit-v1','status':'passed-negative-diagnostic','execution_panel_sha256':sha(root/'execution/panel.json'),'execution_summary_sha256':sha(root/'execution/summary.json'),'learning_panel_sha256':sha(root/'learning/panel.json'),'learning_summary_sha256':sha(root/'learning/summary.json'),'execution':execution,'learning':learning,'optimizer_updates':0,'promotion':False,'protected_outputs_training_eligible':False,'checker_sha256':sha(__file__)}
 a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_POST_STAGE_B_DIAGNOSTIC_AUDIT',sha(a.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
