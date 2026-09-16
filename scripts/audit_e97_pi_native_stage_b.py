#!/usr/bin/env python3
"""Audit immutable Stage-B Pi-native outcomes without regrading them."""
import argparse,hashlib,json
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit(a):
 plan=json.loads(a.plan.read_text());panel=json.loads(Path(plan['panel']).read_text());summary=json.loads((a.results/'summary.json').read_text());terminal=json.loads(a.terminal.read_text());before=json.loads((a.results/'model-before-private.json').read_text());after=json.loads((a.results/'model-after-private.json').read_text());cases=[c for c in panel['cases'] if c['stage']=='B']
 corrected='stage_b_tools' in plan
 expected_terminal={'original_exit':0,'audited_exit':0,'automatic_retries':0,'optimizer_updates':0,**({'corrected_protocol':True} if corrected else {})}
 if sha(a.plan)!=summary['plan_sha256'] or sha(plan['panel'])!=plan['panel_sha256'] or sha(plan['checkpoint'])!=plan['checkpoint_sha256'] or {k:terminal.get(k) for k in expected_terminal}!=expected_terminal:raise ValueError('authority/terminal')
 if corrected and (sha(plan['stage_b_tools'])!=plan['stage_b_tools_sha256'] or sha(plan['cli_image'])!=plan['cli_image_sha256']):raise ValueError('corrected executor authority')
 if before['fingerprint']!=after['fingerprint'] or not after['unchanged'] or not after['no_gradients'] or not after['all_parameters_bf16']:raise ValueError('model changed')
 if len(cases)!=14 or len(summary['rows'])!=14 or [c['id'] for c in cases]!=[r['id'] for r in summary['rows']]:raise ValueError('coverage/order')
 receipts=[];valid=correct=successes=0
 for case,row in zip(cases,summary['rows']):
  path=a.results/case['id']/'episode-private.json';generation=a.results/case['id']/'generations-private.json';episode=json.loads(path.read_text())
  if episode['id']!=case['id'] or not episode['closed'] or not (episode['close_verified'] or episode['model_failure_verified']) or not generation.exists():raise ValueError('episode closure')
  public={k:episode[k] for k in ('id','family','delay_tokens','model_generations','reason','failed','close_verified','model_failure_verified','grade')}
  if public!=row:raise ValueError('summary projection')
  first=episode['actions'][0] if episode['actions'] else None;observed_valid=first is not None;observed_correct=bool(first and first['name']==case['expected_first_action'])
  if row['grade']['checks']['valid_first_frame']!=observed_valid or row['grade']['checks']['first_action_correct']!=observed_correct:raise ValueError('first action reconstruction')
  valid+=observed_valid;correct+=observed_correct;successes+=row['grade']['success'];receipts.append({'id':case['id'],'episode_sha256':sha(path),'generations_sha256':sha(generation)})
 if (valid,correct,successes)!=(summary['valid_first_frames'],summary['correct_first_actions'],summary['successes']) or summary['gate']['passed'] or valid>=12 or correct>=10 or summary['automatic_retries'] or summary['optimizer_updates'] or summary['checkpoint_promotion']:raise ValueError('negative gate accounting')
 report={'schema':'emender-e97-pi-native-stage-b-audit-v1','status':'qualified-negative-gate-failure-corrected-protocol' if corrected else 'qualified-negative-gate-failure','panel_sha256':plan['panel_sha256'],'plan_sha256':sha(a.plan),'checkpoint_sha256':plan['checkpoint_sha256'],'corrected_protocol':corrected,'episodes':14,'valid_first_frames':valid,'valid_first_frame_required':12,'correct_first_actions':correct,'correct_first_action_required':10,'successes_as_frozen':successes,'gate_passed':False,'episode_receipts':receipts,'panel_expected_final_inconsistencies_not_regraded':['choice-read-explicit expects done although prompt requests facts.txt content','choice-grep-symbol expects done although prompt requests relative path'],'automatic_retries':0,'optimizer_updates':0,'checkpoint_promotion':False,'promotion_blocked':True,'diagnostic_evaluation_allowed':True,'further_optimizer_updates_authorized':False,'checker_sha256':sha(__file__)}
 a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_STAGE_B_AUDIT_NEGATIVE',valid,correct,successes,sha(a.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
