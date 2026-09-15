import json
from scripts.audit_e97_repository_discovery_proposal import inventory


def test_inventory_counts_supervision_suffix_and_authentic_receipts(tmp_path):
 results=[]
 for i in range(8):
  family='scale_all' if i<4 else 'sum_all';name=f'training-discovery-{family}-{i:02d}'
  start=i%2; d=tmp_path/name;d.mkdir()
  episode=dict(training_eligible=False,generations=[{'token_ids':[1,2,3]},{'token_ids':[4,5]}],calls=[])
  (d/'episode-private.json').write_text(json.dumps(episode))
  results.append(dict(name=name,supervise_assistant_from=start,authored_failure_prefix_turns=start,
   training_eligible=False,evaluation_trajectory_reused=False,native_calls=5,native_errors=1))
 (tmp_path/'summary.json').write_text(json.dumps({'results':results}))
 actual,files=inventory(tmp_path)
 assert actual['records']==8 and actual['families']=={'scale_all':4,'sum_all':4}
 assert actual['assistant_turns']==16 and actual['supervised_assistant_turns']==12
 assert actual['masked_authored_assistant_turns']==4 and actual['supervised_target_tokens']==28
 assert actual['native_calls']==40 and actual['authentic_errors']==8 and len(files)==8
