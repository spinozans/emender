import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROPOSAL=ROOT/'configs/pi/e97-pi-native-repair-training-proposal-v1.json'

def run_audit(proposal_path,tmp):
 out=tmp/'audit.json'
 return subprocess.run([sys.executable,'scripts/audit_e97_pi_native_repair_proposal.py',
  '--proposal',str(proposal_path),'--output',str(out)],
  cwd=ROOT,capture_output=True,text=True)

def test_audit_accepts_frozen_proposal_and_rejects_tampering(tmp_path):
 assert PROPOSAL.exists()
 r=run_audit(PROPOSAL,tmp_path)
 assert r.returncode==0,(r.stdout,r.stderr)
 receipt=json.loads((tmp_path/'audit.json').read_text())
 assert receipt['status']=='qualified-proposal-not-authorized'
 assert receipt['optimizer_updates_authorized']==0 and receipt['checkpoint_promotion'] is False
 assert receipt['per_update_cohort_stratification'].startswith('verified')
 tampered=tmp_path/'tampered.json'
 data=json.loads(PROPOSAL.read_text());data['optimizer_updates_authorized']=3
 tampered.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
 r2=run_audit(tampered,tmp_path)
 assert r2.returncode!=0
 data2=json.loads(PROPOSAL.read_text());data2['automatic_retry']=True
 (tmp_path/'tampered2.json').write_text(json.dumps(data2,sort_keys=True)+'\n')
 assert run_audit(tmp_path/'tampered2.json',tmp_path).returncode!=0
