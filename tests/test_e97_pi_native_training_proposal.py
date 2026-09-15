import json
from pathlib import Path

def test_pi_native_training_proposal_is_non_authorizing_and_bounded():
 proposal=json.loads(Path('configs/pi/e97-pi-native-curriculum-training-proposal-v1.json').read_text())
 assert proposal['status']=='frozen-proposal-not-authorized'
 assert proposal['operator_internal_training_authorized'] is False
 assert proposal['optimizer_updates_authorized']==0
 assert proposal['packing_authorized'] is False
 assert proposal['checkpoint_promotion'] is False
 assert proposal['automatic_retry'] is False and proposal['automatic_expansion'] is False
 assert proposal['proposed_updates']==32 and proposal['proposed_learning_rate']==1e-5
 assert proposal['scheduled_unique_records']==sum(proposal['scheduled_source_unique_records'].values())
 assert proposal['scheduled_assistant_targets']==sum(proposal['scheduled_source_targets'].values())
 assert proposal['scheduled_source_unique_records']['pi-native-curriculum']>=1200
