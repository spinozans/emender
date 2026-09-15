import json,tiktoken
from scripts.build_e97_pi_native_baselines import DELAYS,TOOL_MANIFEST_REF,build,filler

def manifest():return json.load(open('configs/pi/e97-active-tool-surface-v1.json'))

def test_filler_has_exact_requested_token_distance():
 enc=tiktoken.get_encoding('p50k_base')
 for n in (128,1024,8192):assert len(enc.encode_ordinary(filler(n,str(n),enc)))==n

def test_panel_has_tiered_copy_binding_path_and_tool_choice_coverage():
 assert TOOL_MANIFEST_REF=='configs/pi/e97-active-tool-surface-v1.json'
 cases=build(manifest());assert len(cases)==26 and sum(c['stage']=='A' for c in cases)==12
 for family in ('copy','bind','path'):
  selected=[c for c in cases if c['family']==family];assert [c['delay_tokens'] for c in selected]==list(DELAYS)
  assert all(c['initial_prompt_tokens']<=65536 for c in selected)
 assert {c['expected_first_action'] for c in cases if c['family']=='tool-choice'}=={'finish','fffind','web_search','read','ffgrep','bash','edit','write','process','source_check','fetch_content'}
