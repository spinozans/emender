from types import SimpleNamespace
from scripts.eval_e97_pi_native_stage_b import grade

def test_stage_b_grades_required_edit_outcome_without_unchanged_source_rule():
 case={'id':'choice-edit-file','expected_final':'done','expected_first_action':'edit'}
 bridge=SimpleNamespace(final='done',history=[{'role':'toolResult','isError':False,'content':[{'text':'ok'}]}])
 action={'name':'edit','arguments':{}}
 result=grade(case,bridge,[action],{'state.txt':'alpha\n','result.txt':None},{'state.txt':'beta\n','result.txt':None})
 assert result['success'] is True

def test_stage_b_rejects_missing_first_frame():
 case={'id':'choice-bash-command','expected_final':'1723','expected_first_action':'bash'}
 bridge=SimpleNamespace(final='1723',history=[])
 assert grade(case,bridge,[None],{}, {})['success'] is False


def test_stage_b_run_metadata_uses_non_default_frozen_checkpoint(tmp_path, monkeypatch):
 import json
 import sys
 from pathlib import Path
 from scripts import eval_e97_pi_native_stage_b as stage_b

 checkpoint_sha = 'a' * 64
 assert checkpoint_sha != stage_b.CHECKPOINT_SHA
 cases = [{'stage':'B', 'id':f'case-{i}', 'files':{}, 'prompt':'finish',
           'expected_final':'done', 'expected_first_action':'finish',
           'family':'test', 'delay_tokens':0} for i in range(14)]
 panel = tmp_path / 'panel.json'
 panel.write_text(json.dumps({'cases':cases, 'tools':[], 'episode_seconds':1}))
 manifest = tmp_path / 'manifest.json'
 manifest.write_text(json.dumps({'model_visible_tools':[]}))
 plan = {'panel':str(panel), 'checkpoint':'non-default.pt', 'args_json':'args.json',
         'parent_model_source_panel':'source-panel.json', 'pi_inventory':'inventory.json',
         'tool_manifest':str(manifest), 'stage_b_tools':'tools.ts', 'cli_image':'image',
         'pi_bin':'pi', 'case_ids':[case['id'] for case in cases],
         'tokenizer_vocabulary_sha256':'vocab-sha'}
 for key in ('panel', 'checkpoint', 'args', 'parent_model_source_panel',
             'pi_inventory', 'tool_manifest', 'stage_b_tools', 'cli_image'):
  plan[key + '_sha256'] = checkpoint_sha if key == 'checkpoint' else 'pin'
 plan_path = tmp_path / 'plan.json'
 plan_path.write_text(json.dumps(plan))
 monkeypatch.setattr(stage_b, 'sha', lambda path: checkpoint_sha if str(path) == plan['checkpoint'] else 'pin')
 monkeypatch.setattr(stage_b, 'PACKAGES', {})
 monkeypatch.setattr(stage_b, 'vocabulary', lambda enc: (None, 'vocab-sha'))
 monkeypatch.setattr(stage_b.tiktoken, 'get_encoding', lambda name: None)
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
 model = SimpleNamespace(eval=lambda:None, parameters=lambda:[])
 loaded = SimpleNamespace(model=model)
 loads = []
 def load(checkpoint, **kwargs):
  loads.append(checkpoint)
  return loaded
 monkeypatch.setitem(sys.modules, 'ndm.e97', SimpleNamespace(load_e97_checkpoint=load))
 monkeypatch.setitem(sys.modules, 'scripts.audit_e97_live_actor_capture',
                     SimpleNamespace(fingerprint=lambda m:'fingerprint', runtime=lambda l,d:{}))
 monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
  cuda=SimpleNamespace(set_device=lambda d:None, max_memory_allocated=lambda:0),
  backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())),
  device=lambda *args:None, bfloat16='bf16'))
 bridge = SimpleNamespace(final='done', history=[], generations=[], reason='finish',
                          failed=False, closed=True, close_verified=True,
                          episode=SimpleNamespace(text=lambda:'', source_messages=lambda:[]))
 monkeypatch.setattr(stage_b, 'NativePiToolBridge', lambda *args:bridge)
 monkeypatch.setattr(stage_b, 'actions', lambda *args:[{'name':'finish', 'arguments':{}}])
 def serve(bridge, output, **kwargs):
  output.mkdir()
  (output / 'pi-events-private.jsonl').write_text('')
  return {'model_failure_verified':False}
 monkeypatch.setattr(stage_b, 'serve_pi_native_tools', serve)
 output = tmp_path / 'results'
 stage_b.run(SimpleNamespace(plan=plan_path, plan_sha='pin', output=output))
 summary = json.loads((output / 'summary.json').read_text())
 before = json.loads((output / 'model-before-private.json').read_text())
 assert loads == [plan['checkpoint']]
 assert summary['checkpoint_sha256'] == plan['checkpoint_sha256'] == checkpoint_sha
 assert before['checkpoint_sha256'] == plan['checkpoint_sha256']
