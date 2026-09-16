from types import SimpleNamespace
from scripts.eval_e97_pi_native_stage_b import grade

def test_stage_b_grades_required_edit_outcome_without_unchanged_source_rule():
 case={'id':'choice-edit-file','expected_final':'done','expected_first_action':'edit'}
 bridge=SimpleNamespace(final='done',history=[{'role':'toolResult'}])
 action={'name':'edit','arguments':{}}
 result=grade(case,bridge,[action],{'state.txt':'alpha\n','result.txt':None},{'state.txt':'beta\n','result.txt':None})
 assert result['success'] is True

def test_stage_b_rejects_missing_first_frame():
 case={'id':'choice-bash-command','expected_final':'1723','expected_first_action':'bash'}
 bridge=SimpleNamespace(final='1723',history=[])
 assert grade(case,bridge,[None],{}, {})['success'] is False
