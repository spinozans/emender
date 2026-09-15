from copy import deepcopy
import pytest
from scripts.prepare_e97_repository_discovery import fixtures, control, validate_disjoint
from scripts.e97_pi_repository_probes import repository_probes, inspect_snapshot
from scripts.qualify_e97_pi_native_transport import texts


def test_separate_preparation_and_truthful_failure_labels():
    rows=fixtures(); assert len(rows)==8
    assert validate_disjoint(rows,repository_probes())['exact_prompt_file_and_repair_overlap'] is False
    assert sum(c['recovery'] for c in rows)==4
    for c in rows:
        assert not c['training_eligible'] and not c['genuine_model_failure'] and not c['evaluation_trajectory_reused']
        assert c['supervise_assistant_from']==c['authored_failure_prefix_turns']==int(c['recovery'])
        recipe=control(c); assert len(recipe['steps'])==6+int(c['recovery'])
        assert recipe['prompt']==c['prompt']
        assert all(t.startswith('Analysis: null\nCommentary: null\n') and 'SENTINEL' not in t for t in texts(recipe))
        discovery=recipe['steps'][int(c['recovery'])]
        assert discovery[0]=='execute_bash' and discovery[1]['command'].startswith('ls -a; cat README.md; ')
        assert recipe['steps'][-1][0]=='finish'
        assert inspect_snapshot(c,recipe['expected'])['semantic_tests_required']
        assert inspect_snapshot(c,c['files'])['semantic_tests_required']


@pytest.mark.parametrize('field',['prompt','files','expected_source'])
def test_evaluation_reuse_rejected(field):
    rows=fixtures(); protected=repository_probes()
    rows[0][field]=deepcopy(protected[0][field])
    with pytest.raises(ValueError): validate_disjoint(rows,protected)
