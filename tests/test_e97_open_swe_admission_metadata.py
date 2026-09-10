from scripts.audit_e97_open_swe_admission_metadata import COLUMNS,census,repo_identity


def row(identity,split,repo,problem,license='MIT'):
    return {'identity':identity,'split':split,'repo_identity':repo,'instance_id':problem,
            'license':license,'language':'python','assistant_target_tokens':100}


def test_repository_normalization_is_not_basename_matching():
    assert repo_identity('https://github.com/Pallets/MarkupSafe.git')=='pallets/markupsafe'
    assert repo_identity('fork/markupsafe')=='fork/markupsafe'
    assert repo_identity('markupsafe') is None
    assert repo_identity('https://elsewhere.example/pallets/markupsafe') is None


def test_problem_and_repository_overlap_are_separate():
    result=census([row('a',0,'org/repo','one'),row('b',1,'org/repo','two'),
                   row('c',1,'org/repo','one')],{'org/repo'})
    assert result['validation_trajectories_with_train_repo']==2
    assert result['validation_trajectories_with_train_problem']==1
    assert result['protected_repository_trajectory_identities']==['a','b','c']
    assert result['groups']['license']['MIT']['1']['assistant_target_tokens']==200


def test_unknowns_do_not_create_a_shared_problem():
    result=census([row('a',0,None,None,None),row('b',1,None,None,None)],set())
    assert result['shared_train_validation_problems']==[]
    assert result['unknown_metadata_counts']['license']==2
    assert result['unknown_metadata_counts']['repo']==2


def test_no_trajectory_or_patch_columns_are_read():
    assert COLUMNS==['trajectory_id','instance_id','repo','license','language','resolved']
