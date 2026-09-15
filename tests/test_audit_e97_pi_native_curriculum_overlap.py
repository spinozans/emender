from scripts.audit_e97_pi_native_curriculum_overlap import significant_contents,significant_scalars,finish_values,records

def test_stage_entity_policy_ignores_only_short_trivial_values():
 assert significant_contents({'alpha\n','a sufficiently distinctive fixture'})=={'a sufficiently distinctive fixture'}
 assert significant_scalars({'1','old','distinctive'})=={'distinctive'}

def test_records_and_finish_values_are_scoped_to_selected_ids():
 cases=[{'id':'a','family':'copy','prompt':'p','files':{'x':'A=distinctive\n'},'steps':[{'name':'finish','arguments':{'message':'DISTINCTIVE'}}]},{'id':'b','family':'copy','prompt':'q','files':{},'steps':[{'name':'finish','arguments':{'message':'excluded-value'}}]}]
 result=records(cases,{'a'},'synthetic/test')
 assert len(result)==1 and result[0]['task_id']=='a' and result[0]['repository']=='synthetic/test'
 assert finish_values(cases,{'a'})=={'DISTINCTIVE'}
