from dataclasses import replace
import pytest
from ndm.e97_decision_metrics import COMPONENTS,DECISIONS,Span,Turn,byte_span_to_tokens,score_panel


def turn(values=None,trajectory='a',identity='0'):
    values=values or {c:[[1.0]] for c in DECISIONS}
    scores=[None];spans=[]
    for c in COMPONENTS:
        for index,field in enumerate(values.get(c,[])):
            start=len(scores);scores.extend(field)
            spans.append(Span(c,str(index),start,len(scores)))
    return Turn(trajectory,identity,'a'*64,(False,)+(True,)*(len(scores)-1),tuple(scores),
                {c:bool(values.get(c)) for c in COMPONENTS},tuple(spans))


def test_long_reasoning_cannot_hide_opening_regression():
    values={c:[[1.0]] for c in DECISIONS};values['reasoning']=[[100.0]*100]
    parent=turn(values)
    values['opening']=[[5.0]];values['reasoning']=[[0.0]*100]
    candidate=turn(values)
    a,b=score_panel([parent]),score_panel([candidate])
    assert sum(x for x in candidate.nll if x is not None)<sum(x for x in parent.nll if x is not None)
    assert b['decision_loss']>a['decision_loss']
    assert a['annotation_sha256']==b['annotation_sha256']


def test_equal_fields_not_long_argument_token_weight():
    values={c:[[0.0]] for c in DECISIONS};values['arguments']=[[0.0]*100,[10.0]]
    report=score_panel([turn(values)])
    assert report['component_means']['arguments']==5
    assert report['decision_loss']==1.25
    assert report['counts']['arguments']=={'tokens':101,'fields':2,'turns':1,'absent_turns':0,'trajectories':1}


def test_trajectories_not_long_trajectory_turn_counts():
    records=[turn({c:[[0.0]] for c in DECISIONS},identity=str(i)) for i in range(9)]
    records.append(turn({c:[[4.0]] for c in DECISIONS},trajectory='b'))
    assert score_panel(iter(records))['decision_loss']==2
    assert score_panel(records)['trajectories']==2


def test_absent_arguments_not_zero_filled():
    records=[turn({c:[[4.0]] for c in DECISIONS})]
    records.append(turn({c:[[0.0]] for c in DECISIONS if c!='arguments'},identity='final'))
    report=score_panel(records)
    assert report['component_means']['arguments']==4
    assert report['counts']['arguments']['absent_turns']==1
    with pytest.raises(ValueError,match='undefined'):score_panel(records[1:])


@pytest.mark.parametrize('change',[
    lambda t:replace(t,spans=t.spans[:-1]),
    lambda t:replace(t,spans=t.spans+(Span('opening','duplicate',1,2),)),
    lambda t:replace(t,spans=(Span('opening','0',0,2),)+t.spans[1:]),
    lambda t:replace(t,applicable={**t.applicable,'arguments':False}),
    lambda t:replace(t,nll=(None,float('nan'))+t.nll[2:]),
    lambda t:replace(t,nll=(None,-0.1)+t.nll[2:]),
    lambda t:replace(t,nll=(None,True)+t.nll[2:]),
])
def test_invalid_annotation_or_likelihood_fails_closed(change):
    with pytest.raises(ValueError):score_panel([change(turn())])


def test_duplicate_turns_rejected():
    t=turn()
    with pytest.raises(ValueError):score_panel([t,t])


def test_full_stream_bpe_boundaries_and_bytes_are_required():
    assert byte_span_to_tokens(b'Analysis:',[b'Analysis',b':'],0,8)==(0,1)
    with pytest.raises(ValueError,match='BPE'):byte_span_to_tokens(b'Analysis:',[b'Analysis',b':'],0,3)
    with pytest.raises(ValueError,match='serialization'):byte_span_to_tokens(b'Analysis:',[b'Analysis',b': '],0,8)
    assert byte_span_to_tokens('€x'.encode(),[b'\xe2',b'\x82\xac',b'x'],0,3)==(0,2)
