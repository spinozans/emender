"""Candidate decision-balanced loss reducer; no protocol adapter or trainer wiring.

Annotations, component applicability, and source identities must be frozen before
scoring. Each field has one contiguous token span. All supervised tokens must be
classified, including reasoning and auxiliary text excluded from primary D.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from statistics import fmean

DECISIONS=('opening','choice','arguments','termination')
COMPONENTS=DECISIONS+('reasoning','auxiliary')
SCHEMA='emender-decision-balanced-loss-candidate-v1'


@dataclass(frozen=True)
class Span:
    component: str
    field: str
    start: int
    end: int


@dataclass(frozen=True)
class Turn:
    trajectory_id: str
    turn_id: str
    # Digest of the frozen causal probe identity (tokenizer, tokens, reset/valid
    # controls), not merely displayed text. Construction belongs to the adapter.
    record_sha256: str
    mask: tuple[bool,...]
    nll: tuple[float | None,...]
    applicable: dict[str,bool]
    spans: tuple[Span,...]


def byte_span_to_tokens(serialized, token_bytes, start, end):
    """Use complete-stream BPE bytes, never separately tokenize a field.

    Returns full-stream token indices, not shifted prediction-target indices.
    An adapter must explicitly align these with its target/mask convention.
    """
    token_bytes=tuple(token_bytes)
    if b''.join(token_bytes)!=serialized:raise ValueError('token bytes differ from serialization')
    if type(start) is not int or type(end) is not int or not 0<=start<end<=len(serialized):
        raise ValueError('invalid byte span')
    boundaries={0:0}; offset=0
    for index,piece in enumerate(token_bytes):
        if not piece:raise ValueError('empty token bytes')
        offset+=len(piece);boundaries[offset]=index+1
    if start not in boundaries or end not in boundaries:
        raise ValueError('span intersects a BPE token; annotation must be reviewed')
    return boundaries[start],boundaries[end]


def score_panel(turns):
    """Fields -> applicable turns -> trajectories -> four equally weighted axes.

    Absent components are declared, never zero-filled. Every decision component
    must occur somewhere in the panel; otherwise D is undefined and rejected.
    No generation success or source-admission claim follows from this number.
    """
    turns=tuple(turns)
    if not turns:raise ValueError('empty panel')
    seen=set();annotations=[]
    by_component={c:{} for c in COMPONENTS}
    counts={c:{'tokens':0,'fields':0,'turns':0,'absent_turns':0} for c in COMPONENTS}
    for turn in turns:
        identity=(turn.trajectory_id,turn.turn_id)
        if not all(type(v) is str and v for v in identity) or identity in seen:
            raise ValueError('invalid or repeated turn identity')
        seen.add(identity)
        if type(turn.record_sha256) is not str or len(turn.record_sha256)!=64 or any(c not in '0123456789abcdef' for c in turn.record_sha256):
            raise ValueError('record SHA-256 required')
        if len(turn.mask)!=len(turn.nll) or not turn.mask or any(type(x) is not bool for x in turn.mask):
            raise ValueError('aligned boolean target mask required')
        for active,value in zip(turn.mask,turn.nll):
            if active:
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('finite nonnegative target NLL required')
            elif value is not None:raise ValueError('unscored context must have null NLL')
        if set(turn.applicable)!=set(COMPONENTS) or any(type(v) is not bool for v in turn.applicable.values()):
            raise ValueError('explicit applicability required for every component')
        covered=set();fields=set();values={c:[] for c in COMPONENTS};layout=[]
        for span in turn.spans:
            c=span.component
            if c not in COMPONENTS or not turn.applicable[c]:raise ValueError('inapplicable span')
            if type(span.field) is not str or not span.field or (c,span.field) in fields:
                raise ValueError('each field requires one uniquely named contiguous span')
            fields.add((c,span.field))
            if type(span.start) is not int or type(span.end) is not int or not 0<=span.start<span.end<=len(turn.mask):
                raise ValueError('invalid token span')
            indices=set(range(span.start,span.end))
            if covered & indices or any(not turn.mask[i] for i in indices):
                raise ValueError('overlapping or unsupervised annotation')
            covered.update(indices)
            values[c].append(fmean(turn.nll[span.start:span.end]))
            counts[c]['tokens']+=span.end-span.start;counts[c]['fields']+=1
            layout.append([c,span.field,span.start,span.end])
        if covered!={i for i,v in enumerate(turn.mask) if v}:raise ValueError('unclassified target tokens')
        for c in COMPONENTS:
            if bool(values[c])!=turn.applicable[c]:raise ValueError('applicability and annotations disagree')
            if values[c]:
                counts[c]['turns']+=1
                by_component[c].setdefault(turn.trajectory_id,[]).append(fmean(values[c]))
            else:counts[c]['absent_turns']+=1
        annotations.append([*identity,turn.record_sha256,list(turn.mask),turn.applicable,sorted(layout)])
    means={}
    for c in COMPONENTS:
        trajectories=by_component[c]
        counts[c]['trajectories']=len(trajectories)
        means[c]=fmean(fmean(v) for v in trajectories.values()) if trajectories else None
    if any(means[c] is None for c in DECISIONS):raise ValueError('D undefined: missing decision component')
    annotation_sha=hashlib.sha256(json.dumps(sorted(annotations),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'schema':SCHEMA,'decision_loss':fmean(means[c] for c in DECISIONS),
            'component_means':means,'counts':counts,'annotation_sha256':annotation_sha,
            'trajectories':len({t.trajectory_id for t in turns}),'turns':len(turns)}
