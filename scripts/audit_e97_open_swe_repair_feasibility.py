#!/usr/bin/env python3
"""Size a PROPOSED source-native, complete-context representation. Never emit training data.

Preserves source paths, observations and executable arguments. Nonempty assistant
content is included as explicitly declared internal commentary in this proposal;
this visibility change requires protocol review before implementation/admission.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

from scripts.audit_e97_open_swe_semantics import publish, sha, verify_file

FRAME = '''Source-tool projection proposal (not an implemented serving protocol).
Use exactly this native text frame for each assistant response:
Analysis: <one JSON string containing internal planning>
Action: <execute_bash or str_replace_editor>
Arguments: <one JSON object>
For completion replace Action/Arguments with Final: <answer>.
Internal think-tool text and source assistant commentary are retained in Analysis,
not dispatched as tools. Preserve native executable tool arguments and observations.
'''


@lru_cache(maxsize=2)
def token_lengths(name):
    encoding=tiktoken.get_encoding(name)
    return np.array([len(encoding.decode_single_token_bytes(i)) for i in range(encoding.n_vocab)],dtype=np.int64)


def project(row, encoding, analysis_cap=2048, max_record=65536):
    pieces, pending = [], []
    provenance = Counter()
    first_user = False
    schemas = []
    for spec in row.get('tools') or []:
        spec = json.loads(spec) if isinstance(spec, str) else spec
        if spec['function']['name'] in {'execute_bash', 'str_replace_editor'}:
            schemas.append(spec)
    system = '\n\n'.join(str(m.get('content') or '') for m in row['messages'] if m['role'] == 'system')
    system += '\n\n'+FRAME+'\nSource executable tool declarations:\n'+json.dumps(schemas,ensure_ascii=False,sort_keys=True,separators=(',', ':'))
    pieces.append(('System:\n'+system, False))
    messages = row['messages']
    finished = False
    overcap = 0
    for i, m in enumerate(messages):
        role = m['role']
        if role == 'user':
            if first_user:
                raise ValueError('multiple_users_require_separate_projection')
            first_user = True
            pieces.append(('\n\nUser:\n'+str(m.get('content') or ''), False))
        if role != 'assistant':
            continue
        calls = m.get('tool_calls') or []
        if len(calls) != 1:
            raise ValueError('requires_single_tool_call')
        fn = calls[0]['function'];name = fn['name']
        args = json.loads(fn.get('arguments') or '{}')
        if not isinstance(args,dict):
            raise ValueError('nonobject_arguments')
        for field in ('reasoning_content', 'content'):
            value = m.get(field)
            if value is not None and not isinstance(value,str):
                raise ValueError('nonstring_assistant_text')
            if value and value.strip():
                pending.append(value)  # exact constituent bytes; no strip/deduplication
                provenance[field+'_pieces'] += 1
        if name == 'think':
            thought = args.get('thought')
            if thought is not None and not isinstance(thought,str):
                raise ValueError('nonstring_thought')
            if thought and thought.strip():
                pending.append(thought)
                provenance['think_thought_pieces'] += 1
            following = messages[i+1] if i+1 < len(messages) else {}
            if following.get('role') != 'tool' or str(following.get('content') or '').strip() != 'Your thought has been logged.':
                raise ValueError('noncanonical_think_acknowledgement')
            continue
        reasoning = '\n\n'.join(pending)
        pending.clear()
        n = len(encoding.encode_ordinary(reasoning))
        if not reasoning.strip():
            raise ValueError('empty_analysis')
        if n > analysis_cap or len(reasoning.encode()) > 65536:
            overcap += 1
        prefix = 'Analysis: '+json.dumps(reasoning,ensure_ascii=False,separators=(',', ':'))+'\n'
        if name == 'finish':
            final = args.get('message')
            if not isinstance(final,str) or not final.strip():
                raise ValueError('missing_final_message')
            body = prefix+'Final: '+final
            finished = True
        elif name in {'execute_bash','str_replace_editor'}:
            body = prefix+'Action: '+name+'\nArguments: '+json.dumps(args,ensure_ascii=False,separators=(',', ':'))
        else:
            raise ValueError('unsupported_source_tool:'+name)
        pieces.extend([('\n\nAssistant:\n',False),(body,True)])
        provenance['assistant_units'] += 1
        if finished:
            break
        following = messages[i+1] if i+1 < len(messages) else {}
        if following.get('role') != 'tool' or not isinstance(following.get('content'),str):
            raise ValueError('missing_original_observation')
        pieces.append(('\n\nTool:\n'+following['content'],False))
    if not first_user or not finished or pending:
        raise ValueError('incomplete_trajectory')
    if any('\x1e' in text for text,_ in pieces):
        raise ValueError('literal_record_separator_requires_lossless_escape_design')
    pieces.append(('\x1e',True))
    text=''.join(part for part,_ in pieces)
    tokens=encoding.encode_ordinary(text)
    # Byte-boundary inventory verifies no token would cross supervision boundaries.
    intervals=[];pos=0
    for part,target in pieces:
        end=pos+len(part.encode())
        if target:intervals.append((pos,end))
        pos=end
    target_tokens=0;crossings=0
    ends=np.cumsum(token_lengths(encoding.name)[tokens])
    starts=np.r_[0,ends[:-1]]
    for low,high in intervals:
        left=int(np.searchsorted(ends,low,side='right'))
        right=int(np.searchsorted(starts,high,side='left'))
        inside=int(((starts[left:right]>=low)&(ends[left:right]<=high)).sum())
        target_tokens+=inside
        crossings+=right-left-inside
    reasons=[]
    if overcap:reasons.append('analysis_cap')
    if len(tokens)>max_record:reasons.append('whole_trajectory_context_cap')
    if crossings:reasons.append('target_boundary_token_crossing')
    return {'within_proposed_envelope':not reasons,'exclusion_reasons':reasons,
            'input_tokens':len(tokens),'assistant_target_tokens':target_tokens,
            'overcap_analysis_units':overcap,'boundary_crossings':crossings,
            'provenance_piece_counts':dict(provenance),'serialization_sha256':hashlib.sha256(text.encode()).hexdigest()},pieces


def run(authority,raw_root,manifest_sha256,output):
    authority,raw_root,output=Path(authority).resolve(),Path(raw_root).resolve(),Path(output)
    output.mkdir(mode=0o700,exist_ok=False)
    if sha(authority/'manifest.json')!=manifest_sha256:raise ValueError('manifest mismatch')
    m=json.loads((authority/'manifest.json').read_text())
    if m['training_eligible'] is not False:raise ValueError('expected candidate')
    metadata=verify_file(authority,m['outputs']['metadata'])
    expected={}
    for line in metadata.read_text().splitlines():
        r=json.loads(line);item=(r['source_file'],r['source_row'],r['split'])
        if expected.setdefault(r['trajectory_identity'],item)!=item:raise ValueError('metadata mismatch')
    if sha(raw_root/'README.md')!=m['dataset_card_sha256']:raise ValueError('card mismatch')
    files=[]
    for d in m['input_files']:
        ps=list(raw_root.rglob(d['name']))
        if len(ps)!=1 or ps[0].is_symlink():raise ValueError('ambiguous source')
        p=ps[0]
        if p.stat().st_size!=d['bytes'] or sha(p)!=d['sha256']:raise ValueError('source mismatch')
        files.append((p,d))
    identity={'manifest_sha256':manifest_sha256,'probe_sha256':sha(__file__),
              'projection':'proposed source-native analysis/action/final; unchanged raw observations and executable arguments; full trajectory in one record',
              'analysis_cap':2048,'record_cap':65536,'training_eligible':False,
              'visibility_change':'source commentary becomes internal planning in this proposal; not approved as runtime behavior',
              'scope':'counts and identity receipts only; no dataset/token files emitted; no tool execution'}
    publish(output/'identity.json',identity)
    counts,exclusions,splits=Counter(),Counter(),{'0':Counter(),'1':Counter()}
    rows=[];seen=set();encoding=tiktoken.get_encoding('p50k_base')
    print('REPAIR_FEASIBILITY_INPUTS_VERIFIED',flush=True)
    for p,d in files:
        index=0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=8,use_threads=False):
            for row in batch.to_pylist():
                at=index;index+=1;tid='open-swe:'+str(row['trajectory_id'])
                if tid not in expected:continue
                if tid in seen or expected[tid][:2]!=(p.name,at):raise ValueError('row identity mismatch')
                seen.add(tid)
                try:
                    result,_=project(row,encoding)
                except ValueError as exc:
                    result={'within_proposed_envelope':False,'exclusion_reasons':['structural:'+str(exc)]}
                counts['examined_trajectories']+=1
                if result['within_proposed_envelope']:
                    counts.update(within_envelope_trajectories=1,input_tokens=result['input_tokens'],assistant_target_tokens=result['assistant_target_tokens'])
                    splits[str(expected[tid][2])].update(trajectories=1,input_tokens=result['input_tokens'],assistant_target_tokens=result['assistant_target_tokens'])
                for reason in result['exclusion_reasons']:exclusions[reason]+=1
                rows.append({'identity':tid,'source_file':p.name,'source_row':at,'split':expected[tid][2],**result})
        print(f'REPAIR_FEASIBILITY_SHARD_COMPLETE file={p.name} examined={len(seen)}',flush=True)
    if seen!=set(expected):raise ValueError('missing sources')
    for p,d in files:
        if p.stat().st_size!=d['bytes'] or sha(p)!=d['sha256']:raise ValueError('source changed')
    verify_file(authority,m['outputs']['metadata'])
    if sha(authority/'manifest.json')!=manifest_sha256:raise ValueError('manifest changed')
    publish(output/'trajectories.json',rows)
    result={'schema':'emender-open-swe-repair-feasibility-v1','status':'completed','identity':identity,
            'counts':dict(counts),'overlapping_exclusion_counts':dict(exclusions),'split_counts':splits,
            'limitations':['Within envelope does not mean admitted, replay-verified, independently held out, or training-ready.',
                           'Source-native tools require a separate versioned runtime contract; current Pi tools are not equivalent.',
                           'This proposal changes commentary visibility and needs explicit protocol review.',
                           'Only the existing 10,905-trajectory candidate pool is examined; no excluded raw sources are newly admitted.']}
    publish(output/'summary.json',result)
    print('REPAIR_FEASIBILITY_COMPLETE '+json.dumps({'counts':dict(counts),'exclusions':dict(exclusions)}),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ('authority','raw-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    a=p.parse_args();run(a.authority,a.raw_root,a.manifest_sha256,a.output)


if __name__=='__main__':main()
