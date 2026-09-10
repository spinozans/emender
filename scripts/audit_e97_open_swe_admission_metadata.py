#!/usr/bin/env python3
"""Read-only declared-license / split-family census of the proposed repair pool.

Reads metadata columns only, never trajectories, patches, commands, or holdout
prompts. Emits no training data and makes no source-admission decision.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
from scripts.audit_e97_open_swe_semantics import publish

COLUMNS=['trajectory_id','instance_id','repo','license','language','resolved']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def repo_identity(value):
    if not isinstance(value,str):return None
    value=value.strip().lower()
    if value.startswith('https://github.com/'):value=value[len('https://github.com/'):].removesuffix('.git')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*/[a-z0-9_.-]+',value):return None
    if value.split('/')[1] in ('.','..'):return None
    return value


def census(rows,protected):
    groups={name:defaultdict(lambda:defaultdict(Counter)) for name in ('repo','license','language')}
    families={0:{'repo':set(),'problem':set()},1:{'repo':set(),'problem':set()}}
    unknown=Counter(); conflicts=[];repo_licenses=defaultdict(set)
    for row in rows:
        split=row['split'];assert split in (0,1)
        repo=row['repo_identity'];problem=row['instance_id']
        for field in groups:
            label=repo if field=='repo' else row[field]
            if not label:unknown[field]+=1;label='<missing-or-unclassified>'
            groups[field][label][str(split)].update(trajectories=1,assistant_target_tokens=row['assistant_target_tokens'])
        if repo:
            families[split]['repo'].add(repo)
            repo_licenses[repo].add(row['license'])
            if isinstance(problem,str) and problem:families[split]['problem'].add((repo,problem))
        if not isinstance(problem,str) or not problem:unknown['instance_id']+=1
        if repo in protected:conflicts.append(row['identity'])
    shared_repo=families[0]['repo'] & families[1]['repo']
    shared_problem=families[0]['problem'] & families[1]['problem']
    validation=[r for r in rows if r['split']==1]
    return {'groups':groups,'unknown_metadata_counts':dict(unknown),
            'shared_train_validation_repositories':sorted(shared_repo),
            'shared_train_validation_problems':[list(x) for x in sorted(shared_problem)],
            'validation_trajectories_with_train_repo':sum(r['repo_identity'] in shared_repo for r in validation),
            'validation_trajectories_with_train_problem':sum((r['repo_identity'],r['instance_id']) in shared_problem for r in validation),
            'unique_problem_counts':{str(s):len(families[s]['problem']) for s in (0,1)},
            'protected_repository_trajectory_identities':conflicts,
            'multiple_license_strings_per_repo':{k:sorted(v,key=str) for k,v in repo_licenses.items() if len(v)>1}}


def main():
    p=argparse.ArgumentParser()
    for name in ('authority','raw-root','cohort','protected-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','cohort-sha256','protected-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();a.output.mkdir(mode=0o700,exist_ok=False)
    inputs=[(a.authority/'manifest.json',a.manifest_sha256),(a.cohort,a.cohort_sha256),(a.protected_manifest,a.protected_sha256)]
    for path,digest in inputs:assert sha(path)==digest,str(path)
    m=json.loads((a.authority/'manifest.json').read_text());assert m['training_eligible'] is False
    cohort=json.loads(a.cohort.read_text())
    selected={r['identity']:r for r in cohort if r['within_proposed_envelope']}
    assert len(selected)==7232
    protected_manifest=json.loads(a.protected_manifest.read_text())
    assert protected_manifest['schema']=='emender-e97-real-repo-holdout-v1'
    protected={repo_identity(v['url']) for v in protected_manifest['repositories'].values()}
    assert len(protected)==4 and None not in protected
    inputs.append((a.raw_root/'README.md',m['dataset_card_sha256']))
    shards=[]
    for item in m['input_files']:
        matches=list(a.raw_root.rglob(item['name']));assert len(matches)==1
        path=matches[0];assert not path.is_symlink() and path.stat().st_size==item['bytes']
        inputs.append((path,item['sha256']));shards.append(path)
    for path,digest in inputs:assert sha(path)==digest,str(path)
    print('ADMISSION_METADATA_INPUTS_VERIFIED',flush=True)
    rows=[];seen=set()
    for path in shards:
        index=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=COLUMNS,use_threads=False):
            for raw in batch.to_pylist():
                at=index;index+=1;tid='open-swe:'+str(raw['trajectory_id'])
                if tid not in selected:continue
                expected=selected[tid]
                assert tid not in seen and (path.name,at)==(expected['source_file'],expected['source_row'])
                assert raw['resolved']==1
                seen.add(tid)
                rows.append({'identity':tid,'split':int(expected['split']),'repo_identity':repo_identity(raw['repo']),
                    'instance_id':raw['instance_id'],'license':raw['license'],'language':raw['language'],
                    'assistant_target_tokens':expected['assistant_target_tokens']})
    assert seen==set(selected)
    assert sum(r['assistant_target_tokens'] for r in rows)==87107899
    result=census(rows,protected)
    for path,digest in inputs:assert sha(path)==digest,str(path)
    publish(a.output/'rows.json',rows)
    publish(a.output/'identity.json',{'inputs':{str(path):digest for path,digest in inputs},'source_sha256':sha(__file__),
        'columns':COLUMNS,'cohort_policy':'all and only 7,232 existing within-envelope identities; original split retained','protected_scope':sorted(protected)})
    result.update(schema='emender-open-swe-admission-metadata-v1',status='completed',training_eligible=False,
        trajectories=len(rows),assistant_target_tokens=87107899,
        limitations=['Declared license strings are not a license/legal audit or admission.',
            'Only normalized identities from the four-repository protected manifest are checked; not all protected panels or semantic overlap.',
            'Shared repository or problem checks do not establish independent holdout when absent.',
            'All rows descend from the earlier-used Open-SWE candidate family; full historical sample exposure is not reconstructed.',
            'No source-tool/runtime or commentary-visibility choice is approved; no training derivative emitted.'])
    publish(a.output/'summary.json',result)
    print('ADMISSION_METADATA_COMPLETE '+json.dumps({'trajectories':len(rows),'shared_repos':len(result['shared_train_validation_repositories']),
        'shared_problems':len(result['shared_train_validation_problems']),'protected_hits':len(result['protected_repository_trajectory_identities'])}),flush=True)


if __name__=='__main__':main()
