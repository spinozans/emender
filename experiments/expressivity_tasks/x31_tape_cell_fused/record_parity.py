"""Attest a successful, complete local JUnit receipt and its source hashes."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent


def source_hashes():
    files=[ROOT/n for n in ('cell_fused.py','kernels.py','backward.py',
                           'all_field_journal.py','journal_mutations.py','lazy_decay.py',
                           'birth_index.py','write_journal.py')]
    files+=sorted((ROOT/'tests').glob('test_*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    path=Path(sys.argv[1]).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('X31 receipt required')
    root=ET.parse(path).getroot();cases=list(root.iter('testcase'))
    if any(x.find(tag) is not None for x in cases for tag in ('failure','error','skipped')):
        raise RuntimeError('failures/errors/skips prevent attestation')
    counts={prefix:sum(x.attrib['name'].startswith(prefix) for x in cases) for prefix in
            ('test_full_first_order_parity[','test_production_dimension_gradient_parity[',
             'test_gradient_continuation_parity','test_all_field_exact_rollback[',
             'test_all_field_snapshot_parent_and_pressure_expiry','test_random_imported_store[',
             'test_retained_first_order_backward','test_mutating_public_trace_does_not_change_backward')}
    expected=(25,4,1,5,1,25,1,1)
    if len(cases)<95 or tuple(counts.values())!=expected:raise RuntimeError(f'incomplete suite: {counts}')
    result=dict(full_parity=True,tests=len(cases),coverage=counts,junit=str(path.relative_to(ROOT)),
                junit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=source_hashes(),
                tolerance=dict(atol=1e-5,rtol=1e-5))
    (ROOT/(sys.argv[2] if len(sys.argv)>2 else 'parity_part5.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
