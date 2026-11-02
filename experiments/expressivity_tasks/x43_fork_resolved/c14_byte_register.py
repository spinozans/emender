"""Freeze C14 byte protocol; consume phase-1 committed receipts, never rerun them."""
import json
from pathlib import Path
import subprocess
from c14_documents import sha

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT/'receipts/c14_byte'


def register():
    OUT.mkdir(exist_ok=True)
    source = ROOT/'receipts/c14_byte_source.json'
    discovery = json.loads(source.read_text())
    manifest = ROOT.parent/'x42_pile_scale/data/pile24/manifest.json'
    pile = json.loads(manifest.read_text())
    assert discovery['documents_index_sha256'] == pile['document_index_sha256']
    assert discovery['status'] == 'passed' and discovery['cross_document_windows'] == 0
    wiki = Path('/mnt/nvme2n1/erikg/wiki/MANIFEST.json')
    index = Path('/mnt/nvme2n1/erikg/wf-finish-as-100gb-20261102T044713Z/corpus.sxi2')
    xsa = Path('/home/erikg/sxgc/xsa/target/release/xsa')
    phase = ROOT/'receipts/c14'
    phase_names = ['coverage.json', 'validation.json', 'content_strata.json', 'provenance.json']
    coverage_binding = {}
    for name in phase_names:
        path = phase/name
        relative = str(path.relative_to(REPO))
        try:
            commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', relative], cwd=REPO).decode().strip()
            raw = subprocess.check_output(['git', 'show', commit+':'+relative], cwd=REPO, stderr=subprocess.PIPE)
            import hashlib
            if hashlib.sha256(raw).hexdigest() != sha(path):
                raise ValueError('phase-1 working bytes not yet committed')
            coverage_binding[name] = dict(path=relative, commit=commit, sha256=sha(path))
        except (subprocess.CalledProcessError, ValueError):
            coverage_binding = {}
            break
    files = ['PREREGISTRATION_C14_BYTE.md', 'c14_documents.py', 'test_c14_documents.py',
             'c14_byte_register.py', 'validate_c14_byte.py', 'c14_byte_models.py',
             'receipts/c14_byte_source.json', 'receipts/c14_byte/model_shapes.json',
             'receipts/c14_byte/index_identity.json', 'RESULT_C14_BYTE_REGISTRATION.md',
             '../../../AGENTS.md', '../../../CLAUDE.md']
    result = dict(schema=1, claim='C14', status='registered', training_started=False,
                  vocabulary=256, context=128, no_cross_document_context=True,
                  source_discovery=discovery,
                  pile_manifest_sha256=sha(manifest), wiki_manifest_path=str(wiki),
                  wiki_manifest_sha256=sha(wiki), index_path=str(index), index_bytes=index.stat().st_size,
                  xsa_path=str(xsa), xsa_sha256=sha(xsa),
                  index_sha256=json.loads((OUT/'index_identity.json').read_text())['sha256'],
                  files={name:sha(ROOT/name) for name in files},
                  coverage_status='bound' if coverage_binding else 'pending-coverage',
                  coverage_binding=coverage_binding,
                  launch_ready=False,
                  launch_blockers=['Full selected natural-byte corpus/offsets not yet attested',
                                   'Phase-1 native branch-count failures; incomplete 255-byte geometry',
                                   'Complete causal train/eval feature cache and cost receipts missing'])
    if not coverage_binding:
        result['launch_blockers'].append('Coverage prediction binding awaits committed phase-1 receipts')
    (OUT/'registration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='registered', coverage_status=result['coverage_status'], launch_ready=False)))


if __name__ == '__main__':
    register()
