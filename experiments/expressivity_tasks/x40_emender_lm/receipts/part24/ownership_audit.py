"""Offline, specialization-bound ownership ledger; no kernel or GPU execution.

Run with /home/erikg/emender/.venv/bin/python. This is an address-set audit,
not a control-flow verifier or evidence that a proposed repair is correct.
"""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'part23'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    audit = json.loads((PREVIOUS / 'audit.json').read_text())
    capture_path, = PREVIOUS.glob('capture_*.json')
    capture = json.loads(capture_path.read_text())
    artifacts = {}
    for kind, expected in capture['artifacts'].items():
        data = gzip.decompress(capture_path.with_suffix('.' + kind + '.gz').read_bytes())
        assert sha(data) == expected, kind
        artifacts[kind] = data
    sass = gzip.decompress(capture_path.with_suffix('.sass.gz').read_bytes())
    assert sha(sass) == audit['captures'][0]['sass_sha256']
    ptx = artifacts['ptx'].decode()
    # Bind the conditional counterexamples to actual unpredicated PTX stores,
    # rather than assuming that lack of a SASS predicate proves path reachability.
    unpredicated_stores = re.findall(r'^\s*(st\.shared\S*\s+.*);', ptx, re.M)
    for operand in ('[%r48]', '[%r49]', '[%r60]', '[%r70]'):
        assert any(operand in store for store in unpredicated_stores), operand
    assert re.search(r'setp\.eq\.b32\s+%p254, %r2, 0;', ptx)
    assert re.search(r'setp\.eq\.b32\s+%p265, %r1, 0;', ptx)
    instructions = {int(pc, 16): text for pc, text in re.findall(
        r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*', sass.decode())}
    isolated, = [r for r in audit['runs'] if r['binding'] == 'exact isolated X40 cubin']
    endpoints = [dict(group=g['id'], occurrence=i, **e)
                 for g in isolated['groups'] for i, e in enumerate(g['accesses'])]
    assert len(endpoints) == 175

    # Address expressions from the persistent setup at SASS 1a060..1a5f0.
    # t is CTA-local thread id, not token time. All addresses are shared bytes.
    expressions = {
        'RZ': ('0', lambda t: 0),
        'R2': ('t & 127', lambda t: t & 127),
        'R15': ('t', lambda t: t),
        'R137': ('4*(t >> 5) + 64*(t & 31)', lambda t: 4*(t >> 5)+64*(t & 31)),
        'R138': ('4*(t >> 5) + 64*(t & 1)', lambda t: 4*(t >> 5)+64*(t & 1)),
        'R146': ('16*(t & 1)', lambda t: 16*(t & 1)),
        'R149': ('64*(t & 31)', lambda t: 64*(t & 31)),
        'R150': ('64*(t & 1)', lambda t: 64*(t & 1)),
        'R151': ('4*(t & 126)', lambda t: 4*(t & 126)),
        'R154': ('t >> 5', lambda t: t >> 5),
        'R155': ('16*(t & 1)', lambda t: 16*(t & 1)),
        'R156': ('((128*(t >> 5)) | (16*t)) & 496',
                 lambda t: ((128*(t >> 5)) | (16*t)) & 496),
        'R159': ('(t & 96) >> 1', lambda t: (t & 96) >> 1),
        'R161': ('t & 112', lambda t: t & 112),
        'R162': ('t & 31', lambda t: t & 31),
        'R163': ('16*(t & 7) + ((t & 24) >> 1)',
                 lambda t: 16*(t & 7) + ((t & 24) >> 1)),
    }
    sites = []
    for pc in sorted({int(e['pc'], 16) for e in endpoints}):
        instruction = instructions[pc]
        matched = [e for e in endpoints if int(e['pc'], 16) == pc]
        assert all(e['sass'] == instruction for e in matched)
        operand, = re.findall(r'\[([^]]+)\]', instruction)
        match = re.fullmatch(r'(R\d+|RZ)(?:\.X(\d+))?(?:\+0x([0-9a-f]+))?', operand)
        assert match, operand
        register, scale, offset = match.groups()
        scale, offset = int(scale or 1), int(offset or '0', 16)
        expression, evaluate = expressions[register]
        width = 16 if '.128' in instruction else 4
        matrix = 'LDSM' in instruction
        write = matched[0]['access'] == 'Write'
        if register == 'R154':
            participation = 'lane == 0'
            threads = [t for t in range(128) if t % 32 == 0]
        elif register == 'R15' and write:
            participation = ('t < 32' if pc == 0x1dcb0 else
                             '(t & 3) == 0' if 0x1c8b0 <= pc <= 0x1c900 else 't == 0')
            threads = [t for t in range(128) if
                       (t < 32 if pc == 0x1dcb0 else
                        (t & 3) == 0 if 0x1c8b0 <= pc <= 0x1c900 else t == 0)]
        elif register == 'R138':
            participation = 'lane < 2'
            threads = [t for t in range(128) if t % 32 < 2]
        else:
            participation = 'see enclosing path; no additional store predicate' if write else 'see enclosing path and instruction predicate'
            threads = list(range(128))
        if 'kernels.py:99' in matched[0]['location']:
            phase = 'query broadcast to key-row layout'
        elif any('kernels.py:' + str(n) in matched[0]['location'] for n in (119, 120)):
            phase = 'tile weighted-value broadcast / reduction / output layout'
        elif any('kernels.py:' + str(n) in matched[0]['location'] for n in (149, 150, 151)):
            phase = 'read-pass denom/scale broadcast / output layout'
        else:
            phase = 'reduction exchange (operation/callsite needs PTX location context)'
        owners = collections.defaultdict(list)
        if write:
            for t in threads:
                base = scale*evaluate(t)+offset
                for byte in range(base, base+width):
                    owners[byte].append(t)
        duplicate = {str(byte): tids for byte, tids in owners.items() if len(tids) > 1}
        sites.append(dict(pc=hex(pc), instruction=instruction,
            locations=sorted({e['location'].split(' [')[0] for e in matched}),
            occurrences=len(matched), phase=phase,
            address_bytes=f'{scale}*({expression}) + {offset}',
            access_width_bytes=None if matrix else width,
            participation=participation,
            ownership=('conditional nonunique-writer witness' if duplicate else
                       'injective writes under stated participation' if write else
                       'collective matrix read; address operand is not the per-thread footprint' if matrix else
                       'reader address; producer/lifetime must be paired by dynamic epoch'),
            duplicate_byte_count=len(duplicate),
            duplicate_example=next(iter(duplicate.items()), None),
            lifetime_status='not globally proved; no repair authorized by this ledger',
            groups=sorted({e['group'] for e in matched})))
    assert len(sites) == 73
    # Preserve exact dynamic access occurrences as well as deduplicated PCs.
    report = dict(status='ownership_prerequisite_incomplete', production_changes=False,
        input_revision='b44020f2', input_audit_sha256=sha((PREVIOUS/'audit.json').read_bytes()),
        cubin_sha256=capture['artifacts']['cubin'], sass_sha256=sha(sass),
        assumptions=['128 threads in one CTA/program; t in [0,127]',
                     'address sets conditional on all stated threads reaching this dynamic site',
                     'scalar-reduction predicates checked against PTX p254/p265 and local SASS',
                     'no inter-CTA alias is implied by reuse of shared offsets'],
        endpoint_count=len(endpoints), site_count=len(sites), sites=sites, endpoints=endpoints,
        not_claimed=['full control-flow lifetime proof', 'certified-shape ownership proof',
                     'accepted race repairs', 'battery green', '9x acceptance'])
    (ROOT/'ownership.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Endpoint site ledger', '',
             'Generated by `ownership_audit.py`; read `ownership.md` for limitations.', '',
             '| PC | Source | Address in shared bytes | Participation | Ownership | Occurrences |',
             '| --- | --- | --- | --- | --- | --- |']
    for site in sites:
        lines.append('| ' + ' | '.join([site['pc'], ', '.join(site['locations']),
            site['address_bytes'].replace('|', '&#124;'), site['participation'],
            site['ownership'], str(site['occurrences'])]) + ' |')
    (ROOT/'endpoint_sites.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(endpoint_count=175, unique_sites=73,
        conditional_duplicate_writer_sites=[s['pc'] for s in sites if s['duplicate_byte_count']],
        status=report['status'])))


if __name__ == '__main__':
    main()
