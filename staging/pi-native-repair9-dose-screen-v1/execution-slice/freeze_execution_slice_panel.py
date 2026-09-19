#!/usr/bin/env python3
"""Freeze the 24-case dose-screen execution-slice panel for ONE probe checkpoint.

STAGED machinery (executed by the orchestrator, per probe, AFTER the 32-update
training run produces its u32 checkpoint). Mirrors the u256 dual-gate execution
panel construction: the frozen 96-case execution panel (the v3-u256 dual-gate
panel, sha cdcd3ac9...; identical case set to the v10 u256 panel) supplies the
cases and every runtime binding; the slice keeps the fixed seeded 24-case
subset (6 id-prefix groups x 4 families, 1 case/cell; slice-definition.json) and
rebinds the models to THIS probe checkpoint (train/saved) plus the bridge
controls (train/saved) — the same-run window reference the dual gates use.

Fail-closed: source panel identity, slice membership + balance, checkpoint
identity. Panels are regenerated per checkpoint (recorded v9-full defect class:
a copied panel silently evaluates the wrong checkpoint).
"""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

SOURCE_PANEL = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/'
                    'pi-native-repair9r-full-arc-v3-u256-dual-gate-v1/execution/panel.json')
SOURCE_PANEL_SHA = 'cdcd3ac9e4bde2325f2dc28e082efd8e74a04cd03a29fe34d2663882cd59aa74'
BRIDGE = (Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt'),
          '9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa')
GROUPS = ('regression-old', 'regression-fresh', 'fresh', 'transfer', 'bridge-fresh', 'bridge-composition')
FAMILIES = ('lookup', 'sum', 'edit', 'recovery')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(args):
    source = json.loads(Path(args.source_panel).read_text())
    if sha(args.source_panel) != SOURCE_PANEL_SHA:
        raise SystemExit('source execution panel identity')
    definition = json.loads(Path(args.slice_definition).read_text())
    if (definition.get('source_panel_sha256') != SOURCE_PANEL_SHA
            or definition.get('seed') != 240977 or len(definition['case_ids']) != 24):
        raise SystemExit('slice definition identity')
    by_id = {c['id']: c for c in source['cases']}
    missing = [i for i in definition['case_ids'] if i not in by_id]
    if missing:
        raise SystemExit(f'slice cases missing from the source panel: {missing}')
    cases = [by_id[i] for i in definition['case_ids']]
    # balance verification: 6 groups x 4 families, one case per cell
    cells = Counter()
    for c in cases:
        group = next((g for g in GROUPS if c['id'].startswith(g + '-')), None)
        if group is None or c['family'] not in FAMILIES:
            raise SystemExit(f'unstructured slice case {c["id"]}')
        cells[(group, c['family'])] += 1
    if set(cells.values()) != {1} or len(cells) != 24:
        raise SystemExit('slice is not the balanced 6x4 grid')
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit('probe checkpoint identity')
    if sha(BRIDGE[0]) != BRIDGE[1]:
        raise SystemExit('bridge control identity')
    panel = dict(source)
    panel['cases'] = cases
    panel['models'] = [
        {'checkpoint': str(BRIDGE[0]), 'mode': 'train', 'name': 'bridge-y-control', 'sha256': BRIDGE[1]},
        {'checkpoint': str(BRIDGE[0]), 'mode': 'saved', 'name': 'bridge-x-control', 'sha256': BRIDGE[1]},
        {'checkpoint': str(Path(args.checkpoint).resolve()), 'mode': 'saved', 'name': 'dose-probe-x', 'sha256': args.checkpoint_sha256},
        {'checkpoint': str(Path(args.checkpoint).resolve()), 'mode': 'train', 'name': 'dose-probe-y', 'sha256': args.checkpoint_sha256},
    ]
    panel['diagnostic_derivation'] = {
        'checkpoint_sha256': args.checkpoint_sha256,
        'note': ('dose-screen probe execution slice: fixed seeded 24-case subset of the frozen 96-case '
                 'execution panel (6 id-prefix groups x 4 families, seed 240977); bridge controls = '
                 'same-run window reference; panel regenerated per checkpoint (v9-full defect class).'),
    }
    panel['source_panel_sha256'] = SOURCE_PANEL_SHA
    panel['upstream_panel_sha256'] = source.get('source_panel_sha256')
    panel['execution_slice'] = {
        'method': definition['method'],
        'seed': definition['seed'],
        'case_ids': definition['case_ids'],
        'slice_definition_sha256': sha(args.slice_definition),
    }
    panel['scope'] = ('Dose-screen probe execution slice (24 of the frozen 96 cases); measurement only; '
                      'the frozen >=64/96 gate remains bound to the full 96-case panel.')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit('refusing to overwrite an existing slice panel; regenerate per checkpoint')
    out.write_text(json.dumps(panel, indent=1, sort_keys=True) + '\n')
    print('DOSE_EXECUTION_SLICE_PANEL_FROZEN', sha(out), 'checkpoint', args.checkpoint_sha256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--slice-definition', default=str(Path(__file__).parent / 'slice-definition.json'))
    p.add_argument('--source-panel', default=str(SOURCE_PANEL))
    p.add_argument('--output', required=True)
    freeze(p.parse_args())


if __name__ == '__main__':
    main()
