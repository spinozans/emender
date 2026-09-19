#!/usr/bin/env python3
"""Freeze a repair-v9r DOSE-SCREEN probe proposal (32 updates; STAGED; NOT AUTHORIZED).

One probe per dose prep (pi-native-repair9-full-preparation-v3-d{25,50,75}):
identical to the v3 full prep except the restored pools (grounded-authored,
representation-bridge) are seeded first-N subsamples at 25/50/75% dose — the
two gate verdicts bracket the dose axis (v2 scrubbed-to-zero: Stage-B 9-10/14,
execution 0/96; v3 full: execution 59-60/96, Stage-B 4-5/14; both fail the
frozen gate). Each probe trains 32 updates from the bridge parent on its dose
prep under its freshly screened key, then measures Stage-B (14-case panel) and
the 24-case execution slice. Measurement only: no promotion, no gate change.

Mirrors the v3-restart segment freeze writer (same identity-binding discipline;
every identity re-hashed from disk) for scripts/audit_e97_pi_native_repair_generic.py
and the standard admission machinery. The parent of every dose probe is the
bridge checkpoint (the v10/v3 restart point, full retention headroom).
"""
import argparse, hashlib, json
from pathlib import Path

BRIDGE = (Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt'),
          '9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa')
BRIDGE_AUTHORITY = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/representation-bridge-v1-data/authority')
AUTHORED_AUTHORITY = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/grounded-expansion-v1-data-r2/authority')
SOURCE_COMMIT = 'd77dcc462e8b25c0b362b196347ddc9f50ffe683'  # trainer: translated-OH reader lineage head
REPO = Path('/home/erikg/emender')

GATE = {
    'composition': '>= 4/16',
    'openhands_execution_overall': '>= 64/96',
    'openhands_fresh': '>= 30/32',
    'openhands_prior_fresh': '= 16/16',
    'openhands_prior_regression': '>= 10/16',
    'openhands_transfer': '>= 5/16',
    'pi_native_stage_b': 'valid first frame >= 12/14 and correct first action >= 10/14',
    'retention': 'both saved-x/train-y retention gates pass',
}


def sha(p):
    p = Path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def must(path, label):
    if not Path(path).is_file():
        raise SystemExit(f'freeze failed: missing {label}: {path}')
    return sha(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dose', required=True, choices=('d25', 'd50', 'd75'))
    ap.add_argument('--preparation', type=Path, required=True)
    ap.add_argument('--schedule', type=Path, required=True)
    ap.add_argument('--key', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    prep = a.preparation
    prep_manifest_sha = must(prep / 'manifest.json', 'prep manifest')
    pack_manifest_sha = must(prep / 'packs/manifest.json', 'packs manifest')
    schedule_sha = must(a.schedule, 'probe schedule')
    prep_manifest = json.loads((prep / 'manifest.json').read_text())
    packs_manifest = json.loads((prep / 'packs/manifest.json').read_text())
    schedule = json.loads(Path(a.schedule).read_text())
    if (schedule.get('status') != 'planning-only-not-authorized' or schedule.get('training_eligible')
            or schedule.get('optimizer_updates_authorized') or len(schedule.get('steps', [])) != 32):
        raise SystemExit('probe schedule is not a planning-only 32-update proposal')
    if (schedule['sampler_key'] != a.key or schedule['world_size'] != 8 or schedule['context_size'] != 65536):
        raise SystemExit('probe schedule binding')
    if set(schedule['source_target_totals']) != set(prep_manifest['source_target_totals']):
        raise SystemExit('schedule cohorts differ from the preparation')
    if (prep_manifest['training_eligible'] or prep_manifest['packing_authorized']
            or prep_manifest['optimizer_updates_authorized'] or packs_manifest['training_eligible']):
        raise SystemExit('prepared authority is not non-authorized')
    for restored in ('representation-bridge-rehearsal', 'grounded-authored-rehearsal'):
        if restored not in prep_manifest['source_target_totals']:
            raise SystemExit(f'dose prep is missing the restored cohort {restored}')
    subs = {}
    for key in ('authored_rehearsal', 'rehearsal_rehearsal'):
        entry = prep_manifest.get(key) or {}
        sub = entry.get('subsample')
        if not sub or sub.get('method') != 'seeded-shuffle-first-N':
            raise SystemExit(f'{key} does not record a seeded subsample (dose prep required)')
        subs[key] = sub
    if must(BRIDGE[0], 'bridge parent checkpoint') != BRIDGE[1]:
        raise SystemExit('bridge parent identity')

    T = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
    bindings = [
        (str(BRIDGE[0]), BRIDGE[1]),
        (str(prep / 'manifest.json'), prep_manifest_sha),
        (str(prep / 'packs/manifest.json'), pack_manifest_sha),
        (str(Path(a.schedule).resolve()), schedule_sha),
        (str(BRIDGE_AUTHORITY / 'manifest.json'), must(BRIDGE_AUTHORITY / 'manifest.json', 'bridge authority')),
        (str(AUTHORED_AUTHORITY / 'manifest.json'), must(AUTHORED_AUTHORITY / 'manifest.json', 'grounded-authored authority')),
        (str(T / 'e97-oh-pi-native-translation-v1/candidate-authority/manifest.json'),
         must(T / 'e97-oh-pi-native-translation-v1/candidate-authority/manifest.json', 'translated OH authority')),
        (str(T / 'e97-oh-pi-native-translation-v1/scrub-report.json'),
         must(T / 'e97-oh-pi-native-translation-v1/scrub-report.json', 'OH scrub report')),
        (str(T / 'e97-oh-pi-native-translation-v1/translate-report.json'),
         must(T / 'e97-oh-pi-native-translation-v1/translate-report.json', 'translate report')),
        (str(T / 'pi-native-reasoning-rehearsal-collection-v1/collect/candidate-authority/manifest.json'),
         must(T / 'pi-native-reasoning-rehearsal-collection-v1/collect/candidate-authority/manifest.json', 'reasoning authority')),
        (str(T / 'pi-native-reasoning-rehearsal-collection-v1/audit.json'),
         must(T / 'pi-native-reasoning-rehearsal-collection-v1/audit.json', 'reasoning collection audit')),
        (str(T / 'pi-native-reasoning-rehearsal-collection-v1/overlap-audit.json'),
         must(T / 'pi-native-reasoning-rehearsal-collection-v1/overlap-audit.json', 'reasoning overlap audit')),
        ('/mnt/nvme1n1/erikg/sft/e97-4b-smoltalk2-admitted-v1/manifest.json',
         must('/mnt/nvme1n1/erikg/sft/e97-4b-smoltalk2-admitted-v1/manifest.json', 'smoltalk2 authority')),
        ('/mnt/nvme1n1/erikg/sft/e97-4b-pi-instruction-mix-v2/manifest.json',
         must('/mnt/nvme1n1/erikg/sft/e97-4b-pi-instruction-mix-v2/manifest.json', 'instruction authority')),
        ('/mnt/nvme1n1/erikg/sft/e97-4b-pi-compositional-retention-mix-v1/manifest.json',
         must('/mnt/nvme1n1/erikg/sft/e97-4b-pi-compositional-retention-mix-v1/manifest.json', 'compositional authority')),
        ('/mnt/nvme1n1/erikg/sft/e97-4b-pi-cumulative-recovery-mix-v1/manifest.json',
         must('/mnt/nvme1n1/erikg/sft/e97-4b-pi-cumulative-recovery-mix-v1/manifest.json', 'cumulative authority')),
        ('/mnt/nvme1n1/erikg/sft/e97-4b-pi-live-aligned-all-assistant-v1/manifest.json',
         must('/mnt/nvme1n1/erikg/sft/e97-4b-pi-live-aligned-all-assistant-v1/manifest.json', 'live-aligned authority')),
        (str(T / 'pi-native-extracterror-collection-v1/collect/candidate-authority/manifest.json'),
         must(T / 'pi-native-extracterror-collection-v1/collect/candidate-authority/manifest.json', 'extracterror authority')),
        (str(T / 'pi-native-longcopy-collection-v1/collect/candidate-authority/manifest.json'),
         must(T / 'pi-native-longcopy-collection-v1/collect/candidate-authority/manifest.json', 'longcopy authority')),
        (str(T / 'pi-native-pointerchase-collection-v1/collect/candidate-authority/manifest.json'),
         must(T / 'pi-native-pointerchase-collection-v1/collect/candidate-authority/manifest.json', 'pointerchase authority')),
        (str(T / 'pi-native-curriculum-2000-selected-v1/candidate-authority/manifest.json'),
         must(T / 'pi-native-curriculum-2000-selected-v1/candidate-authority/manifest.json', 'selected curriculum authority')),
        (str(T / 'pi-native-curriculum-2000-selected-v1/selection-audit.json'),
         must(T / 'pi-native-curriculum-2000-selected-v1/selection-audit.json', 'selected curriculum selection audit')),
        (str(T / 'pi-native-curriculum-2000-selected-v1/overlap-audit.json'),
         must(T / 'pi-native-curriculum-2000-selected-v1/overlap-audit.json', 'selected curriculum overlap audit')),
        ('configs/pi/e97-active-tool-surface-v1.json',
         must(REPO / 'configs/pi/e97-active-tool-surface-v1.json', 'active tool surface')),
    ]
    # the repo-relative binding string above resolves from the repo cwd, matching the v3 segment-writer precedent
    triple = {Path(b[0]).name: b[1] for b in bindings}
    for name, field in (('manifest.json', 'selected_authority_sha256'),
                        ('selection-audit.json', 'selected_selection_audit_sha256'),
                        ('overlap-audit.json', 'selected_overlap_audit_sha256')):
        got = triple[name]
        if prep_manifest[field] != got:
            raise SystemExit(f'prep manifest {field} does not match the bound file on disk')

    cohort_bindings = {}
    for key in ('openhands_rehearsal', 'conversation_rehearsal', 'loopbreak_rehearsal',
                'extra_rehearsal', 'extra2_rehearsal', 'extra3_rehearsal',
                'rehearsal_rehearsal', 'authored_rehearsal'):
        entry = prep_manifest.get(key)
        if entry is None:
            raise SystemExit(f'cohort binding {key} missing from the preparation manifest')
        binding = {'authority_sha256': entry['authority_sha256'], 'records': entry['records']}
        for field in ('cohort', 'consumed_target_tokens', 'seed'):
            if field in entry:
                binding[field] = entry[field]
        if 'subsample' in entry:
            binding['subsample'] = entry['subsample']
        cohort_bindings[key] = binding

    authored, bridge = prep_manifest['authored_rehearsal'], prep_manifest['rehearsal_rehearsal']
    proposal = {
        'arc': {
            'arc_id': 'repair-v9r-dose-screen',
            'dose': a.dose,
            'note': ('diagnostic dose-screen probe, not an arc segment: 32 updates from the bridge '
                     'parent on the %s dose prep; measurement only, no promotion, no gate change' % a.dose),
            'probe_updates': 32,
            'throughput_note': ('chunk 2048/16384 qualified recipe; trainer worktree pinned at %s '
                                '(the translated-OH reader lineage head)' % SOURCE_COMMIT),
        },
        'authority_manifest_sha256': prep_manifest_sha,
        'automatic_expansion': False,
        'automatic_retry': False,
        'checkpoint_promotion': False,
        'context_size': 65536,
        'cohort_bindings': cohort_bindings,
        'data_world_size': 8,
        'declared_cohorts': sorted(prep_manifest['source_target_totals']),
        'dose': {
            'axis': 'restored-pool dose (grounded-authored + representation-bridge)',
            'authored_subsample': subs['authored_rehearsal'],
            'bridge_subsample': subs['rehearsal_rehearsal'],
            'bracketing_evidence': ('v2 prep (restored pools scrubbed to zero): Stage-B 9-10/14 valid, '
                                    'execution 0/96; v3 prep (both restored at FULL pool): execution '
                                    '59-60/96, Stage-B 4-5/14 — both extremes fail the frozen gate'),
        },
        'evidence_basis': {
            'dose_rationale': ('The restored cohorts carry the execution dialect (OpenHands-compatible '
                               'runtime) training data, but at full pool they damage the Pi-native Stage-B '
                               'first frame; at zero dose the execution dialect collapses. The dose axis '
                               'between the two measured extremes is unexplored; these probes measure it.'),
            'nested_subsample': ('first-N by fixed seeded shuffle: authored seed 882843 (the slice seed; '
                                 'prefix of the full-pool selection order), bridge seed 882844; '
                                 'd25 subset of d50 subset of d75 subset of the v3 full restoration '
                                 '(machine-verified on the built preps by source_record_id sets)'),
            'v10_u256_stop': ('repair-v9r (v2 prep) stopped fail-closed at u256: execution 0/96 '
                              '(bridge control 67/96 in the same run). Ledger: '
                              'docs/validation/e97-pi-native-tool-copy-progress.md (v10 section).'),
            'v3_u256_gate': ('v3 restart arc u256 dual gate: execution 59/60 of 96 (bridge control 67/96), '
                             'Stage-B 4/14 valid 4/14 correct. Retained at '
                             'pi-native-repair9r-full-arc-v3-u256-dual-gate-v1/.'),
            'eval_protocol': ('Stage-B 14-case panel (pi-native-baseline-v1-r3-panel binding) + the fixed '
                              'seeded 24-case execution slice (seed 240977, 6 id-prefix groups x 4 '
                              'families); the frozen >=64/96 execution gate stays bound to the full '
                              '96-case panel; no threshold changes.'),
        },
        'identity_bindings': bindings,
        'new_rl_updates': 0,
        'operator_internal_training_authorized': False,
        'optimizer_updates_authorized': 0,
        'pack_manifest_sha256': pack_manifest_sha,
        'packing_authorized': False,
        'parent_checkpoint': str(BRIDGE[0]),
        'parent_checkpoint_sha256': BRIDGE[1],
        'per_update_cohort_stratification': {
            'min_target_fraction': 0.02,
            'proposed_updates': 32,
            'scheme': 'window-coverage',
            'window_updates': 32,
        },
        'preparation_root': str(prep),
        'proposed_behavioral_gate': GATE,
        'proposed_learning_rate': 1e-05,
        'proposed_updates': 32,
        'purpose': ('Repair-v9r dose screen, probe %s: 32-update diagnostic from the bridge parent on '
                    'the %s preparation (v3 with the restored pools at %d%%/%d%% seeded first-N dose: '
                    'grounded-authored %d of 2,583 records, representation-bridge %d of 2,223); '
                    'measures Stage-B + the 24-case execution slice to bracket the dose axis between the '
                    'v2 (zero-dose) and v3 (full-dose) gate failures. Measurement only; no promotion.' 
                    % (a.dose, a.dose,
                       round(100 * subs['authored_rehearsal']['max_records'] / subs['authored_rehearsal']['full_pool_records']),
                       round(100 * subs['rehearsal_rehearsal']['max_records'] / subs['rehearsal_rehearsal']['full_pool_records']),
                       subs['authored_rehearsal']['max_records'], subs['rehearsal_rehearsal']['max_records'])),
        'sampler_key': a.key,
        'schedule_selection': ('fresh probe-key screen from 1402000 under the window-coverage rule '
                               '(majors >=2%% every update, minors every 32 updates) for this manifest '
                               'sha; key %d re-verified with the real planner and the window-rule '
                               'verifier on the 32-update schedule' % a.key),
        'schedule_sha256': schedule_sha,
        'scheduled_assistant_targets': sum(schedule['source_target_totals'].values()),
        'scheduled_input_tokens': sum(schedule['source_token_totals'].values()),
        'scheduled_source_targets': schedule['source_target_totals'],
        'scheduled_source_unique_records': schedule['source_unique_records'],
        'scheduled_unique_packs': schedule['unique_packs'],
        'scheduled_unique_records': schedule['unique_records'],
        'schema': 'emender-e97-pi-native-repair9-dose-probe-proposal-v1',
        'source_commit': SOURCE_COMMIT,
        'status': 'frozen-proposal-not-authorized',
    }
    out = a.output
    if out.exists():
        raise SystemExit('refusing to overwrite an existing proposal')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proposal, indent=2, sort_keys=True) + '\n')
    print('DOSE_PROBE_PROPOSAL_FROZEN', a.dose, sha(out))


if __name__ == '__main__':
    main()
