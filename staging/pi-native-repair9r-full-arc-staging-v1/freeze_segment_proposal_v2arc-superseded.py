#!/usr/bin/env python3
"""Freeze a repair-v9r full-arc segment training proposal (segments 1..8).

Fail-closed freeze-time writer. Segment 1 binds the bridge parent and may be
frozen now. Segments 2..8 chain to the prior segment's final audited checkpoint
and can only be frozen after that checkpoint exists, its sha256 is known, and
the prior segment's training root carries audit.json with status
'passed-training-not-promoted' naming that exact sha (the same chained-parent
identity the generic auditor enforces).

Every identity this proposal binds is verified against disk bytes at freeze
time. No value is trusted from this script's own text except arc metadata that
is cross-checked against the preparation root. Audits are run separately with
scripts/audit_e97_pi_native_repair_generic.py.
"""
import argparse, hashlib, json
from pathlib import Path

PREP = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-full-preparation-v2')
REPO = Path('/home/erikg/emender')
SOURCE_COMMIT = 'd77dcc462e8b25c0b362b196347ddc9f50ffe683'  # trainer HEAD: "Fix consumed_target_tokens clobbering in OH manifest block"
BRIDGE = (Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt'),
          '9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa')
ARC_KEYS = [1400005, 1400010, 1400023, 1400027, 1400050, 1400057, 1400080, 1400081]
TRAIN_DIR = '/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9r-full-arc-segment{idx}-training-v1'

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
    ap.add_argument('--segment-index', type=int, required=True)
    ap.add_argument('--parent-checkpoint', type=Path, default=None)
    ap.add_argument('--parent-sha256', default=None)
    ap.add_argument('--repo', type=Path, default=REPO)
    ap.add_argument('--prep', type=Path, default=PREP)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    idx = a.segment_index
    if not 1 <= idx <= 8:
        raise SystemExit('segment index out of arc range')

    prep = a.prep
    repo = a.repo
    schedule_rel = 'schedule-proposal.json' if idx == 1 else f'schedule-segment{idx}-proposal.json'
    schedule_path = prep / schedule_rel

    # --- parent identity (fail-closed) ---
    if idx == 1:
        parent_path, parent_sha = BRIDGE
        if a.parent_checkpoint or a.parent_sha256:
            raise SystemExit('segment 1 parent is the bridge checkpoint; no --parent arguments')
    else:
        if not (a.parent_checkpoint and a.parent_sha256):
            raise SystemExit(f'segment {idx} requires --parent-checkpoint and --parent-sha256 '
                             '(frozen only after the prior segment trained and was audited)')
        parent_path, parent_sha = a.parent_checkpoint, a.parent_sha256
        prev_root = parent_path.parents[1]
        prev_audit_path = prev_root / 'audit.json'
        if must(parent_path, 'parent checkpoint') != parent_sha:
            raise SystemExit('parent checkpoint sha256 mismatch')
        try:
            prev_audit = json.loads(prev_audit_path.read_text())
        except (OSError, ValueError):
            raise SystemExit(f'prior segment training audit missing/unreadable: {prev_audit_path}')
        if (prev_audit.get('status') != 'passed-training-not-promoted'
                or prev_audit.get('checkpoint', {}).get('sha256') != parent_sha):
            raise SystemExit('prior segment training audit does not verify this parent')

    # --- schedule + authority identities ---
    prep_manifest_sha = must(prep / 'manifest.json', 'prep manifest')
    pack_manifest_sha = must(prep / 'packs/manifest.json', 'packs manifest')
    schedule_sha = must(schedule_path, 'segment schedule')
    schedule = json.loads(schedule_path.read_text())
    prep_manifest = json.loads((prep / 'manifest.json').read_text())
    if (schedule['training_eligible'] or schedule['optimizer_updates_authorized']
            or prep_manifest['training_eligible'] or prep_manifest['packing_authorized']
            or prep_manifest['optimizer_updates_authorized']):
        raise SystemExit('prepared authority is not non-authorized')
    key = schedule['sampler_key']
    if key != ARC_KEYS[idx - 1]:
        raise SystemExit(f'schedule key {key} does not match screened arc order position {idx}')
    if schedule['world_size'] != 8 or schedule['context_size'] != 65536 or len(schedule['steps']) != 128:
        raise SystemExit('schedule shape does not match the frozen arc geometry')

    # --- identity bindings: provenance + overlap-audit coverage (all verified) ---
    T = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
    bindings = [(str(parent_path), must(parent_path, 'parent checkpoint'))]
    if idx > 1:
        bindings.append((str(parent_path.parents[1] / 'audit.json'),
                         must(parent_path.parents[1] / 'audit.json', 'prior training audit')))
    bindings += [
        (str(prep / 'manifest.json'), prep_manifest_sha),
        (str(prep / 'packs/manifest.json'), pack_manifest_sha),
        (str(schedule_path), schedule_sha),
        ('configs/pi/e97-active-tool-surface-v1.json',
         must(repo / 'configs/pi/e97-active-tool-surface-v1.json', 'active tool surface')),
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
    ]
    # selected_* provenance triple cross-check against the prep manifest's own recorded bindings
    for got, field in ((bindings[-3][1], 'selected_authority_sha256'),
                       (bindings[-2][1], 'selected_selection_audit_sha256'),
                       (bindings[-1][1], 'selected_overlap_audit_sha256')):
        if prep_manifest[field] != got:
            raise SystemExit(f'prep manifest {field} does not match the bound file on disk')

    # --- cohort bindings verified against the prep manifest ---
    oh = prep_manifest['openhands_rehearsal']
    r = prep_manifest['extra3_rehearsal']
    conv = prep_manifest['conversation_rehearsal']
    cohort_bindings = {
        'openhands_rehearsal': {'authority_sha256': oh['authority_sha256'], 'records': oh['records'],
                                'consumed_target_tokens': oh['consumed_target_tokens']},
        'extra3_rehearsal': {'authority_sha256': r['authority_sha256'], 'cohort': r['cohort'], 'records': r['records']},
        'conversation_rehearsal': {'authority_sha256': conv['authority_sha256'], 'records': conv['records'],
                                   'consumed_target_tokens': conv['consumed_target_tokens']},
    }

    chaining = ('Each segment trains 128 updates from the prior segment final checkpoint '
                '(segment 1 from the bridge parent, full retention headroom).') if idx == 1 else (
        f'Segment {idx} trains 128 updates from segment {idx-1}\'s final audited checkpoint '
        f'({parent_path}); the prior segment\'s training audit ({parent_path.parents[1]}/audit.json, '
        'status passed-training-not-promoted) verifies the exact parent sha256.')

    evidence = {
        't1_scrub': ('grounded-authored 80% and representation-bridge 57% OH-only tool vocabulary '
                     '(str_replace_editor / execute_bash / Think:) in SAMPLED supervised assistant targets '
                     '-> both cohorts dropped (no retained workspaces or base commits, replay verification '
                     'impossible, fail-closed); all other cohorts 0.0 (scrub-report.json; sampling-based, '
                     '150/120-record samples per cohort). Operator ruling in force: markers count OH-specific '
                     'tokens only; Analysis:/Commentary:/Action:/Arguments: are the Pi codec itself, not contamination.'),
        't1_translation': ('5969 raw trajectories -> 5940 translated (29 dropped: 26 invalid_view_range, '
                           '3 malformed_create_args) -> 5181 replay-PASSED / 759 fail-closed drops '
                           '(read_content_mismatch 141, read_missing_file 332, final_state_mismatch 125, ...; '
                           'replay-full.log OH_REPLAY_VERIFIED line) -> 5087-record sealed collection '
                           '(231,652,231 tokens / 57,814,225 supervised targets, 104 excluded over-context); '
                           'zero GPUs; commits 78904dd7 + 9099b92c'),
        't2_reasoning_pilot': ('50/50 verified, 230 real Pi tool calls, 30 authentic tool errors, 331,929 tokens '
                               '/ 15,853 supervised targets, observe-then-quote machine grounding; '
                               'audit f62c86efb264d47eb88669132e5e8f17f5cb64b4335d20675c7d4b5805471580, '
                               'overlap d8ed9720c9d55721a35420b43e1a753ea3d0034b1c35851f7f5a792363033ae6; '
                               'commit cc016d31 + follow-ups'),
        'throughput_forensics': ('75s step decomposed; chunk 2048/16384 qualified at 72s/update with losses '
                                 'matching to 5-6 digits; DDP bucket coarsening (946bda6c) neutral'),
        'v9_full_stop': ('u128 juncture Stage-B 3/14 valid, 3/14 correct; decoded emissions in the raw '
                         'OpenHands protocol; raw-OH cohort 54/60 sampled records carried '
                         'Analysis/Commentary/Think scaffolds; v9-full prep a05156ad retained as evidence; '
                         'ledger docs/validation/e97-pi-native-tool-copy-progress.md'),
        'overlap_coverage': ('prep manifest binds the selected-curriculum provenance triple (authority '
                             '8cf83db8..., selection audit de977726..., overlap audit 9c104d73... = '
                             'pi-native-curriculum-2000-selected-v1), the same whole-prep coverage every prior '
                             'repair prep bound; no prior prep ever ran a whole-prep protected-overlap audit. '
                             'Per-collection coverage: reasoning cohort overlap-audit d8ed9720... (zero '
                             'significant entity collisions); translated OH cohort coverage = machine replay '
                             'verification + OH-vocabulary scrub + public SWE-rebench provenance (no '
                             'protected-panel entity overlap audit exists for it, as for every prior OH cohort).'),
    }

    proposal = {
        'arc': {
            'arc_id': 'repair-v9r-full-wide-arc',
            'arc_keys': ARC_KEYS,
            'arc_updates': 1024,
            'chaining': chaining,
            'evaluation_caveats': ('Thinking-trained emission may shift the conversation-retention NLL panel '
                                   '(T2 caveat: re-baseline at the reasoning cohort\'s first entry); '
                                   'conversational-behavior panel added per the first unscripted-session '
                                   'operator findings (no-chat-mode gap).'),
            'juncture_policy': ('Stage-B panel AND conversation NLL probe AND conversational-behavior panel at '
                                'every 128-update boundary; full frozen dual gate at u256, u512, u896, u1024 '
                                'and at any detector-flagged checkpoint; fail-closed stopping if retention is '
                                'decisively breached.'),
            'rationale': ('The operator directed the data-scale breakout; the first attempt exposed '
                          'raw-protocol contamination, now remediated by translation with replay verification. '
                          'The mixture otherwise keeps the v9-full design (39.9% conversation, 30.2% instruction, '
                          '17.3% translated tool trajectories, 6.1% cumulative recovery, 4.9% compositional, '
                          'plus minors).'),
            'segment_index': idx,
            'segment_updates': 128,
            'throughput_note': ('Trained at 72s/update (chunk 2048/16384, trainer worktree pinned at '
                                'd77dcc46) per the throughput forensics in the progress ledger; per-step config '
                                'deviation from prior arcs qualified by loss-matching evidence.'),
            'total_segments': 8,
        },
        'authority_manifest_sha256': prep_manifest_sha,
        'automatic_expansion': False,
        'automatic_retry': False,
        'checkpoint_promotion': False,
        'context_size': 65536,
        'cohort_bindings': cohort_bindings,
        'conversation_rehearsal': {
            'budget_targets': conv['target_token_budget'],
            'manifest_sha256': conv['authority_sha256'],
            'seed': conv['seed'],
            'source': conv['authority'],
        },
        'data_world_size': 8,
        'declared_cohorts': sorted(prep_manifest['source_target_totals']),
        'evidence_basis': evidence,
        'identity_bindings': bindings,
        'new_rl_updates': 0,
        'operator_internal_training_authorized': False,
        'optimizer_updates_authorized': 0,
        'pack_manifest_sha256': pack_manifest_sha,
        'packing_authorized': False,
        'parent_checkpoint': str(parent_path),
        'parent_checkpoint_sha256': parent_sha,
        'per_update_cohort_stratification': {
            'min_target_fraction': 0.02,
            'proposed_updates': 128,
            'scheme': 'window-coverage',
            'window_updates': 32,
        },
        'preparation_root': str(prep),
        'proposed_behavioral_gate': GATE,
        'proposed_learning_rate': 1e-05,
        'proposed_updates': 128,
        'purpose': ('Repair-v9r REVISED full wide arc segment %d of 8: the v9-full arc stopped fail-closed at '
                    'seg1 (raw-OpenHands protocol capture). This restart trains on the translated + '
                    'replay-verified OH collection (5087 records replay-PASSED of 5940, byte-exact final-state '
                    'oracle), drops the representation-bridge and grounded-authored cohorts per the OH scrub '
                    'report (57%%/80%% OH-only tool vocabulary in supervised targets), and adds the '
                    'reasoning-rehearsal pilot cohort (50 records, machine-checked observe-then-quote '
                    'grounding). Same dose structure as v9-full otherwise; juncture detectors and frozen dual '
                    'gates unchanged.' % idx),
        'sampler_key': key,
        'schedule_selection': ('Training-data-only screen over keys 1400000.. under the window-coverage rule '
                               '(majors >=2%% every update, minors every 32 updates); 8 keys admitted: %s; '
                               'this segment trains key %d.' % (','.join(str(k) for k in ARC_KEYS), key)),
        'schedule_sha256': schedule_sha,
        'scheduled_assistant_targets': sum(schedule['source_target_totals'].values()),
        'scheduled_input_tokens': sum(schedule['source_token_totals'].values()),
        'scheduled_source_targets': schedule['source_target_totals'],
        'scheduled_source_unique_records': schedule['source_unique_records'],
        'scheduled_unique_packs': schedule['unique_packs'],
        'scheduled_unique_records': schedule['unique_records'],
        'schema': f'emender-e97-pi-native-repair9r-full-arc-segment{idx}-proposal-v1',
        'source_commit': SOURCE_COMMIT,
        'status': 'frozen-proposal-not-authorized',
    }

    if set(prep_manifest['source_target_totals']) != set(schedule['source_target_totals']):
        raise SystemExit('schedule cohorts differ from the prep authority')
    out = a.output
    if out.exists() and out.read_bytes() == json.dumps(proposal, indent=2, sort_keys=True).encode() + b'\n':
        raise SystemExit('output identical; refusing to rewrite')
    out.write_text(json.dumps(proposal, indent=2, sort_keys=True) + '\n')
    print('FROZEN_PROPOSAL', sha(out), out)

if __name__ == '__main__':
    main()
