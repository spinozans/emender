"""Render part-seven acceptance from actual receipts; never extrapolate rates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / 'receipts/part6'


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    summary = load(R / 'summary.json')
    memory = load(R / 'memory_breakdown.json')
    validation = load(R / 'part7_validation.json')
    best = summary.get('best_sustained_pure') or {}
    lines = [
        '# X40 part 7 — OOM fixes, complete surface, fresh gate measurement', '',
        f"100k acceptance gate: **{summary.get('status', 'missing').upper()}**. "
        f"Sustained surface arms: {summary.get('sustained_arms', 0)}/24; "
        f"matched ratios: {summary.get('matched_ratios', 0)}/12.", '',
        'A rate requires three warmups and 20 synchronized forward, backward, '
        'gradient clipping and AdamW updates. Failures retain their errors and '
        'memory peaks, with no invented rate. The surface is a capacity-matched '
        'compute experiment, not a model-quality comparison. Scales name pure '
        'trainable counts; controls match pure total capacity, including frozen '
        'features, within 0.1%. This is not matched trainable capacity.', '',
        '| Scale | T | Effective B | Pure tok/s | Transformer tok/s | Ratio | Pure peak GiB | Control peak GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'
    ]
    def rate(cell):
        value = cell.get('tokens_per_second')
        return f'{value:,.1f}' if value is not None else cell['status']
    def gib(value):
        return f'{value / 2**30:.3f}' if value is not None else 'unmeasured'
    for row in summary.get('surface', []):
        p, t = row['pure'], row['transformer']
        ratio = row['ratio_to_transformer']
        lines.append(f"| {row['scale']} | {row['context']} | {row['batch']} | "
                     f"{rate(p)} | {rate(t)} | {ratio:.4f} | "
                     f"{gib(p.get('peak_allocated'))} | {gib(t.get('peak_allocated'))} |"
                     if ratio is not None else
                     f"| {row['scale']} | {row['context']} | {row['batch']} | {rate(p)} | {rate(t)} | — | {gib(p.get('peak_allocated'))} | {gib(t.get('peak_allocated'))} |")
    lines += ['', '![Completed scale surface](receipts/part6/scale_surface.png)', '',
        f"Receipt validation: {validation.get('status', 'pending')}; "
        f"{validation.get('two_update_controls', 0)} two-update controls "
        f"({validation.get('controls_passed', 0)} pass / {validation.get('controls_oom', 0)} actual OOM), "
        f"{validation.get('native_fixture_executions', 0)} native fixture executions and 15 local tests. "
        'See receipts/part6/part7_validation.json for independently recomputed rates, '
        'immutable core-source bindings, token/replay checks and explicit failed controls.', '',
        '## Memory fixes and attribution', '',
        'Recorded reentrant checkpointing forces the certified recording path in '
        'both passes, returns the auxiliary gate through checkpoint, and clears '
        'Cell.gate_mean so it cannot retain prior replay graphs. Physical B32 '
        'accumulation weights each microbatch by its token fraction, then clips '
        'and updates once per effective batch. Completed-journal packing clones '
        'only the measured maximum used prefix while retaining writer overflow '
        'assertions and the transient full writer reservation. Its host maximum '
        'synchronization is included in timing. Bounded fp32 projection is used '
        'for both architectures; this differs from part5 dense bf16 projection.', '',
        'Before/after and isolated controls below are observed peak allocated GiB. '
        'Control status is explicit: OOM peak is a partial-run high water, not '
        'the memory required to finish. Controls use two updates to include '
        'initialized optimizer state; only the surface columns claim sustained '
        'throughput. Combined baseline-to-surface savings cannot be attributed '
        'to any single lever. Transformer physical/B32 controls hold reentrant '
        'checkpointing and bounded projection fixed to isolate accumulation; '
        'native journal packing is not applicable to them.', '',
        '| Scale/T/B/arm | Historical rerun | Legacy bounded | Checkpoint physical | Accumulated B32 | Packed B32 | Final surface |',
        '|---|---|---|---|---|---|---|']
    def control(c):
        return f"{c.get('status', 'missing')} / {gib(c.get('peak_allocated'))}"
    for row in memory.get('rows', []):
        if row.get('historical_status') == 'passed':
            continue
        controls = row.get('two_update_controls', {})
        before = f"{row.get('before_status')} / {gib(row.get('before_peak_allocated'))}"
        cells = ([control(controls.get(v, {})) for v in ('legacy', 'recorded_physical', 'accumulated', 'packed')]
                 if row['arm'] == 'pure' else
                 ['not isolated', control(controls.get('recorded_physical', {})),
                  control(controls.get('accumulated', {})), 'not applicable'])
        lines.append('| ' + '/'.join(str(row[k]) for k in ('scale', 'context', 'batch', 'arm')) +
                     ' | ' + ' | '.join([before] + cells + [control(row['after'])]) + ' |')
    lines += ['',
        'The first 22 completed arms used the installed allocator without verified '
        'expandable segments; only the final two 1.3B/T2048 pure arms use the verified '
        'expandable allocator, with GPU checkpoint inputs. This backend difference '
        'is retained in the ratios; the transformer arms retain their earlier '
        'allocator setting. All new isolated memory controls use the verified allocator. '
        'PYTORCH_ALLOC_CONF alone did not activate expandable segments; the legacy '
        'PYTORCH_CUDA_ALLOC_CONF alias was also required and verified in allocator_settings. '
        'Earlier OOMs and the interrupted CPU-input-offload diagnostic remain archived; '
        'offload has exactness gates but no accepted sustained rate.', '',
        'Native stores retain horizons 128/512/2048/8192 and heterogeneous '
        'budgets 64/256/736/1472. Measured journal retention, not a native-store '
        'floor, motivated checkpointing and packing. If future scale snapshots '
        'show native payload dominance after these fixes, a scale-only proposed '
        'quarter budget 16/64/184/368 would reduce payload by 75%. It is '
        'unvalidated and unapplied: require explicit architecture approval, '
        'production parity against that declared policy, and new matched quality '
        'curves before accepting a changed model. No pilot-family budget changes.', '',
        '## Fresh 100k gate', '']
    if best:
        arithmetic = summary['arithmetic']
        lines += [f"Best sustained pure receipt: **{best['tokens_per_second']:,.1f} tok/s**, "
                  f"{best.get('label', best['scale'])}, B{best['batch']}/T{best['context']}. "
                  f"Mean update {arithmetic['mean_seconds']:.6f} s; 100k budget "
                  f"{arithmetic['gate_budget_seconds']:.6f} s; additional required "
                  f"saving {arithmetic['required_savings_seconds']:.6f} s "
                  f"({arithmetic['required_speedup']:.3f}x).", '']
    else:
        lines += ['No fully qualified fresh pilot rate exists.', '']
    decomposition = load(R / 'part7_backward_decomposition.json')
    if decomposition:
        lines += [f"The best point's separate profile measures "
                  f"{decomposition['replay_device_ms']:.3f} ms replay and "
                  f"{decomposition['core_backward_device_ms']:.3f} ms core backward "
                  f"({100 * decomposition['replay_fraction_of_backward_device_busy']:.2f}% replay "
                  "of backward device busy time). These are separate-profile busy times, "
                  "not an additive wall identity, an achieved saving, or an irreducible floor.", '']
    lines += ['| Fresh pilot | B | Raw sustained tok/s | Matched curve | Qualification |',
              '|---|---:|---:|---|---|']
    for qualified, key in ((True, 'pilots'), (False, 'unqualified_pilots')):
        for pilot in summary.get(key, []):
            value = pilot.get('tokens_per_second')
            displayed = f'{value:,.1f}' if value is not None else pilot['status']
            lines.append(f"| {pilot['label']} | {pilot['batch']} | {displayed} | "
                         f"{pilot.get('matched_curve_status', 'see curve receipt')} | "
                         f"{'qualified' if qualified and summary.get('surface_status') == 'passed' else 'unqualified'} |")
    lines.append('')
    fusion = load(R / 'fusion_comparison.json')
    if fusion.get('status') == 'passed':
        lines += [f"Pool fusion removes {fusion['removed_kernel_launches_per_update']} kernels "
                  f"and {fusion['removed_inclusive_activities_per_update']} inclusive device activities "
                  f"per full update. Mean inclusive activities per replay: "
                  f"{fusion['off']['mean']['inclusive_gpu_activities']:.2f} -> "
                  f"{fusion['on']['mean']['inclusive_gpu_activities']:.2f}; sustained speedup "
                  f"{fusion['measured_speedup']:.3f}x. This is partial launch fusion; "
                  "it does not establish complete inclusive replay closure or a hardware floor.", '']
    lines += [f"Integer pool fusion comparison: **{fusion.get('status', 'missing')}**. "
              'See fusion_comparison.json for inclusive correlated device activity '
              'counts; separate profiles are excluded from wall timing. '
              'Unaccepted register, stream and deferred-packing candidates are '
              'retained as experimental sources, with no performance claim.', '',
              '## Validation and residual risks', '',
              f"Production parity: `{json.dumps(summary.get('production_parity', {}), sort_keys=True)}`.", '',
              f"Strict checkpoint byte gates: `{json.dumps(summary.get('checkpoint_byte_parity', {}), sort_keys=True)}`.", '',
              f"Native fixtures: `{json.dumps(summary.get('native_fixtures', {}), sort_keys=True)}`.", '',
              f"Local regressions: `{json.dumps(summary.get('local_tests', {}), sort_keys=True)}`.", '',
              f"Accumulated pilot-v1 curve: `{json.dumps({key: summary.get('curve', {}).get(key) for key in ('status', 'steps', 'tokens', 'max_loss_difference', 'tolerance')}, sort_keys=True)}`.", '',
              'The token-equivalence receipt proves disjoint microbatch coverage, '
              'weighted means, native replay counts and one optimizer update per '
              'effective batch. It does not claim bitwise dense-gradient equality '
              'between physical and accumulated batches. Separate pilot curves '
              'retain the 1e-5 loss contract.', '',
              f"Residual risks: 100k gate status is {summary.get('status', 'missing')}; "
              'no irreducible hardware floor is established. '
              'B1/B32 checkpoint gates do not prove every width/shape bytewise; '
              'the scale quality curve is not measured. Synthetic data are not '
              'identical token tensors across architectures. Some controls can '
              'still OOM. Packing retains transient reservations and host waits; '
              'allocator and pinned-input behavior are backend dependent. '
              'Production defaults retain their existing path; candidates are '
              'scoped probe helpers, not automatically promoted defaults.', '',
              'Evidence: receipts/part6/summary.json, memory_breakdown.json, '
              'token_equivalence.json, part7_commands.log, all raw surface/log '
              'receipts and archived failures. Tool discovery found no '
              'contact_supervisor. GPU work is sequential via run_leased.sh '
              'on physical 1/5/6; protected PID 634691 and cards 0/2/3/4/7 '
              'are untouched. No .pt artifacts, WG, subagents, Frontier, Slurm '
              'or resilient DiLoCo changes. AGENTS.md and CLAUDE.md are unchanged.']
    (ROOT / 'REPORT_PART7.md').write_text('\n'.join(lines) + '\n')
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(R.rglob('*')) if p.is_file() and p.name != 'part7_manifest.json'}
    (R / 'part7_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
