"""Measurements gated by fresh part12 source-bound parity and real-data curves."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from ..x36_tape_perf import adjoint_pool
from . import part10_profile, part8_surface

NATIVE_LAUNCH = part10_profile.Launch
COMPILER_RECORDS = []

class CompilerLaunch(NATIVE_LAUNCH):
    def __getitem__(self, grid):
        native = super().__getitem__(grid)
        def launch(*args, **kw):
            compiled = native(*args, **kw)
            names = getattr(self.native, 'arg_names', [])
            shape = {name: args[names.index(name)] for name in ('T', 'S', 'DK', 'DV', 'BS', 'PS', 'RS') if name in names and names.index(name) < len(args)}
            COMPILER_RECORDS.append(dict(region=self.name, grid=list(grid), shape=shape,
                registers=getattr(compiled, "n_regs", None), spills=getattr(compiled, "n_spills", None),
                shared_bytes=getattr(getattr(compiled, "metadata", None), "shared", None),
                num_warps=kw.get("num_warps")))
            return compiled
        return launch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part12'


def gate(context, spec):
    assert os.environ.get('X40_LONG_POOL') == '1'
    assert os.environ.get('X40_FRESH_POOL') == '1'
    assert os.environ.get('X40_FUSED_INIT') == '1'
    assert os.environ.get('X40_STABLE_MAPPING') == '1'
    batched = os.environ.get('X40_BATCHED_READS')
    assert batched in ('0', '1')
    if batched == '0':
        assert json.loads((OUT/'parity_fused_forward_random.json').read_text())['status'] == 'passed'
    import torch
    assert torch.cuda.memory._snapshot().get('allocator_settings', {}).get('expandable_segments')
    sha = hashlib.sha256((ROOT / 'backward.py').read_bytes()).hexdigest()
    support = json.loads((OUT/'support_gate.json').read_text())
    assert support['status'] == 'passed'
    for name in ('forward.py', 'model.py', 'kernels.py', 'fused_init.py'):
        assert support['sources'][name] == hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for fixture in ('', '_empty', '_popped', '_decay', '_correction'):
        result = json.loads((OUT / f'parity{fixture}.json').read_text())
        assert result['status'] == 'passed' and result['sources']['backward.py'] == sha
        assert result['sources']['forward.py'] == hashlib.sha256((ROOT/'forward.py').read_bytes()).hexdigest()
        assert len(result['rows']) == 8
    for arm in ('base', 'candidate'):
        curve = json.loads((OUT / f'curve_{arm}_T2048.json').read_text())
        assert curve['status'] == 'passed' and curve['tokens'] >= 2_000_000
        assert curve['sources']['backward.py'] == sha
        for name in ('model.py', 'scale_surface.py'):
            assert curve['sources'][name] == hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('profile', 'sustained', 'split', 'fused-forward'))
    parser.add_argument('--context', type=int, choices=(512, 2048), required=True)
    parser.add_argument('--arm', choices=('pure', 'transformer'), default='pure')
    a = parser.parse_args()
    out = OUT / a.mode
    out.mkdir(exist_ok=True)
    support = []
    original_build = adjoint_pool.build
    def pooled_build(entry, allocation):
        rows, inverse, counts = original_build(entry, allocation)
        support.append(dict(slots=entry.shape[-1], stores=entry.shape[0], reserved_rows=rows.shape[-1], union_counts=counts.detach()))
        return rows, inverse, counts
    if a.mode == 'fused-forward':
        fusion = json.loads((OUT/'parity_fused_forward_random.json').read_text())
        assert fusion['status'] == 'passed'
        for name in ('backward.py', 'forward.py'):
            assert fusion['sources'][name] == hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        os.environ['X40_BATCHED_READS'] = '0'
    (out/f'environment_{a.arm}_T{a.context}.json').write_text(json.dumps(dict(
        flags={name:os.environ.get(name, '0') for name in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_LONG_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')},
        python=sys.executable, physical_gpu=os.environ.get('GPU_LEASE_HELD'),
        allocator_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        driver_sha256=hashlib.sha256((ROOT/'part12_measure.py').read_bytes()).hexdigest()), indent=2) + '\n')
    if a.mode in ('profile', 'split', 'fused-forward'):
        sys.argv = [sys.argv[0], '--context', str(a.context), '--arm', a.arm]
        if a.mode == 'split':
            from . import backward
            from .part12_decompose import Split
            # A split cannot be timed until its separate byte gate passes.
            split_gate = json.loads((OUT / 'parity_split_random.json').read_text())
            assert split_gate['status'] == 'passed'
            for name in ('backward.py', 'part12_decompose.py', 'part12_split_generated.py'):
                assert split_gate['sources'][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            split = Split()
            with patch.object(backward, 'reverse_kernel', split), patch.object(part10_profile, 'OUT', out), patch.object(part10_profile, 'validate_parity', gate):
                part10_profile.main()
            profile = json.loads((out/f'profile_pure_T{a.context}.json').read_text())
            phases = ['undo', 'strength_restore_and_store_event_adjoints', 'score_softmax_recompute', 'read_adjoints']
            seconds = [sum(row[i] for row in split.records[-192:]) for i in range(4)]
            exclusive = dict(profile['exclusive_seconds'])
            wrapper = exclusive.pop('reverse_fused_read_math_and_adjoints')
            overhead = wrapper - sum(seconds)
            assert overhead >= 0
            exclusive.update(dict(zip(phases, seconds)))
            exclusive['split_reverse_launch_setup_overhead'] = overhead
            assert abs(sum(exclusive.values()) + profile['unattributed_seconds'] - profile['wall_seconds']) < 1e-7
            (out / f'regions_T{a.context}.json').write_text(json.dumps(dict(
                phases=phases, seconds=seconds, raw_reverse_call_seconds=split.records[-192:], exclusive_update_seconds=exclusive,
                wall_seconds=profile['wall_seconds'], unattributed_seconds=profile['unattributed_seconds'],
                reverse_wrapper_seconds=wrapper, reverse_wrapper_overhead_seconds=overhead,
                reverse_calls=len(split.records[-192:]), split_launches=192*a.context*4, total_calls_including_warmup=split.calls,
                limitation='Split-kernel synchronized wall, includes spill/reload and extra launches; not a decomposition of the unchanged fused binary.'), indent=2) + '\n')
        else:
            with patch.object(part10_profile, 'OUT', out), patch.object(part10_profile, 'validate_parity', gate), patch.object(part10_profile, 'Launch', CompilerLaunch), patch.object(adjoint_pool, 'build', pooled_build):
                part10_profile.main()
            import torch
            properties = torch.cuda.get_device_properties(0)
            compiler = dict(records=COMPILER_RECORDS, multiprocessors=properties.multi_processor_count,
                note='Compiler resources and launch grid, not measured achieved occupancy. A 32-CTA store-owner launch cannot occupy more than 32 SMs concurrently.')
            (out / f'compiler_{a.arm}_T{a.context}.json').write_text(json.dumps(compiler, indent=2) + '\n')
            if a.arm == 'pure':
                for row in support:
                    row['union_counts'] = row['union_counts'].cpu().tolist()
                (out / f'support_T{a.context}.json').write_text(json.dumps(dict(rows=support[-192:], scope='Last full diagnostic update, actual 1.3B model; not inferred from live slots'), indent=2) + '\n')
    else:
        def specs(batch, arm):
            return [dict(name='matched_checkpoint_B32', microbatch=32, checkpoint=True, packed=arm=='pure', offload=False)]
        sys.argv = [sys.argv[0], '--scale', '1.3B', '--arm', a.arm, '--batch', '64', '--context', str(a.context), '--steps', '20']
        with patch.object(part8_surface, 'OUT', out), patch.object(part8_surface, 'validate_parity', gate), patch.object(part8_surface, 'envelope_specs', specs):
            part8_surface.main()


if __name__ == '__main__':
    main()
