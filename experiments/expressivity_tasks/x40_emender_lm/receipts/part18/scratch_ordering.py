"""Rejected-until-certified PTX scheduling experiment, never a production hook.

Bracket each generated shared-memory access with a CTA barrier. This preserves
the original arithmetic, assertion text, reduction instructions and four-warps
launch. It tests whether ordering *within* compiler-generated scratch phases is
needed; it does not prove the barriers legal under every control-flow path.
"""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from experiments.expressivity_tasks.x40_emender_lm import forward

OUT = Path(__file__).resolve().parent


def install(*, unique_writers=False):
    original = forward.scan_kernel

    class Ordered:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kernel = original.warmup(*args, grid=grid, **kwargs)
                if not getattr(kernel, 'part18_ordered', False):
                    assert kernel.module is None
                    ptx = kernel.asm['ptx']
                    original_ptx = ptx
                    if unique_writers:
                        # Shape-bound diagnostic: these register/address maps
                        # were derived from this exact PTX, not guessed for
                        # other compiler specializations.
                        assert hashlib.sha256(ptx.encode()).hexdigest() == 'e6ec7da79ae4a299657f2f062438d811ad453c614fbee9a118c3922c5c3c9070'
                        ptx, count = re.subn(r'(\.reg\s+\.pred\s+%p<\d+>;)',
                            r'\1\n\t.reg .pred %p_part18_owner32, %p_part18_owner2;', ptx, count=1)
                        assert count == 1
                        marker = 'mov.u32 \t%r1, %tid.x;'
                        assert ptx.count(marker) == 1
                        ptx = ptx.replace(marker, marker + '\n\tsetp.lt.u32 %p_part18_owner32, %r1, 32;\n\tsetp.lt.u32 %p_part18_owner2, %r1, 2;')
                        for address, owner in (('60', '32'), ('70', '2')):
                            pattern = r'(?m)^(\s*)st.shared.v4.b32(\s+\[%r' + address + r'\].*)$'
                            ptx, count = re.subn(pattern, r'\1@%p_part18_owner' + owner + r' st.shared.v4.b32\2', ptx)
                            assert count == 1
                    lines = ptx.splitlines(keepends=True)
                    changed = []
                    count = 0
                    for line in lines:
                        if re.search(r'\b(?:ld|st|ldmatrix|stmatrix)\.[^;]*\bshared\.', line):
                            changed.extend(['\tbar.sync 0; // part18 scratch before\n', line,
                                            '\tbar.sync 0; // part18 scratch after\n'])
                            count += 1
                        else:
                            changed.append(line)
                    changed = ''.join(changed)
                    assert count > 0
                    assert ''.join(x for x in changed.splitlines(keepends=True)
                                   if '// part18 scratch ' not in x) == ptx
                    arch = re.search(r'^\.target\s+(sm_\d+)', ptx, re.M).group(1)
                    prefix = OUT / (('scratch_v3_' if unique_writers else 'scratch_v2_') + kernel.hash)
                    for name, content in (('original.ptx', original_ptx), ('ordered.ptx', changed)):
                        with gzip.open(str(prefix) + '.' + name + '.gz', 'wt') as f:
                            f.write(content)
                    with tempfile.TemporaryDirectory(prefix='x40_part18_') as directory:
                        source = Path(directory) / 'ordered.ptx'
                        binary = Path(directory) / 'ordered.cubin'
                        source.write_text(changed)
                        command = ['/home/erikg/emender/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas',
                                   '-lineinfo', '--fmad=false', '--gpu-name=' + arch,
                                   str(source), '-o', str(binary)]
                        subprocess.run(command, check=True)
                        kernel.kernel = binary.read_bytes()
                    with gzip.open(str(prefix) + '.ordered.cubin.gz', 'wb') as f:
                        f.write(kernel.kernel)
                    record = dict(kernel_hash=kernel.hash, inserted_barriers=2*count,
                                  original_ptx_sha256=hashlib.sha256(original_ptx.encode()).hexdigest(),
                                  ordered_ptx_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                                  cubin_sha256=hashlib.sha256(kernel.kernel).hexdigest(),
                                  scheduling_only_text_proof=not unique_writers,
                                  unique_writers=unique_writers, production=False,
                                  control_flow_legality='unproved; diagnostic only', arch=arch)
                    Path(str(prefix) + '.json').write_text(json.dumps(record, indent=2) + '\n')
                    print(json.dumps(record), flush=True)
                    kernel.part18_ordered = True
                return original[grid](*args, **kwargs)
            return launch
    forward.scan_kernel = Ordered()
