import json
import pytest
from scripts.audit_e97_triton_cache_overlap import normalized_ptx,audit


def test_debug_paths_ignored_but_later_executable_text_retained():
    template='.version 8.7\n.file 1 "{path}"\n.loc 1 2 3\nmov.u32 %r1, 1;\n.section .debug_info\n{{\n.b8 {byte}\n}}\nadd.u32 %r2, %r1, 2;\n'
    a=normalized_ptx(template.format(path='a',byte=12));b=normalized_ptx(template.format(path='b',byte=15))
    assert a==b and 'add.u32' in a
    assert a!=normalized_ptx(template.format(path='a',byte=12).replace('mov.u32 %r1, 1','mov.u32 %r1, 2'))


def test_multiple_debug_sections_removed_without_skipping_alternate_sections():
    source='.version 8.7\n'+''.join('.section .debug_'+name+'\n{\n.b8 12\n}\n' for name in ('info','abbrev','line'))
    assert normalized_ptx(source)=='.version 8.7'


def test_unterminated_debug_rejected():
    with pytest.raises(ValueError,match='unterminated'):normalized_ptx('.version 8.7\n.section .debug_info\n{\n')


def test_ptx_overlap_is_not_compiler_metadata_or_cubin_identity(tmp_path):
    for name,warps in [('left',4),('right',8)]:
        root=tmp_path/name;root.mkdir()
        (root/'kernel.ptx').write_text('.version 8.7\nmov.u32 %r1, 1;\n')
        (root/'kernel.json').write_text(json.dumps(dict(hash=name,num_warps=warps)))
        (root/'kernel.cubin').write_bytes(name.encode())
    result=audit(tmp_path/'left',tmp_path/'right')['kernels']['kernel']
    assert result['right_ptx_subset_of_left']
    assert not result['right_ptx_and_metadata_subset_of_left']
