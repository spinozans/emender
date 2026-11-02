"""Fresh six-arm C6 run with the corrected NVIDIA GDN-2 constructor.

c6.py is retained byte-for-byte as the hash-bound first execution source.
The corrected Head has identical Emender/transformer branches.
"""
from . import c6
from .c6_gdn2 import Head


if __name__=='__main__':
    c6.OUT.mkdir(exist_ok=True)
    c6.Head=Head
    c6.train()
    meta=c6.json.loads((c6.OUT/'training.json').read_text())
    meta['gdn2_code_sha256']=c6.digest(c6.ROOT/'c6_gdn2.py')
    meta['gdn2_protocol_sha256']=c6.digest(c6.ROOT/'PREREGISTRATION_C6_GDN2.md')
    meta['gdn2_physical_gpu']=meta['physical_gpu']
    meta['entry_sha256']=c6.digest(c6.ROOT/'c6_entry.py')
    c6.write('training.json',meta)
