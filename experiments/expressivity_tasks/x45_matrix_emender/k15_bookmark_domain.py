"""Report-only exact pile_set_name lookup; no domain input to the model."""
import mmap
from pathlib import Path
import struct


def domains(offsets, path=Path('/mnt/nvme2n1/erikg/pilemap_full/meta.bin')):
    result={}
    with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
        magic,version,n,_,shards,names,p,q,width,_=struct.unpack_from('<4sI6QIQ',m)
        base=68+p+6*names+q+6*shards
        assert (magic,version,width)==(b'XPM1',1,34) and len(m)==base+n*width
        for offset in offsets:
            lo,hi=0,n
            while lo<hi:
                mid=(lo+hi)//2
                if struct.unpack_from('<Q',m,base+mid*width)[0]<offset:lo=mid+1
                else:hi=mid
            pos,length,_,label=struct.unpack_from('<QQHI',m,base+lo*width)
            assert pos==offset and length>=129 and label<names
            start=68+struct.unpack_from('<I',m,68+p+4*label)[0]
            size=struct.unpack_from('<H',m,68+p+4*names+2*label)[0]
            result[offset]=m[start:start+size].decode()
    return result
