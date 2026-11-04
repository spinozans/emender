"""Fast raw suffix metadata; consultation bits use the unchanged v2 decoder."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np
from .c14_documents import sha
from .c14_fast_features import decode_shard


def load_context_decoder():
    source=Path(__file__).with_name('c14_b_contexts.c')
    path=Path('/home/erikg/c14_consultation_features/native_decode')/(sha(source)+'.contexts.so')
    if not path.exists():
        temporary=path.with_suffix('.tmp.so')
        subprocess.run(['cc','-O3','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
        temporary.replace(path)
    f=ctypes.CDLL(str(path)).c14_contexts
    f.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t];f.restype=ctypes.c_int
    return f


def eval_shard(data,count,decoder,context_decoder):
    ids,lengths,answers=decode_shard(data,count,decoder)
    contexts=np.zeros((count,128),dtype=np.int64)
    result=context_decoder(data,len(data),contexts.ctypes.data,count)
    if result!=count:raise ValueError(f'raw context decoder failed: {result}')
    return ids,lengths,answers,contexts


def load_owner_decoder():
    load_context_decoder()
    source=Path(__file__).with_name('c14_b_contexts.c')
    path=Path('/home/erikg/c14_consultation_features/native_decode')/(sha(source)+'.contexts.so')
    f=ctypes.CDLL(str(path)).c14_owner_metadata
    f.argtypes=[ctypes.c_void_p]*5+[ctypes.c_size_t]*2;f.restype=ctypes.c_int
    return f


def owner_metadata(raw,contexts,valid,decoder):
    raw=np.ascontiguousarray(raw,dtype=np.uint8)
    contexts=np.ascontiguousarray(contexts,dtype=np.int64)
    valid=np.ascontiguousarray(valid,dtype=np.int32)
    owners=np.empty(contexts.shape,dtype=np.int32);expiry=np.empty_like(owners)
    result=decoder(raw.ctypes.data,contexts.ctypes.data,valid.ctypes.data,
        owners.ctypes.data,expiry.ctypes.data,*contexts.shape)
    if result:raise ValueError(f'owner metadata bounds failure: {result}')
    return owners,expiry
