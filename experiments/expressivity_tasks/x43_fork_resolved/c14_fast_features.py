"""Bounded in-memory shard decoding, shared by consultation architectures.

No dense corpus copy is written. Native arithmetic uses doubles in original
branch order and emits float32 once, as does the registered Python path.
"""
import ctypes
import hashlib
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_decoder():
    source = ROOT/'c14_fast_features.c'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = Path('/home/erikg/c14_consultation_features/native_decode')
    directory.mkdir(exist_ok=True)
    library = directory/(digest+'.so')
    if not library.exists():
        if shutil.disk_usage(directory).free < 150_000_000_000:
            raise RuntimeError('STOP: less than 150 GB free')
        temporary = library.with_suffix('.tmp.so')
        subprocess.run(['cc', '-O3', '-fno-fast-math', '-ffp-contract=off', '-shared', '-fPIC',
                        str(source), '-lm', '-o', str(temporary)], check=True)
        temporary.replace(library)
    lib = ctypes.CDLL(str(library))
    decode = lib.c14_decode
    decode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    decode.restype = ctypes.c_int
    return decode


def decode_shard(data, capacity, decoder=None):
    decoder = decoder or load_decoder()
    values = np.zeros((capacity, 128, 260), dtype=np.float32)
    ids = np.empty((capacity, 2), dtype=np.uint64)
    lengths = np.empty(capacity, dtype=np.uint16)
    count = decoder(data, len(data), values.ctypes.data, ids.ctypes.data,
                    lengths.ctypes.data, capacity)
    if count < 0:
        raise ValueError(f'invalid sparse shard: native error {count}')
    if count != capacity:
        raise ValueError(f'shard window count {count} != {capacity}')
    return ids, lengths, values
