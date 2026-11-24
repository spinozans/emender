"""Require a fresh local adopted lease held by an ancestor of this process."""
from pathlib import Path
import os
import socket
import time


def approved():
    gpu=os.getenv('GPU_LEASE_HELD'); pid=os.getpid()
    if gpu not in ('0','1') or os.getenv('CUDA_VISIBLE_DEVICES')!=gpu: return False
    try:
        path=Path(os.getenv('GPU_LEASE_DIR',Path(__file__).resolve().parents[3]/'.wg/gpu_leases'))/gpu
        row=dict(s.split('=',1) for s in path.read_text().splitlines())
        while pid>1:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            if str(pid)==row['pid']:
                return (fields[19]==row['starttime'] and row['gpu']==gpu and row['host']==socket.gethostname() and row['adopted']=='1' and 0<=time.time()-int(row['heartbeat'])<30)
            pid=int(fields[1])
    except (OSError,ValueError,KeyError): pass
    return False
