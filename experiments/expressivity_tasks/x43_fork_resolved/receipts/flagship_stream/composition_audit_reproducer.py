"""Reproduce the stream composition audit (read-only, by-offset).
Usage: python composition_audit_reproducer.py"""
import os, math, zlib
P = '/mnt/nvme2n1/erikg/pile.txt'
fd = os.open(P, os.O_RDONLY)
def sample(off, CH=16384):
    b = os.pread(fd, CH, off); n = len(b)
    counts = [0]*256
    for x in b: counts[x] += 1
    ent = -sum((c/n)*math.log2(c/n) for c in counts if c)
    return dict(off=off, n=n, entropy=round(ent,3),
                letters=round((sum(counts[97:123])+sum(counts[65:91]))/n,3),
                digits=round(sum(counts[48:58])/n,3),
                nl_per_kb=round(counts[10]/n*1000,1),
                zlib_ratio=round(len(zlib.compress(b,6))/n,3))
for name, offs in {
  'A search/C14 0-59GB': [1_000_000_000, 8e9, 16e9, 24e9, 32e9, 40e9, 48e9, 56e9],
  'B flagship 100-101.5GB': [100_050_000_000, 100.2e9, 100.4e9, 100.6e9, 100.8e9, 101.0e9, 101.2e9, 101.4e9]}.items():
    print(name)
    for off in offs: print(' ', sample(int(off)))
os.close(fd)
