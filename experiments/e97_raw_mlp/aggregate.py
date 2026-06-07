"""Parse e97-raw-plus run logs -> matched comparison table.
Extracts param count, final train loss (last-100 avg), last held-out val loss,
mean tok/s, and converts val loss to BPB. REAL logs only."""
import re, sys, glob, os, math

BPT = 3.8645  # bytes/token of held-out slice (p50k_base)
BPB_FACTOR = 1.0 / (math.log(2) * BPT)  # 0.37332

LABELS = {
    'e97raw_mixer':       'e97-raw (mixer-only)  [baseline]',
    'e97raw_mlp_bolt':    'e97-raw + MLP bolt-on (d10,h323,r1.5)',
    'e97raw_mlp_realloc': 'e97-raw + MLP realloc (d21,h128,r2.0)',
    'gdn2_mlp':           'gdn2-mlp (d17,h8,r2.854) [ref]',
}
ORDER = ['e97raw_mixer','e97raw_mlp_bolt','e97raw_mlp_realloc','gdn2_mlp']

def parse(path):
    txt = open(path).read()
    d = {}
    m = re.search(r'Model: Level \S+, ([\d,]+) parameters', txt)
    d['params'] = int(m.group(1).replace(',','')) if m else None
    m = re.search(r'FINAL_LOSS_LAST100: ([\d.]+)', txt)
    d['train_loss'] = float(m.group(1)) if m else None
    vals = re.findall(r'validation loss: ([\d.]+)', txt)
    d['val_loss'] = float(vals[-1]) if vals else None
    d['val_history'] = [float(v) for v in vals]
    toks = [float(t) for t in re.findall(r'tok/s (\d+)', txt)]
    d['tok_s'] = sum(toks[2:])/len(toks[2:]) if len(toks) > 3 else (sum(toks)/len(toks) if toks else None)
    steps = re.findall(r'step +(\d+) \|', txt)
    d['last_step'] = int(steps[-1]) if steps else 0
    return d

def main(rundir):
    rows = []
    for name in ORDER:
        p = os.path.join(rundir, name + '.log')
        if not os.path.exists(p):
            continue
        d = parse(p); d['name'] = name
        d['bpb'] = d['val_loss']*BPB_FACTOR if d['val_loss'] else None
        rows.append(d)
    hdr = f"{'candidate':40s} {'params(M)':>10s} {'train':>7s} {'val(held)':>9s} {'BPB':>6s} {'tok/s':>7s} {'step':>5s}"
    print(hdr); print('-'*len(hdr))
    for d in rows:
        print(f"{LABELS[d['name']]:40s} {d['params']/1e6:10.1f} "
              f"{d['train_loss'] if d['train_loss'] else float('nan'):7.4f} "
              f"{d['val_loss'] if d['val_loss'] else float('nan'):9.4f} "
              f"{d['bpb'] if d['bpb'] else float('nan'):6.4f} "
              f"{d['tok_s'] if d['tok_s'] else float('nan'):7.0f} {d['last_step']:5d}")
    print()
    for d in rows:
        print(f"  {d['name']:22s} val_history: {d['val_history']}")
    return rows

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '/mnt/nvme1n1/erikg/e97_raw_mlp_runs')
