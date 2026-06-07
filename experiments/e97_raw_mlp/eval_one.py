"""Per-candidate held-out eval (shardable across GPUs). REAL data, REAL forward.
Usage: eval_one.py NAME VALFILE [MAX_BATCHES]
Rebuilds the model from its known config, loads the step-3000 checkpoint,
computes token-weighted CE (nats/token) + BPB on the held-out file, writes JSON.
Identical fixed-seed batches across candidates => apples-to-apples."""
import os, sys, glob, math, json, time, torch
from ndm.models import LadderLM
from ndm.data.tokenized_dataset import TokenizedStreamDataset

VOCAB = 50281


def e97(dim, depth, h, r):
    return dict(level='E97', dim=dim, depth=depth, n_heads=h, n_state=32,
                expansion=1.0, mlp_ratio=r, e88_raw_write=True)

CONFIGS = {
    'e97raw_mixer':       e97(1536, 10, 354, 0.0),
    'e97raw_mlp_bolt':    e97(1536, 10, 323, 1.5),
    'e97raw_mlp_realloc': e97(1536, 21, 128, 2.0),
    'gdn2_mlp':           dict(level='gdn2', dim=2304, depth=17, n_heads=8,
                               expansion=2.0, mlp_ratio=2.854),
    'e97raw_mlp_r0p5':    e97(1536, 10, 344, 0.5),
    'e97raw_mlp_r1p0':    e97(1536, 10, 334, 1.0),
    'e97raw_mlp_r2p0':    e97(1536, 10, 313, 2.0),
    'e97raw_mlp_r2p694':  e97(1536, 10, 299, 2.694),
}
RUNDIR = '/mnt/nvme1n1/erikg/e97_raw_mlp_runs'


def bytes_per_token(path, enc_name='p50k_base', sample=20_000_000):
    import tiktoken
    enc = tiktoken.get_encoding(enc_name)
    raw = open(path, 'rb').read(sample)
    txt = raw.decode('utf-8', errors='ignore')
    nb = len(txt.encode('utf-8')); nt = len(enc.encode(txt, disallowed_special=()))
    return nb / nt


def build(cfg):
    common = dict(vocab_size=VOCAB, use_gate=True, gate_activation='silu', use_triton=True)
    return LadderLM(**{**common, **cfg})


@torch.no_grad()
def evaluate(model, val_path, device, batch_size=2, chunk=2048, max_batches=200,
             tokenizer='p50k_base', seed=1234):
    ds = TokenizedStreamDataset(data_path=val_path, chunk_size=chunk + 1,
                                seed=seed, tokenizer_name=tokenizer)
    model.eval()
    tot_loss = 0.0; tot_tok = 0
    for i in range(max_batches):
        chunk_b, is_doc_end, actual_lengths = ds.get_batch(batch_size, device=device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(chunk_b, return_loss=True)
        ntok = actual_lengths.sum().item()
        tot_loss += loss.item() * ntok; tot_tok += ntok
    return tot_loss / tot_tok


def main():
    name = sys.argv[1]
    val_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/e97_heldout_rep.txt'
    max_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    cfg = CONFIGS[name]
    device = 'cuda'
    t0 = time.time()
    bpt = bytes_per_token(val_path)
    factor = 1.0 / (math.log(2) * bpt)
    hits = glob.glob(f"{RUNDIR}/{name}/*/checkpoint_step_003000_*.pt")
    if not hits:
        print(f"NO CHECKPOINT for {name}"); sys.exit(2)
    ckpt = hits[0]
    train_loss = float(ckpt.split('_loss_')[1].replace('.pt', ''))
    model = build(cfg).to(device).bfloat16()
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(sd)
    val = evaluate(model, val_path, device, max_batches=max_batches)
    out = dict(name=name, val=val, bpb=val * factor, train=train_loss,
               bytes_per_token=bpt, max_batches=max_batches,
               n_params=sum(p.numel() for p in model.parameters()),
               secs=round(time.time() - t0, 1))
    os.makedirs(f"{RUNDIR}/heldout", exist_ok=True)
    with open(f"{RUNDIR}/heldout/{name}.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
