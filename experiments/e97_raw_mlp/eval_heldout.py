"""Re-evaluate final checkpoints on a held-out set. REAL data, REAL forward.
Rebuilds each model from its known config, loads averaged (schedule-free eval)
weights, computes token-weighted CE (nats/token) and BPB."""
import os, sys, glob, math, torch
from ndm.models import LadderLM
from ndm.data.tokenized_dataset import TokenizedStreamDataset

VOCAB = 50281
BPT = None  # computed per held-out file below

def e97(dim, depth, h, r):
    return dict(level='E97', dim=dim, depth=depth, n_heads=h, n_state=32,
                expansion=1.0, mlp_ratio=r, e88_raw_write=True)

CONFIGS = {
    # Wave 1
    'e97raw_mixer':       e97(1536, 10, 354, 0.0),
    'e97raw_mlp_bolt':    e97(1536, 10, 323, 1.5),
    'e97raw_mlp_realloc': e97(1536, 21, 128, 2.0),
    'gdn2_mlp':           dict(level='gdn2', dim=2304, depth=17, n_heads=8,
                               expansion=2.0, mlp_ratio=2.854),
    # Wave 2: mlp_ratio sweep (depth=10 bolt-on)
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
    kw = {**common, **cfg}
    return LadderLM(**kw)


@torch.no_grad()
def evaluate(model, val_path, device, batch_size=2, chunk=2048, max_batches=300,
             tokenizer='p50k_base', seed=1234):
    # SAME tokenizer as training (p50k_base). Fixed seed => every model sees the
    # IDENTICAL held-out batches, so the comparison is apples-to-apples.
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
    val_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/e97_heldout_rep.txt'
    device = 'cuda'
    bpt = bytes_per_token(val_path)
    factor = 1.0 / (math.log(2) * bpt)
    print(f"held-out: {val_path}  bytes/token={bpt:.4f}  BPB=val*{factor:.5f}")
    print(f"{'candidate':22s} {'val(nats)':>9s} {'BPB':>7s} {'train':>7s}")
    results = {}
    for name, cfg in CONFIGS.items():
        hits = glob.glob(f"{RUNDIR}/{name}/*/checkpoint_step_003000_*.pt")
        if not hits:
            continue
        ckpt = hits[0]
        train_loss = float(ckpt.split('_loss_')[1].replace('.pt', ''))
        model = build(cfg).to(device).bfloat16()
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
        model.load_state_dict(sd)
        val = evaluate(model, val_path, device)
        results[name] = dict(val=val, bpb=val * factor, train=train_loss)
        print(f"{name:22s} {val:9.4f} {val*factor:7.4f} {train_loss:7.4f}")
        del model; torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    main()
