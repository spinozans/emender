#!/usr/bin/env python3
"""s5-symmetric-budget: give ALL THREE 1.3B models (E88 / GDN / M2RNN) the SAME
gentler/longer budget on S5, symmetric, with NO S5-tuning, and report where each
lands on S5 length-extrapolation.

Resolves the protagonist's unfinished business from refinetune-s3-s5-from-v03:
is E88's S5 shortfall at length (0.162 @ T=512 from the 2500-step matched run, or
0.162 @T512 from the to-competence run) its real expressivity/capacity ceiling, or
just under-training? The prior run bought out the BASELINE's trainability confound
(M2RNN got a gentler/longer recipe) but never gave E88 the same extra budget on S5.

This script fixes the asymmetry:
  - IDENTICAL recipe across all three (the gentler/longer recipe that worked for
    M2RNN: lr 5e-5 + grad-clip 0.5 + warmup 300), with a FIXED, A-PRIORI budget
    chosen in advance = 24000 steps (2x M2RNN's 12000-step to-competence budget).
    The budget / LR / stopping are NOT set using S5 performance.
  - Const LR (not cosine-to-zero) chosen a-priori specifically so the
    climbing/plateau/ceiling readout is unambiguous: a cosine schedule decaying to
    ~0 would confound "plateaued at the cap" with "LR went to zero". With const LR,
    a curve still rising at step 24000 == under-training; a flat curve == ceiling.
  - Train ON S5 with the SAME length curriculum as before (prefixes up to T=64),
    eval LENGTH-EXTRAPOLATION out to T=1024.
  - Record the full trajectory: periodic eval at several lengths (incl. T=512) over
    training, so we can SEE whether E88 is still climbing, plateaued, or at ceiling.

Init: STRICT load of the public @v0.3 model.safetensors (verified bit-identical to
the pinned checkpoints by v03_init_verify.py). REAL training + eval. One model/GPU.
Lengths are multiples of 16 (E88 triton checkpoint_interval=16).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace

_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument("--gpu", type=int, required=True)
_known, _ = _ap0.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_known.gpu)
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
CLEAN = "/tmp/v03-init-clean-cache"
os.environ["HF_HOME"] = CLEAN
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CLEAN, "hub")

ELMAN_DIR = "/home/erikg/elman"
sys.path.insert(0, ELMAN_DIR)
sys.path.insert(0, os.path.join(ELMAN_DIR, "elman", "cuda"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from experiments.expressivity_tasks.tasks import ALL_TASKS  # noqa: E402

REPOS = {
    "e88":   ("poietic-pbc/emender-e88-1.3b",  "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/e88"),
    "gdn":   ("poietic-pbc/gdn-1.3b",          "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/gdn"),
    "m2rnn": ("poietic-pbc/m2rnn-cma-1.3b",    "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/m2rnn"),
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_level(s):
    if isinstance(s, str) and s.startswith("log_"): return s
    try: return int(s)
    except (ValueError, TypeError): return s


def resolve_r_h_mode(level):
    full = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}
    li = int(level) if str(level).isdigit() else 0
    return "spectral_norm" if li in full else "none"


def build_model(a_json, vocab_size):
    a = SimpleNamespace(**a_json)
    level = parse_level(a.level)
    if str(a.level).lower() == "m2rnn":
        from elman.models.m2rnn_baseline import M2RNNLM
        return M2RNNLM(
            vocab_size=vocab_size, dim=a.dim, depth=a.depth, n_heads=a.n_heads,
            n_state=a.n_state, expansion=a.expansion,
            paper_shape=bool(getattr(a, "m2rnn_paper_shape", False)),
            k_head_dim=getattr(a, "m2rnn_k_head_dim", None),
            v_head_dim=getattr(a, "m2rnn_v_head_dim", None),
            num_q_heads=getattr(a, "m2rnn_q_heads", None),
            num_k_heads=getattr(a, "m2rnn_k_heads", None),
            num_v_heads=getattr(a, "m2rnn_v_heads", None),
            num_f_heads=getattr(a, "m2rnn_f_heads", None),
            num_g_heads=getattr(a, "m2rnn_g_heads", None),
            num_weight_heads=getattr(a, "m2rnn_weight_heads", None),
            use_gate=bool(getattr(a, "use_gate", 1)),
            use_residual=bool(getattr(a, "m2rnn_use_residual", 1)),
            state_weight_trainable=not bool(getattr(a, "m2rnn_freeze_state_weight", 0)),
            use_conv=bool(getattr(a, "use_conv", 0)), d_conv=getattr(a, "d_conv", 4),
            output_norm=bool(getattr(a, "m2rnn_output_norm", 0)),
            normalize_qk=bool(getattr(a, "m2rnn_normalize_qk", 0)),
            dropout=getattr(a, "dropout", 0.0),
            gradient_clipping=getattr(a, "m2rnn_state_grad_clip", None),
            gradient_checkpointing=False,
            loss_chunk_size=getattr(a, "loss_chunk_size", 0))
    from elman.models import LadderLM
    return LadderLM(
        vocab_size=vocab_size, dim=a.dim, depth=a.depth, level=level,
        expansion=getattr(a, "expansion", 1.0), n_groups=getattr(a, "n_groups", 32),
        n_state=getattr(a, "n_state", 64), n_slots=getattr(a, "n_slots", 64),
        n_heads=getattr(a, "n_heads", None), top_k=getattr(a, "top_k", None),
        k_fast=getattr(a, "k_fast", None), k_slow=getattr(a, "k_slow", None),
        use_gate=bool(getattr(a, "use_gate", 1)),
        gate_activation=getattr(a, "gate_activation", "sigmoid"),
        linear_state=bool(getattr(a, "linear_state", 0)),
        use_write_gate=bool(getattr(a, "use_write_gate", 0)),
        e88_decay_mode=getattr(a, "e88_decay_mode", "mamba"),
        e88_value_residual=bool(getattr(a, "e88_value_residual", 0)),
        e88_raw_write=bool(getattr(a, "e88_raw_write", 0)),
        state_expansion=getattr(a, "state_expansion", 2),
        r_h_mode=resolve_r_h_mode(level), use_conv=bool(getattr(a, "use_conv", 0)),
        d_conv=getattr(a, "d_conv", 4), dropout=getattr(a, "dropout", 0.0),
        checkpoint_interval=getattr(a, "checkpoint_interval", 16),
        gradient_checkpointing=False,
        projection_chunk_size=getattr(a, "projection_chunk_size", 0),
        loss_chunk_size=getattr(a, "loss_chunk_size", 0),
        use_triton=bool(getattr(a, "use_triton", 0)))


def load_v03(model, repo):
    """Strict-load the public @v0.3 safetensors into the harness model."""
    st_path = hf_hub_download(repo, "model.safetensors", revision="v0.3")
    from safetensors import safe_open
    with safe_open(st_path, framework="pt") as f:
        meta = f.metadata() or {}
    sd = {(k[len("model."):] if k.startswith("model.") else k): v
          for k, v in load_file(st_path).items()}
    hk = set(model.state_dict().keys())
    if "lm_head.weight" in sd and "lm_head.weight" not in hk:
        del sd["lm_head.weight"]
    model.load_state_dict(sd, strict=True)
    log(f"  STRICT v0.3 load OK (checkpoint_step={meta.get('checkpoint_step')} "
        f"ymode={meta.get('ymode_export')})")
    return meta.get("checkpoint_step")


def make_batch(task, B, T, rng, device):
    inp, tgt, msk = task.generate_batch(B, T, rng)
    return (torch.from_numpy(inp).to(device), torch.from_numpy(tgt).to(device),
            torch.from_numpy(msk).to(device))


@torch.no_grad()
def eval_acc(model, task, B, T, n_batches, rng, device):
    model.eval(); correct = total = 0
    for _ in range(n_batches):
        x, y, m = make_batch(task, B, T, rng, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += ((preds == y) & m).sum().item(); total += m.sum().item()
    return correct / max(total, 1)


def lr_at(step, base_lr, total, warmup, schedule):
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "cosine":
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))
    return base_lr


def finetune_one(model_name, repo, task_name, args, device, a_json):
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    task = ALL_TASKS[task_name](mode="running")
    log(f"[{model_name}/{task_name}] vocab={task.vocab_size} "
        f"baseline={task.random_baseline_acc():.4f}")
    model = build_model(a_json, args.vocab_size)
    v03_step = load_v03(model, repo)
    model = model.to(device).bfloat16()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    train_lens = [int(x) for x in args.train_lens.split(",")]
    eval_lens = [int(x) for x in args.eval_lens.split(",")]
    traj_lens = [int(x) for x in args.traj_lens.split(",")]
    rec = {"model": model_name, "task": task_name, "v03_checkpoint_step": v03_step,
           "n_classes": getattr(task, "n_classes", None),
           "random_baseline_acc": task.random_baseline_acc(), "train_curve": [],
           "traj_curve": [], "traj_lens": traj_lens,
           "train_lens": train_lens, "eval_lens": eval_lens}
    t0 = time.time(); model.train()
    rng_traj = np.random.default_rng(args.seed + 2000)
    for step in range(args.steps):
        cur_lr = lr_at(step, args.lr, args.steps, args.warmup, args.lr_schedule)
        for pg in opt.param_groups: pg["lr"] = cur_lr
        T = int(rng.choice(train_lens))
        x, y, m = make_batch(task, args.batch_size, T, rng, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   y.view(-1), reduction="none").view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            acc_short = eval_acc(model, task, args.batch_size, max(train_lens), 4, rng, device)
            log(f"[{model_name}/{task_name}] step {step:>5d} lr={cur_lr:.2e} "
                f"loss={loss.item():.4f} acc@T{max(train_lens)}={acc_short:.4f} ({time.time()-t0:.0f}s)")
            rec["train_curve"].append({"step": step, "lr": cur_lr,
                "loss": float(loss.item()), "acc_train_maxT": float(acc_short)})
            model.train()
        # Periodic LENGTH-EXTRAPOLATION trajectory: eval at several lengths incl.
        # T=512 so we can see whether the model is still climbing / plateaued / at
        # ceiling at extrapolated length as the budget is spent.
        if step % args.traj_every == 0 or step == args.steps - 1:
            accs = {}
            for T in traj_lens:
                accs[T] = float(eval_acc(model, task, args.eval_batch,
                                         T, args.traj_nbatch, rng_traj, device))
            rec["traj_curve"].append({"step": step, "lr": cur_lr, "acc_vs_T": accs})
            log(f"[{model_name}/{task_name}] TRAJ step {step:>5d} "
                + " ".join(f"T{T}={accs[T]:.3f}" for T in traj_lens))
            model.train()
    rng_eval = np.random.default_rng(args.seed + 1000)
    acc_vs_T = {}
    for T in eval_lens:
        acc = eval_acc(model, task, args.eval_batch, T, args.eval_nbatch, rng_eval, device)
        acc_vs_T[T] = float(acc)
        log(f"[{model_name}/{task_name}] EVAL T={T:>4d} acc={acc:.4f}")
    rec["acc_vs_T"] = acc_vs_T; rec["train_max_T"] = max(train_lens)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    del model, opt; torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--model", required=True, choices=list(REPOS.keys()))
    ap.add_argument("--tasks", default="s5_permutation")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr_schedule", default="const", choices=["const", "cosine"])
    ap.add_argument("--train_lens", default="16,32,48,64")
    ap.add_argument("--eval_lens", default="16,32,48,64,96,128,192,256,384,512,768,1024")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--eval_nbatch", type=int, default=16)
    ap.add_argument("--traj_every", type=int, default=1000)
    ap.add_argument("--traj_lens", default="64,128,256,512")
    ap.add_argument("--traj_nbatch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu)
    device = torch.device("cuda")
    repo, ckptdir = REPOS[args.model]
    log(f"device={torch.cuda.get_device_name(0)} model={args.model} repo={repo}")
    a_json = json.loads(Path(ckptdir, "args.json").read_text())
    import tiktoken
    args.vocab_size = tiktoken.get_encoding(a_json["tokenizer"]).n_vocab

    recipe = {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
              "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
              "warmup": args.warmup, "lr_schedule": args.lr_schedule,
              "optimizer": "AdamW(betas=0.9,0.95)", "train_lens": args.train_lens,
              "eval_lens": args.eval_lens, "traj_lens": args.traj_lens,
              "traj_every": args.traj_every, "seed": args.seed, "dtype": "bf16",
              "finetune": "full (all params)", "mode": "running",
              "init": "PUBLIC HF v0.3 safetensors (strict load)",
              "note": "SYMMETRIC a-priori budget, identical across all 3 models, "
                      "no S5-tuning; const LR for clean climbing/plateau/ceiling readout"}
    results = {"model": args.model, "gpu": args.gpu, "repo": repo,
               "init": "hf_v0.3_safetensors", "recipe": recipe, "tasks": {}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for task_name in args.tasks.split(","):
        rec = finetune_one(args.model, repo, task_name, args, device, a_json)
        results["tasks"][task_name] = rec
        Path(args.out).write_text(json.dumps(results, indent=2))
        log(f"wrote {args.out} (after {task_name})")
    log(f"DONE {args.model}: {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
