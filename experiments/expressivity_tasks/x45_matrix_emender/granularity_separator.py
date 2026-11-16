"""GRANULARITY SEPARATOR: chunk-granular decision *generation* as an expressivity
weakening, tested at probe scale.

Hypothesis (the operator's): with state-dependent, irreversible management ops
(no inverses by design -- ``formal/lean/OPERATOR_ALGEBRA_STATUS.md``), a decision
delayed past the information it acts on is permanent loss; therefore a C-granular
decision schedule is a strictly weaker machine, not an implementation detail.
The Lean theorems cover only *fixed* op sequences (chunked == sequential replay);
nothing covers decision *generation* at chunk granularity.  This module is the
empirical witness for the generation-level gap.

Two properties of the registered chunked router are EXACT and strategy-independent
(they hold for every payload the model can emit, so no learnable geometry escapes
them -- this is what licenses an expressivity claim rather than a learnability
one):

* **P1 (entry-pinned free-slot set)** -- ``schedule.route_chunk`` enumerates free
  slots from the CHUNK-ENTRY liveness, so a slot freed mid-chunk cannot be
  APPENDed into until the next chunk boundary.
* **P2 (entry-pinned read weights)** -- ``read.read_weights`` is evaluated against
  the chunk-entry state; a row that is dead at chunk entry receives attention
  EXACTLY 0 for every read in that chunk (dead rows score ``-inf``), whatever its
  content is by read time.  The in-chunk write-then-read channel therefore does
  not exist at granularity C.

The per-token arm is ``routed_chunk_scan(chunk_tokens=1)``: the registered
eager scan with the registered five-op ``RoutingPolicy`` evaluated against the
CURRENT state at every token -- the K6 semantic (``PREREGISTRATION_SEQUENTIAL_MATRIX.md``;
``test_granularity_separator.py`` holds this arm to K6's ``sequential_oracle_scan``
decision for decision).  The chunked arms are the same scan, same policy, same
algebra, with ``chunk_tokens`` in {8, 32, 128} (the K5B whole-scan path is a
bit-identical host-cost specialisation of this same scan per its receipts).
Only the dial differs.

Task SEP-AR ("write-then-read inside the decision interval"): each window is
BOS, PAD x3, six associative pairs presented as combined (key, value) tokens,
then three probes; a probe is a bare key token and the target is the paired
value token.  A pair whose row is created in the same chunk as its probe is
exactly unreadable by the chunked arm (P2); per-token reads it.  Task SEP-EVICT
is the same window at rank 8 (store full through the probe phase), adding
forced POP/CORRECT destruction events whose victims differ between entry-stale
and fresh argmins -- corroboration, direction pre-stated, magnitudes best-effort.

Pre-stated predictions live in ``RESULT_GRANULARITY_SEPARATOR.md`` (committed
before any training run).  Protocol: per-arm LR sweep, >=3 seeds, probe-scale
runs of minutes, mechanistic receipts (weights channel + occupancy + op
histogram + destruction events) per run.

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    /home/erikg/emender/.venv/bin/python -m \
        experiments.expressivity_tasks.x45_matrix_emender.granularity_separator \
        --task both
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .algebra import APPEND, CORRECT, NONE, POP, op_histogram
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS
from .schedule import RoutingPolicy, route_chunk
from .scan import routed_chunk_scan

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"

# ---------------------------------------------------------------- task family
# Vocabulary: BOS, PAD, 8 key tokens (probes), 32 value tokens (targets), and
# 256 combined pair tokens (key, value).  A combined token is how a fresh
# key->value binding enters the store as ONE write whose payload can carry both
# the key cluster direction and the value direction: the layer-1 payload is a
# fixed function of the token identity, so presenting key and value as separate
# tokens would make the window-specific binding unrepresentable at depth 1 --
# a different (and not the property under test) failure.
BOS, PAD = 0, 1
KEY_BASE, N_KEYS = 2, 8
VAL_BASE, N_VALUES = KEY_BASE + N_KEYS, 32
PAIR_BASE = VAL_BASE + N_VALUES
VOCAB = PAIR_BASE + N_KEYS * N_VALUES

WINDOW = 16
N_PAIRS = 6
N_PROBES = 3
PAIR_POSITIONS = tuple(4 + i for i in range(N_PAIRS))          # 4..9
PROBE_POSITIONS = tuple(10 + 2 * j for j in range(N_PROBES))  # 10, 12, 14
# The answer is the next token at each probe position (revealed there, then
# teacher-forced as an input at the following position).
VALUE_SLICE = slice(VAL_BASE, VAL_BASE + N_VALUES)
CHANCE = 1.0 / N_VALUES


def pair_id(k, v):
    return PAIR_BASE + k * N_VALUES + v


def same_chunk(pair_index: int, probe_index: int, chunk_tokens: int) -> bool:
    """Was the pair's write executed inside the probe's own chunk?

    The probe's read weights were generated at the probe's chunk ENTRY, i.e.
    before every write positioned at or after ``(probe // C) * C``; a pair
    written there is dead-at-entry from the read's point of view, so its read
    weight is exactly 0 for the whole chunk (P2).  This predicate IS the
    pre-registered stratification of every probe.
    """
    return (PAIR_POSITIONS[pair_index] // chunk_tokens) \
        == (PROBE_POSITIONS[probe_index] // chunk_tokens)


def same_chunk_table(chunk_tokens: int) -> torch.Tensor:
    """The (N_PAIRS, N_PROBES) blindness table for one chunk size."""
    return torch.tensor([[same_chunk(p, j, chunk_tokens) for j in range(N_PROBES)]
                         for p in range(N_PAIRS)])


def make_batch(rng: torch.Generator, count: int):
    """Sample ``count`` SEP-AR windows (CPU tensors; move to the device after).

    Keys, values and probe order are drawn without replacement per window, so a
    probe's answer is unambiguous and no earlier probe reveals a later answer.
    """
    key_order = torch.argsort(torch.rand(count, N_KEYS, generator=rng), dim=1)[:, :N_PAIRS]
    val_order = torch.argsort(torch.rand(count, N_VALUES, generator=rng), dim=1)[:, :N_PAIRS]
    probe_order = torch.argsort(torch.rand(count, N_PAIRS, generator=rng), dim=1)[:, :N_PROBES]
    ids = torch.full((count, WINDOW), PAD, dtype=torch.long)
    ids[:, 0] = BOS
    for i, position in enumerate(PAIR_POSITIONS):
        ids[:, position] = PAIR_BASE + key_order[:, i] * N_VALUES + val_order[:, i]
    targets = torch.zeros(count, N_PROBES, dtype=torch.long)
    probe_pairs = torch.zeros(count, N_PROBES, dtype=torch.long)
    for j, position in enumerate(PROBE_POSITIONS):
        picked = probe_order[:, j]
        probe_pairs[:, j] = picked
        keys = key_order.gather(1, picked.unsqueeze(1)).squeeze(1)
        values = val_order.gather(1, picked.unsqueeze(1)).squeeze(1)
        ids[:, position] = KEY_BASE + keys
        ids[:, position + 1] = VAL_BASE + values
        targets[:, j] = VAL_BASE + values
    return ids, targets, probe_pairs


# ---------------------------------------------------------------- probe model
# The registered block shape at probe scale: RMSNorm -> mixer -> residual, then
# RMSNorm -> SwiGLU -> residual.  The mixer is the registered store: query/value/
# output projections, the DECAY gate with the registered bias init, and the
# five-op scan.  Cross-token information flows ONLY through store reads (every
# other component is position-wise), so the task's dependence on the read
# channel is total.
class SeparatorMixer(nn.Module):
    def __init__(self, width: int, value_dim: int, heads: int, rank: int,
                 policy: RoutingPolicy, chunk_tokens: int):
        super().__init__()
        self.heads, self.value_dim, self.rank = heads, value_dim, rank
        self.policy, self.chunk_tokens = policy, chunk_tokens
        self.query = nn.Linear(width, heads * value_dim, bias=False)
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))

    def forward(self, x, collect: bool = False):
        batch, tokens, _ = x.shape
        query = self.query(x).view(batch, tokens, self.heads, self.value_dim)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR, max=1.0 - 1e-6)
        result = routed_chunk_scan(
            query, payload, decay, router=route_chunk, policy=self.policy,
            rank=self.rank, rank_max=self.rank, chunk_tokens=self.chunk_tokens,
            collect=collect, fused=False)
        reads = result.reads.reshape(batch, tokens, self.heads * self.value_dim)
        return self.out(reads), result


class SeparatorBlock(nn.Module):
    def __init__(self, width: int, hidden: int, value_dim: int, heads: int,
                 rank: int, policy: RoutingPolicy, chunk_tokens: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5)
        self.mixer = SeparatorMixer(width, value_dim, heads, rank, policy, chunk_tokens)
        self.norm2 = nn.RMSNorm(width, eps=1e-5)
        self.w1 = nn.Linear(width, hidden, bias=False)
        self.w2 = nn.Linear(width, hidden, bias=False)
        self.w3 = nn.Linear(hidden, width, bias=False)

    def forward(self, x, collect: bool = False):
        mixed, result = self.mixer(self.norm1(x), collect=collect)
        h = x + mixed
        gate = F.silu(self.w1(self.norm2(h))) * self.w2(self.norm2(h))
        return h + self.w3(gate), result


class SeparatorModel(nn.Module):
    """Depth-2 store model; the ONLY knob that differs between arms is
    ``chunk_tokens`` (the decision granularity dial)."""

    def __init__(self, chunk_tokens: int, vocab: int = VOCAB, width: int = 64,
                 hidden: int = 128, depth: int = 2, heads: int = 1,
                 value_dim: int = 32, rank: int = 16,
                 policy: RoutingPolicy | None = None):
        super().__init__()
        policy = RoutingPolicy() if policy is None else policy
        self.chunk_tokens = chunk_tokens
        self.policy = policy
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(
            SeparatorBlock(width, hidden, value_dim, heads, rank, policy, chunk_tokens)
            for _ in range(depth))
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids, collect: bool = False):
        x = self.embed(ids)
        results = [] if collect else None
        for block in self.blocks:
            x, result = block(x, collect=collect)
            if collect:
                results.append(result)
        return self.head(self.norm(x)), results

    def loss(self, ids, targets):
        logits, _ = self.forward(ids)
        probe = logits[:, PROBE_POSITIONS]                     # (B, N_PROBES, V)
        return F.cross_entropy(probe.reshape(-1, probe.shape[-1]), targets.reshape(-1))


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def probe_predictions(model, ids, batch: int = 1024):
    """Value-restricted argmax at the probe positions: (N, N_PROBES) predicted
    value-token ids.  The task is "which value", so accuracy is measured over
    the 32 value tokens; chance = 1/32."""
    model.eval()
    picks = []
    for start in range(0, ids.shape[0], batch):
        logits, _ = model.forward(ids[start:start + batch])
        probe = logits[:, PROBE_POSITIONS, VALUE_SLICE]        # (b, J, N_VALUES)
        picks.append(probe.argmax(-1) + VAL_BASE)
    return torch.cat(picks)


@torch.no_grad()
def evaluate(model, ids, targets, probe_pairs, chunk_tokens: int) -> dict:
    """Overall / readable / blind value accuracy under the pre-registered
    same-chunk stratification."""
    picks = probe_predictions(model, ids)
    correct = picks == targets
    table = same_chunk_table(chunk_tokens).to(correct.device)
    blind = table[probe_pairs, torch.arange(N_PROBES, device=correct.device)]
    readable = ~blind
    return dict(value_acc=float(correct.float().mean()),
                value_acc_readable=float(correct[readable].float().mean())
                if bool(readable.any()) else None,
                value_acc_blind=float(correct[blind].float().mean())
                if bool(blind.any()) else None,
                n_readable=int(readable.sum()), n_blind=int(blind.sum()),
                n_probes=int(correct.numel()), chance=CHANCE)


def row_destroyed_before_probe(schedule, probe_pairs) -> torch.Tensor:
    """(N, N_PROBES) bool: a POP or a foreign CORRECT hit the probed pair's row
    strictly between the pair's write and its probe.  The probe's own CORRECT at
    the probe position is excluded (reads precede writes within a token), and
    so is the value reveal after the probe (the answer was already demanded
    there).  What remains is exactly the irreversible destruction the separator
    is about: the decision that would have protected the row was generated too
    late, and no inverse exists."""
    windows, probes = probe_pairs.shape
    destroyed = torch.zeros(windows, probes, dtype=torch.bool,
                            device=probe_pairs.device)
    pair_slot = schedule.slot[:, PAIR_POSITIONS, 0, 0]
    pair_op = schedule.op[:, PAIR_POSITIONS, 0, 0]
    for window in range(windows):
        for j in range(probes):
            pi = int(probe_pairs[window, j])
            if int(pair_op[window, pi]) == NONE:
                continue                      # the pair never entered the store
            slot, write = int(pair_slot[window, pi]), PAIR_POSITIONS[pi]
            probe = PROBE_POSITIONS[j]
            for t in range(write + 1, probe):
                op = int(schedule.op[window, t, 0, 0])
                if op in (POP, CORRECT) and int(schedule.slot[window, t, 0, 0]) == slot:
                    destroyed[window, j] = True
                    break
    return destroyed


@torch.no_grad()
def collect_channels(model, ids, targets, probe_pairs, chunk_tokens: int) -> dict:
    """The mechanistic receipt pass: for every (window, probe) record the op
    that created the probed pair's row, the read weight the probe actually put
    on that row, whether the row was still live at the probe, and whether the
    answer was right -- stratified by the pre-registered blindness predicate.
    ``weights`` is the scan's recorded read-weight channel (softmax x strength,
    computed at the chunk entry)."""
    model.eval()
    logits, results = model.forward(ids, collect=True)
    probe = logits[:, PROBE_POSITIONS, VALUE_SLICE]
    correct = probe.argmax(-1) + VAL_BASE == targets
    table = same_chunk_table(chunk_tokens).to(correct.device)
    blind = table[probe_pairs, torch.arange(N_PROBES, device=correct.device)]
    per_layer = []
    for layer, result in enumerate(results):
        schedule, weights = result.schedule, result.weights
        pair_slot = schedule.slot[:, PAIR_POSITIONS, 0, 0]     # (N, N_PAIRS)
        pair_op = schedule.op[:, PAIR_POSITIONS, 0, 0]
        slot = pair_slot.gather(1, probe_pairs)                # (N, J)
        op = pair_op.gather(1, probe_pairs)
        stored = op.ne(NONE)
        probe_weights = weights[:, PROBE_POSITIONS, 0, :]     # (N, J, rank)
        weight_at_probe = probe_weights.gather(2, slot.unsqueeze(-1)).squeeze(-1)
        destroyed = row_destroyed_before_probe(schedule, probe_pairs)
        live_at_probe = stored & ~destroyed
        masks = dict(all=stored, readable=stored & ~blind, blind=stored & blind,
                     live_blind=live_at_probe & blind,
                     # the churn channel: a blind pair that CORRECTed onto an
                     # ENTRY-LIVE register is the one blind write the chunked
                     # router can still read; a blind APPEND is dead-at-entry
                     # and must carry exactly zero weight.
                     blind_append=blind & (op == APPEND),
                     blind_correct=blind & (op == CORRECT))

        def stats(mask):
            if not bool(mask.any()):
                return None
            w = weight_at_probe[mask]
            return dict(n=int(mask.sum()), mean=float(w.float().mean()),
                        max=float(w.max()), min=float(w.min()))

        per_layer.append(dict(
            layer=layer,
            pair_op_counts={name: int((pair_op == code).sum())
                            for code, name in ((APPEND, "append"), (CORRECT, "correct"),
                                              (NONE, "none"))},
            op_histogram=op_histogram(schedule),
            stored_frac=float(stored.float().mean()),
            # The headline receipt: the blind probes' answer rows are live and
            # holding the pair payload (stored, not destroyed before the probe)
            # while the chunked read puts EXACTLY 0 weight on them.
            weight_on_answer_row={name: stats(mask) for name, mask in masks.items()},
            accuracy=dict(
                readable=float(correct[stored & ~blind].float().mean())
                if bool((stored & ~blind).any()) else None,
                blind=float(correct[stored & blind].float().mean())
                if bool((stored & blind).any()) else None,
                blind_append=float(correct[blind & (op == APPEND)].float().mean())
                if bool((blind & (op == APPEND)).any()) else None,
                blind_correct=float(correct[blind & (op == CORRECT)].float().mean())
                if bool((blind & (op == CORRECT)).any()) else None,
                blind_intact=float(correct[live_at_probe & blind].float().mean())
                if bool((live_at_probe & blind).any()) else None)))
    return dict(chunk_tokens=chunk_tokens, per_layer=per_layer)


@torch.no_grad()
def destruction_events(model, ids, probe_pairs, targets, chunk_tokens: int,
                       layer: int = 0) -> dict:
    """SEP-EVICT receipt: pair-row destruction rates and the accuracy of probes
    whose row was destroyed vs intact -- the operator's mechanism (i) made
    countable."""
    model.eval()
    _, results = model.forward(ids, collect=True)
    logits, _ = model.forward(ids)
    probe = logits[:, PROBE_POSITIONS, VALUE_SLICE]
    correct = probe.argmax(-1) + VAL_BASE == targets
    schedule = results[layer].schedule
    destroyed = row_destroyed_before_probe(schedule, probe_pairs)
    ops = schedule.op[:, :, 0, 0]
    table = same_chunk_table(chunk_tokens).to(correct.device)
    blind = table[probe_pairs, torch.arange(N_PROBES, device=correct.device)]
    out = dict(layer=layer,
               pop_count=int((ops == POP).sum()),
               correct_count=int((ops == CORRECT).sum()),
               probes=int(destroyed.numel()),
               destroyed_frac=float(destroyed.float().mean()),
               acc_when_destroyed=float(correct[destroyed].float().mean())
               if bool(destroyed.any()) else None,
               acc_when_intact=float(correct[~destroyed].float().mean())
               if bool((~destroyed).any()) else None)
    if bool(blind.any()):
        out["blind_acc_when_intact"] = float(correct[blind & ~destroyed].float().mean()) \
            if bool((blind & ~destroyed).any()) else None
    return out


# ------------------------------------------------------------------- one run
def run_one(task: str, chunk_tokens: int, lr: float, seed: int, steps: int,
            batch: int, rank: int, device: str, eval_sets):
    """Train one (arm, lr, seed) cell of the best-effort sweep and return its
    receipt and state dict.  Every cell of the sweep shares one fixed eval set
    (seed 4999) so the arms are compared on identical probes."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SeparatorModel(chunk_tokens=chunk_tokens, rank=rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed)
    eval_ids, eval_targets, eval_pairs = eval_sets
    curve = []
    started = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        ids, targets, _ = make_batch(rng, batch)
        loss = model.loss(ids.to(device), targets.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == steps:
            picks = probe_predictions(model, eval_ids[:1024])
            curve.append(dict(step=step, loss=float(loss.detach()),
                              value_acc=float((picks == eval_targets[:1024]).float().mean())))
    walls = time.monotonic() - started
    metrics = evaluate(model, eval_ids, eval_targets, eval_pairs, chunk_tokens)
    receipt_ids, receipt_targets, receipt_pairs = (eval_ids[:512], eval_targets[:512],
                                                   eval_pairs[:512])
    channels = collect_channels(model, receipt_ids, receipt_targets, receipt_pairs,
                                 chunk_tokens)
    receipt = dict(
        task=task, chunk_tokens=chunk_tokens, lr=lr, seed=seed, rank=rank,
        steps=steps, batch=batch, device=device,
        policy=RoutingPolicy().record(),
        window=dict(window=WINDOW, n_pairs=N_PAIRS, n_probes=N_PROBES,
                    pair_positions=list(PAIR_POSITIONS),
                    probe_positions=list(PROBE_POSITIONS), chance=CHANCE),
        metrics=metrics, curve_tail=curve[-4:], walls_s=round(walls, 1),
        channels=channels, torch=torch.__version__)
    if task == "evict":
        receipt["destruction"] = destruction_events(
            model, receipt_ids, receipt_pairs, receipt_targets, chunk_tokens)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    return receipt, state


def parse_floats(text: str):
    return [float(x) for x in text.split(",")]


def parse_ints(text: str):
    return [int(x) for x in text.split(",")]


def sweep(task: str, arms, lrs, seeds, steps: int, batch: int, rank: int,
          device: str, eval_sets, quick: bool = False):
    """Run the full (arm, lr, seed) grid, write a per-run receipt, keep and save
    the best model per arm, and write the sweep summary receipt."""
    runs, best, best_state = [], {}, {}
    for chunk_tokens in arms:
        for lr in lrs:
            for seed in seeds:
                receipt, state = run_one(task, chunk_tokens, lr, seed, steps,
                                        batch, rank, device, eval_sets)
                name = f"separator_{task}_c{chunk_tokens}_lr{lr:.0e}_seed{seed}.json"
                if not quick:
                    (RECEIPTS / name).write_text(json.dumps(receipt, indent=1))
                runs.append({k: receipt[k] for k in
                             ("task", "chunk_tokens", "lr", "seed", "metrics")})
                if chunk_tokens not in best or \
                        receipt["metrics"]["value_acc"] > best[chunk_tokens]["metrics"]["value_acc"]:
                    best[chunk_tokens] = receipt
                    best_state[chunk_tokens] = state
                print(f"{task} C={chunk_tokens} lr={lr:.0e} seed={seed}: "
                      f"value_acc={receipt['metrics']['value_acc']:.4f} "
                      f"readable={receipt['metrics']['value_acc_readable']} "
                      f"blind={receipt['metrics']['value_acc_blind']} "
                      f"[{receipt['walls_s']}s]", flush=True)
    summary = dict(task=task, rank=rank, steps=steps, batch=batch, seeds=seeds,
                   chance=CHANCE, runs=runs,
                   best={str(k): dict(chunk_tokens=v["chunk_tokens"], lr=v["lr"],
                                      seed=v["seed"], metrics=v["metrics"],
                                      walls_s=v["walls_s"])
                         for k, v in best.items()})
    if not quick:
        for chunk_tokens, state in best_state.items():
            torch.save(state, RECEIPTS / f"separator_{task}_best_c{chunk_tokens}.pt")
        (RECEIPTS / f"separator_{task}_sweep.json").write_text(json.dumps(summary, indent=1))
    return summary, best, best_state


def mechanism_pass(arms, rank: int, device: str, eval_ids, eval_targets,
                   eval_pairs):
    """The headline mechanistic receipt, from the sweep's saved best models:
    the recorded read channel at the probe on the row the probed pair's write
    created, stratified by the pre-registered blindness predicate."""
    receipt_ids = eval_ids[:512].to(device)
    receipt_targets = eval_targets[:512].to(device)
    receipt_pairs = eval_pairs[:512].to(device)
    mechanism = dict(experiment="granularity_separator_mechanism",
                      chance=CHANCE,
                      note=("weight_on_answer_row is the scan's recorded read "
                            "channel at the probe token on the row the probed "
                            "pair's write created; blind = the pre-registered "
                            "same-chunk predicate (P2: dead at the probe chunk's "
                            "entry). live_blind = stored and not destroyed "
                            "before the probe."))
    for chunk_tokens in arms:
        path = RECEIPTS / f"separator_ar_best_c{chunk_tokens}.pt"
        if not path.exists():
            continue
        model = SeparatorModel(chunk_tokens=chunk_tokens, rank=rank).to(device)
        model.load_state_dict(torch.load(path, map_location=device,
                                         weights_only=True))
        mechanism[f"c{chunk_tokens}"] = collect_channels(
            model, receipt_ids, receipt_targets, receipt_pairs, chunk_tokens)
    (RECEIPTS / "separator_mechanism.json").write_text(json.dumps(mechanism, indent=1))
    return mechanism


def main():
    parser = argparse.ArgumentParser(
        description="Granularity separator sweep (SEP-AR headline, SEP-EVICT corroboration)")
    parser.add_argument("--task", choices=["ar", "evict", "both", "mechanism"],
                        default="both")
    parser.add_argument("--arms", default=None, help="comma list of chunk sizes (AR)")
    parser.add_argument("--lrs", default="3e-3,1e-3,3e-4,1e-4")
    parser.add_argument("--seeds", default="3401,3402,3403")
    parser.add_argument("--evict-arms", default="1,8")
    parser.add_argument("--evict-lrs", default="3e-3,1e-3,3e-4")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-windows", type=int, default=4096)
    parser.add_argument("--quick", action="store_true",
                        help="smoke: two arms, one lr, one seed, 200 steps, no receipts")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.quick:
        args.task, args.arms, args.lrs = "ar", "1,8", "1e-3"
        args.seeds, args.steps, args.eval_windows = "3401", 200, 512

    eval_rng = torch.Generator().manual_seed(4999)
    eval_ids, eval_targets, eval_pairs = make_batch(eval_rng, args.eval_windows)
    eval_sets = (eval_ids.to(args.device), eval_targets.to(args.device),
                 eval_pairs.to(args.device))
    ar_arms = parse_ints(args.arms) if args.arms else [1, 8, 32, 128]

    if args.task in ("ar", "both"):
        summary, best, states = sweep("ar", ar_arms, parse_floats(args.lrs),
                                      parse_ints(args.seeds), args.steps, args.batch,
                                      16, args.device, eval_sets, args.quick)
        if not args.quick:
            print(json.dumps(mechanism_pass(ar_arms, 16, args.device,
                                            eval_ids, eval_targets, eval_pairs),
                             indent=1))
        print(json.dumps(summary["best"], indent=1))
    if args.task == "mechanism":
        ar_arms = parse_ints(args.arms) if args.arms else [1, 8, 32, 128]
        print(json.dumps(mechanism_pass(ar_arms, 16, args.device,
                                        eval_ids, eval_targets, eval_pairs),
                         indent=1))
    if args.task in ("evict", "both"):
        summary, _, _ = sweep("evict", parse_ints(args.evict_arms),
                              parse_floats(args.evict_lrs), parse_ints(args.seeds),
                              args.steps, args.batch, 8, args.device, eval_sets,
                              args.quick)
        print(json.dumps(summary["best"], indent=1))


if __name__ == "__main__":
    main()
