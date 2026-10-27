"""X35 part 2: TRAINED-tape event density — the speculative-parallelism verdict datum.

The frozen X28E package is READ-ONLY (verify_freeze re-verified at start, the
x33_pressure_probe/probe.py pattern): trainer.build_tape_model with the frozen
arm config, best.pt loaded read-only, eval mode, no training. The part-1
instrumentation (density.summarize + classifier) is reused unchanged: C=64
store chunks, MUTATION_CANDIDATE / ENTRY_PURE labels, event-token fraction
(case >= R-CONFLICT, POP fired, or any strength death), reasons breakdown.

Data (registered in RECEIPTS.md BEFORE measurement):
  - exact positive ladder words [SOS] a^k b^k [EOS] at the bin boundaries
    k = 128 / 256 / 512 / 1152 (bins ID / 2x / 4x / 9x), 8 copies per bin;
  - a secondary band probe: per bin, 8 DISTINCT-k exact positive words evenly
    spaced in the registered depth band (clamped below at k=31), one word per
    forward (no padding);
  - the first 8 words of the registered stage-1 MQAR validation pool
    (T=128, seed 28302 — byte-identical to part 1's words). The stage-1
    embedding is stage-local and NOT persisted in best.pt; it is reconstructed
    by replaying the frozen deterministic construction sequence
    (manual_seed(0) -> build_tape_model -> attach_multiclass_stage(65, 64);
    F12f stage-local embeddings are frozen random feature maps, never
    trained). Labeled a hybrid: trained layer + training-time stage-1
    addressing geometry.

Outputs density_trained.json (part-1 schema + trained-checkpoint provenance).
Eval-only: no writes outside this experiment directory.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PKG = (ROOT.parent / "x28e_memory_head").resolve()
RESULTS = (ROOT.parent / "results_x28e_memory_head").resolve()
REPO_ROOT = ROOT.parents[2]

sys.path.insert(0, str(PKG))

from trainer import build_tape_model, verify_freeze  # noqa: E402 (frozen, read-only)
from generators.stage1_mqar import (  # noqa: E402
    STAGE1_SPLITS,
    VOCAB_SIZE as S1_VOCAB,
    generate_pool as gen_mqar,
)
from generators.stage3_ladder import (  # noqa: E402
    A,
    B,
    EOS,
    LADDER_VOCAB,
    SOS,
)

from .density import summarize  # noqa: E402  (the part-1 instrumentation, unchanged)

ARM = "tape_B0x4"
N_COPIES = 8
BINS = (("id_k128", 128), ("2x_k256", 256), ("4x_k512", 512), ("9x_k1152", 1152))
BANDS = {"id": (1, 128), "2x": (129, 256), "4x": (257, 512), "9x": (1025, 1152)}
BAND_K_MIN = 31          # 2k+2 >= 64: every band word has at least one full chunk
CHUNK = 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def positive_word(k: int):
    return [SOS] + [A] * k + [B] * k + [EOS]


def band_ks(lo: int, hi: int, n: int = N_COPIES):
    lo = max(lo, BAND_K_MIN)
    return [int(x) for x in np.round(np.linspace(lo, hi, n))]


def build_trained():
    """Verify the freeze, rebuild the arm, load best.pt read-only (x33 pattern)."""
    manifest_hash = json.loads(
        (PKG / "freeze_manifest.json").read_text())["manifest_hash"]
    manifest = verify_freeze(manifest_hash)
    print(f"freeze verified: {manifest['manifest_hash']} "
          f"({len(manifest['files'])} files)", flush=True)
    arm_cfg = json.loads((PKG / "configs" / f"{ARM}.json").read_text())

    # Reconstruct the stage-1 frozen feature map by replaying the trainer's
    # deterministic construction sequence: manual_seed(0) -> build_tape_model
    # -> attach_multiclass_stage(65, 64) is the first torch-RNG consumer pair.
    # Two independent replays must agree bit-for-bit.
    torch.manual_seed(0)
    probe = build_tape_model(arm_cfg)
    probe.attach_multiclass_stage(S1_VOCAB, S1_VOCAB - 1)
    stage1_embed = probe.embed.weight.detach().clone()
    del probe
    torch.manual_seed(0)
    probe = build_tape_model(arm_cfg)
    probe.attach_multiclass_stage(S1_VOCAB, S1_VOCAB - 1)
    assert torch.equal(stage1_embed, probe.embed.weight.detach().clone()), \
        "stage-1 embedding reconstruction is not deterministic"
    del probe

    torch.manual_seed(0)
    model = build_tape_model(arm_cfg)
    model.attach_binary_stage(LADDER_VOCAB)
    ckpt = RESULTS / f"{ARM}_seed0" / "best.pt"
    state = torch.load(ckpt, map_location="cpu")
    # best.pt was saved during stage 3 with the stage-2 head_multi still
    # attached; tolerate EXACTLY those two stale keys and nothing else.
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing}"
    assert sorted(unexpected) == ["head_multi.bias", "head_multi.weight"], \
        f"unexpected keys: {unexpected}"
    model.eval()
    meta = json.loads((RESULTS / f"{ARM}_seed0" / "checkpoint.json").read_text())
    assert meta["arm"] == ARM
    return model, stage1_embed, manifest, meta, arm_cfg


@torch.no_grad()
def ladder_run(model, ids: torch.Tensor):
    """One forward on token ids; returns (summary-ready trace, accept preds)."""
    y, _, _, trace = model.forward(ids, collect_trace=True)
    idx = torch.arange(ids.shape[0], device=ids.device)
    logits = model.head_binary(y[idx, ids.shape[1] - 1])
    if not bool(torch.isfinite(logits).all()):
        raise SystemExit("nonfinite eval logits on a positive ladder word")
    preds = (logits[:, 1] > logits[:, 0]).long().cpu().tolist()
    return trace, preds


@torch.no_grad()
def main():
    held = os.environ.get("GPU_LEASE_HELD")
    if held not in tuple(map(str, range(7))) or \
            os.environ.get("CUDA_VISIBLE_DEVICES") != held:
        raise RuntimeError("broker lease required; card 7 forbidden")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    walls = {}

    t0 = time.time()
    model, stage1_embed, manifest, meta, arm_cfg = build_trained()
    model.to(device)
    stage1_embed = stage1_embed.to(device)
    walls["build_and_load_s"] = round(time.time() - t0, 3)
    ckpt = RESULTS / f"{ARM}_seed0" / "best.pt"

    micro = arm_cfg["microbatch"]
    result = dict(
        seed=0,
        physical_gpu=held,
        torch=torch.__version__,
        slots=arm_cfg["budget_slots"],
        model=(f"trained X28E {ARM} FusedTapeLayer (d_model=256, K64 V128 H4), "
               f"best.pt update {meta['update']}, validation "
               f"{meta['validation_metric']}, eval mode, read-only"),
        scope=("8 exact positive ladder words per bin (ID k=128, 2x k=256, "
               "4x k=512, 9x k=1152), a secondary 8-distinct-k band probe per "
               "bin, and 8 MQAR validation-prefix words; store-chunks "
               "(C=64), not whole-batch certificates; part-1 classifier"),
        trained_checkpoint=dict(
            arm=ARM,
            path=f"results_x28e_memory_head/{ARM}_seed0/best.pt",
            sha256=sha256_file(ckpt),
            update=meta["update"],
            validation_metric=meta["validation_metric"],
            selected_by=meta["selected_by"],
        ),
        freeze_manifest=dict(hash=manifest["manifest_hash"],
                             files=len(manifest["files"])),
        stage1_embedding=dict(
            reconstruction=("replayed frozen construction sequence: "
                            "torch.manual_seed(0) -> trainer.build_tape_model "
                            "-> attach_multiclass_stage(65, 64); "
                            "deterministic-replay asserted bitwise"),
            role=("frozen random feature map (F12f), never trained; the "
                  "stage-1 draw is not persisted in best.pt — hybrid: "
                  "trained stage-3 layer weights + training-time stage-1 "
                  "addressing geometry; no MQAR accuracy receipt possible "
                  "(the stage-1 64-class head is not persisted)"),
        ),
        ladder_bins={},
        ladder_bands={},
        mqar=None,
    )

    # -- ladder bins: 8 copies of the exact positive word per bin ----------
    for name, k in BINS:
        t0 = time.time()
        ids = torch.tensor([positive_word(k)] * N_COPIES, device=device)
        n_tokens = int(ids.numel())
        assert n_tokens <= micro["token_budget"] and \
            ids.shape[0] <= micro["max_sequences"], "microbatch budget exceeded"
        trace, preds = ladder_run(model, ids)
        row = summarize(trace, ids)
        row["k"] = k
        row["predictions_accept"] = preds
        # Structure evidence: the first 16 in-word case codes (word 0, head 0).
        row["case_prefix_word0_head0"] = trace.case[0, :16, 0].cpu().tolist()
        result["ladder_bins"][name] = row
        walls[f"ladder_bin_{name}_s"] = round(time.time() - t0, 3)
        b = row["buckets"]
        print(f"ladder_bin {name} k={k} "
              f"event_fraction={b['all']['event_token_fraction']} "
              f"a={b['all_a']['event_token_fraction']} "
              f"b={b['all_b']['event_token_fraction']} "
              f"preds={preds} "
              f"counts={json.dumps(row['counts'])}", flush=True)
        del trace

    # -- band probe: 8 distinct-k positive words per bin, one per forward ---
    for bname, (lo, hi) in BANDS.items():
        entries = []
        t0 = time.time()
        for k in band_ks(lo, hi):
            ids = torch.tensor([positive_word(k)], device=device)
            assert ids.numel() <= micro["token_budget"]
            trace, preds = ladder_run(model, ids)
            row = summarize(trace, ids)
            row["k"] = k
            row["predictions_accept"] = preds
            entries.append(row)
            del trace
        ev = sum(e["buckets"]["all"]["event_tokens"] for e in entries)
        tk = sum(e["buckets"]["all"]["chunks"] for e in entries) * CHUNK
        result["ladder_bands"][bname] = dict(
            band=[lo, hi],
            ks=[e["k"] for e in entries],
            words=entries,
            aggregated_event_token_fraction=(ev / tk) if tk else None,
            event_tokens=ev,
            event_token_denominator=tk,
            predictions_accept=[e["predictions_accept"][0] for e in entries],
        )
        walls[f"ladder_band_{bname}_s"] = round(time.time() - t0, 3)
        band = result["ladder_bands"][bname]
        print(f"ladder_band {bname} ks={band['ks']} "
              f"aggregated_event_fraction={band['aggregated_event_token_fraction']} "
              f"preds={band['predictions_accept']}", flush=True)

    # -- MQAR validation-prefix words (hybrid embedding, see provenance) ----
    t0 = time.time()
    split = STAGE1_SPLITS["validation"]
    assert (split.length, split.seed) == (128, 28302)
    words = torch.from_numpy(
        gen_mqar(split.length, N_COPIES, split.seed)[0]).to(device)
    w_mask = (words != 0).to(torch.float32)
    x = torch.nn.functional.embedding(words, stage1_embed)
    _, _, _, trace = model.layer(x, collect_trace=True, w_mask=w_mask)
    row = summarize(trace, words)
    # Token IDs 1/2 are content classes, not ladder phases (part-1 convention).
    row["buckets"] = {key: row["buckets"][key]
                      for key in ("all", "all_pad", "other")}
    row["prediction_head"] = "none (stage-1 64-class head not persisted)"
    # Positional evidence: where the corrections fire (word 0, per head),
    # plus the query mask positions for cross-checking.
    cases_cpu = trace.case[0].cpu()
    row["correction_positions_word0_per_head"] = [
        (cases_cpu[:, h] == 4).nonzero().flatten().tolist() for h in range(cases_cpu.shape[1])]
    row["word0_tokens"] = words[0].cpu().tolist()
    result["mqar"] = row
    walls["mqar_s"] = round(time.time() - t0, 3)
    b = row["buckets"]
    print(f"mqar event_fraction={b['all']['event_token_fraction']} "
          f"counts={json.dumps(row['counts'])}", flush=True)
    del trace

    result["walls"] = walls
    result["walls_note"] = ("single-tenant walls for this eval process only; "
                            "other cards may serve other agents concurrently")
    paths = [Path(__file__), ROOT / "classifier.py", ROOT / "density.py",
             ROOT / "CHUNK_PARALLEL.md", ROOT / "RECEIPTS.md",
             PKG / "freeze_manifest.json", PKG / "configs" / f"{ARM}.json",
             PKG / "trainer.py",
             PKG / "generators" / "stage1_mqar.py",
             PKG / "generators" / "stage3_ladder.py"]
    result["source_sha256"] = {
        str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in paths}
    (ROOT / "density_trained.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print("wrote density_trained.json", flush=True)


if __name__ == "__main__":
    main()
