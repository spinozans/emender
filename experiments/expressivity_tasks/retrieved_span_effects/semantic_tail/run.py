"""CPU-only single-use FIT -> verify all finals -> paired synthetic PROBE."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from ..corpus_kernel.token_lm import CorpusTokenLM
from .editions import (
    ARMS,
    K,
    SEEDS,
    TOKENIZER,
    VOCAB,
    digest,
    fit_editions,
    order,
    packet,
    probe_pairs,
)
from .measure import Means, compare, view


REGISTRATION = "75fa4b70bdbfcec81bdb92a5ba6cdc47a108376f"


def make_model(seed):
    torch.manual_seed(seed)
    return (
        CorpusTokenLM(
            VOCAB,
            32,
            32,
            tokenizer=TOKENIZER,
            leaf=4,
            feature_seed=5700,
            values="vector",
        )
        .cpu()
        .float()
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(m):
    return digest(
        {
            k: hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest()
            for k, v in m.state_dict().items()
        }
    )


def save(path, obj):
    # The terminal manifest is installed only after all referenced files close.
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.link(tmp, path)  # unlike replace(), never overwrite an existing receipt
    tmp.unlink()


def read(path):
    return json.loads(path.read_text())


def check_cpu(ref):
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or any(
            k in os.environ
            for k in ("GPU_LEASE_HELD", "PYTHONOPTIMIZE", "TRITON_INTERPRET")
        )
        or any(
            os.environ.get(k) != "1"
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        )
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError(
            "requires explicit single-thread CPU environment, no GPU lease or Triton interpreter"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[4]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if (
        len(ref) != 40
        or git("rev-parse", "HEAD") != ref
        or git("status", "--porcelain")
    ):
        raise RuntimeError("run from the clean checkout at the qualified public ref")


def fit(out, ref):
    out.mkdir(mode=0o700, exist_ok=False)
    editions = fit_editions()
    geometry = make_model(SEEDS[0])
    # Identical immutable packets across arms with the same corpus setting; no
    # learned state, including archive values, lives in these cached objects.
    packets = {
        use: [packet(geometry, p, corpus=use) for p in editions]
        for use in (True, False)
    }
    files = {}
    for seed in SEEDS:
        seq = order(seed)
        for arm in ARMS:
            key = f"{seed}.{arm}"
            m = make_model(seed).train()
            initial = state_sha(m)
            opt = torch.optim.AdamW(
                m.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=0.01
            )
            law = "indexed" if arm == "indexed" else "dense"
            touched = set()
            params = dict(m.named_parameters())
            targets = 0
            start = time.perf_counter()
            for i in seq:
                w = packets[arm != "local"][i]
                opt.zero_grad(set_to_none=True)
                loss = w.backward_nll(m, mode=law)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite FIT NLL")
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), 1.0, error_if_nonfinite=True
                )
                for name in params.keys() - touched:
                    grad = params[name].grad
                    if grad is not None and torch.count_nonzero(grad):
                        touched.add(name)
                opt.step()
                targets += len(w.targets)
            elapsed = time.perf_counter() - start
            if len(seq) != 1024 or targets != 16384:
                raise ValueError("dose mismatch")
            weights = out / (key + ".pt")
            with weights.open("xb") as f:
                torch.save(m.state_dict(), f)
            meta = dict(
                seed=seed,
                arm=arm,
                initial=initial,
                final=state_sha(m),
                weights_sha256=sha(weights),
                order_sha256=digest(seq),
                updates=len(seq),
                targets=targets,
                fit_loop_s=elapsed,
                trainable_scalars=sum(v.numel() for v in params.values()),
                nonzero_gradient_parameters=sorted(touched),
                untouched_parameters=sorted(params.keys() - touched),
            )
            side = out / (key + ".json")
            save(side, meta)
            files[key] = sha(side)
            print("closed", key, targets, flush=True)
    save(
        out / "frozen.json",
        dict(
            schema=1,
            registered=REGISTRATION,
            qualified=ref,
            tokenizer=TOKENIZER,
            fit_editions_sha256=digest(editions),
            fits=files,
        ),
    )


def verified(out, ref, frozen_sha):
    path = out / "frozen.json"
    if sha(path) != frozen_sha:
        raise ValueError("unwitnessed freeze")
    d = read(path)
    if (
        d["schema"] != 1
        or d["qualified"] != ref
        or d["registered"] != REGISTRATION
        or d["tokenizer"] != TOKENIZER
        or d["fit_editions_sha256"] != digest(fit_editions())
        or set(d["fits"]) != {f"{s}.{a}" for s in SEEDS for a in ARMS}
    ):
        raise ValueError("incomplete or foreign freeze")
    models, prices = {}, {}
    for seed in SEEDS:
        order_sha = digest(order(seed))
        for arm in ARMS:
            key = f"{seed}.{arm}"
            path = out / (key + ".json")
            if sha(path) != d["fits"][key]:
                raise ValueError("altered fit receipt")
            z = read(path)
            m = make_model(seed)
            weights = out / (key + ".pt")
            if (
                sha(weights) != z["weights_sha256"]
                or z["initial"] != state_sha(m)
                or (z["seed"], z["arm"], z["order_sha256"], z["updates"], z["targets"])
                != (seed, arm, order_sha, 1024, 16384)
            ):
                raise ValueError("invalid frozen fit")
            m.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True), strict=True
            )
            if state_sha(m) != z["final"] or not all(
                v.isfinite().all() for v in m.state_dict().values()
            ):
                raise ValueError("foreign or nonfinite frozen weights")
            models[(seed, arm)] = m.eval().requires_grad_(False)
            prices[key] = z
    return models, prices


def conclusions(values):
    out = {}
    for arm in ("dense", "indexed"):
        by_seed = {}
        for s in SEEDS:
            m = values[f"{s}.{arm}"]
            gain = values[f"{s}.local"]["edited.answer_nll"] - m["edited.answer_nll"]
            joint, stale = m["changed.old_and_new_correct"], m["changed.gain_vs_stale"]
            by_seed[str(s)] = dict(
                gain_vs_local=gain,
                changed_old_and_new_correct=joint,
                changed_gain_vs_stale=stale,
                passes=(gain > 0 and joint > 0.5 and stale > 0),
            )
        out[arm] = dict(
            by_seed=by_seed, passes=all(v["passes"] for v in by_seed.values())
        )
    return out


@torch.no_grad()
def probe(out, ref, frozen_sha):
    models, prices = verified(out, ref, frozen_sha)
    save(
        out / "probe-opened.json",
        dict(frozen_sha256=frozen_sha, verified_fits=len(models)),
    )
    pairs = tuple(probe_pairs())
    geometry = next(iter(models.values()))
    sums = {key: Means() for key in models}
    for p, q, keys in pairs:
        for corpus in (False, True):
            wa, wb = (packet(geometry, ed, corpus=corpus) for ed in (p, q))
            if not torch.equal(wa.inputs.ids, wb.inputs.ids) or not torch.equal(
                wa.archive.inputs.ids, wb.archive.inputs.ids
            ):
                raise ValueError("edit changed inputs or keys")
            for key, m in models.items():
                _, arm = key
                if (arm != "local") != corpus:
                    continue
                law = "indexed" if arm == "indexed" else "dense"
                a, b = (view(w, m, law) for w in (wa, wb))
                other = (
                    view(wb, m, "dense" if law == "indexed" else "indexed")
                    if corpus
                    else None
                )
                compare(sums[key], a, wa.targets, b, wb.targets, keys, cross=other)
    values = {}
    for (seed, arm), total in sums.items():
        n = total.counts
        if (
            n["edited.all_nll"],
            n["edited.answer_nll"],
            n["changed.gain_vs_stale"],
            n["unchanged.gain_vs_stale"],
        ) != (3584, 1792, 448, 1344):
            raise ValueError("probe denominator mismatch")
        result = total.result()
        if not all(map(math.isfinite, result.values())):
            raise ValueError("nonfinite probe aggregate")
        values[f"{seed}.{arm}"] = result
    save(
        out / "result.json",
        dict(
            schema=1,
            scope="enumerated invented-token edit fixture; not natural language or a GPU test",
            registered=REGISTRATION,
            qualified=ref,
            tokenizer=TOKENIZER,
            frozen_sha256=frozen_sha,
            fit_editions_sha256=digest(fit_editions()),
            probe_pairs_sha256=digest(pairs),
            counts_per_fit=dict(
                probe_edition_pairs=len(pairs),
                editions=2 * len(pairs),
                archive_rows_per_edition=K,
                exact_refined_pairs=0,
                all_shifted_targets=4 * K * len(pairs),
                edited_answers=K * len(pairs),
                changed_bindings=2 * len(pairs),
                unchanged_bindings=(K - 2) * len(pairs),
            ),
            fits=prices,
            metrics=values,
            primary=conclusions(values),
            cuda_launches=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "probe"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--public-ref", required=True)
    parser.add_argument("--freeze-sha")
    args = parser.parse_args()
    check_cpu(args.public_ref)
    if args.phase == "fit" and args.freeze_sha is None:
        fit(args.out, args.public_ref)
    elif args.phase == "probe" and args.freeze_sha:
        probe(args.out, args.public_ref, args.freeze_sha)
    else:
        parser.error("only PROBE requires --freeze-sha")
