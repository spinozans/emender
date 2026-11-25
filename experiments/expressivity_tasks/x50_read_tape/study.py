"""Paired CPU fits, frozen evaluation and report functions for the registered study."""

from time import monotonic, process_time

import numpy as np
import torch

from . import data
from .kernel import Kernel
from .program import ARMS, T, Tape, execute, score

BATCH, EPOCHS = 16, 3


def setup():
    data.cpu()
    data.check(torch.get_default_device().type == "cpu", "unexpected default device")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data.check(
        not torch.cuda.is_initialized() and not torch.cuda.is_available(),
        "CPU-only study",
    )


def order(seed):
    i = data.SEEDS.index(seed)
    return ARMS[i:] + ARMS[:i]


def model_at_birth(seed):
    torch.manual_seed(seed)
    return Tape()


def phase(model, pages, windows, seed, arm, fit, tick):
    # New immutable-page LRU and forward state for every arm/seed/phase.
    tic, cpu = monotonic(), process_time()
    bank = Kernel(pages)
    model.train(fit)
    opt = (
        torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        if fit
        else None
    )
    shuffler = np.random.default_rng(seed)
    draw = torch.Generator().manual_seed(seed + 32_000) if fit else None
    n, length = windows.shape
    data.check(n % BATCH == 0 and length == T + 1 and arm in ARMS, "bad dose or arm")
    epochs = EPOCHS if fit else 1
    epoch_means = []
    steps = changed = 0
    for _ in range(epochs):
        doc_loss = np.full(n, np.nan)
        base_loss = np.full(n, np.nan)
        ranks = shuffler.permutation(n) if fit else np.arange(n)
        for ix in ranks.reshape(-1, BATCH):
            w = windows[ix]
            trace = execute(model, bank, w[:, :-1], arm, draw)
            with torch.set_grad_enabled(fit):
                objective, losses, base = score(model, trace, w[:, 1:])
                if fit:
                    opt.zero_grad(set_to_none=True)
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    opt.step()
            doc_loss[ix] = losses.detach().double().sum(1).numpy()
            base_loss[ix] = base.double().sum(1).numpy()
            if trace.action is not None:
                changed += (trace.action != trace.parent.argmax(-1)).sum().item()
            steps += 1
            tick(BATCH * T)
        data.check(
            np.isfinite(doc_loss).all() and np.isfinite(base_loss).all(),
            "nonfinite or missing document",
        )
        epoch_means.append(float(doc_loss.sum() / (n * T)))
    nt = n * T * epochs
    data.check(
        bank.counts["parent_reads"] == nt
        and bank.counts["child_reads"] == (0 if arm == "ONE" else nt),
        "missing/extra branch reads",
    )
    facts = dict(
        targets=nt,
        batches=steps,
        documents=n,
        epochs=epochs,
        mean_nll_by_epoch=epoch_means,
        changes_vs_parent_argmax=changed,
        archive=dict(bank.counts),
        wall_s=monotonic() - tic,
        cpu_s=process_time() - cpu,
        action_prior_scale=float(model.scale.detach()) if arm == "ADAPT" else None,
    )
    return facts, doc_loss, base_loss


def compare(losses, *, draws=10_000):
    """[fixed seed, paired document], each entry a sum over exactly T real targets."""
    data.check(set(losses) == set(ARMS) | {"PRIOR"}, "missing arm")
    shape = losses["ONE"].shape
    data.check(
        len(shape) == 2
        and shape[0] == len(data.SEEDS)
        and shape[1] > 0
        and all(x.shape == shape and np.isfinite(x).all() for x in losses.values()),
        "unpaired data",
    )
    per_seed = {arm: (ds.mean(1) / T).tolist() for arm, ds in losses.items()}
    policies = {
        arm: dict(by_seed=ys, mean=float(np.mean(ys))) for arm, ys in per_seed.items()
    }
    rng = np.random.default_rng(5000)
    indices = rng.integers(0, shape[1], size=(draws, shape[1]))
    pairs = [("ADAPT", c) for c in ("ONE", "BLIND", "GREEDY")] + [
        ("GREEDY", "ONE"),
        ("BLIND", "ONE"),
    ]
    contrasts = {}
    for a, b in pairs:
        ds = (losses[b] - losses[a]) / T
        # Mean over fixed weights first; resampling preserves every seed pairing.
        boots = ds.mean(0)[indices].mean(1)
        ci = np.quantile(boots, [0.025, 0.975], method="linear")
        contrasts[a + ":" + b] = dict(
            gain=float(ds.mean()), by_seed=ds.mean(1).tolist(), ci95=ci.tolist()
        )
    return policies, contrasts


def report():
    r = data.RUN
    seal = data.frozen()
    data.check(
        data.read(r / "probe.json")["fit_seal_sha"] == data.sha(r / "frozen.json"),
        "probe did not follow current seal",
    )
    fit = {}
    probe = {}
    losses = {k: [] for k in (*ARMS, "PRIOR")}
    for seed in data.SEEDS:
        reference = None
        for arm in ARMS:
            k = f"{seed}-{arm}"
            p = data.read(r / (k + "-probe.json"))
            f = data.read(r / (k + "-fit.json"))
            data.check(
                p["state"] == seal["states"][k]
                and p["scores"] == data.sha(r / (k + "-loss.npz"))
                and p["probe"] == data.sha(r / "probe.json")
                and f["state"] == p["state"]
                and f["fit"] == data.sha(r / "fit.json"),
                "changed fit/probe journal",
            )
            fit[k], probe[k] = f["facts"], p["facts"]
            with np.load(r / (k + "-loss.npz"), allow_pickle=False) as z:
                loss, base = z["model"], z["prior"]
            data.check(
                loss.shape == base.shape == (data.N["probe"],)
                and float(loss.sum() / (len(loss) * T))
                == p["facts"]["mean_nll_by_epoch"][0],
                "score sum or target count mismatch",
            )
            if reference is None:
                reference = base
            else:
                np.testing.assert_array_equal(base, reference)
            losses[arm].append(loss)
        losses["PRIOR"].append(reference)
    for v in losses["PRIOR"][1:]:
        np.testing.assert_array_equal(v, losses["PRIOR"][0])
    policies, contrasts = compare({a: np.stack(d) for a, d in losses.items()})
    positive = all(
        min(contrasts["ADAPT:" + c]["by_seed"]) > 0
        and contrasts["ADAPT:" + c]["ci95"][0] > 0
        for c in ("ONE", "BLIND", "GREEDY")
    )
    different = all(
        probe[f"{s}-ADAPT"]["changes_vs_parent_argmax"] > 0 for s in data.SEEDS
    )
    return dict(
        spec=data.SPEC,
        public_registration=data.REG,
        public_qualification=(data.OUT / "qualified-ref").read_text().strip(),
        fit_seal_sha=data.sha(r / "frozen.json"),
        seed_order=data.SEEDS,
        documents=data.N,
        context=T,
        probe_targets_per_seed=T * data.N["probe"],
        policies=policies,
        contrasts=contrasts,
        useful_learned_followup=bool(positive and different),
        changed_from_greedy_at_all_seeds=different,
        fit=fit,
        probe=probe,
        bootstrap=dict(draws=10_000, seed=5000, unit="paired whole document"),
        scope="fresh native TRAIN probe; P100 materialized kernel; no native final or scale decision",
    )
