"""Two consecutive TRAIN-only value updates under an immutable executed selector. No host training."""

import argparse
from copy import deepcopy
from pathlib import Path
import os
import resource
import subprocess
import sys
from time import perf_counter_ns as clock, time, time_ns

import torch
from ndm.memory_program.frozen_routes import digest, record_training_routes, replay_routes
from ndm.memory_program.paired_frontier import FixedFrontierPair
from ndm.memory_program.serving import StreamUniforms
from .branch_domain_bench import WavePages, high_water
from .direct_control_bench import setup
from .frozen_packet_bench import OPT, canonical, no_card, tensor_sha
from .frozen_route_bench import PURE
from .paired_frontier_fixture import ORDER, GATES, SEED, Physical, Report, inputs
from .pinned_branch import p100_domain
from .scoped_records import ScopedRecords
from .serving_verify import HERE, load, require
from .workbench_run import dump
from .workbench_witness import digest_state


def packet_fn(packet, ss, ix, verify):
    enc = packet.encoder(ss, row_indices=ix)

    def get(x):
        h, b = enc(x)
        require(not verify and not h.requires_grad and not b.requires_grad, "additional host features")
        verify.append(tensor_sha(torch.cat([h, b], -1)))
        return h, b

    return get


def separate(source, values, records, shared, rows, words, ix, packet, authority, progress):
    reports = []
    phys = []
    inputs = []
    tape = None
    clears = {}
    for label, model in [("select", source), ("values", values)]:
        progress(label + "-binding", 0)
        t = clock()
        with Physical(records, shared, rows, model=model) as mount:
            ss = mount.sessions
            rep = Report(ss)
            h = []
            common = dict(authority=authority, max_steps=16 * 128 * 6)
            encode = packet_fn(packet, ss, ix, h)
            if label == "select":
                tape, batch = record_training_routes(
                    model, encode, ss, words, **common, generator=StreamUniforms(ss, ix, 129, SEED, "cpu")
                )
            else:
                batch = replay_routes(model, encode, ss, words, tape, **common)
                require(batch.objective is None, "forced score-function training")
            require(batch.targets == 2048 and tape.authority == authority, "partial separate rollout")
            for j in range(128):
                rep.note([doc[j] for doc in batch.decisions], [digest(doc[j]).hex() for doc in tape.programs])
            reports.append(rep.finish(batch.mean_nll))
            obs = mount.observed(words)
            inputs += h
            if label == "values":
                loss = batch.mean_nll
            del batch
        phys.append(dict(mount.final, observed129_sha=obs))
        clears[label + "_ns"] = clock() - t
        progress(label + "-complete", 2048)
    require(len(inputs) == 2 and inputs[0] == inputs[1], "unmatched causal inputs")
    return reports, phys, inputs[0], loss, None, clears


def paired(source, values, records, shared, rows, words, ix, packet, progress):
    progress("pair-binding", 0)
    t = clock()
    features = []
    with Physical(records, shared, rows) as mount:
        ss = mount.sessions
        rng = StreamUniforms(ss, ix, 129, SEED, "cpu")
        rep = [Report(ss), Report(ss)]
        losses = [[[] for _ in ss] for _ in rep]
        h, b = packet_fn(packet, ss, ix, features)(torch.tensor([list(w[:-1]) for w in words]))
        with FixedFrontierPair(source, values, ss, guard=mount.guard, byte_limit=129) as engine:
            engine.observe([w[:1] for w in words])
            for j in range(128):
                out = engine.predict(
                    torch.stack([v[j] for v in h]), torch.stack([v[j] for v in b]), generator=rng
                )
                y = [w[j + 1] for w in words]
                label = torch.tensor(y, dtype=torch.long)
                for rr, store, ds in zip(rep, losses, (out.selectors, out.values), strict=True):
                    rr.note(ds, out.choice_digests)
                    ls = -torch.stack([d.log_probs for d in ds]).gather(1, label[:, None])[:, 0]
                    for row, v in zip(store, ls, strict=True):
                        row.append(v)
                engine.observe([bytes([v]) for v in y])
                del out
                if (j + 1) % 32 == 0:
                    progress("pair-scored", len(ss) * (j + 1))
            obs = mount.observed(words)
            stage = engine.work()
        means = [torch.stack([torch.stack(row).sum() for row in role]).sum() / 2048 for role in losses]
        reports = [r.finish(v) for r, v in zip(rep, means, strict=True)]
    return (
        reports,
        [dict(mount.final, observed129_sha=obs)],
        features[0],
        means[1],
        stage,
        dict(pair_ns=clock() - t),
    )


def optim_state(model, opt):
    return {
        k: {n: tensor_sha(t) for n, t in opt.state.get(p, {}).items()} for k, p in model.named_parameters()
    }


def update(model, opt, loss, selector):
    t = clock()
    require(
        not any(p.grad is not None for p in model.parameters()) and loss.requires_grad and loss.isfinite(),
        "unbound value loss",
    )
    loss.backward()
    back = clock()
    ps = dict(model.named_parameters())
    grads = {k: None if p.grad is None else tensor_sha(p.grad) for k, p in ps.items()}
    require(
        all(p.grad is None for p in selector.parameters()) and all(grads[k] is None for k in PURE),
        "selector learned in fixed control",
    )
    norm = ps["writer.weight"].grad.norm().item()
    require(norm > 0, "missing writer gradient")
    preclip = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, foreach=False, error_if_nonfinite=True
    ).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return dict(
        grads=grads,
        writer_norm=norm,
        preclip_norm=preclip,
        params={k: tensor_sha(p) for k, p in ps.items()},
        moments=optim_state(model, opt),
        after=digest_state(model),
    ), dict(backward_ns=back - t, check_update_ns=clock() - back)


def usage():
    return sum(
        z.ru_utime + z.ru_stime
        for z in map(resource.getrusage, (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    )


def worker(root, name):
    ref = setup(root, gate_count=GATES)
    start = clock()
    epoch = time_ns()
    cpu = usage()
    run = root / name
    pair = name.startswith("pair")
    require(
        load(root / "driver.json")["pid"] == os.getppid() and not (run / "identity.json").exists(),
        "unbound/repeated cell",
    )
    source, packet, rows, ws, man, ident = inputs()
    values = deepcopy(source)
    opt = torch.optim.AdamW(values.parameters(), **OPT)
    dump(
        run / "identity.json",
        dict(
            ref=ref,
            name=name,
            inputs=ident,
            pid=os.getpid(),
            parent=os.getppid(),
            start_epoch_ns=epoch,
            threads=torch.get_num_threads(),
            torch=str(torch.__version__),
            strict=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
        ),
    )
    receipt = load(HERE / "b1_mounts.json")["P100"]["files"][1]
    steps = 0
    for step in range(2):

        def progress(phase, n):
            dump(
                run / "progress.json",
                dict(pid=os.getpid(), step=step, phase=phase, scored_in_role=n, updates=steps),
            )

        progress("admission", 0)
        begin = clock()
        ix = list(range(16 * step, 16 * (step + 1)))
        selected = rows[ix[0] : ix[-1] + 1]
        words = ws[ix[0] : ix[-1] + 1]
        pages = []

        def factory(**kw):
            require(not pages, "duplicate m20 worker")
            p = WavePages(**kw)
            p.begin()
            pages.append(p)
            return p

        initial = digest_state(values)
        initial_opt = optim_state(values, opt)
        params0 = {k: tensor_sha(p) for k, p in values.named_parameters()}
        authority = canonical(dict(identity=ident, rows=ix, selector=ident["initial"], ref=ref))
        with p100_domain(man, scope="whole", max_grants=1, pages_type=factory) as (domain, desc):
            with (
                domain.borrow("P100", man["mount"], scope="whole") as g,
                ScopedRecords(desc, receipt, scope="whole", branches=g) as records,
            ):
                shared = (g.reader(), *records.readers())
                opened = clock()
                args = source, values, records, shared, selected, words, ix, packet
                reports, phys, hs, loss, stage, phases = (
                    paired(*args, progress) if pair else separate(*args, authority, progress)
                )
                progress("value-backward", 2048)
                state, clocks = update(values, opt, loss, source)
                del loss
                steps += 1
                require(
                    digest_state(source) == ident["initial"] and steps == step + 1, "mutated fixed selector"
                )
                pos_peak = high_water(records.pid)
                upd = clock()
            wire = pages[0].finish()
        end = clock()
        require(domain.closed, "unclosed branch domain")
        progress("released", 2048)
        dump(
            run / f"batch-{step}.json",
            dict(
                step=step,
                ix=ix,
                authority=authority,
                feature_sha=hs,
                before=initial,
                before_params=params0,
                before_moments=initial_opt,
                selectors=reports[0],
                values=reports[1],
                physical=phys,
                stage=stage,
                update=state,
                position=records.final,
                position_peak_kib=pos_peak,
                wire=pages[0].receipt(),
                actual_rowhash=wire,
                clocks=dict(
                    **phases, **clocks, setup_ns=opened - begin, close_ns=end - upd, complete_ns=end - begin
                ),
                selector_after=digest_state(source),
            ),
        )
    no_card()
    dump(
        run / "cell.json",
        dict(
            ref=ref,
            pid=os.getpid(),
            name=name,
            wall_ns=clock() - start,
            end_epoch_ns=time_ns(),
            core_s=usage() - cpu,
            main_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scored=8192,
            backward_targets=4096,
            updates=steps,
            card_s=0,
            cuda_available=torch.cuda.is_available(),
            cuda_initialized=torch.cuda.is_initialized(),
        ),
    )


def main(root):
    ref = setup(root, gate_count=GATES)
    t = clock()
    require(
        time() + 490 < 1795531800 and not (root / "driver.json").exists(), "too late or repeated T12 driver"
    )
    dump(
        root / "driver.json", dict(ref=ref, pid=os.getpid(), order=ORDER, start_epoch_ns=time_ns(), card_s=0)
    )
    from .paired_frontier_verify import check_cell

    first = None
    for name in ORDER:
        run = root / name
        run.mkdir()
        start = clock()
        with (run / "console").open("xb") as f:
            try:
                p = subprocess.run(
                    [sys.executable, "-m", __spec__.name, str(root), "--cell", name],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    timeout=max(1, 469 - (start - t) / 1e9),
                )
            except subprocess.TimeoutExpired:
                dump(run / "exit.json", dict(timeout=True, wall_ns=clock() - start))
                raise
        dump(run / "exit.json", dict(rc=p.returncode, wall_ns=clock() - start))
        require(p.returncode == 0, "incomplete fixed control " + name)
        signature, *_ = check_cell(root, name, ref)
        if first is None:
            first = signature
        require(
            signature == first,
            "different frontier, executed route, value scores or optimizer transition: " + name,
        )
        print(name, "two value steps verified", flush=True)
    dump(
        root / "complete.json",
        dict(ref=ref, pid=os.getpid(), end_epoch_ns=time_ns(), wall_ns=clock() - t, updates=8, card_s=0),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    a = ap.parse_args()
    if a.cell:
        worker(a.root, a.cell)
    else:
        main(a.root)
