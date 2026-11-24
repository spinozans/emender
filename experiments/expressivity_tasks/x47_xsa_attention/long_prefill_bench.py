"""S9: qualify native long chunk→one-byte recurrent handoff before any timing cells."""

from collections import Counter
from pathlib import Path
from time import perf_counter_ns as clock
import argparse, copy, hashlib, json, os
import torch
from ndm.memory_program.resident_host import ResidentArena
from .large_host_bench import prepare_large, adapt, EXTERNAL, digest
from .large_host_check import stock
from .wide_host_bench import Bound
from .host_vjp_bench import CHUNK_SHA
from .serving_bench import measure
from .w3_data import HERE, getwindow
from .workbench_run import dump
from .workbench_witness import require


class SingleCache(dict):
    """One native past_key_values per document. Directly holds the layer's own returned tensors."""

    def update(self, *, layer_idx, recurrent_state, conv_state, offset):
        self[layer_idx] = dict(recurrent_state=recurrent_state, conv_state=conv_state)


def independent(net, cache, data):
    ids = torch.as_tensor([list(data)], device=net.embed.weight.device)
    x = net.embed(ids)
    for block in net.blocks:
        delta, state = block.layer(block.norm(x), cache)
        require(state is cache, "lost stock split cache")
        x = x + delta
    h = net.norm(x[:, -1])
    return h[0], net.head(h).float().log_softmax(-1)[0]


def arena(net, b, widths):
    return ResidentArena(
        net, b, 2048, widths, byte_budget=256 << 20, private_budget=64 << 20, chunk_prefill=True
    )


@torch.no_grad()
def check(run, net, windows):
    lengths = [63, 65, 128, 1025, 2047]
    cases = []
    failures = []
    start = clock()
    with arena(net, len(lengths), (1, len(lengths))) as a:
        ids = [a.admit(str(i)) for i in range(5)]
        orig = list(range(5))
        ws = [windows[i][:n] for i, n in enumerate(lengths)]
        cache = [SingleCache() for _ in ids]

        def audit(out, ix, inputs, tag):
            for j, (i, data) in enumerate(zip(ix, inputs, strict=True)):
                hs, ps = independent(net, cache[i], data)
                h, p = stock(net, [ws[i]], limit=2048)
                ah, ap = (x[j] for x in out)
                cc = dict(tag=tag, row=orig[i], frontier=len(ws[i]), errors={})

                def pair(k, x, y):
                    finite = (x.isfinite() & y.isfinite()).all().item()
                    err = (x - y).abs().max().item() if finite else None
                    ok = finite and torch.allclose(x, y, atol=5e-4, rtol=2e-4)
                    cc["errors"][k] = dict(elements=x.numel(), max_abs=err, ok=ok)
                    if not ok:
                        failures.append((tag, i, k, err))

                pair("stock_full_hidden", ah, h[0])
                pair("stock_full_logp", ap, p[0])
                pair("split_hidden", ah, hs)
                pair("split_logp", ap, ps)
                cc["argmax"] = bool(ap.argmax() == p[0].argmax() and ap.argmax() == ps.argmax())
                if not cc["argmax"]:
                    failures.append((tag, i, "argmax", None))
                states = [cache[i][z] for z in range(len(net.blocks))]
                for k in ("recurrent", "q", "k", "v"):
                    yy = torch.stack(
                        [
                            s["recurrent_state"][0]
                            if k == "recurrent"
                            else s["conv_state"][("q", "k", "v").index(k)][0]
                            for s in states
                        ]
                    )
                    pair("state_" + k, a.slots[k][ids[i].slot], yy)
                cases.append(cc)
                dump(
                    run / "progress.json",
                    dict(
                        cell="qualification",
                        phase=tag,
                        oracle_vectors=len(cases),
                        total_emitted=0,
                        server_pid=os.getpid(),
                    ),
                )

        audit(a.prefill(ids, ws), list(range(5)), ws, "prefill")
        ix = [4, 2, 0, 3, 1]
        suffix = [windows[i][len(ws[i]) : len(ws[i]) + 1] for i in ix]
        out = a.advance([ids[i] for i in ix], [len(ws[i]) for i in ix], [x[0] for x in suffix])
        for i, s in zip(ix, suffix):
            ws[i] += s
        audit(out, ix, suffix, "reordered-step")
        s = windows[0][64:65]
        out = a.advance(ids[:1], [64], [s[0]])
        ws[0] += s
        audit(out, [0], [s], "cross64")
        peer = [x.clone() for x in a.read(ids[:1], [65])]
        old = ids[-1]
        try:
            a.advance([old], [2048], [0])
            raise AssertionError("past 2048")
        except ValueError:
            pass
        a.release([old])
        assert all(t[old.slot].count_nonzero().item() == 0 for t in a.slots.values())
        try:
            a.read([old], [2048])
            raise AssertionError("stale client")
        except ValueError:
            pass
        orig[-1] = 16
        ids[-1] = a.admit("opposite-block")
        ws[-1] = windows[16][:513]
        cache[-1] = SingleCache()
        audit(a.prefill(ids[-1:], ws[-1:]), [4], ws[-1:], "reuse")
        assert all(torch.equal(x, y) for x, y in zip(peer, a.read(ids[:1], [65])))
        a.clear()
        assert all(not t.count_nonzero().item() for t in a.slots.values())
    results = dict(
        cases=cases,
        failures=failures,
        qualified=not failures,
        zeroed_and_reused=True,
        wall_ns=clock() - start,
    )
    dump(run / "device-parity.json", results)
    if failures:
        raise ValueError(f"chunk/recurrent conformance veto: {failures[:4]}; no timed cells")


class Full:
    def __init__(self, net, ss, lives):
        self.net = net
        self.ss = ss
        self.lives = lives
        self.tokens = 0

    def inputs(self, ids):
        ws = [bytes(self.lives[i].data) for i in ids]
        require(all(len(w) == self.ss[i].visible for i, w in zip(ids, ws, strict=True)), "full frontier")
        self.tokens += sum(map(len, ws))
        return adapt(*stock(self.net, ws, limit=2048))

    def clear(self):
        pass

    def report(self):
        return dict(tensor_bytes=0, encoded_tokens=self.tokens)


@torch.no_grad()
def prepare_long(run):
    net, eager, fused, selected, short, man = prepare_large(run)
    require(digest(EXTERNAL / "lit_gpt/gdn2_ops/chunk_gdn2.py") == CHUNK_SHA, "wrong long chunk kernel")
    cpu = copy.deepcopy(eager).cpu()
    del fused
    panel = json.loads((HERE / "receipts/w3/panels.json").read_text())
    assert selected == panel["panels"]["train"][:32]
    with Path(panel["source"]["path"]).open("rb") as f:
        ws = [getwindow(f, d) for d in selected]
    assert all(len(w) == 2049 and w[:129] == v for w, v in zip(ws, short, strict=True))
    meta = json.loads((run / "identity.json").read_text())
    meta.update(
        prompt_bytes=512,
        generated_length=8,
        prompt_sha=[hashlib.sha256(w[:512]).hexdigest() for w in ws],
        chunk_sha=CHUNK_SHA,
    )
    meta["generation_base"]["max_context"] = 2048
    dump(run / "identity.json", meta)
    return net, eager, cpu, selected, ws, man


@torch.no_grad()
def main(run):
    net, eager, cpu, selected, ws, man = prepare_long(run)
    check(run, net, ws)
    total = 0
    t0 = clock()
    with arena(net, 16, (1, 16)) as pool:
        torch.cuda.synchronize()
        dump(
            run / "setup.json",
            dict(wall_ns=clock() - t0, tensor_bytes=pool.tensor_bytes, private_bytes=pool.backend.pool_bytes),
        )

        def bound(n, s, l):
            return Bound(pool, n, s, l)

        for block in (0, 1):
            for b in (1, 16):
                ix = list(range(block * 16, block * 16 + b))
                arms = [
                    (m, h) for m in ("direct", "fused-greedy", "fused-sample") for h in ("full", "resident")
                ]
                for m, h in arms[::-1] if block else arms:
                    total += measure(
                        run,
                        net,
                        eager if m == "direct" else cpu,
                        m,
                        block,
                        [selected[i] for i in ix],
                        [ws[i] for i in ix],
                        ix,
                        man,
                        total,
                        label=f"{h}-{m}",
                        host_factory=bound if h == "resident" else Full,
                        length=8,
                        prefix=512,
                    )
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
