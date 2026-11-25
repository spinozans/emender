"""One local DR FIT (observed writes only), private decoder seal, and all-byte TRAIN PROBE."""

import shutil
from time import perf_counter
import torch
from ..x48_muon_continuation.data import check, load, sha
from . import (
    decode_data as data,
    decode_panel as p,
    decode_record as dr,
    decode_store as store,
    followup_loop as loop,
    joint_reader as joint,
    literal_mark as ls,
    local_u,
    panel,
    surprise_run as prior,
)
from .decode_fit import Moments as Moments
from .decode_store import tables as load_decoder
from .endpoint import frozen_prefill
from .meter import MODULE
from .panels import OUT
from .surprise_run import model as model

MOD = MODULE + ".decode_run"
AUTH = OUT / "decode-record-authorization-1.json"
GATES = 142
CORE = "5997d05eccf0b910ae69ee9bf42fe92acb5360d2154ef850ec1b5e2ab2d2bff6"


def source():
    ref = prior.source()
    p.bindings(hash_weights=False)
    check(
        sha(OUT / "decode-record-mechanism-gate-2.log") == CORE,
        "unbound DR synthetic check",
    )
    return ref


def qualification():
    return prior.qualify_frozen(
        source,
        "decode-record",
        data.SPEC,
        GATES,
        "DR teacher-free responses, centered ridge, private decoder seal/replay, terminal paired analysis, three strict real CPU loads; no natural DR access",
        writer_build=joint.COST / "build",
    )


def checked_fixture_ref():
    return prior.checked_gate(source, "decode-record", data.SPEC, GATES)


def authority():
    check(AUTH.is_file(), "no separate DR FIT/PROBE budget")
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient DR disk")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong DR stage")
    return loop.admit(job, p, authority)


def document_options(plan, seed):
    ck, ub = local_u.contract(seed)
    jk, jb = joint.contract(seed)
    k = str(seed)
    check(
        plan["endpoints"][k]["sha256"] == ck == jk
        and plan["u_beta"][k] == list(ub)
        and plan["j_beta"][k] == list(jb)
        and plan["writer_binary_sha"] == p.jt.lineage()["writer_binary_sha"],
        "foreign seed, writer or DR trust readout",
    )
    return dict(ub=ub, jb=jb, writer=joint._writer())


def document(*a, beta=None, **kw):
    return dr.document(*a, decoder=beta, **kw)


@torch.inference_mode()
def finish_fit(d, seed, mom, host, head, options):
    tic = perf_counter()
    dec = mom.solve()
    secs = dict(solve=perf_counter() - tic)
    tic = perf_counter()
    state = store.store(d, seed, mom, dec)
    secs["store"] = perf_counter() - tic
    tic = perf_counter()
    w = p.fixture_word()
    width = head.direct.host_dim
    # Fresh invented bytes only. Load the actual FP32 storage before exercising it.
    dec = load_decoder(d, seed, state)
    a = dr.panel_answers(host, head, w, dec, **options, width=width)
    h, z, _ = frozen_prefill(host, w, width)
    ref = ls.fields(head, w, h, z)
    check(
        torch.equal(ref.base.double(), a.scores["B"])
        and all(
            torch.equal(ls.response(ref, k, options[k.lower() + "b"]), a.scores[k])
            for k in ("U", "J")
        ),
        "DR full siblings disagree on invented inputs",
    )

    def forbid(*a, **kw):
        raise ValueError("standalone decoder used the teacher writer")

    hook = head.writer.register_forward_pre_hook(forbid)
    try:
        compact = dr.compact_predict(
            host, head, w, dec, options["jb"], writer=options["writer"], width=width
        )
    finally:
        hook.remove()
    check(
        torch.equal(compact, a.scores["D"]) and len(a.query) > 0,
        "DR teacher-disabled checkpoint parity failed",
    )
    state["synthetic"] = dict(
        observed_sha=p.fixture_sha(),
        hit_rows=len(a.query),
        all_256_B_U_J_equal=True,
        all_256_teacher_disabled_D_equal=True,
        synthetic_host_prefills=3,
    )
    secs["synthetic_teacher_disabled"] = perf_counter() - tic
    return state, secs


def evaluate(job, ctx):
    return prior.evaluate(job, ctx, study="DR")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32)
