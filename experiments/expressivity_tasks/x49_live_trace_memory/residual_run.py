"""One RB FIT/PROBE job. Qualification binds the actual head axes before any RB preparation."""

import shutil
from time import perf_counter
import torch
from ..x48_muon_continuation.data import check, load, sha
from . import (
    residual_byte as rb,
    residual_codec as codec,
    residual_data as data,
    residual_panel as p,
    residual_store as store,
    decode_record as dr,
    decode_run,
    followup_loop as loop,
    joint_reader as joint,
    literal_mark as ls,
    panel,
    surprise_run as prior,
)
from .residual_codec import Moments as Moments
from .residual_store import tables as load_decoder
from .endpoint import frozen_prefill
from .meter import MODULE
from .panels import OUT
from .surprise_run import model as model

MOD = MODULE + ".residual_run"
AUTH = OUT / "residual-byte-authorization-1.json"
GATES = 153
CORE = "3c38bdb598917e098a6f6a8c48849e4bfa8cc7c9034d81d7e64db9c2515c16c0"


def source():
    ref = prior.source()
    p.bindings(hash_weights=False)
    check(
        sha(OUT / "residual-byte-mechanism-gate-1.log") == CORE,
        "changed RB mechanism gate",
    )
    return ref


def audit_head(s, e, head):
    r = data.checked_spec()
    k = str(s)
    row = r["decoders"][k]
    check(
        e["sha256"] == row["checkpoint_sha"] == r["checkpoints"][k],
        "foreign base for RB axes",
    )
    _, note = codec.response_axes(head)
    return dict(checkpoint_sha=e["sha256"], coarse_sha=row["tables_sha"], **note)


def qualification():
    return prior.qualify_frozen(
        source,
        "residual-byte",
        data.SPEC,
        GATES,
        "RB invented strict sign code, conditioned donor and lifecycle; actual three frozen CPU heads, axes and sealed DR tables; no natural RB text",
        writer_build=joint.COST / "build",
        head_audit=audit_head,
    )


def checked_fixture_ref():
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient RB disk")
    ref = prior.checked_gate(source, "residual-byte", data.SPEC, GATES)
    w, _, c = p.bindings(hash_weights=False)
    axes = p.checked_axes(ref, w, c)
    # The public, source-only qualification includes the axis hash per checkpoint.
    # No changing these between qualification, selection and the leased inference worker.
    prior.check_public_qualification("residual-byte", ref, data.SPEC, axes)
    return ref


def authority():
    check(AUTH.is_file(), "no separate RB FIT/PROBE budget")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong RB stage")
    return loop.admit(job, p, authority)


def document_options(plan, seed, head):
    k = str(seed)
    opts = decode_run.document_options(plan, seed)
    axes, note = codec.response_axes(head)
    cs = plan["coarse_sha"][k]
    e = dict(checkpoint_sha=plan["endpoints"][k]["sha256"], coarse_sha=cs, **note)
    check(plan["qualified_axes"][k] == e, "wrong RB seed/axes or DR table")
    return dict(**opts, coarse=p.bound_coarse(seed, cs), axes=axes)


def document(*a, beta=None, **kw):
    return rb.document(*a, book=beta, **kw)


@torch.inference_mode()
def finish_fit(d, seed, mom, host, head, options):
    tic = perf_counter()
    b = mom.solve()
    secs = dict(solve=perf_counter() - tic)
    tic = perf_counter()
    state = store.store(d, seed, mom, b)
    secs["store"] = perf_counter() - tic
    tic = perf_counter()
    b = load_decoder(d, seed, state)
    w = p.dr.fixture_word()
    width = head.direct.host_dim
    a, _ = rb.panel_answers(host, head, w, book=b, **options, width=width)
    h, z, _ = frozen_prefill(host, w, width)
    ref = ls.fields(head, w, h, z)
    ub, jb, writer = (options[k] for k in ("ub", "jb", "writer"))
    d_only = dr.compact_predict(
        host, head, w, options["coarse"], jb, writer=writer, width=width
    )
    check(
        torch.equal(a.scores["D"], d_only)
        and torch.equal(ref.base.double(), a.scores["B"])
        and all(
            torch.equal(a.scores[k], ls.response(ref, k, v))
            for k, v in [("U", ub), ("J", jb)]
        ),
        "RB changed a full or compact anchor",
    )
    calls = []

    def once(_, inputs):
        check(
            not calls and inputs[0].shape == (len(w), head.writer.in_features),
            "RB compact path recomputed the teacher",
        )
        calls.append(True)

    hook = head.writer.register_forward_pre_hook(once)
    try:
        r = rb.compact_predict(
            host,
            head,
            w,
            options["coarse"],
            options["axes"],
            b,
            jb,
            writer=writer,
            width=width,
        )
    finally:
        hook.remove()
    check(
        calls == [True] and torch.equal(r, a.scores["R"]) and len(a.query) > 0,
        "RB emitted-record parity failure",
    )
    initial = load(d / "frozen.json")
    k = str(seed)
    state.update(
        input_tables=dict(
            axes_sha256=initial["qualified_axes"][k]["axes_sha256"],
            coarse_sha=initial["coarse_sha"][k],
        ),
        synthetic=dict(
            observed_sha=p.dr.fixture_sha(),
            hit_rows=len(a.query),
            synthetic_host_prefills=4,
            full_anchors_equal=True,
            one_write_standalone_R_equal=True,
        ),
    )
    secs["synthetic_emit_read"] = perf_counter() - tic
    return state, secs


def evaluate(job, ctx):
    return prior.evaluate(job, ctx, study="RB")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32)
