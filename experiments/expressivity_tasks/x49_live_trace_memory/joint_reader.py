"""No-fit J evaluation candidate. No corpus, allocator, sham, other responses or persistent document state."""

from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import torch
from . import literal_mark as ls, local_u, native_text, surprise as u
from .endpoint import frozen_prefill
from .panels import OUT

RECEIPTS = Path(__file__).with_name("receipts")
RESULT = "4e2789566883cbaf11131919bfa72781f0ac0c0d74a0c58289caea2728fe99a2"
FIT = "443f3161f2dbfe075e6ece00329e6c2dc0ebedb19693100cf40292a672e47847"
REG = "00f06b9b97172a572695e8e1068b1c596772453ad7b38e844977ad681157dae1"
COST = OUT / "joint-reader-cost-1"


def contract(seed):
    ck, ub = local_u.contract(seed)
    raw = (RECEIPTS / "literal_surprise_result.json").read_bytes()
    if sha256(raw).hexdigest() != RESULT:
        raise ValueError("changed frozen LS result")
    z = json.loads(raw)
    k = str(seed)
    if (
        z["fit_freeze_sha"] != FIT
        or z["u_confirmation_sha"] != local_u.CONFIRMATION
        or z["checkpoints"][k] != ck
        or tuple(z["u_beta"][k]) != ub
        or not z["decision"]["consider_joint_confirmation"]
        or z["neural_updates"]
        or z["u_refits"]
    ):
        raise ValueError("J readout does not belong to this confirmed base")
    return ck, tuple(u.checked_vector(ls.DIMS, "J", z["beta"][k]["J"]).tolist())


@torch.inference_mode()
def _predict(host, head, beta, observed, width, writer):
    h, z, _ = frozen_prefill(host, observed, width)
    p = u.bound_fields(head, observed, h, z)
    tape, _ = writer(observed)
    hfeat = p.phi["U"]
    tc = tape[p.latest].double()[:, None] / 64
    # J = (three current fields, true literal mark, true neural mark). U isn't scored.
    phi = torch.cat((hfeat[:, :3], tc, hfeat[:, 3:]), dim=1)
    return u.copy_response(p, phi, u.checked_vector(ls.DIMS, "J", beta))


def _writer():
    pub = RECEIPTS / "joint_reader_cost.json"
    private = COST / "aggregate.json"
    if (
        not pub.is_file()
        or not private.is_file()
        or pub.read_bytes() != private.read_bytes()
    ):
        raise ValueError("no published matching synthetic writer qualification")
    note = json.loads(pub.read_bytes())
    if (
        note["protocol_sha256"] != REG
        or note["ls_result_sha256"] != RESULT
        or note["synthetic_parity"] is not True
        or note["natural_documents"] != 0
    ):
        raise ValueError("foreign writer qualification")
    return native_text._load(COST / "build", note["build"])


def load_candidate(seed):
    """For a separately admitted comparison, NOT a replacement for the confirmed local U predictor."""
    from .query_panel import checkpoint, module
    from ..x47_xsa_attention.workbench_run import check_gpu_owner

    ck, beta = contract(seed)
    check_gpu_owner()
    _, endpoint = checkpoint(seed, hash_weights=False)
    if endpoint["sha256"] != ck:
        raise ValueError("J readout and base checkpoint differ")
    writer = _writer()
    host, head = module(seed, endpoint)
    host = host.to("cuda")

    def predict(observed):
        return _predict(host, head, beta, observed, 256, writer)

    return predict
