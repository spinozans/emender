"""Opt-in, frozen local U predictor for an already admitted inference worker. No fit or data reader."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .endpoint import frozen_prefill
from .surprise import bound_fields, checked_beta, response

CONFIRMATION = "9c9bbabdf8cbb146af32f6b031e2c946c3840834c42779c6fc8cfd135bbc5626"
RECEIPT = Path(__file__).with_name("receipts") / "surprise_confirmation_result.json"


def contract(seed):
    """Evidenced checkpoint and its own four U coefficients; not transferable between seeds."""
    if type(seed) is not int or seed not in (4901, 4902, 4903):
        raise ValueError("unconfirmed U seed")
    raw = RECEIPT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIRMATION:
        raise ValueError("changed U confirmation")
    z = json.loads(raw)
    if (
        not z["decision"]["recommend_local_frozen_U"]
        or z["neural_updates"]
        or z["new_fits"]
    ):
        raise ValueError("no confirmed frozen U predictor")
    e = z["predictors"][str(seed)]
    return e["checkpoint"], tuple(checked_beta("U", e["beta"]["U"]).tolist())


@torch.inference_mode()
def _predict(host, head, beta, observed, width):
    h, z, _ = frozen_prefill(host, observed, width)
    return response(bound_fields(head, observed, h, z), "U", beta)


def load(seed):
    """Return predict(observed_bytes) -> [positions,256] log scores. Does NOT acquire a GPU.

    For the existing qualified lambda01 workspace/runtime. The host runs on its leased GPU,
    the head on CPU. Each call starts fresh with 1..4096 observed bytes from one document.
    This prefills a bounded window; it isn't an incremental decoder or latency benchmark.
    """
    from .query_panel import checkpoint, module
    from ..x47_xsa_attention.workbench_run import check_gpu_owner

    ck, beta = contract(seed)
    check_gpu_owner()
    _, endpoint = checkpoint(seed, hash_weights=False)
    if endpoint["sha256"] != ck:
        raise ValueError("U readout does not belong to this checkpoint")
    host, head = module(
        seed, endpoint
    )  # re-hash the file and strict-load both state dictionaries
    host = host.to("cuda")

    def predict(observed_bytes):
        return _predict(host, head, beta, observed_bytes, 256)

    return predict
