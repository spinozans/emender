"""Incremental decoding parity on invented byte streams; no historical data loading."""

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks.x50_read_tape.kernel import Kernel
from experiments.expressivity_tasks.x50_read_tape.program import (
    ARMS,
    T,
    Tape,
    execute,
    predict,
)
from experiments.expressivity_tasks.x50_read_tape.online.stream import ReadSession
from experiments.expressivity_tasks.x50_read_tape.test_read_tape import Fixture


@pytest.mark.parametrize("arm", ARMS)
def test_real_byte_api_matches_full_causal_tape_and_reset(arm):
    torch.set_num_threads(1)
    torch.manual_seed(274)
    m = Tape()
    with torch.no_grad():
        m.act.weight.normal_(std=0.15)
        m.emit.weight.normal_(std=0.15)
    raw = (b"AB\xffC\0\x1eAY" * 8)[:T]
    x = np.frombuffer(raw, np.uint8)[None, :]
    tr = execute(m, Kernel(Fixture()), x, arm)
    expected, _ = predict(m, tr)
    bank = Kernel(Fixture())
    session = ReadSession(m, bank, arm)
    for i, b in enumerate(raw):
        out = session.observe(b)
        torch.testing.assert_close(out.logits, expected[0, i], atol=4e-6, rtol=2e-5)
        assert (
            out.extension == (None if arm == "ONE" else tr.action[0, i].item())
            and session.observed == raw[: i + 1]
        )
        assert session.rows == (i + 1) * (2 if arm == "ONE" else 3)
    with pytest.raises(OverflowError):
        session.observe(7)
    assert bank.counts["parent_reads"] == T and bank.counts["child_reads"] == (
        0 if arm == "ONE" else T
    )
    session.reset()
    assert session.rows == 0 and not session.observed
    torch.testing.assert_close(
        session.observe(raw[0]).logits, expected[0, 0], rtol=2e-5, atol=4e-6
    )


def test_failed_follow_does_not_append_a_byte_or_a_work_row():
    m = Tape()
    k = Kernel(Fixture())
    p = ReadSession(m, k, "ADAPT")
    with pytest.raises(ValueError):
        p.observe(256)
    assert k.counts["parent_reads"] == 0

    def broken(*args):
        raise OSError("lost reply")

    k.follow = broken
    with pytest.raises(OSError):
        p.observe(65)
    assert p.rows == 0 and p.observed == b""
