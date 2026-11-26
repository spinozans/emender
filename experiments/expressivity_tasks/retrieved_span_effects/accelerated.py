"""Drop-in trained-fixture backend; same parameters, candidate scope and objective."""

import torch
from torch.nn import functional as F

from .joint_fixture import HEAD, JointLM, read
from .triton_controls import pack
from .triton_read import dual_edges, merge, terminal, weights
from .triton_scan import pool


class AcceleratedJointLM(JointLM):
    def __init__(self, *, host="gdn2", reader="gdn2", backend="dual", rows=4):
        super().__init__()
        if (
            host not in ("gdn2", "e97")
            or reader not in ("gdn2", "e97")
            or backend not in ("matrix", "dual")
            or (backend == "dual" and reader != "gdn2")
        ):
            raise ValueError("dual read requires affine GDN2 reader")
        self.host_family = host
        self.reader_family = reader
        self.backend = backend
        self.rows = rows

    def forward(self, prompt, pages, mode):
        x = self.embedding(prompt)
        b, t, _ = x.shape
        c = pack(self.host.controls(x), HEAD, HEAD, family=self.host_family)
        live = terminal(
            c[:, None],
            x.new_zeros((b, HEAD, HEAD)),
            torch.full((b, 1), t, device=x.device, dtype=torch.int32),
            family=self.host_family,
            seeded=False,
        )[:, 0]
        last = x[:, -1]
        h = self.norm(
            last + self.host_out(read(self.host_query(last), live).to(last.dtype))
        )
        y = self.embedding(pages)
        _, p, t, _ = y.shape
        lengths = torch.full((b, p), t, device=y.device, dtype=torch.int32)
        prob = weights(
            self.address(h), self.title_key(y[:, :, 0]), y.new_zeros((b, p)), lengths, t
        )
        if mode == "none":
            return self.decoder(h), prob
        y = y + self.left_context(F.pad(y[..., :-1, :], (0, 0, 1, 0)))
        c = pack(self.passage.controls(y), HEAD, HEAD, family=self.reader_family)
        q = self.read_query(h)
        if mode not in ("state", "neutral", "effect", "wrong_a"):
            raise ValueError(mode)
        if self.backend == "dual":
            edge = dual_edges(c, q, lengths, self.rows)
            if mode == "wrong_a":
                edge = torch.cat((edge[..., :HEAD].roll(1, 1), edge[..., HEAD:]), -1)
            hint = merge(edge, prob, live, seeded=mode in ("effect", "wrong_a"))
        else:
            m = terminal(
                c,
                live,
                lengths,
                family=self.reader_family,
                seeded=mode in ("effect", "wrong_a"),
            )
            if mode == "wrong_a":
                blank = terminal(
                    c, live, lengths, family=self.reader_family, seeded=False
                )
                m = blank + (m - blank).roll(1, 1)
            hint = read(q, pool(m, prob))
        if mode == "neutral":
            hint = hint + read(q, live)
        return self.decoder(
            h + self.gate(h).sigmoid() * self.hint_out(hint.to(h.dtype))
        ), prob
