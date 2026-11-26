"""Capture the *whole* mixed-precision update. Warm/capture work is invented and reset."""

import torch

from .natural_model import TokenLM


def invented(shape, batch, device="cuda", seed=7744):
    sh = shape
    g = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randint(
            sh.vocab,
            (batch, sh.time + 1),
            generator=g,
            device=device,
            dtype=torch.int32,
        ),
        torch.full((batch,), sh.time, device=device, dtype=torch.int32),
        torch.randint(
            sh.vocab,
            (batch, sh.hits, sh.page),
            generator=g,
            device=device,
            dtype=torch.int32,
        ),
        torch.full((batch, sh.hits), sh.page, device=device, dtype=torch.int32),
        torch.randint(
            1, 16, (batch, sh.hits), generator=g, device=device, dtype=torch.int32
        ),
    )


class Step:
    def __init__(self, shape, batch, host, arm, seed, *, graph=True):
        torch.manual_seed(seed)
        self.model = TokenLM(shape, host, arm).cuda()
        self.lr = torch.tensor(0.0005, device="cuda")
        self.graph = None
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            weight_decay=0.01,
            capturable=True,
            fused=True,
        )
        self.static = invented(shape, batch)
        self.stat = torch.zeros(3, device="cuda", dtype=torch.float64)
        if graph:
            origin = {k: p.detach().clone() for k, p in self.model.state_dict().items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._run()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.opt.zero_grad(set_to_none=True)
            graph_obj = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_obj):
                self._run()
            self.graph = graph_obj
            self.model.load_state_dict(origin)
            for state in self.opt.state.values():
                for val in state.values():
                    val.zero_()
            self.stat.zero_()
            torch.cuda.synchronize()

    def _run(self):
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = self.model.losses(*self.static)
        total = self.static[1].sum()
        nll = losses.sum()
        loss = nll / total.clamp_min(1)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0, foreach=True
        )
        self.opt.step()
        self.stat.add_(
            torch.stack((nll.detach(), total.float(), norm.detach())).double()
        )

    def train(self, packet, lr=None):
        for dst, src in zip(self.static, packet, strict=True):
            dst.copy_(src)
        if lr is not None:
            self.lr.fill_(lr)
        if self.graph is None:
            self._run()
        else:
            self.graph.replay()


@torch.no_grad()
def eval_losses(model, packet, intervention="intact"):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.losses(*packet, intervention=intervention)
