"""Entirely trainable, invented-token fixture. See JOINT_FIXTURE.md for scope."""

import json
import math
import os
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TITLES, FIELDS, VALUES, HITS = 8, 2, 8, 4
VOCAB = TITLES + FIELDS + VALUES
WIDTH, HEAD, STEPS, BATCH = 24, 8, 320, 64


@dataclass
class Episode:
    prompt: Tensor
    pages: Tensor
    row: Tensor  # for scoring only; never passed to the model

    def answer(self, pages: Tensor) -> Tensor:
        field = self.prompt[:, 1] - TITLES
        return pages[torch.arange(len(field)), self.row, 2 + 2 * field]


def episodes(gen: torch.Generator, n: int) -> Episode:
    titles = torch.rand(n, TITLES, generator=gen).argsort(-1)[:, :HITS]
    settings = torch.rand(n, FIELDS, VALUES, generator=gen).argsort(-1)[:, :, :HITS]
    settings = settings + TITLES + FIELDS
    day = torch.full_like(titles, TITLES)
    pages = torch.stack((titles, day, settings[:, 0], day + 1, settings[:, 1]), -1)
    row = torch.randint(HITS, (n,), generator=gen)
    field = torch.randint(FIELDS, (n,), generator=gen)
    prompt = torch.stack((titles[torch.arange(n), row], TITLES + field), -1)
    return Episode(prompt, pages, row)


def change_manual(pages: Tensor) -> Tensor:
    other = pages.clone()
    other[..., 2::2] = pages[..., 2::2].roll(1, dims=1)
    return other


def read(query: Tensor, state: Tensor) -> Tensor:
    return (query.unsqueeze(-1) * state).sum(dim=-2)


class Delta(nn.Module):
    """Passage-only or prefix-only controls; explicit differentiable recurrence."""

    def __init__(self):
        super().__init__()
        self.controls = nn.Linear(WIDTH, 5 * HEAD)

    def forward(self, tokens: Tensor, seed: Tensor) -> Tensor:
        state = seed
        for step in self.controls(tokens).unbind(-2):
            key, decay, erase, gate, value = step.chunk(5, dim=-1)
            key = F.normalize(key, dim=-1)
            state = (decay + 2).sigmoid().unsqueeze(-1) * state
            error = gate.sigmoid() * value - read(key * erase.sigmoid(), state)
            state = state + key.unsqueeze(-1) * error.unsqueeze(-2)
        return state


class JointLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, WIDTH)
        self.host = Delta()
        self.host_query = nn.Linear(WIDTH, HEAD)
        self.host_out = nn.Linear(HEAD, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.address = nn.Linear(WIDTH, HEAD, bias=False)
        self.title_key = nn.Linear(WIDTH, HEAD, bias=False)
        self.left_context = nn.Linear(WIDTH, WIDTH, bias=False)
        self.passage = Delta()
        self.read_query = nn.Linear(WIDTH, HEAD)
        self.hint_out = nn.Linear(HEAD, WIDTH, bias=False)
        self.gate = nn.Linear(WIDTH, 1)
        self.decoder = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, VOCAB))

    def forward(
        self, prompt: Tensor, pages: Tensor, mode: str
    ) -> tuple[Tensor, Tensor]:
        # Only actual prompt tokens advance the host. No detach on the live seed.
        x = self.embedding(prompt)
        live = self.host(x, x.new_zeros((len(x), HEAD, HEAD)))
        last = x[:, -1]
        h = self.norm(last + self.host_out(read(self.host_query(last), live)))
        y = self.embedding(pages)
        score = (self.address(h)[:, None] * self.title_key(y[:, :, 0])).sum(-1)
        p = (score / math.sqrt(HEAD)).softmax(dim=1)
        if mode == "none":
            return self.decoder(h), p
        # A 1-token left context stops at the start of each document.
        y = y + self.left_context(F.pad(y[..., :-1, :], (0, 0, 1, 0)))
        seed = live[:, None]
        if mode == "state":
            m = self.passage(y, torch.zeros_like(seed))
        elif mode in ("effect", "wrong_a"):
            m = self.passage(y, seed)
            if mode == "wrong_a":
                b = self.passage(y, torch.zeros_like(seed))
                m = b + (m - b).roll(1, dims=1)
        else:
            raise ValueError(mode)
        pooled = (p[:, :, None, None] * m).sum(1)
        r = read(self.read_query(h), pooled)
        return self.decoder(h + self.gate(h).sigmoid() * self.hint_out(r)), p


def module_norms(model: nn.Module, tensors: dict[str, Tensor]) -> dict[str, float]:
    return {
        name: sum(
            tensors[f"{name}.{k}"].double().square().sum().item()
            for k, _ in module.named_parameters()
        )
        ** 0.5
        for name, module in model.named_children()
    }


def fit(mode: str):
    torch.manual_seed(5401)
    model = JointLM()
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    g = torch.Generator().manual_seed(5402)
    witness = {}
    start = time.perf_counter()
    for step in range(STEPS):
        batch = episodes(g, BATCH)
        logits, _ = model(batch.prompt, batch.pages, mode)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(logits, batch.answer(batch.pages)).backward()
        if step == 0:
            old = {k: v.detach().clone() for k, v in model.named_parameters()}
            witness["gradient_l2"] = module_norms(
                model, {k: v.grad for k, v in model.named_parameters()}
            )
        nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        if step == 0:
            witness["update_l2"] = module_norms(
                model, {k: v.detach() - old[k] for k, v in model.named_parameters()}
            )
            if not all(
                math.isfinite(v) and v > 0
                for group in witness.values()
                for v in group.values()
            ):
                raise ValueError(f"broken joint gradient/update path: {witness}")
    return model.eval().requires_grad_(False), {
        "first_step": witness,
        "fit_wall_s": time.perf_counter() - start,
    }


@torch.no_grad()
def score(models):
    g = torch.Generator().manual_seed(5403)
    tally = {}

    def add(key, logits, target):
        count, loss, correct = tally.get(key, (0, 0.0, 0))
        tally[key] = (
            count + len(target),
            loss + F.cross_entropy(logits.double(), target, reduction="sum").item(),
            correct + (logits.argmax(-1) == target).sum().item(),
        )

    for _ in range(64):
        batch = episodes(g, BATCH)
        other = change_manual(batch.pages)
        target, new_target = batch.answer(batch.pages), batch.answer(other)
        for mode, model in models.items():
            true, p = model(batch.prompt, batch.pages, mode)
            changed, p2 = model(batch.prompt, other, mode)
            if not torch.equal(p, p2) or (new_target == target).any():
                raise ValueError(
                    "manual change must fix search weights and change every answer"
                )
            add((mode, "intact"), true, target)
            add((mode, "changed_manual_old_target"), changed, target)
            add((mode, "changed_manual_new_target"), changed, new_target)
            add((mode, "no_hint"), model(batch.prompt, batch.pages, "none")[0], target)
            if mode == "effect":
                add(
                    (mode, "wrong_a"),
                    model(batch.prompt, batch.pages, "wrong_a")[0],
                    target,
                )
    return {
        mode: {
            case: {"tokens": n, "correct": c, "nats_per_token": loss / n}
            for (arm, case), (n, loss, c) in tally.items()
            if arm == mode
        }
        for mode in models
    }


def main():
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or "GPU_LEASE_HELD" in os.environ
        or not __debug__
    ):
        raise ValueError(
            "CPU fixture requires CUDA_VISIBLE_DEVICES='', no GPU lease and checks enabled"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models, training = {}, {}
    for mode in ("state", "effect"):
        models[mode], training[mode] = fit(mode)
    print(
        json.dumps(
            {
                "settings": {
                    "torch": str(torch.__version__),
                    "rng_init_fit_check": [5401, 5402, 5403],
                    "steps_per_arm": STEPS,
                    "fit_tokens_per_arm": BATCH * STEPS,
                    "parameters_per_arm": sum(
                        p.numel() for p in models["effect"].parameters()
                    ),
                    "device": "cpu",
                },
                "training": training,
                "score": score(models),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
