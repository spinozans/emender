"""Optional occurrence memory for the singleton phrase machine (no parser gradients)."""
import torch
from torch import nn


class PhraseQuery(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.byte = nn.Embedding(256, rank)
        self.key = nn.Linear(rank, rank, bias=False)
        self.value = nn.Linear(rank, width, bias=False)
        self.query = nn.Linear(width, rank, bias=False)
        nn.init.zeros_(self.query.weight)  # key must NOT also be zero
        self.cursor = nn.Linear(rank + 1, 1, bias=False)
        self.register_buffer('center', torch.zeros(width))
        self.register_buffer('scale', torch.ones(()))

    def encode(self, content):
        # Position-weighted observed content, not an ordinal type/occurrence ID.
        x = self.byte(torch.tensor(list(content), device=self.center.device))
        pos = torch.arange(1, len(x)+1, device=x.device)
        return (x * pos[:, None]).sum(0) / pos.sum()

    def relative(self, h, rows):
        q = self.query((h-self.center)/self.scale)
        return self.key(rows) @ q / q.numel()**.5

    def continue_logit(self, shared, row, offset):
        return shared + self.cursor(torch.cat((row, row.new_tensor([offset/4])))).squeeze(-1)
