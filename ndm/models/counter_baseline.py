"""Additive / non-saturating counter-baseline layers (PROBE 1 positive control).

These are the arms that CAN count. Weiss, Goldberg & Yahav (2018) prove that
finite-precision LSTMs and ReLU-Elman RNNs implement *unbounded counters*
(recognizing a^n b^n, a^n b^n c^n) via an additive, non-saturating cell,
whereas squashing (tanh) RNNs and GRUs cannot. The tanh arm (E88-tanh) is a
known false negative for counting and must NOT be the only nonlinear baseline;
these layers supply the real positive control.

Both wrap a canonical PyTorch recurrent cell (REAL layers, not stubs):

  ReLURNNLayer  ->  torch.nn.RNN(nonlinearity='relu')   (additive ReLU-Elman)
  LSTMLayer     ->  torch.nn.LSTM                        (gated additive cell)

with the standard in_proj / out_proj wrapping used by the other ladder layers,
exposing the HybridLadderLM layer contract:
    forward(x[B,T,dim]) -> (out[B,T,dim], h_final)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RecurrentBaseline(nn.Module):
    """Shared in_proj/RNN/out_proj scaffold for the counter baselines."""

    def __init__(self, dim, expansion=1.0, dropout=0.0, num_layers=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.num_layers = num_layers
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)
        self.rnn = self._make_rnn()
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _make_rnn(self) -> nn.Module:
        raise NotImplementedError

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        B, T, D = x.shape
        x_proj = self.in_proj(x)                 # [B, T, d_inner]
        out, h_n = self.rnn(x_proj)              # cuDNN scan, batch_first
        out = self.out_proj(self.dropout(out))   # [B, T, dim]
        # h_n is [num_layers, B, d_inner] (RNN) or a tuple (LSTM); take last layer.
        h_last = (h_n[0] if isinstance(h_n, tuple) else h_n)[-1]
        return out, h_last


class ReLURNNLayer(_RecurrentBaseline):
    """Additive ReLU-Elman RNN: h_t = relu(W_x x_t + W_h h_{t-1} + b).

    Non-saturating cell -> can realize an unbounded counter (WGY 2018)."""

    def _make_rnn(self):
        return nn.RNN(
            input_size=self.d_inner,
            hidden_size=self.d_inner,
            num_layers=self.num_layers,
            nonlinearity='relu',
            batch_first=True,
            bias=True,
        )

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, cell=ReLU-Elman'


class LSTMLayer(_RecurrentBaseline):
    """Standard LSTM. The additive cell state c_t = f_t*c_{t-1} + i_t*g_t with
    a non-saturating accumulation path implements unbounded counters (WGY 2018,
    Délétang et al. 2022)."""

    def _make_rnn(self):
        return nn.LSTM(
            input_size=self.d_inner,
            hidden_size=self.d_inner,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
        )

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, cell=LSTM'


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for cls in (ReLURNNLayer, LSTMLayer):
        m = cls(dim=128, expansion=2.0).to(device)
        x = torch.randn(2, 32, 128, device=device, requires_grad=True)
        out, h = m(x)
        loss = out.sum()
        loss.backward()
        n = sum(p.numel() for p in m.parameters())
        print(f"{cls.__name__}: out={tuple(out.shape)} h={tuple(h.shape)} "
              f"params={n:,} backward OK")
