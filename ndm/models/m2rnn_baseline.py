"""
M2RNN baseline.

This implements the matrix-to-matrix recurrent update used by
Mishra/Tan/Stoica/Gonzalez/Dao:

    z_t = tanh(h_{t-1} W + k_t v_t^T)
    h_t = f_t h_{t-1} + (1 - f_t) z_t
    y_t = q_t^T h_t + D * v_t

It mirrors the open-lm-engine M2RNN block and prefers the upstream XMA Triton
kernel when available. A simple PyTorch loop remains as a correctness fallback.
"""

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

_xma_path = os.environ.get("XMA_PATH")
if _xma_path and _xma_path not in sys.path:
    sys.path.insert(0, _xma_path)

try:
    from xma import KernelBackend
    from xma.layers.m2rnn import m2rnn as xma_m2rnn
    XMA_M2RNN_AVAILABLE = True
except Exception:
    KernelBackend = None
    xma_m2rnn = None
    XMA_M2RNN_AVAILABLE = False


class M2RNNLayer(nn.Module):
    """Matrix-to-matrix nonlinear recurrent layer."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 128,
        n_state: int = 16,
        expansion: float = 1.0,
        paper_shape: bool = False,
        k_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        num_q_heads: Optional[int] = None,
        num_k_heads: Optional[int] = None,
        num_v_heads: Optional[int] = None,
        num_f_heads: Optional[int] = None,
        num_g_heads: Optional[int] = None,
        num_weight_heads: Optional[int] = None,
        use_gate: bool = True,
        use_residual: bool = True,
        state_weight_trainable: bool = True,
        use_conv: bool = False,
        d_conv: int = 4,
        output_norm: bool = False,
        normalize_qk: bool = False,
        dropout: float = 0.0,
        gradient_clipping: Optional[float] = None,
        linear_state: bool = False,
        **kwargs,
    ):
        super().__init__()
        if n_heads is None:
            raise ValueError("M2RNN requires --n_heads")
        if n_state <= 0:
            raise ValueError("n_state must be positive")

        self.dim = dim
        if paper_shape:
            # Match the released M2RNN-family configs: one q/k head, many
            # value/forget/gate/weight heads, K=64 and V=16 by default.
            k_head_dim = 64 if k_head_dim is None else k_head_dim
            v_head_dim = n_state if v_head_dim is None else v_head_dim
            num_q_heads = 1 if num_q_heads is None else num_q_heads
            num_k_heads = 1 if num_k_heads is None else num_k_heads
            num_v_heads = n_heads if num_v_heads is None else num_v_heads
            num_f_heads = n_heads if num_f_heads is None else num_f_heads
            num_g_heads = n_heads if num_g_heads is None else num_g_heads
            num_weight_heads = n_heads if num_weight_heads is None else num_weight_heads
        else:
            k_head_dim = n_state if k_head_dim is None else k_head_dim
            v_head_dim = max(1, int(round(n_state * expansion))) if v_head_dim is None else v_head_dim
            num_q_heads = n_heads if num_q_heads is None else num_q_heads
            num_k_heads = n_heads if num_k_heads is None else num_k_heads
            num_v_heads = n_heads if num_v_heads is None else num_v_heads
            num_f_heads = n_heads if num_f_heads is None else num_f_heads
            num_g_heads = n_heads if num_g_heads is None else num_g_heads
            num_weight_heads = n_heads if num_weight_heads is None else num_weight_heads

        head_counts = (num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads)
        if use_gate:
            head_counts = head_counts + (num_g_heads,)
        if any(h is None or h <= 0 for h in head_counts):
            raise ValueError("M2RNN head counts must be positive")

        self.n_heads = n_heads
        self.num_q_heads = int(num_q_heads)
        self.num_k_heads = int(num_k_heads)
        self.num_v_heads = int(num_v_heads)
        self.num_f_heads = int(num_f_heads)
        self.num_g_heads = int(num_g_heads)
        self.num_weight_heads = int(num_weight_heads)
        self.num_heads = max(int(h) for h in head_counts)
        for name, count in (
            ("num_q_heads", self.num_q_heads),
            ("num_k_heads", self.num_k_heads),
            ("num_v_heads", self.num_v_heads),
            ("num_f_heads", self.num_f_heads),
            ("num_g_heads", self.num_g_heads if use_gate else 1),
            ("num_weight_heads", self.num_weight_heads),
        ):
            if self.num_heads % count != 0:
                raise ValueError(f"num_heads={self.num_heads} must be divisible by {name}={count}")

        self.paper_shape = paper_shape
        self.k_head_dim = int(k_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.use_gate = use_gate
        self.use_residual = use_residual
        self.state_weight_trainable = state_weight_trainable
        self.use_conv = use_conv and d_conv > 1
        self.d_conv = d_conv
        self.output_norm_enabled = output_norm
        self.normalize_qk = normalize_qk
        self.gradient_clipping = gradient_clipping
        # When True, the state update drops the tanh state-nonlinearity:
        #   nonlinear (as built): Z = tanh(h W + k v^T)
        #   linear  (this knob): Z =      h W + k v^T
        # Exact raw-write analogue of E88's `linear_state` (e88_fla_hybrid.py:1709).
        self.linear_state = linear_state

        q_shape = self.num_q_heads * self.k_head_dim
        k_shape = self.num_k_heads * self.k_head_dim
        v_shape = self.num_v_heads * self.v_head_dim
        g_shape = self.num_g_heads * self.v_head_dim
        self.conv_dim = q_shape + k_shape + v_shape
        self.g_shape = g_shape

        gate_shape = g_shape if use_gate else 0
        self.input_projection = nn.Linear(
            dim,
            self.conv_dim + self.num_f_heads + gate_shape,
            bias=False,
        )

        if self.use_conv:
            self.conv1d = nn.Conv1d(
                self.conv_dim,
                self.conv_dim,
                d_conv,
                padding=d_conv - 1,
                groups=self.conv_dim,
                bias=False,
            )
        else:
            self.conv1d = None

        self.A_log = nn.Parameter(torch.empty(self.num_f_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_f_heads, dtype=torch.float32))

        state_weight = torch.eye(self.v_head_dim).unsqueeze(0).repeat(self.num_weight_heads, 1, 1)
        self.state_weight = nn.Parameter(state_weight)
        if not state_weight_trainable:
            self.state_weight.requires_grad_(False)

        if use_residual:
            self.D = nn.Parameter(torch.ones(self.num_heads, self.v_head_dim))
        else:
            self.D = None

        output_shape = self.num_heads * self.v_head_dim
        self.output_norm = nn.RMSNorm(output_shape) if output_norm else nn.Identity()
        self.output_projection = nn.Linear(output_shape, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.conv1d is not None:
            nn.init.normal_(self.conv1d.weight, std=0.02)

        A = torch.empty(self.num_f_heads, dtype=torch.float32).uniform_(1e-4, 16)
        self.A_log.data.copy_(torch.log(A))

        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.num_f_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)

        with torch.no_grad():
            eye = torch.eye(self.v_head_dim, device=self.state_weight.device, dtype=self.state_weight.dtype)
            self.state_weight.copy_(eye.unsqueeze(0).repeat(self.num_weight_heads, 1, 1))
            if self.D is not None:
                self.D.fill_(1)

    @staticmethod
    def _repeat_heads(x: torch.Tensor, target_heads: int, head_dim: int = -2) -> torch.Tensor:
        heads = x.size(head_dim)
        if heads == target_heads:
            return x
        if target_heads % heads != 0:
            raise ValueError(f"Cannot repeat {heads} heads to {target_heads}")
        return x.repeat_interleave(target_heads // heads, dim=head_dim)

    def _project(self, x: torch.Tensor):
        B, T, _ = x.shape
        Nq = self.num_q_heads
        Nk = self.num_k_heads
        Nv = self.num_v_heads
        Nf = self.num_f_heads
        Ng = self.num_g_heads
        K = self.k_head_dim
        V = self.v_head_dim

        proj = self.input_projection(x)
        qkv, f_raw, g = proj.split(
            [self.conv_dim, Nf, self.g_shape if self.use_gate else 0],
            dim=-1,
        )

        if self.conv1d is not None:
            qkv = self.conv1d(qkv.transpose(1, 2))[:, :, :T].transpose(1, 2)
            qkv = F.silu(qkv.float()).to(dtype=x.dtype)

        q_raw, k_raw, v_raw = qkv.split([Nq * K, Nk * K, Nv * V], dim=-1)
        q = q_raw.view(B, T, Nq, K)
        k = k_raw.view(B, T, Nk, K)
        v = v_raw.view(B, T, Nv, V)
        if self.normalize_qk:
            q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype)
            k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype)

        f = F.softplus(f_raw.float() + self.dt_bias.float().view(1, 1, Nf))
        f = torch.exp(-torch.exp(self.A_log.float()).view(1, 1, Nf) * f)
        f = f.to(dtype=x.dtype)

        if self.use_gate:
            g = g.view(B, T, Ng, V)
        else:
            g = None
        return q, k, v, f, g

    def forward(self, x: torch.Tensor, prev_hidden: Optional[torch.Tensor] = None, **kwargs):
        B, T, _ = x.shape
        N = self.num_heads
        K = self.k_head_dim
        V = self.v_head_dim

        q, k, v, forget, gate = self._project(x)

        # The XMA Triton kernel hardcodes the tanh state-nonlinearity, so it
        # cannot express the linear-state ablation. Fall back to the PyTorch
        # loop (which honors self.linear_state) whenever linear state is asked
        # for. With nonlinear state we use the kernel when available.
        if XMA_M2RNN_AVAILABLE and x.is_cuda and not self.linear_state:
            h0 = None if prev_hidden is None else prev_hidden.to(dtype=q.dtype).contiguous()
            y, h = xma_m2rnn(
                query=q.contiguous(),
                key=k.contiguous(),
                value=v.contiguous(),
                weight=self.state_weight.to(dtype=q.dtype).contiguous(),
                forget_input=forget.contiguous(),
                input_state=h0,
                gradient_clipping=self.gradient_clipping,
                kernel_backend=KernelBackend.triton,
            )

            if self.D is not None:
                v_res = self._repeat_heads(v, N)
                y = y + v_res * self.D.to(dtype=y.dtype).view(1, 1, N, V)
            if self.use_gate and gate is not None:
                gate = self._repeat_heads(gate, N)
                y = y * F.silu(gate.float()).to(dtype=y.dtype)

            y = y.reshape(B, T, N * V)
            y = self.output_norm(y)
            y = self.output_projection(y)
            y = self.dropout(y)
            return y, h

        if prev_hidden is None:
            h = torch.zeros(B, N, K, V, device=x.device, dtype=torch.float32)
        else:
            h = prev_hidden.float()

        q = self._repeat_heads(q, N).float()
        k = self._repeat_heads(k, N).float()
        v = self._repeat_heads(v, N).float()
        forget = self._repeat_heads(forget, N, head_dim=-1).float()
        W = self._repeat_heads(self.state_weight, N, head_dim=0).float().unsqueeze(0)
        D = self.D.float().unsqueeze(0) if self.D is not None else None
        if self.use_gate and gate is not None:
            gate = self._repeat_heads(gate, N).float()
        outputs = []

        for t in range(T):
            q_t = q[:, t].float()
            k_t = k[:, t].float()
            v_t = v[:, t].float()
            f_t = forget[:, t].float().view(B, N, 1, 1)

            outer = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            pre = torch.matmul(h, W) + outer
            candidate = pre if self.linear_state else torch.tanh(pre)
            h = f_t * h + (1.0 - f_t) * candidate

            if self.gradient_clipping is not None:
                h = h.clamp(-self.gradient_clipping, self.gradient_clipping)

            y_t = torch.matmul(q_t.unsqueeze(-2), h).squeeze(-2)
            if D is not None:
                y_t = y_t + v_t * D
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        if self.use_gate and gate is not None:
            y = y * F.silu(gate.float())

        y = y.reshape(B, T, N * V).to(dtype=self.output_projection.weight.dtype)
        y = self.output_norm(y)
        y = self.output_projection(y)
        y = self.dropout(y)
        return y, h


class M2RNNLM(nn.Module):
    """Language-model wrapper with the same interface as LadderLM."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 1024,
        depth: int = 20,
        n_heads: int = 128,
        n_state: int = 16,
        expansion: float = 1.0,
        paper_shape: bool = False,
        k_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        num_q_heads: Optional[int] = None,
        num_k_heads: Optional[int] = None,
        num_v_heads: Optional[int] = None,
        num_f_heads: Optional[int] = None,
        num_g_heads: Optional[int] = None,
        num_weight_heads: Optional[int] = None,
        use_gate: bool = True,
        use_residual: bool = True,
        state_weight_trainable: bool = True,
        use_conv: bool = False,
        d_conv: int = 4,
        output_norm: bool = False,
        normalize_qk: bool = False,
        dropout: float = 0.0,
        gradient_clipping: Optional[float] = None,
        linear_state: bool = False,
        gradient_checkpointing: bool = False,
        loss_chunk_size: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads
        self.n_state = n_state
        self.expansion = expansion
        self.paper_shape = paper_shape
        self.normalize_qk = normalize_qk
        self.use_residual = use_residual
        self.state_weight_trainable = state_weight_trainable
        self.gradient_checkpointing = gradient_checkpointing
        self.loss_chunk_size = loss_chunk_size

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layer_norms = nn.ModuleList([nn.RMSNorm(dim) for _ in range(depth)])
        self.layers = nn.ModuleList([
            M2RNNLayer(
                dim=dim,
                n_heads=n_heads,
                n_state=n_state,
                expansion=expansion,
                paper_shape=paper_shape,
                k_head_dim=k_head_dim,
                v_head_dim=v_head_dim,
                num_q_heads=num_q_heads,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                num_f_heads=num_f_heads,
                num_g_heads=num_g_heads,
                num_weight_heads=num_weight_heads,
                use_gate=use_gate,
                use_residual=use_residual,
                state_weight_trainable=state_weight_trainable,
                use_conv=use_conv,
                d_conv=d_conv,
                output_norm=output_norm,
                normalize_qk=normalize_qk,
                dropout=dropout,
                gradient_clipping=gradient_clipping,
                linear_state=linear_state,
            )
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
        return_prev_hiddens: bool = False,
        prev_hiddens=None,
        actual_length=None,
        **kwargs,
    ):
        if return_loss:
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        hiddens = [None] * self.depth if prev_hiddens is None else prev_hiddens
        x = self.embedding(inp)
        new_hiddens = []

        for ln, layer, hidden in zip(self.layer_norms, self.layers, hiddens):
            residual = x
            x = ln(x)
            if self.gradient_checkpointing and self.training:
                if hidden is None:
                    x, h_final = torch_checkpoint(lambda y: layer(y, None), x, use_reentrant=False)
                else:
                    x, h_final = torch_checkpoint(layer, x, hidden, use_reentrant=False)
            else:
                x, h_final = layer(x, hidden)
            x = residual + x
            new_hiddens.append(h_final)

        x = self.norm(x)

        if return_loss:
            if actual_length is not None:
                device = x.device
                positions = torch.arange(target.size(1), device=device).unsqueeze(0)
                valid_mask = positions < (actual_length.unsqueeze(1) - 1)
                target = target.clone()
                target[~valid_mask] = -100

            loss_chunk = self.loss_chunk_size
            if loss_chunk > 0 and x.size(1) > loss_chunk:
                total_sum = x.new_zeros(())
                total_count = 0
                for t0 in range(0, x.size(1), loss_chunk):
                    t1 = min(t0 + loss_chunk, x.size(1))
                    logits_c = self.lm_head(x[:, t0:t1])
                    target_c = target[:, t0:t1]
                    total_sum = total_sum + F.cross_entropy(
                        logits_c.reshape(-1, self.vocab_size),
                        target_c.reshape(-1),
                        ignore_index=-100,
                        reduction='sum',
                    )
                    total_count = total_count + (target_c != -100).sum()
                loss = total_sum / total_count.clamp(min=1)
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    target.reshape(-1),
                    ignore_index=-100,
                )
            if return_prev_hiddens:
                return loss, (new_hiddens, None)
            return loss

        logits = self.lm_head(x)
        if return_prev_hiddens:
            return logits, (new_hiddens, None)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return (
            f"M2RNN baseline, dim={self.dim}, depth={self.depth}, "
            f"heads={self.n_heads}, state={self.n_state}, expansion={self.expansion}, "
            f"paper_shape={self.paper_shape}"
        )


def create_m2rnn_model(target_params: str = "100m", vocab_size: int = 256):
    from calc_dim import calc_m2rnn_params, find_dim_for_params

    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    depth = 20
    n_heads = 128
    n_state = 16
    dim, _ = find_dim_for_params(
        calc_m2rnn_params,
        target_count,
        depth=depth,
        n_heads=n_heads,
        n_state=n_state,
        vocab_size=vocab_size,
    )
    model = M2RNNLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        n_heads=n_heads,
        n_state=n_state,
    )
    print(
        f"Created M2RNN model: dim={dim}, depth={depth}, "
        f"heads={n_heads}, state={n_state}, params={model.get_num_params():,}"
    )
    return model
