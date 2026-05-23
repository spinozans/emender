"""
E71 Delta Rule: Matrix Gated with Delta Rule - State-dependent learning rate

Architecture (delta rule version):
    k_norm = k / (||k|| + eps)                  # Normalize key
    retrieved = S @ k_norm                       # Query what's stored at key
    beta = sigmoid(W_beta @ x + d_beta * retrieved + b_beta)  # State-dependent learning rate
    delta = v - retrieved                        # Error signal
    S_t = S_{t-1} + beta * outer(delta, k_norm)  # Delta rule update
    out_t = (S @ q) * silu(S @ q)

Why delta rule?
- Global decay wastes ~85% of matrix capacity
- Delta rule provides selective update: only overwrites at queried key direction
- Preserves orthogonal information indefinitely
- Full rank utilization possible
- Exact retrieval for orthogonal keys

beta (learning rate) interpretation:
- High beta = "I need to learn this" (large update)
- Low beta = "I already know this" (small/no update)
- State-dependent: the model learns when to update based on current memory

Stability notes:
- Key normalization bounds update magnitude: ||S_new - S|| <= ||delta||
- Spectral norm on W_k, W_v, W_q bounds ||k||, ||v||, ||q||
- beta in (0, 1) via sigmoid provides additional stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class E71DeltaCell(nn.Module):
    """
    E71 Delta Rule cell - state-dependent learning rate with delta rule update.

    k_norm = k / (||k|| + eps)
    retrieved = S @ k_norm
    beta = sigmoid(W_beta @ x + d_beta * retrieved + b_beta)
    S = S + beta * outer(v - retrieved, k_norm)  # Delta rule
    out = (S @ q) * silu(S @ q)
    """

    def __init__(self, dim, n_state=64, init_beta_bias=0.0, init_d_beta=0.1):
        super().__init__()
        self.dim = dim
        self.n_state = n_state

        # Projections with spectral norm for stability
        self.W_k = nn.utils.spectral_norm(nn.Linear(dim, n_state, bias=False))
        self.W_v = nn.utils.spectral_norm(nn.Linear(dim, n_state, bias=False))
        self.W_q = nn.utils.spectral_norm(nn.Linear(dim, n_state, bias=False))

        # Beta gate with S-dependence (state-dependent learning rate)
        self.W_beta = nn.Linear(dim, n_state, bias=False)
        self.d_beta = nn.Parameter(torch.full((n_state,), init_d_beta))
        self.b_beta = nn.Parameter(torch.full((n_state,), init_beta_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_beta.weight)

    def forward(self, x, S=None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial matrix state

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final matrix state
        """
        T, B, D = x.shape

        if S is None:
            S = torch.zeros(B, self.n_state, self.n_state, device=x.device, dtype=x.dtype)

        # Batch projections
        x_flat = x.reshape(T * B, D)
        k_all = self.W_k(x_flat).reshape(T, B, self.n_state)
        v_all = self.W_v(x_flat).reshape(T, B, self.n_state)
        q_all = self.W_q(x_flat).reshape(T, B, self.n_state)
        beta_x_all = self.W_beta(x_flat).reshape(T, B, self.n_state)

        outputs = []
        for t in range(T):
            k = k_all[t]
            v = v_all[t]
            q = q_all[t]
            beta_x = beta_x_all[t]

            # Key normalization for stability
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieve current content at normalized key direction
            retrieved = torch.einsum('bij,bj->bi', S, k_norm)  # [B, n_state]

            # State-dependent learning rate (beta)
            beta = torch.sigmoid(beta_x + self.d_beta * retrieved + self.b_beta)

            # Delta rule update: S = S + beta * outer(v - retrieved, k_norm)
            delta = v - retrieved  # Error signal
            outer_delta_k = torch.einsum('bi,bj->bij', delta, k_norm)
            S = S + beta.unsqueeze(-1) * outer_delta_k

            # Self-gating output
            out = torch.einsum('bij,bj->bi', S, q)
            out = out * F.silu(out)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E71Delta(nn.Module):
    """
    E71 Delta Rule: Matrix Gated Elman with Delta Rule.

    Uses state-dependent learning rate (beta) instead of global decay.
    Beta determines how much to update the memory based on current state.

    Architecture:
    - k_norm = k / (||k|| + eps)
    - retrieved = S @ k_norm
    - beta = sigmoid(W_beta @ x + d_beta * retrieved + b_beta)
    - S = S + beta * outer(v - retrieved, k_norm)
    - out = (S @ q) * silu(S @ q)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_state=64,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.use_conv = use_conv

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        self.cell = E71DeltaCell(self.d_inner, n_state=n_state)

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, S=None, **kwargs):
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        cell_out, S_final = self.cell(x_rnn, S)

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, S_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, LEVEL=71_DELTA'


if __name__ == "__main__":
    print("Testing E71 Delta Rule...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test E71 Delta
    print("\n--- Testing E71 Delta ---")
    model = E71Delta(dim=256, expansion=1.0, n_state=32).to(device).bfloat16()

    x = torch.randn(4, 32, 256, device=device, dtype=torch.bfloat16)
    S0 = torch.zeros(4, 32, 32, device=device, dtype=torch.bfloat16)

    out, S = model(x)
    print(f"Output: {out.shape}, State: {S.shape}")

    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test cell directly
    print("\n--- Testing E71DeltaCell directly ---")
    cell = E71DeltaCell(dim=256, n_state=32).to(device).bfloat16()

    x_rnn = torch.randn(32, 4, 256, device=device, dtype=torch.bfloat16)
    out_cell, S_cell = cell(x_rnn, S0.clone())
    print(f"Cell output: {out_cell.shape}, State: {S_cell.shape}")

    # Verify delta rule property: retrieving with same key should return close to stored value
    print("\n--- Delta rule retrieval test ---")
    cell.eval()
    with torch.no_grad():
        # Create a simple key-value pair
        test_x = torch.randn(1, 1, 256, device=device, dtype=torch.bfloat16)
        S_init = torch.zeros(1, 32, 32, device=device, dtype=torch.bfloat16)

        # Forward once to store
        _, S_after = cell(test_x, S_init)

        # Forward again with same input
        _, S_after2 = cell(test_x, S_after)

        # The state should be relatively stable (small update on second pass)
        state_change = (S_after2 - S_after).abs().mean().item()
        print(f"State change on repeat: {state_change:.6f}")

    print("\n" + "=" * 60)
    print("E71 Delta: State-dependent learning rate with delta rule")
    print("=" * 60)
