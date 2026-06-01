"""Configuration for private NDM/Emender staging checkpoints."""

from __future__ import annotations

from transformers import PretrainedConfig


class NdmConfig(PretrainedConfig):
    model_type = "ndm"

    def __init__(
        self,
        vocab_size: int = 50281,
        dim: int = 1664,
        depth: int = 12,
        level: str = "E88",
        n_heads: int | None = None,
        n_state: int = 32,
        expansion: float = 1.0,
        n_groups: int = 32,
        n_slots: int = 64,
        use_gate: bool = True,
        gate_activation: str = "sigmoid",
        linear_state: bool = False,
        use_write_gate: bool = False,
        e88_decay_mode: str = "mamba",
        e88_value_residual: bool = False,
        r_h_mode: str = "auto",
        state_expansion: int = 2,
        use_conv: bool = False,
        d_conv: int = 4,
        top_k: int | None = None,
        k_fast: int | None = None,
        k_slow: int | None = None,
        checkpoint_interval: int = 16,
        projection_chunk_size: int = 0,
        loss_chunk_size: int = 0,
        use_triton: bool = False,
        m2rnn_paper_shape: bool = False,
        m2rnn_k_head_dim: int | None = None,
        m2rnn_v_head_dim: int | None = None,
        m2rnn_q_heads: int | None = None,
        m2rnn_k_heads: int | None = None,
        m2rnn_v_heads: int | None = None,
        m2rnn_f_heads: int | None = None,
        m2rnn_g_heads: int | None = None,
        m2rnn_weight_heads: int | None = None,
        m2rnn_use_residual: bool = True,
        m2rnn_freeze_state_weight: bool = False,
        m2rnn_output_norm: bool = False,
        m2rnn_normalize_qk: bool = False,
        m2rnn_state_grad_clip: float | None = None,
        tokenizer_name: str = "p50k_base",
        private_staging: bool = True,
        release_revision_name: str = "staging",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            bos_token_id=kwargs.pop("bos_token_id", 50256),
            eos_token_id=kwargs.pop("eos_token_id", 50256),
            pad_token_id=kwargs.pop("pad_token_id", 50256),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            is_decoder=kwargs.pop("is_decoder", True),
            is_encoder_decoder=kwargs.pop("is_encoder_decoder", False),
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.num_hidden_layers = kwargs.get('num_hidden_layers', depth)
        self.level = level
        self.n_heads = n_heads
        self.n_state = n_state
        self.expansion = expansion
        self.n_groups = n_groups
        self.n_slots = n_slots
        self.use_gate = use_gate
        self.gate_activation = gate_activation
        self.linear_state = linear_state
        self.use_write_gate = use_write_gate
        self.e88_decay_mode = e88_decay_mode
        self.e88_value_residual = e88_value_residual
        self.r_h_mode = r_h_mode
        self.state_expansion = state_expansion
        self.use_conv = use_conv
        self.d_conv = d_conv
        self.top_k = top_k
        self.k_fast = k_fast
        self.k_slow = k_slow
        self.checkpoint_interval = checkpoint_interval
        self.projection_chunk_size = projection_chunk_size
        self.loss_chunk_size = loss_chunk_size
        self.use_triton = use_triton
        self.m2rnn_paper_shape = m2rnn_paper_shape
        self.m2rnn_k_head_dim = m2rnn_k_head_dim
        self.m2rnn_v_head_dim = m2rnn_v_head_dim
        self.m2rnn_q_heads = m2rnn_q_heads
        self.m2rnn_k_heads = m2rnn_k_heads
        self.m2rnn_v_heads = m2rnn_v_heads
        self.m2rnn_f_heads = m2rnn_f_heads
        self.m2rnn_g_heads = m2rnn_g_heads
        self.m2rnn_weight_heads = m2rnn_weight_heads
        self.m2rnn_use_residual = m2rnn_use_residual
        self.m2rnn_freeze_state_weight = m2rnn_freeze_state_weight
        self.m2rnn_output_norm = m2rnn_output_norm
        self.m2rnn_normalize_qk = m2rnn_normalize_qk
        self.m2rnn_state_grad_clip = m2rnn_state_grad_clip
        self.tokenizer_name = tokenizer_name
        self.private_staging = private_staging
        self.release_revision_name = release_revision_name
