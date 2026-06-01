# [fix-hf-v03] packaging patch: vendored-import fallback + transformers-robust tying
"""HF wrapper for private NDM/Emender staging checkpoints.

This custom-code loader expects the `ndm` source package to be installed in the
runtime environment. The private staging Docker smoke installs the repository
before loading these Hugging Face artifacts.
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_ndm import NdmConfig


def _bool(value) -> bool:
    return bool(value)


def _build_ndm_model(config: NdmConfig) -> nn.Module:
    if str(config.level).lower() == "m2rnn":
        try:
            module = importlib.import_module("ndm.models.m2rnn_baseline")
        except ModuleNotFoundError:
            module = importlib.import_module("elman.models.m2rnn_baseline")
        return module.M2RNNLM(
            vocab_size=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            n_heads=config.n_heads,
            n_state=config.n_state,
            expansion=config.expansion,
            paper_shape=_bool(config.m2rnn_paper_shape),
            k_head_dim=config.m2rnn_k_head_dim,
            v_head_dim=config.m2rnn_v_head_dim,
            num_q_heads=config.m2rnn_q_heads,
            num_k_heads=config.m2rnn_k_heads,
            num_v_heads=config.m2rnn_v_heads,
            num_f_heads=config.m2rnn_f_heads,
            num_g_heads=config.m2rnn_g_heads,
            num_weight_heads=config.m2rnn_weight_heads,
            use_gate=_bool(config.use_gate),
            use_residual=_bool(config.m2rnn_use_residual),
            state_weight_trainable=not _bool(config.m2rnn_freeze_state_weight),
            use_conv=_bool(config.use_conv),
            d_conv=config.d_conv,
            output_norm=_bool(config.m2rnn_output_norm),
            normalize_qk=_bool(config.m2rnn_normalize_qk),
            dropout=0.0,
            gradient_clipping=config.m2rnn_state_grad_clip,
            gradient_checkpointing=False,
            loss_chunk_size=0,
        )

    try:
        module = importlib.import_module("ndm.models.ladder_lm")
    except ModuleNotFoundError:
        module = importlib.import_module("elman.models.ladder_lm")
    return module.LadderLM(
        vocab_size=config.vocab_size,
        dim=config.dim,
        depth=config.depth,
        level=config.level,
        expansion=config.expansion,
        n_groups=config.n_groups,
        n_state=config.n_state,
        n_slots=config.n_slots,
        n_heads=config.n_heads,
        top_k=config.top_k,
        k_fast=config.k_fast,
        k_slow=config.k_slow,
        use_gate=_bool(config.use_gate),
        gate_activation=config.gate_activation,
        linear_state=_bool(config.linear_state),
        use_write_gate=_bool(config.use_write_gate),
        e88_decay_mode=config.e88_decay_mode,
        e88_value_residual=_bool(config.e88_value_residual),
        state_expansion=config.state_expansion,
        r_h_mode=config.r_h_mode,
        use_conv=_bool(config.use_conv),
        d_conv=config.d_conv,
        dropout=0.0,
        checkpoint_interval=config.checkpoint_interval,
        gradient_checkpointing=False,
        projection_chunk_size=0,
        loss_chunk_size=0,
        use_triton=_bool(config.use_triton),
    )


class NdmForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NdmConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = ["model.lm_head.weight"]
    all_tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: NdmConfig):
        super().__init__(config)
        self.model = _build_ndm_model(config)

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def tie_weights(self, *args, **kwargs):
        if hasattr(self.model, "lm_head") and hasattr(self.model, "embedding"):
            self.model.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        del attention_mask, kwargs
        if input_ids is None:
            raise ValueError("NdmForCausalLM requires input_ids")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        logits = self.model(input_ids, return_loss=False)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        del past_key_values
        return {"input_ids": input_ids}
