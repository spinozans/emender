"""One explicit precision policy for E97 recurrent state (not model weights)."""
import json

MODES = ('legacy', 'fp32')
FIXED_RECURRENT_KERNEL = 'e88-sequential-fp32-bh1-nw4-v1'


def validate_state_precision(mode):
    if mode not in MODES:
        raise ValueError(f'recurrent_state_precision must be one of {MODES}, got {mode!r}')
    return mode


def recurrent_launch_config(mode, override=None):
    """FP32 policy fixes arithmetic geometry; legacy retains historical tuning.

    Overrides are a low-level kernel-testing API, not an operator setting.
    """
    validate_state_precision(mode)
    if override is None:
        return (1, 4) if mode == 'fp32' else None
    if (not isinstance(override, (tuple, list)) or len(override) != 2
            or any(type(x) is not int for x in override)
            or override[0] not in (1, 2, 4, 8, 16) or override[1] not in (1, 2, 4, 8)):
        raise ValueError('invalid recurrent launch configuration')
    return tuple(override)


def recurrent_checkpoint_dtype(mode, state, projection_dtype):
    """Legacy inference has FP32 carry but projection-dtype replay storage."""
    import torch
    validate_state_precision(mode)
    if mode == 'fp32':
        if state.dtype != torch.float32:
            raise ValueError('fp32 recurrent policy requires FP32 initial state')
        return torch.float32
    return projection_dtype


def inference_workspace_precision(mode, *, training, grad_enabled):
    """Keep discarded eval/no-grad workspace compatible, never retained state.

    Training previews keep the training allocation even under no_grad. Any
    gradient-enabled call (including eval) retains the requested replay policy.
    """
    validate_state_precision(mode)
    return 'legacy' if not training and not grad_enabled else mode


def configure_recurrent_precision(model, mode=None):
    modules = [m for m in model.modules() if hasattr(m, 'recurrent_state_precision')]
    current = {validate_state_precision(m.recurrent_state_precision) for m in modules}
    if len(current) > 1:
        raise ValueError('mixed recurrent precision policies')
    resolved = validate_state_precision(mode if mode is not None else next(iter(current), 'legacy'))
    if not modules and resolved != 'legacy':
        raise ValueError('model has no recurrent precision-aware layers')
    for module in modules:
        module.recurrent_state_precision = resolved
    return resolved


def restore_checkpoint_precision(config, checkpoint):
    """Retain saved numerical semantics even when separate architecture args are used."""
    policy = checkpoint.get('sft_precision', {})
    if not isinstance(policy, dict):
        raise ValueError('checkpoint sft_precision must be an object')
    recorded = policy.get('recurrent_state_precision')
    if 'recurrent_kernel' in policy and (
            recorded != 'fp32' or policy['recurrent_kernel'] != FIXED_RECURRENT_KERNEL):
        raise ValueError('unsupported checkpoint recurrent kernel policy')
    layers = config.get('layer_kwargs') or {}
    if isinstance(layers, str):
        layers = json.loads(layers)
    if not isinstance(layers, dict):
        raise ValueError('layer_kwargs must be an object')
    layers = dict(layers)
    explicit = layers.get('recurrent_state_precision')
    if explicit is not None:
        validate_state_precision(explicit)
    if recorded is not None:
        validate_state_precision(recorded)
        if explicit is not None and explicit != recorded:
            raise ValueError('checkpoint/config recurrent precision mismatch')
        layers['recurrent_state_precision'] = recorded
        config = dict(config, layer_kwargs=layers)
    return config
