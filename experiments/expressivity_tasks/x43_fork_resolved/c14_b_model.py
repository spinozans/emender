"""Enabled B1 store door, with no changes to the hash-bound C14 arms.

CUDA uses a B-only native scan with reset-state autograd replay. CPU uses the
independent eager implementation. Both reuse the actual native stream K.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .c11 import Boosted
from .c14_b_suffix import queried_suffix
from .c14_b_store import SharedStore


class StoreMetadata(tuple):
    """Window-local immutable metadata with reusable integer encodings."""
    def __new__(cls,values):
        obj=super().__new__(cls,values);obj.encodings={}
        return obj


class StoreIntegrated(Boosted):
    def __init__(self):
        super().__init__('emender', 2000000)
        self.telemetry = []

    def forward(self, ids, answers, context_lengths, *, windows=None,
                valid_lengths=None, enabled=True, checkpointed=True):
        if not enabled:
            # The dense arm has no consultation embedding hook.
            return self.head(ids, torch.zeros_like(answers), False), torch.ones_like(context_lengths)
        b,t = ids.shape
        if answers.shape != (b,t,260) or context_lengths.shape != (b,t):
            raise ValueError('answer/context shape mismatch')
        if not bool(torch.isfinite(answers).all()):
            raise ValueError('nonfinite answer')
        if answers.dtype != torch.float32:
            raise ValueError('cache answers must remain float32')
        windows = list(windows) if windows is not None else [(j,0) for j in range(b)]
        valid_lengths = list(valid_lengths) if valid_lengths is not None else [t]*b
        if len(windows) != b or len(valid_lengths) != b or any(not 0 <= n <= t for n in valid_lengths):
            raise ValueError('window/padding metadata mismatch')
        raw = [bytes(row) for row in ids.detach().cpu().tolist()]
        contexts = context_lengths.detach().cpu().tolist()
        # Validate even dropped/blank answers. No silently clipped query IDs.
        owners = [[queried_suffix(windows[j], raw[j][:i+1], contexts[j][i])
                   if i < valid_lengths[j] else None for i in range(t)] for j in range(b)]
        keep = answers.detach().abs().sum(-1).ne(0).cpu().tolist()
        predictions = answers[...,:256].detach().argmax(-1).cpu().tolist()
        metadata = StoreMetadata((owners,raw,windows,valid_lengths,keep,predictions))
        x = self.head.base.embed(ids)
        rows = []
        for block in self.head.base.blocks:
            captured = {}
            def run(value, consultation, block=block, captured=captured):
                with torch.autocast(device_type=ids.device.type, enabled=False):
                    stream = block.input(block.norm1(value)).float()
                    # Consultation content is separate from the native key path.
                    content = self.device_input(consultation.float()).float()
                    cell = block.cell
                    z = cell.W_in(stream)
                    native_values = cell.W_v(stream).reshape(b,t,cell.policy.N,128)
                    array_values = cell.W_v(content).reshape(b,t,cell.policy.N,128)
                    pop = cell.W_pop(stream).sigmoid()
                    append = cell.W_append(stream)
                    logits = cell.W_write(stream)
                    reads = []; reports = []
                    for h,cfg in enumerate(cell.configs):
                        features = z+cell.features[h](ids)
                        # Exactly the same key tensor, computed once for both
                        # stream and array writes. Never keyed by answer content.
                        keys = cell.geometry[h]['key'](features).detach()
                        queries = cell.geometry[h]['query'](features)
                        from .c14_b_native import eager, native_forward, NativeReplay, native_reports
                        inputs = (keys,queries,native_values[:,:,h],array_values[:,:,h],
                                  pop[:,:,h],append[:,:,h],logits[:,:,h],
                                  cell.append_err_coef[h],cell.match_coef[h],cell.error_coef[h])
                        if ids.is_cuda:
                            if torch.is_grad_enabled():
                                output,trace = NativeReplay.apply(cfg,cell.gate_policies[h],metadata,*inputs)
                            else:
                                output,(_,trace) = native_forward(inputs,cfg,cell.gate_policies[h],metadata)
                            # The initial checkpoint forward already owns the
                            # complete telemetry; do not serialize it a second
                            # time during the discarded backward recomputation.
                            head_reports = [] if checkpointed and self.training and torch.is_grad_enabled() else native_reports(trace,cfg,valid_lengths)
                        else:
                            output,head_reports = eager(*inputs,cfg,cell.gate_policies[h],metadata)
                        for j,report in enumerate(head_reports):
                            reports.append(dict(head=h,batch=j,**report))
                        reads.append(output)
                    read = torch.stack(reads,2).flatten(2)
                    captured['reports'] = reports
                    cell.gate_mean = value.new_zeros(())
                value = value+block.read_out(read)
                return value+block.ff(block.norm2(value))
            if checkpointed and self.training and torch.is_grad_enabled():
                x = checkpoint(run,x,answers,use_reentrant=True,preserve_rng_state=True)
            else:
                x = run(x,answers)
            rows.append(captured['reports'])
        self.telemetry = rows
        x = self.head.base.norm(x)
        lp = F.linear(x.float(), self.head.base.head.weight.float()).log_softmax(-1)
        return lp, torch.ones_like(context_lengths)
