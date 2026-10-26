"""Mutation-time before-images for the live scan's key/value writes.

This is deliberately a *payload* journal. Scheduler, strength and liveness
mutations are not rolled back by it; it must not be used as a training
checkpoint or as proof of the complete mutation-journal requirement.
There is at most one payload write per token and store. Slots of -1 denote
tokens without a write. Repeated writes retain distinct before-images.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def capture_payload(Keys, Values, Slots, BeforeKeys, BeforeValues, event, slot,
                    N, S: tl.constexpr, T: tl.constexpr, DK: tl.constexpr,
                    DV: tl.constexpr, BLOCK: tl.constexpr):
    """Called by the owning scan program immediately before its HBM write."""
    d = tl.arange(0, BLOCK)
    old_key = tl.load(Keys+(N*S+slot)*DK+d, d<DK, 0.)
    old_value = tl.load(Values+(N*S+slot)*DV+d, d<DV, 0.)
    tl.store(Slots+N*T+event, slot)
    tl.store(BeforeKeys+(N*T+event)*DK+d, old_key, d<DK)
    tl.store(BeforeValues+(N*T+event)*DV+d, old_value, d<DV)
    # The caller may now overwrite the same addresses. Keep the before-image
    # loads/stores ordered across all warps in the owning program.
    tl.debug_barrier()


@triton.jit
def undo_payload(Keys, Values, Slots, BeforeKeys, BeforeValues,
                 S: tl.constexpr, T: tl.constexpr, DK: tl.constexpr,
                 DV: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    for reverse in range(T):
        event = T-1-reverse
        slot = tl.load(Slots+n*T+event)
        if slot >= 0:
            key = tl.load(BeforeKeys+(n*T+event)*DK+d, d<DK, 0.)
            value = tl.load(BeforeValues+(n*T+event)*DV+d, d<DV, 0.)
            tl.store(Keys+(n*S+slot)*DK+d, key, d<DK)
            tl.store(Values+(n*S+slot)*DV+d, value, d<DV)
        tl.debug_barrier()


@dataclass
class WriteJournal:
    slots: torch.Tensor  # [B*H,T], -1 for no payload mutation
    keys: torch.Tensor   # [B*H,T,K], before-images (zero for padding)
    values: torch.Tensor # [B*H,T,V], before-images (zero for padding)

    @classmethod
    def allocate(cls, stores, tokens, key_dim, value_dim, device):
        return cls(torch.full((stores,tokens),-1,device=device,dtype=torch.int64),
                   torch.zeros(stores,tokens,key_dim,device=device,dtype=torch.float32),
                   torch.zeros(stores,tokens,value_dim,device=device,dtype=torch.float32))

    def snapshot(self):
        return WriteJournal(self.slots.clone(),self.keys.clone(),self.values.clone())

    @property
    def nbytes(self):
        return sum(x.numel()*x.element_size() for x in
                   (self.slots,self.keys,self.values))

    def undo_payload_(self, keys, values):
        """Undo only keys/values, in place, on detached owned storage.

        Accepts native [B,H,S,D] or flattened [B*H,S,D] tensors. The returned
        pair aliases the supplied storage. Other state/metadata is untouched.
        Calling this on a FusedState does not yield a valid continuation.
        """
        if keys.ndim not in (3,4) or values.ndim != keys.ndim:
            raise ValueError('payload storage must have rank 3 or 4')
        if keys.shape[:-1] != values.shape[:-1]:
            raise ValueError('key/value store shape mismatch')
        stores = keys.shape[0] if keys.ndim == 3 else keys.shape[0]*keys.shape[1]
        slots, dk, dv = keys.shape[-2], keys.shape[-1], values.shape[-1]
        tensors = (keys,values,self.slots,self.keys,self.values)
        if any(x.requires_grad for x in tensors):
            raise ValueError('payload rollback requires detached storage')
        if any(x.device != keys.device or not x.is_contiguous() for x in tensors):
            raise ValueError('payload storage must be contiguous on one device')
        if (keys.dtype != torch.float32 or values.dtype != torch.float32
            or self.keys.dtype != torch.float32 or self.values.dtype != torch.float32
            or self.slots.dtype != torch.int64):
            raise ValueError('payload journal requires fp32 fields and int64 IDs')
        if (self.slots.ndim != 2 or self.slots.shape[0] != stores
            or self.keys.shape != (*self.slots.shape,dk)
            or self.values.shape != (*self.slots.shape,dv)):
            raise ValueError('payload journal/store shape mismatch')
        if bool(((self.slots < -1) | (self.slots >= slots)).any()):
            raise ValueError('payload journal slot outside store')
        tokens = self.slots.shape[1]
        if tokens:
            undo_payload[(stores,)](keys,values,self.slots,self.keys,self.values,
                                    slots,tokens,dk,dv,triton.next_power_of_2(max(dk,dv)))
        return keys, values
