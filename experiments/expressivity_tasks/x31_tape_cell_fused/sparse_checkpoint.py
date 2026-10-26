"""Sparse before-image journal building block, not an autograd implementation.

A producer must record each before-image at mutation time, not gather all old
values from a boundary state (which is wrong for repeated writes). One base
store may be retained per sequence. This module adds no boundary store copies.
"""
from dataclasses import dataclass
import torch
import triton
from .kernels import undo_kernel


@dataclass
class DeltaJournal:
    slots: torch.Tensor  # [stores,events], -1 denotes padding
    before: torch.Tensor  # [stores,events,fields]

    @property
    def nbytes(self):
        return self.slots.numel()*self.slots.element_size()+self.before.numel()*self.before.element_size()

    def undo_(self, store):
        """Restore a packed [stores,slots,fields] tensor in reverse event order.

        The caller owns the mutable store. No gradients, allocation, threshold
        or aiming decisions are evaluated. This is state rollback, not BPTT.
        """
        if store.requires_grad or self.before.requires_grad:
            raise ValueError('rollback requires detached storage')
        if (store.ndim != 3 or self.slots.shape != self.before.shape[:2]
            or store.shape[0] != self.slots.shape[0]
            or store.shape[2] != self.before.shape[2]):
            raise ValueError('journal/store shape mismatch')
        if not all(x.is_contiguous() for x in (store,self.slots,self.before)):
            raise ValueError('journal and store must be contiguous')
        if any(x.device != store.device for x in (self.slots,self.before)):
            raise ValueError('journal/store device mismatch')
        if self.slots.dtype != torch.int64 or self.before.dtype != store.dtype:
            raise ValueError("journal IDs must be int64 and field dtypes must match")
        if bool(((self.slots < -1) | (self.slots >= store.shape[1])).any()):
            raise ValueError("journal slot outside store")
        n,s,d = store.shape
        e = self.slots.shape[1]
        if e:
            undo_kernel[(n,)](store,self.slots,self.before,e,s,d,triton.next_power_of_2(d))
        return store
