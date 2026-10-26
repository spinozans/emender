"""Typed mutation-time undo log for owned GPU storage.

Scalars skip unchanged bits; vectors emit at their mutation sites. Ordered
undo groups preserve repeated-write before-images. Reservation is bounded by
initial live rows plus possible token appends, rather than the physical slot
budget (apart from heap depth). Capacity uses a host-known conservative live bound; valid lengths stay on device.
Addresses are private to the owned state; callers cannot supply arbitrary undo
pointers. Snapshots explicitly rebase addresses onto cloned storage.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@triton.jit
def journal_store(pointer, value, mask, J, Record: tl.constexpr, Base=None, Count=None, Scratch=None):
    if Record:
        Addresses, Before, Widths, Counts, n, capacity = J
        dtype: tl.constexpr = pointer.dtype.element_ty
        width: tl.constexpr = ((dtype.primitive_bitwidth+7)//8)*8
        if len(pointer.shape)>0 and pointer.numel<=128:
            old=tl.load(pointer,mask,0)
            new=tl.cast(value,dtype)
            if width==64:
                bits=old.to(tl.uint64,bitcast=True)
                proposed=new.to(tl.uint64,bitcast=True)
            elif width==32:
                bits=old.to(tl.uint32,bitcast=True).to(tl.uint64)
                proposed=new.to(tl.uint32,bitcast=True).to(tl.uint64)
            else:
                bits=old.to(tl.uint8).to(tl.uint64)
                proposed=new.to(tl.uint8).to(tl.uint64)
            changed=mask
            ranks=tl.cumsum(changed.to(tl.int32),0)
            length=tl.sum(changed.to(tl.int32),0)
            start=tl.load(Counts+n)
            tl.store(Counts+n,start+length)
            tl.device_assert(start+length<=capacity,'all-field journal capacity exceeded')
            index=n*capacity+start+ranks-1
            active=changed & (start+ranks<=capacity)
            tl.store(Addresses+index,pointer.to(tl.uint64),active)
            tl.store(Before+index,bits,active)
            tl.store(Widths+index,width | (length<<8),active)
        elif len(pointer.shape)>0:
            # The caller identifies the mutated payload; emit its before-image
            # directly, without staging proposed data or discovering changes.
            old=tl.load(pointer,mask,0)
            if width==64:
                bits=old.to(tl.uint64,bitcast=True)
            elif width==32:
                bits=old.to(tl.uint32,bitcast=True).to(tl.uint64)
            else:
                bits=old.to(tl.uint8).to(tl.uint64)
            ranks=tl.cumsum(mask.to(tl.int32),0)
            length=tl.sum(mask.to(tl.int32),0)
            start=tl.load(Counts+n)
            tl.store(Counts+n,start+length)
            tl.device_assert(start+length<=capacity,'all-field journal capacity exceeded')
            index=n*capacity+start+ranks-1
            active=mask & (start+ranks<=capacity)
            tl.store(Addresses+index,pointer.to(tl.uint64),active)
            tl.store(Before+index,bits,active)
            tl.store(Widths+index,width | (length<<8),active)
        else:
            old=tl.load(pointer,mask,0)
            new=tl.cast(value,dtype)
            if width==64:
                bits=old.to(tl.uint64,bitcast=True)
                new_bits=new.to(tl.uint64,bitcast=True)
            elif width==32:
                bits=old.to(tl.uint32,bitcast=True).to(tl.uint64)
                new_bits=new.to(tl.uint32,bitcast=True).to(tl.uint64)
            else:
                bits=old.to(tl.uint8).to(tl.uint64)
                new_bits=new.to(tl.uint8).to(tl.uint64)
            changed=mask & (bits!=new_bits)
            start=tl.load(Counts+n)
            tl.store(Counts+n,start+changed.to(tl.int32))
            tl.device_assert((~changed)|(start<capacity),'all-field journal capacity exceeded')
            active=changed & (start<capacity)
            index=n*capacity+start
            tl.store(Addresses+index,pointer.to(tl.uint64),active)
            tl.store(Before+index,bits,active)
            tl.store(Widths+index,width | (1<<8),active)
    tl.store(pointer,value,mask)


@triton.jit
def undo_to(J, start, stop, Block: tl.constexpr):
    Addresses, Before, Widths, Counts, n, capacity = J
    d = tl.arange(0,Block)
    while stop > start:
        packed = tl.load(Widths+n*capacity+stop-1)
        count = packed >> 8
        width = packed & 255
        tl.device_assert((count>0) & (stop-count>=start), 'invalid undo group')
        for tile in range(tl.cdiv(count,Block)):
            lane=tile*Block+d
            index = n*capacity+stop-count+lane
            valid = lane<count
            address = tl.load(Addresses+index,valid,0)
            bits = tl.load(Before+index,valid,0)
            if width == 64:
                tl.store(address.to(tl.pointer_type(tl.uint64)), bits,valid)
            elif width == 32:
                tl.store(address.to(tl.pointer_type(tl.uint32)), bits.to(tl.uint32),valid)
            else:
                tl.store(address.to(tl.pointer_type(tl.uint8)), bits.to(tl.uint8),valid)
        tl.debug_barrier()
        stop -= count


@triton.jit
def _undo(Addresses, Before, Widths, Counts, Capacity, Block: tl.constexpr):
    n = tl.program_id(0)
    undo_to((Addresses,Before,Widths,Counts,n,Capacity), 0, tl.load(Counts+n),Block)


@dataclass
class AllFieldJournal:
    addresses: torch.Tensor
    before: torch.Tensor
    widths: torch.Tensor
    counts: torch.Tensor
    boundaries: torch.Tensor
    gates: torch.Tensor
    errors: torch.Tensor
    owners: tuple
    start_tokens: int
    fresh: bool
    block: int = 1
    used: bool = False
    parent: object = None

    @staticmethod
    def capacity_for(stores, tokens, slots, dk, dv, initial_live):
        if stores*slots*(dk+dv)>2**31:
            raise ValueError('journal capacity exceeds checked int32 index range; split batch/chunk')
        active = min(slots, triton.next_power_of_2(max(1,initial_live + tokens)))
        capacity = ((6*tokens+8)*active+tokens*(dk+dv+64)
                    +(active+3*tokens)*(16*slots.bit_length()+32))
        if stores*capacity>2**31:
            raise ValueError('journal capacity exceeds checked int32 index range; split batch/chunk')
        return capacity

    @classmethod
    def allocate(cls, stores, tokens, slots, dk, dv, owners, start_tokens, fresh, initial_live=None):
        # Fail before inspecting owners or allocating an oversized address space.
        cls.capacity_for(stores,tokens,slots,dk,dv,0)
        # A token can introduce at most one live row. Dead imported payloads
        # remain untouched; exact rollback still includes every changed bit.
        capacity = cls.capacity_for(stores,tokens,slots,dk,dv,
                                    slots if initial_live is None else initial_live)
        device = owners[0].device
        return cls(torch.empty(stores,capacity,device=device,dtype=torch.uint64),
                   torch.empty(stores,capacity,device=device,dtype=torch.uint64),
                   torch.empty(stores,capacity,device=device,dtype=torch.int32),
                   torch.zeros(stores,device=device,dtype=torch.int64),
                   torch.empty(stores,tokens,device=device,dtype=torch.int64),
                   torch.empty(stores,tokens,device=device),
                   torch.empty(stores,tokens,device=device),
                   tuple(owners),start_tokens,fresh,min(128,triton.next_power_of_2(max(slots,dk,dv))))

    def compact_(self):
        """Compatibility no-op: fixed capacity avoids host length reads."""
        return self

    @property
    def capacity(self):
        return self.addresses.shape[1]

    @property
    def nbytes(self):
        return sum(x.numel()*x.element_size() for x in
                   (self.addresses,self.before,self.widths,self.counts,self.boundaries,self.gates,self.errors))

    def snapshot(self, owners):
        from .backward import rebase_owned
        addresses=self.addresses.clone()
        rebase_owned(addresses,self.counts,self.owners,owners,self.capacity)
        return AllFieldJournal(addresses,self.before.clone(),self.widths.clone(),
            self.counts.clone(),self.boundaries.clone(),self.gates.clone(),self.errors.clone(),
            tuple(owners),self.start_tokens,self.fresh,self.block,False,
            None if self.parent is None else self.parent.snapshot(owners))

    def undo_(self, state):
        """Consume the final handle and restore its pre-chunk owned state."""
        if self.used or state.consumed or state.all_field_journal is not self:
            raise ValueError('journal requires its unconsumed owning state')
        if any(x.requires_grad for x in self.owners):
            raise ValueError('rollback requires detached owned storage')
        _undo[(len(self.counts),)](self.addresses,self.before,self.widths,
                                  self.counts,self.capacity,self.block)
        from .cell_fused import FusedState
        state.consumed = True
        self.used = True
        return FusedState(state.native,state.history,self.start_tokens,
                          None if self.fresh else state.metadata,
                          None if self.fresh else state.configuration,
                          all_field_journal=self.parent)
