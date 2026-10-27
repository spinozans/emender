"""Mutation-count reservation for the store-local persistent core."""
import torch
import triton
import triton.language as tl
from ..x31_tape_cell_fused.all_field_journal import AllFieldJournal as OracleJournal


class AllFieldJournal(OracleJournal):
    @staticmethod
    def capacity_for(stores, tokens, slots, dk, dv, initial_live):
        # One pop schedule/token <=12*height+6 scalar writes (cancel's
        # up/down plus insertion's up). Allocation always owns a free slot,
        # whose inverse heap position is -1, so append schedule <=4*height+3.
        # One death/cell lifetime, <=8*height
        # heap writes plus <=32 payload/bitmap/birth writes. At most
        # initial_live+tokens lifetimes exist. Fixed writes/token <=64;
        # allocation and correction are exclusive, payload <=dk+dv.
        # Initial low removals and final strength/age materialization fit
        # the 32 fixed entries/lifetime retained below.
        height=slots.bit_length()
        lifetimes=initial_live+tokens
        capacity=tokens*(dk+dv+64+16*height)+lifetimes*(8*height+32)
        if stores*slots*(dk+dv)>2**31 or stores*capacity>2**31:
            raise ValueError('journal capacity exceeds checked int32 index range; split batch/chunk')
        return capacity


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
            tl.device_assert(start+length<=capacity,'all-field journal capacity exceeded')
            index=n*capacity+start+ranks-1
            active=changed & (start+ranks<=capacity)
            tl.store(Addresses+index,pointer.to(tl.uint64),active)
            tl.store(Before+index,bits,active)
            tl.store(Widths+index,width | (length<<8),active)
            # Publish the count only after every warp has emitted its group.
            tl.debug_barrier()
            tl.store(Counts+n,start+length)
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
            tl.device_assert(start+length<=capacity,'all-field journal capacity exceeded')
            index=n*capacity+start+ranks-1
            active=mask & (start+ranks<=capacity)
            tl.store(Addresses+index,pointer.to(tl.uint64),active)
            tl.store(Before+index,bits,active)
            tl.store(Widths+index,width | (length<<8),active)
            # Publish the count only after every warp has emitted its group.
            tl.debug_barrier()
            tl.store(Counts+n,start+length)
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
            tl.device_assert((~changed)|(start<capacity),'all-field journal capacity exceeded')
            active=changed & (start<capacity)
            index=n*capacity+start
            tl.store(Addresses+index,pointer.to(tl.uint64),active)
            tl.store(Before+index,bits,active)
            tl.store(Widths+index,width | (1<<8),active)
            tl.debug_barrier()
            tl.store(Counts+n,start+changed.to(tl.int32))
    tl.store(pointer,value,mask)
    if Record:
        # The next call must observe both the count and this mutation.
        tl.debug_barrier()



@triton.jit
def _audit(Addresses, Counts, Bad, Capacity:tl.constexpr, Block:tl.constexpr):
    n=tl.program_id(0).to(tl.int64)
    x=tl.program_id(1)*Block+tl.arange(0,Block)
    active=x<tl.load(Counts+n)
    address=tl.load(Addresses+n*Capacity+x,active,1)
    if tl.sum((active & (address==0)).to(tl.int32),0)>0:
        tl.atomic_or(Bad,1)


def assert_initialized(log):
    addresses,_,_,counts,*_=log
    bad=torch.zeros((),device=addresses.device,dtype=torch.int32)
    _audit[(len(counts),triton.cdiv(addresses.shape[1],1024))](
        addresses,counts,bad,addresses.shape[1],1024)
    if bad.item():
        raise AssertionError('journal published an unwritten address')
