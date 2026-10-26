import torch
import triton
import triton.language as tl

from experiments.expressivity_tasks.x31_tape_cell_fused.device_free import set_free, lowest_free
from .test_parity import device


@triton.jit
def _transactions(L0, L1, Root, Slots, Output, E: tl.constexpr):
    for event in range(E):
        slot = tl.load(Slots+event)
        if slot >= 0:
            set_free(L0, L1, Root, slot, True)
        else:
            taken = lowest_free(L0, L1, Root)
            tl.store(Output+event, taken)
            if taken >= 0:
                set_free(L0, L1, Root, taken, False)


def test_bitmap_high_bits_all_three_levels_and_reuse():
    dev = device()
    # Begin full; releases include high bit 63 at every hierarchy level.
    events = [262143, 4096, 64, 63, 0] + [-1]*6 + [262143, 0, -1, -1, -1]
    l0 = torch.zeros(4096, device=dev, dtype=torch.uint64)
    l1 = torch.zeros(64, device=dev, dtype=torch.uint64)
    root = torch.zeros(1, device=dev, dtype=torch.uint64)
    output = torch.full((len(events),), -9, device=dev, dtype=torch.int32)
    _transactions[(1,)](l0,l1,root,torch.tensor(events,device=dev),output,len(events),num_warps=1)
    assert output[5:11].tolist() == [0,63,64,4096,262143,-1]
    assert output[-3:].tolist() == [0,262143,-1]
    assert root.eq(0).all() and l0.eq(0).all() and l1.eq(0).all()
