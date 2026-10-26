"""Three-level 64-ary lowest-free bitmap; one owner per tape store.

The supported maximum is 64**3 slots, matching X30 FreeList. No atomics or
host allocation decisions are needed. Initialization is a boundary operation.
"""
import triton
import triton.language as tl


@triton.jit
def _first(bits):
    return tl.inline_asm_elementwise(
        "{ .reg .b64 reversed; brev.b64 reversed, $1; clz.b64 $0, reversed; }",
        constraints="=r,l", args=[bits.to(tl.uint64)], dtype=tl.int32,
        is_pure=True, pack=1)


@triton.jit
def set_free(L0, L1, Root, slot, available):
    word = slot//64
    bit = tl.full((), 1, tl.uint64) << (slot%64).to(tl.uint64)
    bits = tl.load(L0+word)
    bits = tl.where(available, bits | bit, bits & ~bit)
    tl.store(L0+word, bits)
    parent = word//64
    bit = tl.full((), 1, tl.uint64) << (word%64).to(tl.uint64)
    summary = tl.load(L1+parent)
    summary = tl.where(bits != 0, summary | bit, summary & ~bit)
    tl.store(L1+parent, summary)
    bit = tl.full((), 1, tl.uint64) << parent.to(tl.uint64)
    root = tl.load(Root)
    tl.store(Root, tl.where(summary != 0, root | bit, root & ~bit))
    tl.debug_barrier()


@triton.jit
def lowest_free(L0, L1, Root):
    root = tl.load(Root)
    slot = tl.full((), -1, tl.int32)
    if root != 0:
        parent = _first(root)
        word = 64*parent+_first(tl.load(L1+parent))
        slot = 64*word+_first(tl.load(L0+word))
    return slot
