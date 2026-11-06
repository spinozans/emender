"""Opt-in attention-style register tiling, exact normalized parity chains."""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from .c14_phase5r3_split_kernel import as_rows, as_cols, fixed_sum

@gluon.jit
def read_page(FinalV,Product,Carry,Out,Valid,t,page,T:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,PAGE:tl.constexpr):
    b=tl.program_id(0)
    vd=tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    rs=tl.arange(0,2,layout=tl.BlockedLayout([1],[32],[8],[0]))
    ts=tl.arange(0,32,layout=tl.BlockedLayout([1],[32],[8],[0]))
    if t < tl.load(Valid+b):
        if page==0:
            acc=tl.full((2,128),0.,tl.float32,layout=tl.BlockedLayout([1,1],[1,32],[2,4],[1,0]))
        else:
            acc=tl.load(Carry+(b*2+as_rows(rs))*128+as_cols(vd))
        for tile in range(tl.minimum(PAGE,BS)//32):
            rows=page*PAGE+tile*32+ts
            w=tl.load(Product+b*BS+rows,rows<BS,0.)
            v=tl.load(FinalV+(b*S+as_rows(rows))*128+as_cols(vd),as_rows(rows)<S,0.)
            products=as_rows(w)*v
            pairs=products.reshape((16,2,128)).permute((1,2,0))
            pairs=tl.convert_layout(pairs,tl.BlockedLayout([1,1,16],[1,32,1],[2,4,1],[1,0,2]))
            n0=pairs.reshape((2,128,2,2,2,2))
            n1_0,n1_1=tl.split(n0)
            n2_0,n2_2=tl.split(n1_0)
            n2_1,n2_3=tl.split(n1_1)
            n3_0,n3_4=tl.split(n2_0)
            n3_2,n3_6=tl.split(n2_2)
            n3_1,n3_5=tl.split(n2_1)
            n3_3,n3_7=tl.split(n2_3)
            n4_0,n4_8=tl.split(n3_0)
            n4_4,n4_12=tl.split(n3_4)
            n4_2,n4_10=tl.split(n3_2)
            n4_6,n4_14=tl.split(n3_6)
            n4_1,n4_9=tl.split(n3_1)
            n4_5,n4_13=tl.split(n3_5)
            n4_3,n4_11=tl.split(n3_3)
            n4_7,n4_15=tl.split(n3_7)
            if (page==0) & (tile==0):
                acc=n4_0
            else:
                acc=acc+n4_0
            acc=acc+n4_1
            acc=acc+n4_2
            acc=acc+n4_3
            acc=acc+n4_4
            acc=acc+n4_5
            acc=acc+n4_6
            acc=acc+n4_7
            acc=acc+n4_8
            acc=acc+n4_9
            acc=acc+n4_10
            acc=acc+n4_11
            acc=acc+n4_12
            acc=acc+n4_13
            acc=acc+n4_14
            acc=acc+n4_15
        tl.store(Carry+(b*2+as_rows(rs))*128+as_cols(vd),acc)
        if (page+1)*PAGE>=BS:
            tl.store(Out+(b*T+t)*128+vd,fixed_sum(acc,0))
    else:
        tl.store(Out+(b*T+t)*128+vd,tl.full((128,),0.,tl.float32,layout=tl.BlockedLayout([1],[32],[8],[0])))

from . import c14_phase5r4_stream_pages as shared
class TiledPages(shared.StreamingPages):
    def __getitem__(self,grid):
        parent=super().__getitem__(grid)
        def launch(*a,**kw):
            original=shared.read_page
            try:
                shared.read_page=read_page
                return parent(*a,**kw)
            finally:
                shared.read_page=original
        return launch
kernel=TiledPages(1024)
