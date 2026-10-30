"""Process-local cache-mask candidate; production source stays unchanged."""
import hashlib
from experiments.expressivity_tasks.x40_emender_lm import kernels

OLD = 'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live,0.)'
NEW = 'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live & (r_touch<Start+t-1),0.)'


def install(*, store_layout=False):
    source = kernels.scan_kernel.src
    assert source.count(OLD) == 1
    replacement = source.replace(OLD, NEW)
    if store_layout:
        stores = {
            'tl.store(ReadValues+r_header*DV+r_d,r_partial,r_d<DV)':
                'tl.store(tl.broadcast_to(ReadValues+r_header*DV+r_d[None,:],(32,BV)),tl.broadcast_to(r_partial[None,:],(32,BV)),(tl.arange(0,32)[:,None]==0)&(r_d[None,:]<DV))',
            'tl.store(Output+(n*T+t)*DV+rr_d,rr_read,rr_d<DV)':
                'tl.store(tl.broadcast_to(Output+(n*T+t)*DV+rr_d[None,:],(BT,8)),tl.broadcast_to(rr_read[None,:],(BT,8)),(rr_tiles[:,None]==0)&(rr_d[None,:]<DV))',
        }
        for old, new in stores.items():
            assert replacement.count(old) == 1
            replacement = replacement.replace(old, new)
    kernels.scan_kernel._unsafe_update_src(replacement)
    kernels.scan_kernel.hash = None
    return dict(before=hashlib.sha256(source.encode()).hexdigest(),
                after=hashlib.sha256(replacement.encode()).hexdigest(),
                mask_only=not store_layout, store_layout=store_layout,
                barriers_added=0, production_changed=False)
