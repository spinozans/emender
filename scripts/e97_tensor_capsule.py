"""Private, bounded operand snapshots preserving values, strides and aliases."""
import math
import torch

DTYPES={str(x):x for x in (torch.bfloat16,torch.float32,torch.float64,torch.int64,torch.int32,torch.bool,torch.uint8)}


def pack(tensors,max_bytes=256*1024**2):
    if type(max_bytes)!=int or max_bytes<=0:raise ValueError('capsule byte limit')
    stores={};descriptors={};ids={};total=0
    for name,t in tensors.items():
        if t is None:
            descriptors[name]=None;continue
        if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or str(t.dtype) not in DTYPES:
            raise ValueError('unsupported capsule tensor')
        if any(s<0 for s in t.stride()) or t.is_conj() or t.is_neg():raise ValueError('unsupported view flags/stride')
        storage=t.untyped_storage();identity=(str(t.device),storage.data_ptr(),storage.nbytes())
        if identity not in ids:
            total+=storage.nbytes()
            if total>max_bytes:raise ValueError('capsule storage bound')
            key=str(len(stores));ids[identity]=key
            raw=torch.empty(0,dtype=torch.uint8,device=t.device).set_(storage,0,(storage.nbytes(),),(1,))
            stores[key]=raw.detach().to('cpu',copy=True)
        descriptors[name]=dict(storage=ids[identity],dtype=str(t.dtype),shape=list(t.shape),stride=list(t.stride()),
                               offset=t.storage_offset(),original_address=t.data_ptr())
    return dict(schema='e97-tensor-capsule-v1',stores=stores,tensors=descriptors,bytes=total)


def unpack(capsule,device,max_bytes=256*1024**2):
    if type(max_bytes)!=int or max_bytes<=0:raise ValueError('capsule byte limit')
    if capsule['schema']!='e97-tensor-capsule-v1':raise ValueError('capsule schema')
    stores=capsule['stores'];total=sum(x.numel() for x in stores.values())
    if total!=capsule['bytes'] or total>max_bytes:raise ValueError('capsule storage bound')
    if any(x.dtype!=torch.uint8 or x.ndim!=1 or x.device.type!='cpu' for x in stores.values()):raise ValueError('capsule storage type')
    # Validate every descriptor before allocating anything on the destination.
    for d in capsule['tensors'].values():
        if d is None:continue
        if d['dtype'] not in DTYPES or d['storage'] not in stores:raise ValueError('capsule dtype/storage')
        item=torch.empty((),dtype=DTYPES[d['dtype']]).element_size()
        values=list(d['shape'])+list(d['stride'])+[d['offset']]
        if len(d['shape'])!=len(d['stride']) or any(type(v)!=int or v<0 for v in values):raise ValueError('capsule layout')
        end=d['offset'] if math.prod(d['shape'])==0 else d['offset']+sum((n-1)*s for n,s in zip(d['shape'],d['stride']))+1
        if end*item>stores[d['storage']].numel():raise ValueError('capsule view exceeds storage')
    result={};copies={k:v.to(device,copy=True) for k,v in stores.items()}
    for name,d in capsule['tensors'].items():
        if d is None:result[name]=None;continue
        dtype=DTYPES[d['dtype']];raw=copies[d['storage']]
        result[name]=torch.empty(0,dtype=dtype,device=device).set_(raw.untyped_storage(),d['offset'],d['shape'],d['stride'])
    return result
