"""Read-only, logical-token-aligned module traces. No model output is replaced."""
import hashlib
import importlib


class Rows:
    def __init__(self,model,length,layer=None,max_bytes=2*1024**3):
        import torch
        if type(length)!=int or not 0<length<=4607 or max_bytes<=0:raise ValueError('trace bounds')
        if layer is not None and (type(layer)!=int or not 0<=layer<len(model.layers)):raise ValueError('layer index')
        self.length=length;self.max_bytes=max_bytes;self.bytes=0;self.offsets={};self.counts={}
        self.bank={};self.stages=[];self.handles=[];self.restoration=None;self.calls={}
        try:
            if layer is None:
                self.tap(model.embedding,'embedding',inputs=False)
                for i,m in enumerate(model.layers):self.tap(m,f'{i:02d}.block')
            else:
                wrapper=model.layers[layer];m=wrapper.mixer
                if getattr(model,'fused_add_norm',False):self.outer_norm(model,layer)
                elif hasattr(model,'layer_norms'):self.tap(model.layer_norms[layer],'outer-norm')
                self.tap(wrapper,'block');self.tap(m,'mixer')
                for n in ('qkv_proj','a_proj','g_proj','erase_gate_proj','value_write_gate_proj'):
                    if not isinstance(getattr(m,n,None),torch.nn.Linear):raise ValueError('expected E97 projection')
                    self.tap(getattr(m,n),n)
                self.tap(m.o_proj,'o_proj');self.tap(wrapper.norm_2,'post-mixer-norm')
                self.tap(wrapper.mlp,'mlp')
                for n in ('w1','w2','w3'):self.tap(getattr(wrapper.mlp,n),f'mlp.{n}')
                # Topological stage order, not registration order. Paired branches
                # have no unique chronological ordering across execution layouts.
                tail=['mixer.output','post-mixer-norm.input','post-mixer-norm.output',
                    'mlp.input','mlp.w1.input','mlp.w1.output','mlp.w2.input','mlp.w2.output',
                    'mlp.w3.input','mlp.w3.output','mlp.output','block.output']
                self.stages=[s for s in self.stages if s not in tail]+tail
        except BaseException:
            self.close();raise

    def add(self,name):
        if name in self.stages:raise ValueError('duplicate trace site')
        self.stages.append(name)

    def store(self,name,value,width=None):
        import torch
        if value is not None:
            if value.device.type not in ('cpu','cuda') or value.ndim!=3 or value.shape[0]!=1 or value.shape[-1]>11520:
                raise ValueError('trace tensor shape/device')
            width=value.shape[1]
        if type(width)!=int or width<=0:raise ValueError('trace width')
        offset=self.offsets.get(name,0);self.offsets[name]=offset+width
        count=max(0,min(width,self.length-offset))
        if not count:return
        descriptor=dict(logical_start=offset,call_rows=width,captured_rows=count,absent=value is None)
        if value is not None:
            descriptor.update(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
                storage_offset=value.storage_offset(),address=value.data_ptr())
        self.calls.setdefault(name,[]).append(descriptor)
        self.counts[name]=self.counts.get(name,0)+count
        if value is None:
            if name in self.bank and self.bank[name] is not None:raise ValueError('mixed absent/present operand')
            self.bank[name]=None;return
        if name in self.bank and self.bank[name] is None:raise ValueError('mixed absent/present operand')
        self.bytes+=count*value.shape[-1]*value.element_size()
        if self.bytes>self.max_bytes:raise ValueError('trace payload budget')
        rows=value[0,:count].detach().to('cpu',copy=True).contiguous()
        if not bool(torch.isfinite(rows).all()):raise ValueError('nonfinite trace')
        self.bank.setdefault(name,[]).append(rows)

    def tap(self,module,name,inputs=True,outputs=True):
        if inputs:
            site=name+'.input';self.add(site)
            def before(_m,args,site=site):self.store(site,args[0]);return None
            self.handles.append(module.register_forward_pre_hook(before))
        if outputs:
            site=name+'.output';self.add(site)
            def after(_m,_args,value,site=site):
                self.store(site,value[0] if isinstance(value,tuple) else value);return None
            self.handles.append(module.register_forward_hook(after))

    def outer_norm(self,model,layer):
        owner=importlib.import_module('ndm.models.ladder_lm');original=owner.rms_norm_fn
        weight=model.layer_norms[layer].weight
        for stage in ('x','residual','normalized','residual-output'):self.add('outer-norm.'+stage)
        def observe(*args,**kwargs):
            selected=len(args)>1 and args[1] is weight
            if selected:
                self.store('outer-norm.x',args[0])
                self.store('outer-norm.residual',kwargs.get('residual'),args[0].shape[1])
            result=original(*args,**kwargs)
            if selected:
                if not isinstance(result,tuple) or len(result)!=2:raise ValueError('prenorm output')
                self.store('outer-norm.normalized',result[0]);self.store('outer-norm.residual-output',result[1])
            return result  # original objects, unchanged
        owner.rms_norm_fn=observe;self.restoration=(owner,original,observe)

    def finish(self):
        import torch
        if set(self.bank)!=set(self.stages) or any(self.counts.get(s)!=self.length for s in self.stages):
            raise ValueError('logical row/site coverage')
        result={s:None if self.bank[s] is None else torch.cat(self.bank[s]) for s in self.stages}
        self.bank.clear();return result

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        if self.restoration:
            owner,original,observer=self.restoration
            if owner.rms_norm_fn is not observer:raise ValueError('observer ownership changed')
            owner.rms_norm_fn=original;self.restoration=None


def digest(value):
    import torch
    if value is None:return 'none'
    return hashlib.sha256(str((str(value.dtype),list(value.shape))).encode()+value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(a,b):
    import torch
    if list(a)!=list(b):raise ValueError('site identity')
    report=[]
    for name,x in a.items():
        y=b[name];hx,hy=digest(x),digest(y)
        row=dict(site=name,exact=hx==hy,actor_sha256=hx,teacher_sha256=hy,first_token=None)
        if x is None or y is None:
            if (x is None)!=(y is None):raise ValueError('operand presence differs')
        else:
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('trace dtype/shape differs')
            row.update(shape=list(x.shape),dtype=str(x.dtype),absolute_max=0.,differing_elements=0)
            for start in range(0,len(x),128):
                u=x[start:start+128].contiguous();v=y[start:start+128].contiguous()
                bits=(u.view(torch.uint8)!=v.view(torch.uint8)).reshape(len(u),u.shape[1],u.element_size()).any(-1)
                if bool(bits.any()):
                    coordinates=bits.nonzero()
                    if row['first_token'] is None:
                        row['first_token']=start+int(coordinates[0,0]);row['first_feature']=int(coordinates[0,1])
                    row['differing_elements']+=int(bits.sum())
                    row['absolute_max']=max(row['absolute_max'],float((u.float()-v.float()).abs().max()))
        report.append(row)
    return report
