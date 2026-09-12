"""Read-only first-mixer component traces over the entire real causal prefix."""
STAGES=('wrapper-input','mixer-input','qkv-input','q-raw','k-raw','v-raw','alpha-raw',
        'gate-raw','erase-raw','write-raw','output-projection-input','output-projection-output','mixer-output')


class FirstMixerProbe:
    def __init__(self,model,length,max_bytes=512*1024**2):
        import torch
        if not 0<length<=4607 or max_bytes<=0:raise ValueError('component trace bounds')
        self.length=length;self.max_bytes=max_bytes;self.bytes=0;self.total=0;self.start=0
        self.offsets={};self.bank={};self.handles=[];self.chunk_lengths=[]
        layer=model.layers[0];m=layer.mixer;self.key_dim=m.key_dim;self.value_dim=m.value_dim
        names=('qkv_proj','a_proj','g_proj','erase_gate_proj','value_write_gate_proj','o_proj')
        if any(not isinstance(getattr(m,n,None),torch.nn.Linear) for n in names):raise ValueError('expected first-mixer linear projections')
        if max(self.key_dim,self.value_dim)>3840:raise ValueError('component feature bound')
        try:
            self.handles.append(layer.register_forward_pre_hook(self.begin))
            self.handles.append(m.register_forward_pre_hook(self.input_hook('mixer-input')))
            self.handles.append(m.qkv_proj.register_forward_hook(self.qkv))
            for name,stage in (('a_proj','alpha-raw'),('g_proj','gate-raw'),('erase_gate_proj','erase-raw'),('value_write_gate_proj','write-raw')):
                self.handles.append(getattr(m,name).register_forward_hook(self.output_hook(stage)))
            self.handles.append(m.o_proj.register_forward_pre_hook(self.input_hook('output-projection-input')))
            self.handles.append(m.o_proj.register_forward_hook(self.output_hook('output-projection-output')))
            self.handles.append(m.register_forward_hook(self.output_hook('mixer-output')))
        except BaseException:
            self.close();raise

    def begin(self,module,inputs):
        self.start=self.total;self.total+=inputs[0].shape[1];self.offsets={}
        self.store('wrapper-input',inputs[0])
        return None

    def store(self,stage,tensor):
        import torch
        if tensor.ndim!=3 or tensor.shape[0]!=1 or tensor.shape[-1]>3840:raise ValueError('component shape')
        offset=self.offsets.get(stage,0);self.offsets[stage]=offset+tensor.shape[1]
        count=max(0,min(tensor.shape[1],self.length-self.start-offset))
        if count:
            self.bytes+=count*tensor.shape[-1]*tensor.element_size()
            if self.bytes>self.max_bytes:raise ValueError('component storage budget')
            row=tensor[0,:count].detach().cpu().clone()
            if not bool(torch.isfinite(row).all()):raise ValueError('nonfinite component')
            self.bank.setdefault(stage,[]).append(row)

    def qkv(self,module,inputs,output):
        if output.shape[-1]!=2*self.key_dim+self.value_dim:raise ValueError('QKV layout')
        self.chunk_lengths.append(output.shape[1]);self.store('qkv-input',inputs[0])
        self.store('q-raw',output[...,:self.key_dim]);self.store('k-raw',output[...,self.key_dim:2*self.key_dim])
        self.store('v-raw',output[...,2*self.key_dim:])
        return None

    def input_hook(self,stage):
        def hook(module,inputs):
            self.store(stage,inputs[0]);return None
        return hook

    def output_hook(self,stage):
        def hook(module,inputs,output):
            self.store(stage,output[0] if isinstance(output,tuple) else output);return None
        return hook

    def finish(self):
        import torch
        if set(self.bank)!=set(STAGES):raise ValueError('component site coverage')
        result={k:torch.cat(self.bank[k]) for k in STAGES}
        if any(len(v)!=self.length for v in result.values()):raise ValueError('component row coverage')
        self.bank.clear()
        return result

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]
