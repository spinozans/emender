import copy
import pytest
from scripts.e97_openhands_native_backend import OpenHandsNativeBackend
from scripts.run_e97_openhands_native_qualification import container_args,validate_container
from scripts.qualify_e97_openhands_native_execution import numbered_rows

IMAGE='sha256:'+'a'*64
NONCE='b'*32


def spec():
    return {'Config':{'Image':IMAGE,'User':'1000:1000','Labels':{'emender.native-qualification':NONCE}},
        'Mounts':[],'HostConfig':{'NetworkMode':'none','ReadonlyRootfs':True,'Privileged':False,'Runtime':'runc',
            'PidsLimit':128,'Memory':4*1024**3,'MemorySwap':4*1024**3,'NanoCpus':2_000_000_000,
            'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges'],'Binds':None,'Devices':[],'DeviceRequests':None,'PortBindings':{},
            'Tmpfs':{'/tmp':'rw,nosuid,nodev,size=536870912,mode=1777',
                '/testbed':'rw,nosuid,nodev,size=536870912,mode=0700,uid=1000,gid=1000',
                '/workspace':'rw,nosuid,nodev,size=536870912,mode=0700,uid=1000,gid=1000'}}}


def test_container_command_has_no_host_mounts_network_or_gpu():
    args=container_args(IMAGE,NONCE,{},1)
    assert args[:3]==['docker','--host','unix:///var/run/docker.sock']
    assert '--mount' not in args and '-v' not in args and '--gpus' not in args and '--privileged' not in args
    assert args[args.index('--network')+1]=='none'
    assert '--read-only' in args and args[args.index('--user')+1]=='1000:1000'
    validate_container(spec(),IMAGE,NONCE)
    with pytest.raises(ValueError):container_args('mutable:tag',NONCE,{},1)


@pytest.mark.parametrize('key,value',[
    ('NetworkMode','host'),('ReadonlyRootfs',False),('Privileged',True),('Runtime','nvidia'),
    ('Memory',8*1024**3),('MemorySwap',-1),('PidsLimit',0),('NanoCpus',0),
    ('CapDrop',[]),('SecurityOpt',[]),('Binds',['/:/host']),('Devices',[{'PathOnHost':'/dev/sda'}]),
    ('DeviceRequests',[{'Count':-1}]),('PortBindings',{'8000/tcp':[{'HostPort':'8000'}]}),
])
def test_container_inspection_fails_closed_on_relaxed_bounds(key,value):
    s=spec();s['HostConfig'][key]=value
    with pytest.raises(ValueError):validate_container(s,IMAGE,NONCE)


def test_tmpfs_and_owner_identity_are_checked():
    for alter in ('tmpfs','owner','image','user','mount'):
        s=spec()
        if alter=='tmpfs':s['HostConfig']['Tmpfs']['/tmp']='rw,size=9999999999'
        elif alter=='owner':s['Config']['Labels']['emender.native-qualification']='other'
        elif alter=='image':s['Config']['Image']='other'
        elif alter=='user':s['Config']['User']='0'
        else:s['Mounts']=[{'Source':'/home','Destination':'/host'}]
        with pytest.raises(ValueError):validate_container(s,IMAGE,NONCE)


def test_backend_refuses_host_before_importing_openhands(monkeypatch):
    def denied():raise RuntimeError('not a sandbox')
    monkeypatch.setattr('scripts.e97_openhands_native_backend.sandbox_evidence',denied)
    with pytest.raises(RuntimeError,match='not a sandbox'):OpenHandsNativeBackend()


def test_recipe_image_mismatch_is_rejected_before_docker(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts.audit_e97_open_swe_semantics import sha
    from scripts.run_e97_openhands_native_qualification import run
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'files':{},'image_id':IMAGE}))
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'recipe.json').write_text(json.dumps({'bundle_files':{},'image_id':'wrong','image_manifest_sha256':sha(manifest)}))
    def unexpected(*args,**kwargs):raise AssertionError('Docker must not run')
    monkeypatch.setattr('subprocess.check_output',unexpected)
    args=SimpleNamespace(output=tmp_path/'output',image_manifest=manifest,image_manifest_sha256=sha(manifest),bundle=bundle)
    with pytest.raises(ValueError,match='recipe_image_identity'):run(args)


def test_numbered_row_oracle_ignores_headers_and_preserves_indices():
    assert numbered_rows('file header\n     1\trow0001\n   500\trow0500\n')==[(1,'row0001'),(500,'row0500')]
