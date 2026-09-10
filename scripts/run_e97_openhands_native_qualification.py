#!/usr/bin/env python3
"""Deliver frozen qualification code through stdin to a bounded owned container."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import uuid
from scripts.audit_e97_open_swe_semantics import publish,sha

DOCKER=['docker','--host','unix:///var/run/docker.sock']
BOOTSTRAP='''import hashlib,io,json,os,runpy,sys,tarfile
from pathlib import Path
expected=json.loads(sys.argv[1]);size=int(sys.argv[2])
if not 0<size<=16777216:raise ValueError('bundle size')
payload=sys.stdin.buffer.read(size)
if len(payload)!=size:raise ValueError('short bundle')
root=Path('/tmp/e97-native-qualification');root.mkdir(mode=0o700)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
    members=archive.getmembers()
    if len(members)!=len(expected) or {m.name for m in members}!=set(expected):raise ValueError('bundle coverage')
    for member in members:
        if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:raise ValueError('bundle entry')
        data=archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest()!=expected[member.name]:raise ValueError('bundle hash')
        target=root/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
os.environ['E97_NATIVE_BUNDLE']=str(root);sys.path.insert(0,str(root))
runpy.run_path(str(root/'scripts/qualify_e97_openhands_native_execution.py'),run_name='__main__')
'''


def container_args(image,nonce,hashes,size):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image):raise ValueError('immutable image required')
    if not re.fullmatch(r'[0-9a-f]{32}',nonce):raise ValueError('invalid owner nonce')
    return DOCKER+['create','--name','e97-native-qual-'+nonce,'--label','emender.native-qualification='+nonce,
        '--interactive','--runtime','runc','--user','1000:1000','--network','none','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--pids-limit','128','--memory','4g','--memory-swap','4g','--cpus','2',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=536870912,mode=1777',
        '--tmpfs','/testbed:rw,nosuid,nodev,size=536870912,mode=0700,uid=1000,gid=1000',
        '--tmpfs','/workspace:rw,nosuid,nodev,size=536870912,mode=0700,uid=1000,gid=1000',
        '--env','USER=agent','--env','HOME=/tmp/agent-home','--env','TERM=xterm-256color',
        '--env','NO_CHANGE_TIMEOUT_SECONDS=10',image,'python','-c',BOOTSTRAP,json.dumps(hashes,sort_keys=True),str(size)]


def inspect_container(identity):
    return json.loads(subprocess.check_output(DOCKER+['inspect',identity],timeout=10))[0]


def validate_container(spec,image,nonce):
    host=spec['HostConfig'];config=spec['Config']
    if config['Image']!=image or config['User']!='1000:1000':raise ValueError('container identity/user')
    if config['Labels'].get('emender.native-qualification')!=nonce:raise ValueError('container ownership')
    if host['NetworkMode']!='none' or not host['ReadonlyRootfs'] or host['Privileged']:raise ValueError('container isolation')
    if host['Runtime']!='runc' or host['PidsLimit']!=128 or host['Memory']!=4*1024**3 or host['MemorySwap']!=4*1024**3 or host['NanoCpus']!=2_000_000_000:raise ValueError('container bounds')
    if host['CapDrop']!=['ALL'] or not any(x.startswith('no-new-privileges') for x in host['SecurityOpt']):raise ValueError('container privileges')
    if spec.get('Mounts') or host.get('Binds') or host.get('Devices') or host.get('DeviceRequests') or host.get('PortBindings'):raise ValueError('container forbidden exposure')
    if set(host['Tmpfs'])!={'/tmp','/testbed','/workspace'}:raise ValueError('container tmpfs')
    for path,options in host['Tmpfs'].items():
        required={'rw','nosuid','nodev','size=536870912'}
        required.update({'mode=1777'} if path=='/tmp' else {'mode=0700','uid=1000','gid=1000'})
        if set(options.split(','))!=required:raise ValueError('container tmpfs bounds')


def run(args):
    out=args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,mode=0o700)
    if sha(args.image_manifest)!=args.image_manifest_sha256:raise ValueError('image manifest identity')
    m=json.loads(args.image_manifest.read_text());image=m['image_id']
    for name,digest in m['files'].items():
        if sha(args.image_manifest.parent/name)!=digest:raise ValueError('image evidence changed')
    recipe=json.loads((args.bundle/'recipe.json').read_text());hashes=recipe['bundle_files']
    if recipe['image_id']!=image or recipe['image_manifest_sha256']!=args.image_manifest_sha256:
        raise ValueError('recipe_image_identity')
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:') as tar:
        for name,digest in sorted(hashes.items()):
            p=args.bundle/name
            if p.is_symlink() or not p.resolve().is_relative_to(args.bundle.resolve()) or sha(p)!=digest:raise ValueError('bundle identity')
            data=p.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o400
            tar.addfile(info,io.BytesIO(data))
    data=payload.getvalue();nonce=uuid.uuid4().hex;identity=None;result=None
    command=container_args(image,nonce,hashes,len(data))
    publish(out/'launch.json',{'image_manifest_sha256':args.image_manifest_sha256,'recipe_sha256':sha(args.bundle/'recipe.json'),
        'bundle_sha256':hashlib.sha256(data).hexdigest(),'owner_nonce':nonce,'argv':command,'timeout_seconds':600,'automatic_retry':False})
    try:
        identity=subprocess.check_output(command,timeout=30).decode().strip()
        if not re.fullmatch(r'[0-9a-f]{64}',identity):raise ValueError('invalid container id')
        before=inspect_container(identity);publish(out/'container-before.json',before)
        validate_container(before,image,nonce)
        with (out/'docker.stdout').open('wb') as stdout,(out/'docker.stderr').open('wb') as stderr:
            completed=subprocess.run(DOCKER+['start','--attach','--interactive',identity],input=data,
                stdout=stdout,stderr=stderr,timeout=600)
        after=inspect_container(identity);publish(out/'container-after.json',after)
        if completed.returncode or after['State']['ExitCode'] or after['State']['Running'] or after['State']['OOMKilled']:
            raise ValueError('qualification container failed; see retained stdout/stderr')
        lines=(out/'docker.stdout').read_text().splitlines();prefix='E97_NATIVE_EXECUTION_RESULT '
        summaries=[json.loads(line[len(prefix):]) for line in lines if line.startswith(prefix)]
        if len(summaries)!=1 or summaries[0].get('status')!='passed':raise ValueError('missing qualification result')
        result=summaries[0]
        result.update(image_id=image,image_manifest_sha256=args.image_manifest_sha256,recipe_sha256=sha(args.bundle/'recipe.json'))
        for name,digest in hashes.items():
            if sha(args.bundle/name)!=digest:raise ValueError('bundle changed during qualification')
    finally:
        if identity:
            owned=inspect_container(identity)
            if owned['Config']['Labels'].get('emender.native-qualification')!=nonce:raise ValueError('refuse cleanup of unowned container')
            if owned['State']['Running']:subprocess.run(DOCKER+['stop','--time','5',identity],check=True,timeout=10)
            subprocess.run(DOCKER+['rm',identity],check=True,timeout=5,stdout=subprocess.DEVNULL)
            publish(out/'cleanup.json',{'container_id':identity,'owner_nonce':nonce,'removed':True})
    if result is not None:
        publish(out/'summary.json',result)
        print('E97_NATIVE_EXECUTION_QUALIFIED '+str(len(result['checks']))+' checks',flush=True)


if __name__=='__main__':
    def terminated(signum,frame):raise TimeoutError('qualification interrupted')
    signal.signal(signal.SIGTERM,terminated)
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-manifest',type=Path,required=True);p.add_argument('--image-manifest-sha256',required=True)
    run(p.parse_args())
