"""Score the next byte of a single raw document using the trained controller + native XSA."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from .xsa_controller import Controller
from .xsa_controller_data import BINARY, digest


@torch.inference_mode()
def predict(exe, bundle, prefix):
    if len(prefix) < 16:
        raise ValueError('provide at least 16 observed bytes; there is no trained padding symbol')
    weights = bundle / 'controller.pt'
    receipt = json.loads((bundle / 'result.json').read_text())
    if digest(weights) != receipt['checkpoint_sha'] or digest(exe) != BINARY:
        raise ValueError('checkpoint or session executable differs from training receipt')
    net = Controller().eval()
    net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    with subprocess.Popen([exe, 'session', '8'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as server:
        def request(command):
            server.stdin.write((command+'\n').encode()); server.stdin.flush()
            return json.loads(server.stdout.readline())
        for i in range(0, len(prefix), 65536):
            end = min(i+65536, len(prefix))
            if request('append '+prefix[i:end].hex()) != {'visible': end}:
                raise RuntimeError('cursor mismatch')
        if request('seal') != {'sealed':len(prefix)}:
            raise RuntimeError('seal failed')
        reply = request('branch '+prefix[-8:].hex())
        server.stdin.close()
        if server.wait(timeout=10) or reply['end']:
            raise RuntimeError('single open document invariant failed')
    h = torch.tensor([reply['counts']], dtype=torch.float32)
    x = torch.tensor([list(prefix[-16:])]); p, gate, _, _ = net(x,h)
    vals, ids = p[0].topk(5)
    return dict(visible=len(prefix), suffix_hex=prefix[-8:].hex(), support=int(h.sum()),
                gate=float(gate[0]), next_bytes=[dict(hex=f'{i:02x}', probability=v)
                for i,v in zip(ids.tolist(),vals.tolist())])


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('exe',type=Path);a.add_argument('bundle',type=Path);a.add_argument('prefix',type=Path)
    args=a.parse_args(); print(json.dumps(predict(args.exe.resolve(),args.bundle,args.prefix.read_bytes()),indent=2))
