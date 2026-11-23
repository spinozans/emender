"""Read-only, matched-choice D payload substitution over one observed live tail."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from . import k15_bookmark_joint as joint


class LastByteSession:
    """Alter only the consumer's post-locate byte, retaining actual native replies."""
    def __init__(self, session):
        self.session = session
        self.last = None
        self.after_locate = False
        self.reads = []
        self.write_pairs = []

    def rpc(self, cmd):
        reply = self.session.rpc(cmd)
        op, args = cmd.split(' ', 1)
        if op == 'append':
            self.last = bytes.fromhex(args)[-1]
        else:
            self.reads.append([cmd, reply])
        if self.after_locate:
            assert op == 'byte' and self.last is not None
            self.write_pairs.append([reply['byte'], self.last])
            reply = dict(reply, byte=self.last)
        self.after_locate = op == 'locate'
        return reply


class LastBytePort(joint.InterventionPort):
    def forward(self, mixer, hidden, ids, session, forced, audit):
        self.observed = LastByteSession(session)
        return super().forward(mixer, hidden, ids, self.observed, forced, audit)


@torch.no_grad()
def main(out):
    assert joint._require_lease() in ('0', '1') and os.environ['OMP_NUM_THREADS'] == '4'
    d = out.with_name('k15_bookmarks_D')
    evidence = d/'eval.jsonl'
    assert joint.digest(evidence) == '04926d1a1e257f4bd3835ab6504ea41f04c8825bfbe9b9b9db61d52996cd5549'
    assert joint.digest(joint.EXE) == '87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    identity = joint.flagship_identity()
    assert identity['manifest_sha256'] == '48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'
    joint.emit(identity=identity)
    stream = joint.Stream(identity['source'], identity['manifest'])
    base, *_ = joint._build_k10(dict(joint.incumbent._tuned_winner()), 3401)
    net = joint.BookmarkK15(joint.convert_model_to_composition(base), joint.EXE)
    net.port = LastBytePort(net.port.byte.embedding_dim, net.port.action.in_features-net.port.byte.embedding_dim)
    net.load_state_dict(joint.load_d_model(d/'joint.pt')); net.cuda().eval()
    deltas, means = [], []
    writes = different = ext = changed_ext = 0
    with evidence.open() as source, (out/'payload.jsonl').open('x') as f, torch.autocast('cuda', dtype=torch.bfloat16):
        for line, (doc, start, ids, y) in zip(source, joint.rows(stream, 'final_test')):
            row = json.loads(line); assert start == 0 and doc == row['doc'] and bytes(ids[0].tolist()).hex() == row['raw']
            ref = row['modes']['none']; trace = ref['trace']
            z, tr, *_ = net(ids, forced=[(e['slot'], e['op']) for e in trace]); obs = net.port.observed
            assert obs.reads == [call for e in trace for call in e['calls']]
            assert [e.get('saved_pos') for e in tr] == [e.get('saved_pos') for e in trace]
            for e, alt in zip(trace, row['modes']['address']['trace'], strict=True):
                if e['op'] == 2:
                    ext += 1; changed_ext += e['calls'][0][1]['byte'] != alt['calls'][0][1]['byte']
            diff = sum(a != b for a,b in obs.write_pairs); writes += len(obs.write_pairs); different += diff
            loss = F.cross_entropy(z[0].float(), y, reduction='none').tolist()
            if not diff: assert loss == ref['loss']
            means.append(np.mean(loss)); deltas.append(means[-1]-np.mean(ref['loss']))
            f.write(json.dumps(dict(doc=doc, pairs=obs.write_pairs, loss=loss))+'\n'); f.flush()
            if len(means)%8 == 0: joint.emit(step=len(means))
    assert len(means) == 64
    sample = np.random.default_rng(3407).choice(deltas, (10000,64)).mean(1)
    joint.emit(documents=len(means), targets=len(means)*128, last_nll=np.mean(means), last_minus_true=np.mean(deltas), ci95=np.quantile(sample,[.025,.975]).tolist(), writes=writes, different_write_byte=different, extensions=ext, different_extension_operand_vs_address=changed_ext)


if __name__ == '__main__': main(Path(sys.argv[1]))
