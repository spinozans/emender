import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from experiments.expressivity_tasks.x43_fork_resolved.flagship_union_service import phase_windows
from experiments.expressivity_tasks.x43_fork_resolved.c14_documents import DocumentWindows

class FlagshipTests(unittest.TestCase):
    def test_offsets_splits_tails_and_causal_reset(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);raw=b'abcdefXYZ12345';(p/'source').write_bytes(raw)
            rows=[dict(document=0,byte_offset=0,byte_length=6,sha256=hashlib.sha256(raw[:6]).hexdigest(),split='train'),dict(document=6,byte_offset=6,byte_length=8,sha256=hashlib.sha256(raw[6:]).hexdigest(),split='final_test')]
            with gzip.open(p/'docs.gz','wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            a=dict(context=3,source=dict(path=str(p/'source')),manifest=str(p/'docs.gz'))
            self.assertEqual(list(phase_windows(a,'train')),[(0,0,[b'a',b'ab',b'abc']),(0,3,[b'd',b'de'])])
            self.assertEqual(list(phase_windows(a,'final_test')),[(6,0,[b'X',b'XY',b'XYZ']),(6,3,[b'1',b'12',b'123']),(6,6,[b'4'])])
            (p/'source').write_bytes(b'z'+raw[1:])
            with self.assertRaises(ValueError):list(phase_windows(a,'train'))
    def test_cap_never_reads_future(self):
        raw=b'a'*1024+b'bcd'
        self.assertEqual(DocumentWindows.causal_context(raw,1026,0),raw[2:1026])

class CodecTests(unittest.TestCase):
    def test_t2048_and_1024_context_roundtrip(self):
        from io import BytesIO
        from experiments.expressivity_tasks.x43_fork_resolved.flagship_cache import MAGIC,write_window,read_windows
        from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import select,features
        rows,_,_=select([b'a'*min(i,1024) for i in range(1,2049)],lambda ps:[dict(width=7,eof=0,branches=[(97,7)]) for _ in ps])
        out=BytesIO();out.write(MAGIC);write_window(out,100000000071,2048,rows)
        decoded=list(read_windows(BytesIO(out.getvalue())))
        self.assertEqual(decoded,[(100000000071,2048,rows)])
        self.assertEqual([features(r) for r in decoded[0][2]],[features(r) for r in rows])

class OracleTests(unittest.TestCase):
    def test_verified_oracle_accepts_native_tuple_branches(self):
        from unittest.mock import patch
        from experiments.expressivity_tasks.x43_fork_resolved.flagship_union_service import oracle_check, ROOT
        oracle=json.loads((ROOT/'receipts/p100_query_verification/native.json').read_text())
        native=[dict(row,branches=[tuple(p) for p in row['branches']]) for row in oracle]
        class Server:
            def query(self, patterns):
                assert len(patterns)==17
                return native
        oracle_check(Server())
        native[0]['width']+=1
        with self.assertRaises(AssertionError):oracle_check(Server())
