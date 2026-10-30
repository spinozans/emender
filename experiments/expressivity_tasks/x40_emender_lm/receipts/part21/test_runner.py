"""Regression: failure/rerun/timeout retain the original console verbatim."""
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parent
PYTHON='/home/erikg/emender/.venv/bin/python'

class RunnerTests(unittest.TestCase):
    def invoke(self,name,seconds,code):
        result=subprocess.run(['bash',str(ROOT/'run.sh'),name,'host',str(seconds),PYTHON,'-c',code],
                              text=True,capture_output=True,timeout=20)
        self.assertTrue(result.stdout.strip(),result.stderr)
        return result,Path('/home/erikg/emender')/result.stdout.strip()

    def test_errors_and_repeat_preserve_log(self):
        first,a=self.invoke('runner_failure',5,"print('ORIGINAL-EVIDENCE',flush=True); raise RuntimeError('EXPECTED-FAILURE')")
        self.assertEqual(first.returncode,1)
        original=(a/'console.txt').read_bytes()
        self.assertIn(b'ORIGINAL-EVIDENCE',original)
        self.assertIn(b'EXPECTED-FAILURE',original)
        second,b=self.invoke('runner_failure',5,"print('SECOND-RUN',flush=True); raise SystemExit(9)")
        self.assertEqual(second.returncode,9)
        self.assertNotEqual(a,b)
        self.assertEqual((a/'console.txt').read_bytes(),original)
        self.assertIn(b'SECOND-RUN',(b/'console.txt').read_bytes())
        self.assertIn(hashlib.sha256(original).hexdigest(),(a/'console.sha256').read_text())

    def test_timeout_retains_partial_output(self):
        result,path=self.invoke('runner_timeout',1,"import time; print('BEFORE-TIMEOUT',flush=True); time.sleep(30)")
        self.assertEqual(result.returncode,124)
        self.assertEqual((path/'status.txt').read_text(),'124\n')
        self.assertIn(b'BEFORE-TIMEOUT',(path/'console.txt').read_bytes())

if __name__=='__main__':
    assert sys.executable==PYTHON
    unittest.main()
