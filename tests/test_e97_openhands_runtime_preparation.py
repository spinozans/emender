import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_preparation_scripts_have_valid_shell_syntax():
    for name in ('prepare_e97_openhands_runtime_source.sh','build_e97_openhands_runtime_image.sh'):
        subprocess.run(['/bin/bash','-n',str(ROOT/'scripts'/name)],check=True,timeout=5)


def test_existing_output_is_rejected_before_git_or_docker(tmp_path):
    out=tmp_path/'existing';out.mkdir();marker=out/'keep';marker.write_text('original')
    env={**os.environ,'PYTHON_BIN':sys.executable,'UVX_BIN':'not-called',
         'DOCKERFILE':'not-called','SOURCE_MANIFEST_SHA256':'not-called','PATH':str(tmp_path)}
    commands=[['prepare_e97_openhands_runtime_source.sh',str(out)],
              ['build_e97_openhands_runtime_image.sh','not-called',str(out)]]
    for name,*args in commands:
        result=subprocess.run(['/bin/bash',str(ROOT/'scripts'/name),*args],env=env,capture_output=True,text=True,timeout=5)
        assert result.returncode==2 and 'command not found' not in result.stderr
        assert marker.read_text()=='original' and sorted(p.name for p in out.iterdir())==['keep']


def test_image_candidate_has_no_automatic_agent_or_tool_startup():
    text=(ROOT/'configs/pi/e97-openhands-native-runtime.Dockerfile').read_text()
    assert '--require-hashes' in text and 'USER 1000:1000' in text
    assert 'ARG BASE_IMAGE\nFROM ${BASE_IMAGE}' in text
    assert 'execution qualification required' in text
    script=(ROOT/'scripts/build_e97_openhands_runtime_image.sh').read_text()
    assert '--host unix:///var/run/docker.sock' in script
    assert '--network none --read-only --cap-drop ALL' in script
    assert "'execution_qualified':False,'training_eligible':False" in script
