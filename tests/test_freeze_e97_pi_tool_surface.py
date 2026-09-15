import json
import pytest
from scripts.freeze_e97_pi_tool_surface import EXPECTED,load_groups


def make(root,alter=None):
 for group,names in EXPECTED.items():
  d=root/group;d.mkdir()
  tools=[dict(name=n,label=n,description='d',parameters={'type':'object'}) for n in names]
  if alter and alter[0]==group:alter[1](tools)
  (d/'tools.json').write_text(json.dumps(dict(schema='emender-e97-active-pi-tool-surface-v1',tools=tools)))


def test_loads_exact_disjoint_surface(tmp_path):
 make(tmp_path);groups=load_groups(tmp_path)
 assert sum(len(g['tools']) for g in groups.values())==11
 assert [t['name'] for t in groups['web']['tools']]==EXPECTED['web']


def test_rejects_missing_or_reordered_tools(tmp_path):
 make(tmp_path,('core',lambda tools:tools.reverse()))
 with pytest.raises(ValueError,match='coverage/order'):load_groups(tmp_path)
