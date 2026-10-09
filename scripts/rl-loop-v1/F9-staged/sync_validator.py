"""Regenerate self-contained v8 validator from retained v4 + staged mechanics."""
from pathlib import Path
S = Path(__file__).resolve().parent
p = S / 'scripts/e97_first_party_validator_diversity.py'
current = p.read_text()
tail = current[current.index('def _diversity_spec'):]
base = Path('/home/erikg/emender/scripts/e97_first_party_validator_protocol_breadth.py').read_text()
base = base[:base.index('def main() -> None:')]
mechanics = (S / 'scripts/e97_diversity.py').read_text()
mechanics = mechanics[mechanics.index('import hashlib'):].replace('        from scripts.e97_offline_shell import action_allowed, assertion_argv_allowed\n','').replace('    from scripts.e97_offline_shell import workspace_path\n','').replace('    from scripts.e97_offline_shell import assertion_argv_allowed\n','')
shell = (S / 'scripts/e97_offline_shell.py').read_text()
shell = shell[shell.index('import re'):].replace('from scripts.e97_diversity import safe_path\n','')
p.write_text(base + '\n# Additive era8 sealed end-state kind; legacy validator functions above remain unchanged.\n' + mechanics + '\n' + shell + '\n' + tail)
