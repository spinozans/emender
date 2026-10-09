"""Fail-closed era8 offline shell grammar, shared reference/policy gate.

Not a general shell sandbox: only these finite command forms are admitted.
No interpreter, substitution, environment expansion, external program loading,
symlink creation, networking or background processes. bwrap is absent on the
qualification host; this gate is required before Pi sees any new-family call.
"""
import re
import shlex

from scripts.e97_diversity import safe_path


def workspace_path(value):
    path = safe_path(value)
    if any(p.startswith('.git') for p in path.split('/')):
        raise ValueError('private git metadata path forbidden')
    return path


def _literal(value):
    if not isinstance(value, str) or any(c in value for c in ('$','`','\\','\x00','\n','\r')):
        raise ValueError('shell literal required')
    return value


def argv_allowed(argv):
    if not argv or len(argv) > 32:
        raise ValueError('bounded offline argv required')
    name, args = argv[0], argv[1:]
    if name in {'mkdir', 'rm', 'cp', 'mv', 'cat', 'sort', 'uniq', 'wc', 'head', 'tail', 'paste'}:
        flags = {'mkdir': {'-p'}, 'rm': {'-f', '-r', '-rf'}, 'cp': set(), 'mv': set(),
                 'cat': set(), 'sort': {'-n', '-r', '-u'}, 'uniq': {'-c'}, 'wc': {'-l', '-w', '-c'},
                 'head': set(), 'tail': set(), 'paste': set()}[name]
        paths = []
        for arg in args:
            if arg in flags:
                continue
            _literal(arg)
            paths.append(workspace_path(arg))
        if not paths or (name in {'cp', 'mv'} and len(paths) != 2):
            raise ValueError('offline file command paths missing')
    elif name == 'awk':
        # A deliberately small data-transform DSL: numeric CSV records,
        # print a field and bounded arithmetic; no arbitrary AWK program.
        if len(args) != 3 or args[0] != '-F,':
            raise ValueError('awk form outside offline grammar')
        if not re.fullmatch(r'/\^\[a-z\]\+,\[0-9\]\+\$/\{print \$1 ":" \$2[+*\-][0-9]{1,6}\}', args[1]):
            raise ValueError('awk program outside data-transform grammar')
        workspace_path(args[2])
    elif name == 'sed':
        if len(args) != 3 or args[0] != '-n' or not re.fullmatch(r'[0-9]{1,4}(,[0-9]{1,4})?p', args[1]):
            raise ValueError('sed form outside offline grammar')
        workspace_path(args[2])
    elif name == 'printf':
        if not args or len(args) > 8 or args[0] not in {'%s', '%s\n', '%s\\n'}:
            raise ValueError('printf format outside offline grammar')
        for arg in args[1:]:
            _literal(arg)
    elif name == 'test':
        if len(args) != 2 or args[0] not in {'-f','-s','-d'}:
            raise ValueError('test form outside offline grammar')
        workspace_path(args[1])
    elif name == 'git':
        # No commit, config, hooks, aliases, remote, clone, submodule, clean,
        # filters, textconv or external diff. New-family HOME/config are empty.
        if args == ['init'] or args in (['status', '--porcelain'], ['log', '--oneline']):
            return
        if args and args[0] == 'add' and len(args) >= 2:
            for arg in args[1:]:
                workspace_path(arg)
        else:
            raise ValueError('git form outside offline grammar')
    else:
        raise ValueError('command outside offline grammar')


def shell_allowed(command):
    if not isinstance(command, str) or not 1 <= len(command) <= 4096 or any(c in command for c in ('\x00', '\n', '\r')):
        raise ValueError('bounded offline shell required')
    lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>')
    lexer.whitespace_split = True
    lexer.commenters = ''
    tokens = list(lexer)
    if len(tokens) > 128:
        raise ValueError('shell token bound')
    command_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {';', '|'}:
            argv_allowed(command_tokens)
            command_tokens = []
        elif token == '>':
            argv_allowed(command_tokens)
            command_tokens = []
            i += 1
            if i >= len(tokens):
                raise ValueError('redirect target missing')
            workspace_path(tokens[i])
            if i+1 < len(tokens) and tokens[i+1] != ';':
                raise ValueError('redirect must end command')
            if i+1 < len(tokens):
                i += 1
        elif any(c in token for c in ('&', '<', '>')):
            raise ValueError('shell operator outside grammar')
        else:
            command_tokens.append(token)
        i += 1
    if command_tokens:
        argv_allowed(command_tokens)
    elif tokens and tokens[-1] in {';', '|'}:
        raise ValueError('trailing shell operator')
    if not tokens:
        raise ValueError('empty shell')


def assertion_argv_allowed(argv):
    argv_allowed(argv)
    if argv[0] not in {'test', 'cat', 'sort', 'uniq', 'wc', 'head', 'tail', 'paste', 'awk', 'sed'} and not (argv[0] == 'git' and argv[1:] in (['status', '--porcelain'], ['log', '--oneline'])):
        raise ValueError('assertion command must be read-only')


def action_allowed(tool, arguments):
    if tool in {'read','write','edit'}:
        workspace_path(arguments.get('path'))
    elif tool == 'bash':
        if set(arguments) - {'command', 'timeout'}:
            raise ValueError('bash arguments outside grammar')
        shell_allowed(arguments.get('command'))
        if 'timeout' in arguments and (type(arguments['timeout']) is not int or not 1 <= arguments['timeout'] <= 30):
            raise ValueError('bash timeout outside bounds')
    elif tool not in {'finish','think'}:
        raise ValueError('tool outside era8 offline surface')
