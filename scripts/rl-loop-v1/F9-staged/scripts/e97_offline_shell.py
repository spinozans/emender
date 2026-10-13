"""Fail-closed era8 offline shell grammar, shared reference/policy gate.

Not a general shell sandbox: only these finite command forms are admitted.
No interpreter, substitution, environment expansion, external program loading,
symlink creation, networking or background processes. bwrap is absent on the
qualification host; this gate is required before Pi sees any new-family call.
"""
import re
import shlex

from scripts.e97_diversity import safe_path

# Single source for raw and canonical bounds; sync_validator embeds this source.
MAX_SHELL_LENGTH = 4096


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


def _shell_tokens(command):
    """Lex the entire finite grammar, retaining word/operator identity.

    Only space/tab separate words. Quotes concatenate literal fragments; no
    escape processing or expansion is supported (even in double quotes).
    Operator runs are never words, including when adjacent to quoted data.
    """
    tokens = []
    i = 0
    operators = ';&|<>'
    while i < len(command):
        char = command[i]
        if char in ' \t':
            i += 1
            continue
        if char in operators:
            end = i + 1
            while end < len(command) and command[end] in operators:
                end += 1
            operator = command[i:end]
            if operator not in {';', '|', '>'}:
                raise ValueError('shell operator outside grammar')
            tokens.append(('operator', operator))
            i = end
            continue
        word = []
        while i < len(command) and command[i] not in ' \t' + operators:
            char = command[i]
            if char in {"'", '"'}:
                quote = char
                i += 1
                while i < len(command) and command[i] != quote:
                    char = command[i]
                    if quote == '"' and char in {'$', '`', '\\'}:
                        raise ValueError('raw shell expansion/escape forbidden')
                    word.append(char)
                    i += 1
                if i == len(command):
                    raise ValueError('unterminated shell quote')
            else:
                if char in '$`\\{}~*?[]()#' or char.isspace():
                    raise ValueError('raw shell expansion/grouping/whitespace forbidden')
                word.append(char)
            i += 1
        tokens.append(('word', ''.join(word)))
    if not tokens or len(tokens) > 128:
        raise ValueError('bounded nonempty shell tokens required')
    return tokens


def parse_shell(command):
    """Fully consume: command ((';' | '|') command)*, terminal > path.

    Each command is a finite approved argv form, optionally redirected (only
    before end or ';'). Quoted operator-looking data stays a word, never syntax.
    """
    if not isinstance(command, str) or not 1 <= len(command) <= MAX_SHELL_LENGTH or any(
            ord(c) < 32 and c != '\t' or ord(c) == 127 for c in command):
        raise ValueError('bounded offline shell required')
    tokens = _shell_tokens(command)
    parsed = []
    i = 0
    while i < len(tokens):
        argv = []
        while i < len(tokens) and tokens[i][0] == 'word':
            argv.append(tokens[i][1])
            i += 1
        argv_allowed(argv)
        target = None
        if i < len(tokens) and tokens[i] == ('operator', '>'):
            i += 1
            if i == len(tokens) or tokens[i][0] != 'word':
                raise ValueError('redirect target missing')
            target = workspace_path(tokens[i][1])
            i += 1
        separator = None
        if i < len(tokens):
            kind, separator = tokens[i]
            if kind != 'operator' or separator not in {';', '|'} or (target is not None and separator != ';'):
                raise ValueError('unsupported command suffix')
            i += 1
            if i == len(tokens):
                raise ValueError('trailing shell operator')
        parsed.append((tuple(argv), target, separator))
    return tuple(parsed)


def serialize_shell(parsed):
    """Emit only parsed syntax and single-quoted words, never original bytes."""
    def literal(value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    parts = []
    for argv, target, separator in parsed:
        parts.append(' '.join(literal(word) for word in argv))
        if target is not None:
            parts.extend(('>', literal(target)))
        if separator is not None:
            parts.append(separator)
    return ' '.join(parts)


def shell_allowed(command):
    canonical_command = serialize_shell(parse_shell(command))
    if len(canonical_command) > MAX_SHELL_LENGTH:
        raise ValueError('canonical offline shell exceeds length bound')
    return canonical_command


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
        command = shell_allowed(arguments.get('command'))
        if 'timeout' in arguments and (type(arguments['timeout']) is not int or not 1 <= arguments['timeout'] <= 30):
            raise ValueError('bash timeout outside bounds')
        return dict(arguments, command=command)
    elif tool not in {'finish','think'}:
        raise ValueError('tool outside era8 offline surface')

    return dict(arguments)
