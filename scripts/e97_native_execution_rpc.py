"""Container-only RPC. No model, host workspace, oracle or GPU enters this process."""
import asyncio
import json
from pathlib import Path
import sys
from scripts.e97_openhands_native_backend import OpenHandsNativeBackend


def send(value):
    print('E97_RPC ' + json.dumps(value, ensure_ascii=False), flush=True)


async def main():
    backend = OpenHandsNativeBackend()
    initialized = False
    try:
        await backend.start()
        send(dict(ready=True, tools=backend.declared_tools(), sandbox=backend.sandbox))
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'setup':
                if initialized:
                    raise ValueError('duplicate setup')
                for name, text in request['files'].items():
                    path = Path(name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('fixture path')
                    path = Path('/testbed') / path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('x') as f:
                        f.write(text)
                initialized = True
                send(dict(id=request['id'], setup=True))
            elif request['op'] == 'execute' and initialized:
                try:
                    result = await backend.execute(**request['call'])
                except Exception as error:
                    # Do not fabricate an upstream observation or rewrite invalid arguments.
                    send(dict(id=request['id'], dispatch_error=type(error).__name__, detail=str(error)))
                else:
                    send(dict(id=request['id'], result=result))
            else:
                raise ValueError('invalid RPC operation')
    finally:
        backend.close()


if __name__ == '__main__':
    asyncio.run(main())
