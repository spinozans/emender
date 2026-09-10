"""Version-pinned OpenHands backend. Import/run only in the isolated image.

No emulation of shell state, editor behavior or observation formatting. The
upstream action parser, executor and conversation formatter own those semantics.
"""
import os
from pathlib import Path


def sandbox_evidence():
    status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    root=[line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4]=='/']
    facts={'nonroot':os.geteuid()!=0,'docker_marker':Path('/.dockerenv').exists(),
           'no_new_privileges':status.get('NoNewPrivs','').strip()=='1',
           'effective_capabilities_zero':int(status.get('CapEff','0').strip(),16)==0,
           'root_readonly':len(root)==1 and 'ro' in root[0][5].split(','),
           'only_loopback_interface':sorted(p.name for p in Path('/sys/class/net').iterdir())==['lo'],
           'docker_socket_absent':not Path('/var/run/docker.sock').exists(),
           'gpu_devices_absent':not list(Path('/dev').glob('nvidia*'))}
    if not all(facts.values()):raise RuntimeError('native_backend_requires_isolated_container: '+str(facts))
    return facts


class OpenHandsNativeBackend:
    def __init__(self):
        self.sandbox=sandbox_evidence()
        # Imports happen after the fail-closed sandbox check.
        Path(os.environ['HOME']).mkdir(parents=True,exist_ok=True)
        from openhands.runtime.action_execution_server import ActionExecutor
        from openhands.core.config import AgentConfig
        from openhands.memory.conversation_memory import ConversationMemory
        self.executor=ActionExecutor([],work_dir='/testbed',username='agent',user_id=1000,
                                     enable_browser=False,browsergym_eval_env=None)
        self.formatter=ConversationMemory(AgentConfig(),None)
        self.counter=0

    async def start(self):
        await self.executor.ainit()
        return self

    @staticmethod
    def declared_tools():
        from openhands.agenthub.codeact_agent.tools import create_cmd_run_tool,create_str_replace_editor_tool,ThinkTool,FinishTool
        return [create_cmd_run_tool(),ThinkTool,FinishTool,create_str_replace_editor_tool()]

    async def execute(self,name,arguments):
        if name not in ('execute_bash','str_replace_editor'):raise ValueError('not_external_native_tool')
        if not isinstance(arguments,dict):raise ValueError('nonobject_arguments')
        import json
        from litellm import ModelResponse
        from openhands.agenthub.codeact_agent.function_calling import response_to_actions
        from openhands.events.action import CmdRunAction,FileReadAction,FileEditAction
        self.counter+=1;call_id=f'e97-native-call-{self.counter:06d}'
        response=ModelResponse(id=call_id,model='e97-source-native',choices=[{'index':0,'finish_reason':'tool_calls',
            'message':{'role':'assistant','content':None,'tool_calls':[{'id':call_id,'type':'function',
                'function':{'name':name,'arguments':json.dumps(arguments,ensure_ascii=False,allow_nan=False)}}]}}])
        actions=response_to_actions(response)
        if len(actions)!=1 or not isinstance(actions[0],(CmdRunAction,FileReadAction,FileEditAction)):
            raise ValueError('unexpected_upstream_action')
        action=actions[0]
        observation=await self.executor.run_action(action)
        observation.tool_call_metadata=action.tool_call_metadata
        output={}
        # Upstream formatter, with no added truncation beyond upstream tool behavior.
        self.formatter._process_observation(observation,output,max_message_chars=None)
        message=output[call_id]
        if len(message.content)!=1 or not hasattr(message.content[0],'text'):raise ValueError('nontext_observation')
        text=message.content[0].text
        if len(text.encode())>8*1024*1024:raise ValueError('observation_byte_cap')
        return {'message':{'role':'tool','content':text,'reasoning_content':None,'think':None,'tool_calls':None},
                'action_type':type(action).__name__,'observation_type':type(observation).__name__,
                'exit_code':getattr(observation,'exit_code',None),
                'upstream_is_input':getattr(action,'is_input',None)}

    def close(self):
        self.executor.close()
