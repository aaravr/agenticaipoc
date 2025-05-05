from typing import Dict, Any
from datetime import datetime
import json

class InteractionLogger:
    def __init__(self):
        self.interactions = []

    def log_interaction(self, 
                       agent_name: str, 
                       step_type: str, 
                       input_data: Dict[str, Any], 
                       output_data: Dict[str, Any],
                       tools_used: list = None):
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "step": step_type,
            "input": input_data,
            "output": output_data,
            "tools_used": tools_used or []
        }
        self.interactions.append(interaction)
        self._print_interaction(interaction)

    def _print_interaction(self, interaction: Dict):
        print("\n" + "="*50)
        print(f"🤖 Agent: {interaction['agent']}")
        print(f"⏰ Time: {interaction['timestamp']}")
        print(f"📋 Step: {interaction['step']}")
        
        # Only show relevant sections based on step type
        if interaction['step'] == "LLM_REQUEST":
            print("\n📤 Sending to LLM:")
            print(json.dumps(interaction['input'], indent=2))
        
        elif interaction['step'] == "LLM_RESPONSE":
            print("\n📥 Received from LLM:")
            print(json.dumps(interaction['output'], indent=2))
        
        elif interaction['step'] == "TOOL_EXECUTION":
            print("\n🔧 Executing Tool:")
            print(f"Tool Name: {interaction['input'].get('tool')}")
            print("Tool Input:")
            print(json.dumps(interaction['input'].get('input', {}), indent=2))
        
        elif interaction['step'] == "TOOL_RESULT":
            print("\n📊 Tool Result:")
            print(json.dumps(interaction['output'].get('result', {}), indent=2))
        
        if interaction['tools_used']:
            print("\n🔄 Tools Used:")
            for tool in interaction['tools_used']:
                print(f"  - {tool}")
        
        print("="*50 + "\n") 