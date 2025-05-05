# fast_mcp_client.py

import os
import json
import requests
import uuid
from typing import List, Dict, Any
from sseclient import SSEClient
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuration
MCP_SERVER_URL = "http://localhost:8080/mcp/message"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MCPAgent:
    def __init__(self, mcp_url: str = MCP_SERVER_URL):
        self.mcp_url = mcp_url
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self._setup_planner()

    def _setup_planner(self):
        plan_prompt = PromptTemplate(
            input_variables=["tools", "task"],
            template=(
                "You are a task planning assistant. You MUST use ONLY the exact tool names provided below.\n\n"
                "Task to accomplish:\n"
                "{task}\n\n"
                "Available tools (use EXACTLY these names, no variations):\n"
                "{tools}\n\n"
                "Return a JSON array of tool calls in this format:\n"
                "[{{\"name\": \"exactToolName\", \"input\": {{}}}}, {{\"name\": \"exactToolName\", \"input\": {{}}}}]\n"
                "Do not include any explanations or markdown, just the JSON array.\n\n"
                "Example for client onboarding:\n"
                "[{{\"name\": \"saveClientInfo\", \"input\": {{\"clientInfo\": {{\"name\": \"John\", \"email\": \"john@example.com\", \"status\": \"NEW\"}}}}}}, {{\"name\": \"qaVerify\", \"input\": {{\"clientInfo\": {{\"name\": \"John\", \"email\": \"john@example.com\", \"status\": \"READY_FOR_QA\"}}}}}}, {{\"name\": \"approve\", \"input\": {{\"clientInfo\": {{\"name\": \"John\", \"email\": \"john@example.com\", \"status\": \"READY_FOR_APPROVAL\"}}}}}}]"
            )
        )
        self.llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
        self.planner = LLMChain(llm=self.llm, prompt=plan_prompt)

    def _create_jsonrpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 request"""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "id": str(uuid.uuid4()),
            "params": params
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server using SSE"""
        print(f"\nAttempting to fetch tools from MCP server at {self.mcp_url}")
        request = self._create_jsonrpc_request("tools/list", {})
        print("Sending request:", json.dumps(request, indent=2))
        
        try:
            response = requests.post(self.mcp_url, json=request, stream=True)
            response.raise_for_status()
            print("Received response status:", response.status_code)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data = line[5:].strip()  # Remove 'data:' prefix
                        try:
                            result = json.loads(data)
                            if "result" in result and "tools" in result["result"]:
                                tools = result["result"]["tools"]
                                print("Retrieved tools:", json.dumps(tools, indent=2))
                                return tools
                        except json.JSONDecodeError:
                            continue
            
            print("No tools found in response")
            return []
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request to MCP server: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error while fetching tools: {e}")
            return []

    def call_tool(self, tool_name: str, input_obj: Dict[str, Any]) -> Any:
        """Call a specific tool with the given input using SSE"""
        print(f"\nCalling tool {tool_name} at {self.mcp_url}")
        request = self._create_jsonrpc_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": input_obj
            }
        )
        print("Sending request:", json.dumps(request, indent=2))
        response = requests.post(self.mcp_url, json=request, stream=True)
        response.raise_for_status()
        print("Received response status:", response.status_code)
        
        client = SSEClient(response)
        last_result = None
        for event in client.events():
            if event.data:
                print("Received event data:", event.data)
                result = json.loads(event.data)
                if "result" in result:
                    last_result = result["result"]
                elif "error" in result:
                    print(f"Error from tool {tool_name}:", result["error"])
                    raise Exception(f"Tool error: {result['error']['message']}")
        
        if last_result is None:
            print("No result found in response")
            return None
            
        print(f"Tool {tool_name} completed with result:", json.dumps(last_result, indent=2))
        return last_result

    def execute_task(self, task: str) -> None:
        """Execute a task by planning and executing tool calls"""
        tools = self.list_tools()
        if not tools:
            print("Error: No tools available. Cannot proceed with task execution.")
            return

        # Format tools for prompt with explicit tool names
        tool_desc = "\n".join(f"Tool name: {t['name']}\nDescription: {t['description']}\n---" for t in tools)
        print("\nAvailable tools:", tool_desc)

        # Generate a plan
        plan_json = self.planner.run(tools=tool_desc, task=task)
        print("\nRaw plan from LLM:", plan_json)
        
        try:
            plan = json.loads(plan_json)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array")
            
            # Validate tool names against available tools
            available_tools = {tool['name'] for tool in tools}
            for step in plan:
                if step['name'] not in available_tools:
                    print(f"Error: Tool '{step['name']}' not found in available tools")
                    print(f"Available tools: {available_tools}")
                    return
                
        except json.JSONDecodeError as e:
            print(f"Error parsing plan JSON: {e}")
            print("Please ensure the LLM returns a valid JSON array")
            return

        # Execute each step
        for step in plan:
            if not isinstance(step, dict) or "name" not in step or "input" not in step:
                print(f"Invalid step format: {step}")
                continue
                
            name = step["name"]
            inp = step["input"]
            print(f"\n>>> Calling {name} with {inp}")
            try:
                result = self.call_tool(name, inp)
                print("Result:", result)
                # Convert step and result to string format for memory
                input_str = f"Tool call: {name} with input {json.dumps(inp)}"
                output_str = f"Tool result: {json.dumps(result)}"
                self.memory.save_context({"input": input_str}, {"output": output_str})
            except Exception as e:
                print(f"Error executing tool {name}: {str(e)}")
                raise

        print("\n✅ Workflow complete!")

def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        print("Example .env file content:")
        print("OPENAI_API_KEY=your_actual_openai_api_key_here")
        return
    
    agent = MCPAgent()
    task = "Onboard client with data name=Test client, email=test@email.com. You need to first create the client, then complete QA verification and  then approve the case using approval task"
    agent.execute_task(task)

if __name__ == "__main__":
    main()
