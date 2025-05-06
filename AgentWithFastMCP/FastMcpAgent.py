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

import logging

# module‑level logger
_logger = logging.getLogger("MCPAgent")
if not _logger.handlers:
    logging.basicConfig(level=logging.INFO)

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
                "You are a workflow planner. Choose tool names **only** from the "
                "comma‑separated list below. *Never invent new names or change "
                "capitalisation.*\n\n"
                "Available tools: {tools}\n\n"
                "Task:\n{task}\n\n"
                "Return a JSON array; for example:\n"
                "[{{\"name\": \"saveClientInfo\", \"input\": {{}}}}]\n"
                "Do **not** add commentary or markdown."
            ),
        )
        self.llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
        self.planner = LLMChain(llm=self.llm, prompt=plan_prompt)

    # ------------------------------------------------------------------ #
    # Persistent SSE subscription (root‑level /sse)
    # ------------------------------------------------------------------ #
    def _ensure_session(self) -> None:
        """
        Spring AI MCP keeps an `SseEmitter` open at **GET /sse** (root path).
        We create one connection and reuse it for all JSON‑RPC calls.
        """
        if hasattr(self, "_sse_client"):
            return  # already connected

        from urllib.parse import urlsplit

        parts = urlsplit(self.mcp_url)
        sse_url = f"{parts.scheme}://{parts.netloc}/sse"
        resp = requests.get(
            sse_url,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
        self._sse_client = SSEClient(resp)
        _logger.info("✅ Subscribed to SSE stream at %s", sse_url)

    def _create_jsonrpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 request"""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "id": str(uuid.uuid4()),
            "params": params
        }

    def _sse_request(self, method: str, params: Dict[str, Any]):
        self._ensure_session()
        req_id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "method": method, "id": req_id, "params": params}
        requests.post(self.mcp_url, json=payload, timeout=30).raise_for_status()

        for event in self._sse_client.events():
            if not event.data:
                continue
            try:
                payload = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            if payload.get("id") != req_id:
                continue
            yield payload

    def list_tools(self) -> List[Dict[str, Any]]:
        for pay in self._sse_request("tools/list", {}):
            tools = pay.get("result", {}).get("tools") or pay.get("tools")
            if tools:
                return tools
        raise RuntimeError("No tools returned by MCP server")

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
                    # Stop once we have the first concrete result payload
                    break
                elif "error" in result:
                    print(f"Error from tool {tool_name}:", result["error"])
                    raise Exception(f"Tool error: {result['error']['message']}")
        print(f"Tool {tool_name} completed with result:", json.dumps(last_result, indent=2))
        return last_result

    def execute_task(self, task: str) -> None:
        """Execute a task by planning and executing tool calls"""
        tools = self.list_tools()
        tool_names = ", ".join(t["name"] for t in tools)

        # Generate a plan
        plan_json = self.planner.run(tools=tool_names, task=task)
        print("\nRaw plan from LLM:", plan_json)

        try:
            plan = json.loads(plan_json)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array")
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
