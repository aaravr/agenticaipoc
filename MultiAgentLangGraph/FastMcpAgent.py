# fast_mcp_client.py

import os
import json
import requests
import uuid
import httpx
from typing import List, Dict, Any, Optional
from sseclient import SSEClient
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
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
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")



class MCPAgent:
    def __init__(self, mcp_url: str = MCP_SERVER_URL):
        self.mcp_url = mcp_url
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self._setup_planner()
        self._session = None
        self._sse_client = None
        self._last_response = None

    def _setup_planner(self):
        plan_prompt = PromptTemplate(
            input_variables=["tools", "task", "last_response"],
            template=(
                "You are a workflow planner. Choose tool names **only** from the "
                "comma‑separated list below. *Never invent new names or change "
                "capitalisation.*\n\n"
                "Available tools: {tools}\n\n"
                "Task:\n{task}\n\n"
                "Last Response:\n{last_response}\n\n"
                "IMPORTANT RULES:\n"
                "1. Each tool call must include both 'name' and 'input' fields.\n"
                "2. The input must be a map object.\n"
                "3. For any operation involving client information, wrap the parameters inside a 'clientInfo' object.\n"
                "4. ALWAYS include ALL fields from the previous response in subsequent calls.\n"
                "5. NEVER drop or modify fields that were set by the server in previous responses.\n"
                "6. For qaVerify and approve calls, use the EXACT clientInfo from the previous response.\n\n"
                "Tool Call Sequence:\n"
                "1. saveClientInfo: Create new client\n"
                "2. qaVerify: Use FULL clientInfo from saveClientInfo response\n"
                "3. approve: Use FULL clientInfo from qaVerify response\n\n"
                "Example Response Format:\n"
                "[\n"
                "  {{\"name\": \"saveClientInfo\", \"input\": {{\"clientInfo\": {{\"name\": \"Test Client\", \"email\": \"test@email.com\"}}}}}},\n"
                "  {{\"name\": \"qaVerify\", \"input\": {{\"clientInfo\": {{\"clientId\": \"[PREVIOUS-CLIENT-ID]\", \"name\": \"[PREVIOUS-NAME]\", \"email\": \"[PREVIOUS-EMAIL]\", \"status\": \"[PREVIOUS-STATUS]\", \"action\": \"verify\"}}}}}},\n"
                "  {{\"name\": \"approve\", \"input\": {{\"clientInfo\": {{\"clientId\": \"[PREVIOUS-CLIENT-ID]\", \"name\": \"[PREVIOUS-NAME]\", \"email\": \"[PREVIOUS-EMAIL]\", \"status\": \"[PREVIOUS-STATUS]\", \"action\": \"approve\"}}}}}}\n"
                "]\n"
                "Do **not** add commentary or markdown."
            ),
        )
        
        # Configure OpenAI with proxy settings
        openai_config = {
            "api_key": OPENAI_API_KEY,
            "temperature": 0
        }
        
        # Add proxy settings only if they exist
        if HTTP_PROXY or HTTPS_PROXY:
            transport = httpx.HTTPTransport(
                proxy=HTTPS_PROXY or HTTP_PROXY,
                verify=True  # Set to False if you have SSL certificate issues
            )
            openai_config["http_client"] = httpx.Client(transport=transport)
                
        self.llm = OpenAI(**openai_config)
        # Use RunnableSequence instead of LLMChain
        self.planner = plan_prompt | self.llm

    # ------------------------------------------------------------------ #
    # Persistent SSE subscription (root‑level /sse)
    # ------------------------------------------------------------------ #
    def _ensure_session(self) -> None:
        """
        Spring AI MCP exposes an SSE endpoint at GET /sse. Open one subscription
        and reuse it for all calls, keeping a single events() iterator.
        """
        if self._session and self._sse_client:
            return  # already subscribed

        from urllib.parse import urlsplit
        parts = urlsplit(self.mcp_url)
        sse_url = f"{parts.scheme}://{parts.netloc}/sse"

        self._session = requests.Session()
        # don't proxy localhost
        if HTTP_PROXY or HTTPS_PROXY:
            self._session.proxies = {"no_proxy": "localhost,127.0.0.1"}

        resp = self._session.get(
            sse_url,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        # one SSEClient + one generator over its events
        self._sse_client = SSEClient(resp)
        self._event_iterator = self._sse_client.events()

        _logger.info("✅ Subscribed to SSE stream at %s", sse_url)

    def _create_jsonrpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 request"""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "id": str(uuid.uuid4()),
            "params": params
        }

    def _sse_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send a JSON-RPC request over POST and block on our single SSE stream
        until we see an event whose JSON payload has the same 'id'.
        """
        self._ensure_session()
        req_id = str(uuid.uuid4())
        rpc = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id,
            "params": params
        }

        # fire-and-forget the POST
        resp = self._session.post(self.mcp_url, json=rpc, timeout=30)
        resp.raise_for_status()

        # now read events until we find the matching JSON-RPC id
        for event in self._event_iterator:
            if not event.data:
                continue
            try:
                payload = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            if payload.get("id") != req_id:
                continue
            # found it!
            return payload.get("result")
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools via JSON-RPC over SSE."""
        result = self._sse_request("tools/list", {})
        if not result or "tools" not in result:
            raise RuntimeError("No tools returned by MCP server")
        return result["tools"]

    def call_tool(self, tool_name: str, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific tool over SSE, block until its JSON-RPC reply arrives,
        unwrap any content[].text JSON, and chain the last response forward.
        """
        _logger.info(f"\n{'='*80}")
        _logger.info(f"Calling tool: {tool_name}")
        _logger.info(f"Input object: {json.dumps(input_obj, indent=2)}")
        _logger.info(f"{'='*80}")

        # ─── Build the RPC args ────────────────────────────────────────────────
        if tool_name in ("qaVerify", "approve"):
            # for QA/approval, send the full last_response as clientInfo
            if not self._last_response:
                raise ValueError(f"No previous response to pass into {tool_name}")
            rpc_payload = {
                "name": tool_name,
                "arguments": {"clientInfo": self._last_response}
            }
            _logger.info(f"Using previous response for {tool_name}:")
            _logger.info(json.dumps(self._last_response, indent=2))
        else:
            # for saveClientInfo, updateClientInfo, etc. use whatever the planner gave us
            rpc_payload = {
                "name": tool_name,
                "arguments": input_obj
            }

        # ─── Fire & wait on SSE ───────────────────────────────────────────────
        raw = self._sse_request("tools/call", rpc_payload)

        # ─── Unwrap content[].text if present ─────────────────────────────────
        if isinstance(raw, dict) and "content" in raw:
            items = raw.get("content") or []
            if items and isinstance(items[0], dict) and "text" in items[0]:
                txt = items[0]["text"]
                try:
                    raw = json.loads(txt)
                except json.JSONDecodeError:
                    _logger.warning("Couldn't parse server content as JSON: %s", txt)
            # if the server flagged an error payload, raise
            if raw.get("isError", False):
                err = items[0].get("text", "Unknown tool error")
                raise RuntimeError(f"Tool {tool_name} error: {err}")

        # ─── Fallback default if nothing came back ─────────────────────────────
        if raw is None:
            if tool_name == "saveClientInfo":
                new_id = str(uuid.uuid4())
                _logger.info("No client ID returned; generated: %s", new_id)
                raw = {"id": new_id, "status": "success"}
            else:
                raw = {"status": "success"}

        # ─── Chain it ─────────────────────────────────────────────────────────
        self._last_response = raw
        _logger.info(f"\n{'='*80}")
        _logger.info(f"Tool {tool_name} completed with result:")
        _logger.info(json.dumps(raw, indent=2))
        _logger.info(f"{'='*80}\n")
        return raw
    
    def execute_task(self, task: str) -> None:
        """
        Execute a planner-derived sequence of tool calls, merging the previous
        call's response into each subsequent call's arguments.
        """
        _logger.info(f"\n{'='*80}")
        _logger.info("Starting task execution")
        _logger.info(f"Task: {task}")
        _logger.info(f"{'='*80}\n")

        # 1) Fetch available tools
        tools = self.list_tools()
        tool_names = ", ".join(t["name"] for t in tools)
        _logger.info(f"Available tools: {tool_names}")

        # 2) Ask the planner for a sequence of tool calls
        plan_json = self.planner.invoke({
            "tools": tool_names,
            "task": task,
            "last_response": json.dumps(self._last_response) if self._last_response else "No previous response"
        })
        tool_calls = json.loads(plan_json)
        
        _logger.info(f"\n{'='*80}")
        _logger.info("Generated tool call sequence:")
        _logger.info(json.dumps(tool_calls, indent=2))
        _logger.info(f"{'='*80}\n")

        # 3) Execute each call in order, merging in state from the last response
        for i, call in enumerate(tool_calls, 1):
            name = call["name"]
            arguments = call["input"]

            _logger.info(f"\n{'='*80}")
            _logger.info(f"Executing call {i}/{len(tool_calls)}: {name}")
            _logger.info(f"Initial arguments: {json.dumps(arguments, indent=2)}")

            # If we have a previous result, merge it into any clientInfo/updatedInfo
            if self._last_response:
                _logger.info(f"Previous response available: {json.dumps(self._last_response, indent=2)}")
                for key in ("clientInfo", "updatedInfo"):
                    if key in arguments and isinstance(arguments[key], dict):
                        # Base = full previous response, overrides = what the plan specified
                        merged = {**self._last_response, **arguments[key]}
                        arguments[key] = merged
                        _logger.info(f"Merged {key}: {json.dumps(merged, indent=2)}")

                        # For QA/approve, enforce the right action
                        if key == "clientInfo":
                            if name == "qaVerify":
                                arguments[key]["action"] = "verify"
                            elif name == "approve":
                                arguments[key]["action"] = "approve"
                            _logger.info(f"Set action for {name}: {arguments[key]['action']}")

            # 4) Call the tool (blocks until its SSE reply) and let call_tool update self._last_response
            result = self.call_tool(name, arguments)
            # self._last_response is now the dict returned by this call

            _logger.info(f"Call {i}/{len(tool_calls)} completed")
            _logger.info(f"{'='*80}\n")

        _logger.info(f"\n{'='*80}")
        _logger.info("Task execution completed")
        _logger.info(f"{'='*80}\n")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self._session:
            self._session.close()
        if self._sse_client:
            self._sse_client.close()


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



