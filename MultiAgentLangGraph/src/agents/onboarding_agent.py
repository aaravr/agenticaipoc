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
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp/message")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")

class OnboardingAgent:
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
                "You are an onboarding workflow planner. Choose tool names **only** from the "
                "comma‑separated list below. *Never invent new names or change "
                "capitalisation.*\n\n"
                "Available tools: {tools}\n\n"
                "Task:\n{task}\n\n"
                "Last Response:\n{last_response}\n\n"
                "IMPORTANT RULES:\n"
                "1. Each tool call must include both 'name' and 'input' fields.\n"
                "2. The input must be a map object.\n"
                "3. For getTaskDetails, use taskDetailRequest with taskId in the input.\n"
                "4. For claimTask, use taskId from the getTaskDetails response.\n"
                "5. For completeTask, use taskId and taskKey from the getTaskDetails response.\n"
                "6. CRITICAL: For EACH Open task, you MUST follow this EXACT sequence:\n"
                "   a. getTaskDetails to get the task\n"
                "   b. claimTask to claim the Open task\n"
                "   c. completeTask to complete the claimed task\n"
                "   d. getTaskDetails again to check for more tasks\n"
                "7. CRITICAL: Never skip any step in the sequence for any Open task.\n"
                "8. CRITICAL: Only process tasks with status='Open'. Skip any tasks with other statuses.\n"
                "9. CRITICAL: The task status is in the getTaskDetails response. Check it before claiming.\n"
                "10. CRITICAL: If no Open tasks are found, the workflow is complete.\n"
                "11. CRITICAL: Pay attention to taskKey in the response. Different task types require different payloads:\n"
                "    - For TASK_1_070: Use this EXACT payload structure:\n"
                "      {{\"completeRequest\": {{\n"
                "        \"taskId\": \"[TASK-ID]\",\n"
                "        \"taskKey\": \"TASK_1_070\",\n"
                "        \"skipBsOutreach\": true,\n"
                "        \"skipBsOutreachComment\": \"Skipping outreach as per workflow requirements\",\n"
                "        \"outcomeVariable\": {{\n"
                "          \"prepareBsOutreachOutcome\": {{\n"
                "            \"label\": \"Skip Outreach\",\n"
                "            \"action\": \"skipOutreach\"\n"
                "          }}\n"
                "        }}\n"
                "      }}}}\n"
                "    - For all other tasks: Use standard taskAction payload:\n"
                "      {{\"completeRequest\": {{\n"
                "        \"taskId\": \"[TASK-ID]\",\n"
                "        \"taskKey\": \"[TASK-KEY]\",\n"
                "        \"taskAction\": {{\"label\": \"Complete\", \"action\": \"complete\"}}\n"
                "      }}}}\n"
                "12. CRITICAL: Always check the taskKey before constructing the completeTask payload\n"
                "13. CRITICAL: For TASK_1_070, NEVER use taskAction. Use the skipBsOutreach structure instead.\n\n"
                "Tool Call Sequence (REPEAT for EACH Open task):\n"
                "1. getTaskDetails: Get all tasks for a case\n"
                "2. Check task status in response:\n"
                "   - If status='Open': Process the task\n"
                "   - If status≠'Open': Skip and check next task\n"
                "3. For EACH Open task:\n"
                "   a. claimTask: Claim the task\n"
                "   b. completeTask: Complete the claimed task with appropriate payload based on taskKey\n"
                "   c. getTaskDetails: Check for more tasks\n"
                "4. If no more Open tasks, the workflow is complete\n\n"
                "Example Response Format:\n"
                "[\n"
                "  {{\"name\": \"getTaskDetails\", \"input\": {{\"taskDetailRequest\": {{\"taskId\": \"CAS-123\"}}}}}},\n"
                "  {{\"name\": \"claimTask\", \"input\": {{\"claimRequest\": {{\"taskId\": \"[TASK-ID-FROM-GETTASKDETAILS]\"}}}}}},\n"
                "  {{\"name\": \"completeTask\", \"input\": {{\"completeRequest\": {{\"taskId\": \"[TASK-ID]\", \"taskKey\": \"[TASK-KEY]\", \"taskAction\": {{\"label\": \"Complete\", \"action\": \"complete\"}}}}}}}},\n"
                "  {{\"name\": \"getTaskDetails\", \"input\": {{\"taskDetailRequest\": {{\"taskId\": \"CAS-123\"}}}}}}\n"
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
                verify=True
            )
            openai_config["http_client"] = httpx.Client(transport=transport)
                
        self.llm = OpenAI(**openai_config)
        self.planner = plan_prompt | self.llm

    def _ensure_session(self) -> None:
        """Ensure SSE subscription is active."""
        if self._session and self._sse_client:
            return

        from urllib.parse import urlsplit
        parts = urlsplit(self.mcp_url)
        sse_url = f"{parts.scheme}://{parts.netloc}/sse"

        self._session = requests.Session()
        if HTTP_PROXY or HTTPS_PROXY:
            self._session.proxies = {"no_proxy": "localhost,127.0.0.1"}

        resp = self._session.get(
            sse_url,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        self._sse_client = SSEClient(resp)
        self._event_iterator = self._sse_client.events()
        logger.info("✅ Subscribed to SSE stream at %s", sse_url)

    def _sse_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC request and wait for SSE response."""
        self._ensure_session()
        req_id = str(uuid.uuid4())
        rpc = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id,
            "params": params
        }

        resp = self._session.post(self.mcp_url, json=rpc, timeout=30)
        resp.raise_for_status()

        for event in self._event_iterator:
            if not event.data:
                continue
            try:
                payload = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            if payload.get("id") != req_id:
                continue
            return payload.get("result")
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server."""
        result = self._sse_request("tools/list", {})
        if not result or "tools" not in result:
            raise RuntimeError("No tools returned by MCP server")
        return result["tools"]

    def call_tool(self, tool_name: str, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool and handle its response."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Calling tool: {tool_name}")
        logger.info(f"Input object: {json.dumps(input_obj, indent=2)}")
        logger.info(f"{'='*80}")

        rpc_payload = {
            "name": tool_name,
            "arguments": input_obj
        }

        raw = self._sse_request("tools/call", rpc_payload)

        # Handle list response
        if isinstance(raw, list):
            if len(raw) > 0 and isinstance(raw[0], dict):
                if "isError" in raw[0] and raw[0]["isError"]:
                    err = raw[0].get("text", "Unknown tool error")
                    raise RuntimeError(f"Tool {tool_name} error: {err}")
                # If it's a list of content items, try to parse the first one
                if "text" in raw[0]:
                    try:
                        raw = json.loads(raw[0]["text"])
                    except json.JSONDecodeError:
                        logger.warning("Couldn't parse server content as JSON: %s", raw[0]["text"])
                        raw = {"status": "success", "content": raw}
            else:
                raw = {"status": "success", "content": raw}
        # Handle dictionary response
        elif isinstance(raw, dict):
            if "content" in raw:
                items = raw.get("content") or []
                if items and isinstance(items[0], dict) and "text" in items[0]:
                    txt = items[0]["text"]
                    try:
                        raw = json.loads(txt)
                    except json.JSONDecodeError:
                        logger.warning("Couldn't parse server content as JSON: %s", txt)
                if isinstance(raw, dict) and raw.get("isError", False):
                    err = items[0].get("text", "Unknown tool error")
                    raise RuntimeError(f"Tool {tool_name} error: {err}")

        if raw is None:
            raw = {"status": "success"}

        self._last_response = raw
        logger.info(f"\n{'='*80}")
        logger.info(f"Tool {tool_name} completed with result:")
        logger.info(json.dumps(raw, indent=2))
        logger.info(f"{'='*80}\n")
        return raw

    def execute_workflow(self, case_id: str) -> None:
        """Execute the onboarding workflow for a case."""
        logger.info(f"\n{'='*80}")
        logger.info("Starting onboarding workflow")
        logger.info(f"Case ID: {case_id}")
        logger.info(f"{'='*80}\n")

        # Get available tools
        tools = self.list_tools()
        tool_names = ", ".join(t["name"] for t in tools)
        logger.info(f"Available tools: {tool_names}")

        while True:
            # Get task details for the case
            logger.info("\nGetting task details...")
            get_task_details_result = self.call_tool("getTaskDetails", {
                "taskDetailRequest": {
                    "taskId": case_id
                }
            })
            logger.info(f"GetTaskDetails response: {json.dumps(get_task_details_result, indent=2)}")

            # Handle task details response
            tasks = []
            if isinstance(get_task_details_result, list):
                tasks = get_task_details_result
            elif isinstance(get_task_details_result, dict):
                if "taskDetails" in get_task_details_result:
                    tasks = [get_task_details_result["taskDetails"]]
                elif "content" in get_task_details_result:
                    content = get_task_details_result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        try:
                            tasks = json.loads(content[0].get("text", "{}")).get("tasks", [])
                        except json.JSONDecodeError:
                            logger.error("Failed to parse task details from content")
                            tasks = []

            if not tasks:
                logger.info("No more tasks found, workflow complete")
                break

            # Find first Open task
            open_task = None
            for task in tasks:
                if task.get("status") == "Open":
                    open_task = task
                    break
                else:
                    logger.info(f"Skipping task {task.get('taskId')} with status '{task.get('status')}'")

            if not open_task:
                logger.info("No more Open tasks found, workflow complete")
                break

            # Get task ID and key from details
            task_id = open_task.get("taskId")
            task_key = open_task.get("taskKey")

            if not task_id or not task_key:
                logger.error("Missing taskId or taskKey in task details")
                raise RuntimeError("Invalid task details response")

            logger.info(f"\nProcessing Open task: {json.dumps(open_task, indent=2)}")

            # Claim the task
            logger.info("\nClaiming task...")
            claim_result = self.call_tool("claimTask", {
                "claimRequest": {
                    "taskId": task_id
                }
            })
            # Check if claim was successful (no error and content contains "Done")
            if not isinstance(claim_result, dict) or claim_result.get("isError", True) or \
               not any(item.get("text") == "Done" for item in claim_result.get("content", [])):
                raise RuntimeError("Failed to claim task")
            logger.info("Task claimed successfully")

            # Complete the task with appropriate payload based on taskKey
            logger.info("\nCompleting task...")
            complete_payload = {
                "completeRequest": {
                    "taskId": task_id,
                    "taskKey": task_key
                }
            }

            # Let LLM determine the correct payload structure
            task_description = f"Complete task with ID {task_id} and key {task_key}"
            plan = self.planner.invoke({
                "tools": tool_names,
                "task": task_description,
                "last_response": json.dumps(open_task)
            })

            # Extract the completeTask payload from the plan
            if isinstance(plan, list):
                for tool_call in plan:
                    if isinstance(tool_call, dict) and tool_call.get("name") == "completeTask":
                        complete_payload = tool_call.get("input", complete_payload)
                        break

            complete_result = self.call_tool("completeTask", complete_payload)
            # Check if complete was successful (no error and content contains "Done")
            if not isinstance(complete_result, dict) or complete_result.get("isError", True) or \
               not any(item.get("text") == "Done" for item in complete_result.get("content", [])):
                raise RuntimeError("Failed to complete task")
            logger.info("Task completed successfully")

            logger.info(f"\nCompleted task {task_id}, checking for more tasks...")

        logger.info(f"\n{'='*80}")
        logger.info("Workflow execution completed - all Open tasks processed")
        logger.info(f"{'='*80}\n")

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
        return

    agent = OnboardingAgent()
    case_id = "CAS-dwdwd-dwedwedew"
    agent.execute_workflow(case_id)

if __name__ == "__main__":
    main() 