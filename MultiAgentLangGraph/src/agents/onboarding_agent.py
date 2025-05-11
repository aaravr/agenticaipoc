import os
import json
import requests
import uuid
import httpx
from typing import List, Dict, Any, Optional
from sseclient import SSEClient
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
from datetime import datetime

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
        self.memory = ConversationBufferWindowMemory(
            memory_key="history",
            return_messages=True,
            output_key="output",
            k=1  # Only keep last message
        )
        self._setup_planner()
        self._session = None
        self._sse_client = None
        self._last_response = None
        self._current_task = None
        self._processed_tasks = set()

    def _setup_planner(self):
        plan_prompt = PromptTemplate(
            input_variables=["tools", "task", "last_response", "history"],
            template=(
                "You are an onboarding workflow planner. Use tools from: {tools}\n\n"
                "Task:\n{task}\n\n"
                "Last Response:\n{last_response}\n\n"
                "Previous Actions:\n{history}\n\n"
                "CRITICAL RULES:\n"
                "1. Each tool call must have 'name' and 'input' fields.\n"
                "2. For getTaskDetails: use taskDetailRequest with taskId\n"
                "3. For claimTask: use taskId from getTaskDetails response\n"
                "4. For completeTask: use taskId and taskKey from getTaskDetails\n"
                "5. For TASK_1_070, use this EXACT payload:\n"
                "{{\"completeRequest\": {{\n"
                "  \"taskId\": \"[TASK-ID]\",\n"
                "  \"taskKey\": \"TASK_1_070\",\n"
                "  \"skipBsOutreach\": true,\n"
                "  \"skipBsOutreachComment\": \"Skipping outreach as per workflow requirements\",\n"
                "  \"outcomeVariable\": {{\n"
                "    \"prepareBsOutreachOutcome\": {{\n"
                "      \"label\": \"Skip Outreach\",\n"
                "      \"action\": \"skipOutreach\"\n"
                "    }}\n"
                "  }}\n"
                "}}}}\n"
                "6. For other tasks, use standard payload:\n"
                "{{\"completeRequest\": {{\n"
                "  \"taskId\": \"[TASK-ID]\",\n"
                "  \"taskKey\": \"[TASK-KEY]\",\n"
                "  \"taskAction\": {{\"label\": \"Complete\", \"action\": \"complete\"}}\n"
                "}}}}\n"
                "7. NEVER use taskAction for TASK_1_070\n"
                "8. Check history to avoid processing same task twice\n\n"
                "Tool Call Sequence:\n"
                "1. getTaskDetails\n"
                "2. claimTask (if status='Open')\n"
                "3. completeTask (with appropriate payload)\n"
                "4. getTaskDetails (check for more tasks)\n\n"
                "Example Response:\n"
                "[\n"
                "  {{\"name\": \"getTaskDetails\", \"input\": {{\"taskDetailRequest\": {{\"taskId\": \"CAS-123\"}}}}}},\n"
                "  {{\"name\": \"claimTask\", \"input\": {{\"claimRequest\": {{\"taskId\": \"[TASK-ID]\"}}}}}},\n"
                "  {{\"name\": \"completeTask\", \"input\": {{\"completeRequest\": {{\"taskId\": \"[TASK-ID]\", \"taskKey\": \"[TASK-KEY]\", \"taskAction\": {{\"label\": \"Complete\", \"action\": \"complete\"}}}}}}}},\n"
                "  {{\"name\": \"getTaskDetails\", \"input\": {{\"taskDetailRequest\": {{\"taskId\": \"CAS-123\"}}}}}}\n"
                "]\n"
                "Do **not** add commentary or markdown."
            ),
        )
        
        # Configure OpenAI with proxy settings
        openai_config = {
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4-turbo",
            "temperature": 0
        }
        
        # Add proxy settings only if they exist
        if HTTP_PROXY or HTTPS_PROXY:
            transport = httpx.HTTPTransport(
                proxy=HTTPS_PROXY or HTTP_PROXY,
                verify=True
            )
            openai_config["http_client"] = httpx.Client(transport=transport)
                
        self.llm = ChatOpenAI(**openai_config)
        self.planner = plan_prompt | self.llm

    def _truncate_json(self, data: Any, max_length: int = 1000) -> Any:
        """Truncate long JSON responses while preserving structure."""
        if isinstance(data, dict):
            return {k: self._truncate_json(v, max_length) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._truncate_json(item, max_length) for item in data]
        elif isinstance(data, str):
            if len(data) > max_length:
                return data[:max_length] + "..."
            return data
        return data

    def _summarize_task_details(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Create a minimal summary of task details."""
        return {
            "taskId": task_details.get("taskId"),
            "taskKey": task_details.get("taskKey"),
            "status": task_details.get("status")
        }

    def _save_to_memory(self, action: str, details: Dict[str, Any]) -> None:
        """Save action and details to memory with truncation."""
        # Truncate details before saving
        truncated_details = self._truncate_json(details)
        
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": len(self._processed_tasks) + 1,
            "action": action,
            **truncated_details
        }
        
        self.memory.save_context(
            {"input": f"Action: {action}"},
            {"output": json.dumps(memory_entry)}
        )
        
        if action == "task_processed":
            self._processed_tasks.add(details.get("taskId"))
            logger.info(f"Task {details.get('taskId')} processed and saved to memory")

    def _is_task_processed(self, task_id: str) -> bool:
        """Check if a task has already been processed."""
        return task_id in self._processed_tasks

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
            
            # Truncate task details
            truncated_result = self._truncate_json(get_task_details_result)
            logger.info(f"GetTaskDetails response: {json.dumps(truncated_result, indent=2)}")
            self._save_to_memory("get_task_details", {"caseId": case_id, "result": truncated_result})

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

            # Find first Open task that hasn't been processed
            open_task = None
            for task in tasks:
                task_id = task.get("taskId")
                if task.get("status") == "Open" and not self._is_task_processed(task_id):
                    open_task = task
                    break
                else:
                    logger.info(f"Skipping task {task_id} with status '{task.get('status')}'")

            if not open_task:
                logger.info("No more Open tasks found, workflow complete")
                break

            # Get task ID and key from details
            task_id = open_task.get("taskId")
            task_key = open_task.get("taskKey")

            if not task_id or not task_key:
                logger.error("Missing taskId or taskKey in task details")
                raise RuntimeError("Invalid task details response")

            # Use truncated task details
            truncated_task = self._truncate_json(open_task)
            logger.info(f"\nProcessing Open task: {json.dumps(truncated_task, indent=2)}")
            self._current_task = truncated_task

            # Let LLM determine the correct payload structure
            task_description = f"Complete task with ID {task_id} and key {task_key}"
            logger.info("\nGetting LLM plan for task completion...")
            
            # Use JSON-safe memory history for LLM input
            llm_history = self.get_memory_history()
            llm_input = {
                "tools": tool_names,
                "task": task_description,
                "last_response": json.dumps(truncated_task),
                "history": llm_history
            }
            
            # Get LLM plan and extract content from AIMessage
            llm_response = self.planner.invoke(llm_input)
            plan = json.loads(llm_response.content) if hasattr(llm_response, 'content') else llm_response
            
            # Log raw LLM input and output
            logger.info("\nLLM Input:")
            logger.info(json.dumps(llm_input, indent=2))
            logger.info("\nLLM Output (Plan):")
            logger.info(json.dumps(plan, indent=2))
            
            # Save LLM interaction to memory (JSON-safe)
            self._save_to_memory("llm_interaction", {
                "taskId": task_id,
                "taskKey": task_key,
                "input": llm_input,
                "output": plan
            })

            # Extract the completeTask payload from the plan
            complete_payload = {
                "completeRequest": {
                    "taskId": task_id,
                    "taskKey": task_key
                }
            }

            if isinstance(plan, list):
                for tool_call in plan:
                    if isinstance(tool_call, dict) and tool_call.get("name") == "completeTask":
                        complete_payload = tool_call.get("input", complete_payload)
                        logger.info("\nSelected payload from LLM plan:")
                        logger.info(json.dumps(complete_payload, indent=2))
                        break

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
            self._save_to_memory("task_claimed", {"taskId": task_id, "result": claim_result})

            # Complete the task with appropriate payload based on taskKey
            logger.info("\nCompleting task...")
            complete_result = self.call_tool("completeTask", complete_payload)
            # Check if complete was successful (no error and content contains "Done")
            if not isinstance(complete_result, dict) or complete_result.get("isError", True) or \
               not any(item.get("text") == "Done" for item in complete_result.get("content", [])):
                raise RuntimeError("Failed to complete task")
            logger.info("Task completed successfully")
            self._save_to_memory("task_completed", {
                "taskId": task_id,
                "taskKey": task_key,
                "payload": complete_payload,
                "result": complete_result
            })
            self._save_to_memory("task_processed", {"taskId": task_id})

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

    def get_memory_history(self) -> List[Dict[str, Any]]:
        """Get the complete memory history as a list of actions and their details."""
        memory_vars = self.memory.load_memory_variables({})
        history = []

        for message in memory_vars.get("history", []):
            # Convert message object to dict or string
            if hasattr(message, "content"):
                try:
                    content = json.loads(message.content)
                    history.append(content)
                except Exception:
                    # If not JSON, add as string
                    history.append({"content": str(message.content)})
            else:
                # If message is not an object with 'content', just add its string representation
                history.append({"content": str(message)})
        return history

    def make_json_safe(self, obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.make_json_safe(i) for i in obj]
        # For LangChain message objects
        if hasattr(obj, 'content'):
            return str(obj.content)
        return str(obj)

    def get_processed_tasks(self) -> List[str]:
        """Get list of all processed task IDs."""
        return list(self._processed_tasks)

    def get_task_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Get the complete history of actions for a specific task."""
        history = self.get_memory_history()
        task_history = []
        
        for entry in history:
            if isinstance(entry, dict) and entry.get("taskId") == task_id:
                task_history.append(entry)
        
        return task_history

    def get_case_summary(self, case_id: str) -> Dict[str, Any]:
        """Get a summary of all actions taken for a case."""
        history = self.get_memory_history()
        summary = {
            "caseId": case_id,
            "totalTasks": len(self._processed_tasks),
            "processedTasks": self.get_processed_tasks(),
            "steps": []
        }
        
        for entry in history:
            if isinstance(entry, dict):
                step = {
                    "step": entry.get("step", 0),
                    "action": entry.get("action", "unknown"),
                    "timestamp": entry.get("timestamp", None),
                    "taskId": entry.get("taskId", None),
                    "taskKey": entry.get("taskKey", None),
                    "details": entry
                }
                summary["steps"].append(step)
        
        # Sort steps by step number
        summary["steps"].sort(key=lambda x: x["step"])
        return summary

    def export_memory_to_file(self, filepath: str) -> None:
        """Export the complete memory history to a JSON file."""
        memory_data = {
            "processed_tasks": self.get_processed_tasks(),
            "history": self.get_memory_history(),
            "current_task": self._current_task,
            "summary": self.get_case_summary(self._current_task.get("caseId") if self._current_task else "unknown")
        }
        
        with open(filepath, 'w') as f:
            json.dump(self.make_json_safe(memory_data), f, indent=2)
        
        logger.info(f"Memory exported to {filepath}")
        logger.info(f"Total steps recorded: {len(memory_data['history'])}")
        logger.info(f"Total tasks processed: {len(memory_data['processed_tasks'])}")

    def clear_memory(self) -> None:
        """Clear all memory and reset the agent's state."""
        self.memory.clear()
        self._processed_tasks.clear()
        self._current_task = None
        self._last_response = None
        logger.info("Memory cleared and agent state reset")

def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return

    agent = OnboardingAgent()
    case_id = "CAS-dwdwd-dwedwedew"
    
    try:
        # Execute the workflow
        agent.execute_workflow(case_id)
        
        # Export memory to file
        memory_file = f"memory_{case_id}.json"
        agent.export_memory_to_file(memory_file)
        print(f"\nMemory exported to {memory_file}")
        
        # Print summary
        summary = agent.get_case_summary(case_id)
        print("\nCase Summary:")
        print(f"Total Tasks Processed: {summary['totalTasks']}")
        print(f"Processed Task IDs: {', '.join(summary['processedTasks'])}")
        
    except Exception as e:
        print(f"Error during workflow execution: {str(e)}")
        # Export memory even if there's an error
        memory_file = f"memory_{case_id}_error.json"
        agent.export_memory_to_file(memory_file)
        print(f"\nMemory exported to {memory_file}")

if __name__ == "__main__":
    main() 