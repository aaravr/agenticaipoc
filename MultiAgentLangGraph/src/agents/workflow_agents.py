from typing import Dict, List, Any, Annotated, TypedDict, Protocol, Callable, Optional, Union
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field  # Updated to use pydantic directly
import json
import logging
import os
from datetime import datetime
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from pathlib import Path
import httpx
import asyncio
import sseclient
import requests
import uuid
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

# OpenAI API Key configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please ensure it exists in your .env file at the project root."
    )

# Proxy configuration
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")

class TaskStatus(str, Enum):
    """Enum for task statuses."""
    OPEN = "Open"
    COMPLETED = "Completed"
    ERROR = "Error"
    UNAVAILABLE = "Unavailable"
    IN_PROGRESS = "In Progress"

@dataclass
class ToolConfig:
    """Configuration for a tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    required_fields: List[str]
    example_input: Dict[str, Any]

@dataclass
class TaskProcessingRule:
    """Configuration for task processing rules."""
    task_key: str
    required_fields: List[str]
    payload_template: Dict[str, Any]
    validation_rules: List[Callable[[Dict[str, Any]], bool]]
    special_handling: bool = False
    priority: int = 0

@dataclass
class StatusConfig:
    """Configuration for status handling."""
    valid_statuses: List[TaskStatus]
    completion_status: TaskStatus
    error_status: TaskStatus
    status_transitions: Dict[TaskStatus, List[TaskStatus]]

@dataclass
class WorkflowConfig:
    """Configuration for the workflow."""
    tools: List[ToolConfig]
    task_rules: List[TaskProcessingRule]
    status_config: StatusConfig
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30

# Default configurations
DEFAULT_STATUS_CONFIG = StatusConfig(
    valid_statuses=[TaskStatus.OPEN, TaskStatus.COMPLETED, TaskStatus.ERROR, TaskStatus.UNAVAILABLE, TaskStatus.IN_PROGRESS],
    completion_status=TaskStatus.OPEN,
    error_status=TaskStatus.ERROR,
    status_transitions={
        TaskStatus.OPEN: [TaskStatus.IN_PROGRESS, TaskStatus.ERROR],
        TaskStatus.IN_PROGRESS: [TaskStatus.COMPLETED, TaskStatus.ERROR],
        TaskStatus.COMPLETED: [],
        TaskStatus.ERROR: [TaskStatus.OPEN],
        TaskStatus.UNAVAILABLE: []
    }
)

DEFAULT_TOOL_CONFIGS = [
    ToolConfig(
        name="getTaskDetails",
        description="Get details of tasks for a case instance",
        input_schema={
            "type": "object",
            "properties": {
                "caseInstanceId": {"type": "string"}
            },
            "required": ["caseInstanceId"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {"type": "array", "items": {"type": "object"}}
            }
        },
        required_fields=["caseInstanceId"],
        example_input={"caseInstanceId": "CAS-123"}
    ),
    ToolConfig(
        name="claimTask",
        description="Claim a task for processing",
        input_schema={
            "type": "object",
            "properties": {
                "claimRequest": {
                    "type": "object",
                    "properties": {
                        "taskId": {"type": "string"}
                    },
                    "required": ["taskId"]
                }
            },
            "required": ["claimRequest"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {"type": "object"}
            }
        },
        required_fields=["claimRequest.taskId"],
        example_input={"claimRequest": {"taskId": "task-123"}}
    ),
    ToolConfig(
        name="completeTask",
        description="Complete a task",
        input_schema={
            "type": "object",
            "properties": {
                "completeRequest": {
                    "type": "object",
                    "properties": {
                        "taskId": {"type": "string"},
                        "taskKey": {"type": "string"},
                        "outcomeVariable": {"type": "object"}
                    },
                    "required": ["taskId"]
                }
            },
            "required": ["completeRequest"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {"type": "object"}
            }
        },
        required_fields=["completeRequest.taskId"],
        example_input={"completeRequest": {"taskId": "task-123"}}
    )
]

DEFAULT_TASK_RULES = [
    TaskProcessingRule(
        task_key="TASK_1_070",
        required_fields=["taskId", "taskKey", "skipBsOutreach", "skipBsOutreachComment", "outcomeVariable"],
        payload_template={
            "completeRequest": {
                "taskId": "[TASK-ID]",
                "taskKey": "TASK_1_070",
                "skipBsOutreach": True,
                "skipBsOutreachComment": "Skipping outreach as per workflow requirements",
                "outcomeVariable": {
                    "prepareBsOutreachOutcome": {
                        "label": "Skip Outreach",
                        "action": "skipOutreach"
                    }
                }
            }
        },
        validation_rules=[
            lambda x: isinstance(x.get("taskId"), str),
            lambda x: x.get("taskKey") == "TASK_1_070"
        ],
        special_handling=True,
        priority=1
    )
]

DEFAULT_WORKFLOW_CONFIG = WorkflowConfig(
    tools=DEFAULT_TOOL_CONFIGS,
    task_rules=DEFAULT_TASK_RULES,
    status_config=DEFAULT_STATUS_CONFIG
)

class MCPClient:
    """MCP Client for making SSE calls."""
    
    def __init__(self, mcp_url: str, sse_url: str):
        self.mcp_url = mcp_url
        self._session = None
        self._sse_client = None
        self._event_iterator = None
        self._last_response = None

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

        self._sse_client = sseclient.SSEClient(resp)
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

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool and handle its response."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Calling tool: {name}")
        logger.info(f"Input object: {json.dumps(arguments, indent=2)}")
        logger.info(f"{'='*80}")

        rpc_payload = {
            "name": name,
            "arguments": arguments
        }

        raw = self._sse_request("tools/call", rpc_payload)

        # Handle list response
        if isinstance(raw, list):
            if len(raw) > 0 and isinstance(raw[0], dict):
                if "isError" in raw[0] and raw[0]["isError"]:
                    err = raw[0].get("text", "Unknown tool error")
                    raise RuntimeError(f"Tool {name} error: {err}")
                # If it's a list of content items, try to parse the first one
                if "text" in raw[0]:
                    try:
                        raw = json.loads(raw[0]["text"])
                    except json.JSONDecodeError:
                        # Handle "Done" response
                        if raw[0]["text"] == "Done":
                            raw = {"status": "success", "data": {"message": "Done"}}
                        else:
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
                        # Handle "Done" response
                        if txt == "Done":
                            raw = {"status": "success", "data": {"message": "Done"}}
                        else:
                            logger.warning("Couldn't parse server content as JSON: %s", txt)
                if isinstance(raw, dict) and raw.get("isError", False):
                    err = items[0].get("text", "Unknown tool error")
                    raise RuntimeError(f"Tool {name} error: {err}")

        if raw is None:
            raw = {"status": "success"}

        self._last_response = raw
        logger.info(f"\n{'='*80}")
        logger.info(f"Tool {name} completed with result:")
        logger.info(json.dumps(raw, indent=2))
        logger.info(f"{'='*80}\n")
        return raw

    def get_last_response(self) -> Optional[Dict[str, Any]]:
        """Get the last response from the server."""
        return self._last_response

    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self._session:
            self._session.close()
        if self._sse_client:
            self._sse_client.close()

# Initialize MCP client
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8080/mcp/message")
MCP_CLIENT = MCPClient(MCP_BASE_URL, MCP_BASE_URL)

class Tool(Protocol):
    """Protocol defining the interface for tools."""
    name: str
    description: str
    execute: Callable[[Dict[str, Any]], Dict[str, Any]]

class BaseTool:
    """Base class for all tools."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool."""
        raise NotImplementedError("Subclasses must implement execute method")

class DynamicTool(BaseTool):
    """Tool that uses MCP client for execution."""
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.mcp_client = MCP_CLIENT

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool using MCP client."""
        return await self.mcp_client.call_tool(self.name, input_data)

class ToolRegistry:
    """Registry for managing available tools."""
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get descriptions of all tools."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

class Agent(ABC):
    """Base class for all agents."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    @abstractmethod
    async def process(self, state: Dict[str, Any], available_tools: List[Tool]) -> Dict[str, Any]:
        """Process the current state and return updated state."""
        pass

class SupervisorAgent(Agent):
    """Supervisor agent responsible for task orchestration."""
    
    async def process(self, state: Dict[str, Any], available_tools: List[Tool]) -> Dict[str, Any]:
        """Process the current state and delegate to appropriate agent."""
        try:
            # Create a prompt that includes available tools
            tools_info = [{"name": t.name, "description": t.description} for t in available_tools]
            
            analysis_prompt = f"""
            You are a supervisor agent that needs to analyze the current state and determine next steps.
            You have access to the following tools:
            {json.dumps(tools_info, indent=2)}
            
            Current State:
            {json.dumps(state, indent=2)}
            
            Based on the current state and available tools, provide a JSON response with the following structure:
            {{
                "task_analysis": "Analysis of what needs to be done",
                "required_tools": ["List of tools that should be used"],
                "next_agent": "Which agent should handle this next",
                "reasoning": "Your reasoning for these decisions"
            }}
            
            Important:
            - If tools are required, set next_agent to "task_handler"
            - If no more tools are needed, set next_agent to "end"
            - Consider which tools are most appropriate for the current state
            - Think about the sequence of tool usage
            - Consider error cases and edge conditions
            - Your response MUST be valid JSON
            - Do not include any text outside the JSON structure
            """
            
            analysis_response = self.llm.invoke(analysis_prompt)
            
            try:
                # First try direct JSON parsing
                analysis = json.loads(analysis_response.content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the response
                content = analysis_response.content
                if "```json" in content:
                    # Extract JSON from markdown code block
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    analysis = json.loads(json_str)
                elif "```" in content:
                    # Extract JSON from any code block
                    json_str = content.split("```")[1].split("```")[0].strip()
                    analysis = json.loads(json_str)
                else:
                    # Try to find JSON-like structure
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())
                    else:
                        raise ValueError(f"Could not parse LLM response as JSON: {content}")
            
            # Validate required fields
            required_fields = ["task_analysis", "required_tools", "next_agent", "reasoning"]
            missing_fields = [field for field in required_fields if field not in analysis]
            if missing_fields:
                raise ValueError(f"Missing required fields in analysis: {missing_fields}")
            
            # Ensure proper agent transition
            if analysis.get("required_tools"):
                analysis["next_agent"] = "task_handler"
            elif not state.get("current_task"):
                analysis["next_agent"] = "task_handler"
            else:
                analysis["next_agent"] = "end"
            
            state["analysis"] = analysis
            state["next_agent"] = analysis["next_agent"]
            
            return state
            
        except Exception as e:
            logger.error(f"Error in SupervisorAgent: {str(e)}")
            state["error"] = str(e)
            return state

class TaskAgent(Agent):
    """Agent for processing tasks."""
    
    def __init__(self, llm: ChatOpenAI, config: WorkflowConfig):
        super().__init__(llm)
        self.config = config
        self._processed_tasks = set()
        self._last_task_details = None
        self._last_claimed_task = None
        self._empty_task_list_count = 0
        self._max_empty_retries = 3

    def _is_task_processed(self, task_id: str) -> bool:
        """Check if a task has been processed."""
        return task_id in self._processed_tasks

    def _get_task_rule(self, task_key: str) -> Optional[TaskProcessingRule]:
        """Get the processing rule for a task."""
        return next((rule for rule in self.config.task_rules if rule.task_key == task_key), None)

    def _format_task_prompt(self, state: Dict[str, Any], available_tools: List[Tool]) -> str:
        """Format the task processing prompt."""
        tools_info = [{"name": t.name, "description": t.description} for t in available_tools]
        
        # Get task context
        task_context = ""
        if isinstance(state.get("current_task"), list):
            task_context = f"""
            Current Tasks:
            {json.dumps(state["current_task"], indent=2)}
            
            Processed Tasks:
            {list(self._processed_tasks)}
            
            Last Task Details:
            {json.dumps(self._last_task_details, indent=2) if self._last_task_details else "None"}
            """
        
        return f"""
        You are a task execution agent that needs to determine and execute the next tool based on the current state.
        
        Available Tools:
        {json.dumps(tools_info, indent=2)}
        
        Current State:
        {json.dumps(state, indent=2)}
        
        {task_context}
        
        Based on the current state and available tools, determine the next tool to execute.
        Provide a JSON response with the following structure:
        {{
            "tool": "name of the tool to use",
            "input": {{
                // Format based on the tool's input schema
                // For getTaskDetails, use: {{"caseInstanceId": "CAS-123"}}
                // For claimTask, use: {{"claimRequest": {{"taskId": "task-123"}}}}
                // For completeTask, use: {{"completeRequest": {{"taskId": "task-123"}}}}
            }},
            "reasoning": "Why this tool should be used now"
        }}
        
        Important:
        - Choose the most appropriate tool based on the current state
        - Format the input according to the tool's schema
        - Always get updated task list after completing a task
        - Your response MUST be valid JSON
        - Do not include any text outside the JSON structure
        - Do not process tasks that have already been processed: {list(self._processed_tasks)}
        - If no tasks are available, use getTaskDetails to check for new tasks
        - If there is an Open task, claim it first
        - After claiming a task, complete it
        """

    def _format_task_rules(self) -> str:
        """Format task processing rules for the prompt."""
        rules = []
        for rule in sorted(self.config.task_rules, key=lambda x: x.priority, reverse=True):
            rules.append(f"""
            Task: {rule.task_key}
            Required Fields: {', '.join(rule.required_fields)}
            Special Handling: {'Yes' if rule.special_handling else 'No'}
            Priority: {rule.priority}
            Payload Template:
            {json.dumps(rule.payload_template, indent=2)}
            """)
        return "\n".join(rules)

    def _analyze_result(self, tool_name: str, tool_input: Dict[str, Any], result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the tool execution result."""
        analysis_prompt = f"""
        Analyze the tool execution result and determine if more tools need to be executed.
        
        Tool: {tool_name}
        Input: {json.dumps(tool_input, indent=2)}
        Result: {json.dumps(result, indent=2)}
        Current State: {json.dumps(state, indent=2)}
        Processed Tasks: {list(self._processed_tasks)}
        Last Claimed Task: {self._last_claimed_task}
        Last Task Details: {json.dumps(self._last_task_details, indent=2) if self._last_task_details else "None"}
        
        Task Processing Rules:
        {self._format_task_rules()}
        
        Provide a JSON response with the following structure:
        {{
            "continue_execution": true/false,
            "reasoning": "Why execution should continue or stop",
            "next_tool": "name of the next tool to use"
        }}
        
        Important:
        - Set continue_execution to true if:
          * There are tasks with status {self.config.status_config.completion_status}
          * Task list needs to be refreshed
        - Set continue_execution to false only when:
          * No tasks with status {self.config.status_config.completion_status} remain
          * All tasks are marked as {self.config.status_config.error_status}
          * getTaskDetails returns no data
        - Your response MUST be valid JSON
        - Do not include any text outside the JSON structure
        """
        
        analysis_response = self.llm.invoke(analysis_prompt)
        try:
            return json.loads(analysis_response.content)
        except json.JSONDecodeError:
            content = analysis_response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse LLM response as JSON: {content}")

    def _should_continue_processing(self, tasks: List[Dict[str, Any]]) -> bool:
        """Determine if there are more tasks to process."""
        # If we get an empty task list, increment counter
        if not tasks:
            self._empty_task_list_count += 1
            # If we've seen too many empty lists, assume we're done
            if self._empty_task_list_count >= self._max_empty_retries:
                logger.info(f"Received {self._max_empty_retries} empty task lists, assuming completion")
                return False
            # Otherwise, keep trying
            return True

        # Reset empty counter if we get tasks
        self._empty_task_list_count = 0
        
        # Check if there are any tasks with Open status
        has_open_tasks = any(t.get("status") == self.config.status_config.completion_status for t in tasks)
        
        # Check if there are any tasks with Unavailable status that might become available
        has_unavailable_tasks = any(t.get("status") == "Unavailable" for t in tasks)
        
        # Check if all tasks are Complete
        all_complete = all(t.get("status") == "Complete" for t in tasks)
        
        # Continue if there are open tasks or unavailable tasks that might become available
        return has_open_tasks or (has_unavailable_tasks and not all_complete)

    async def process(self, state: Dict[str, Any], available_tools: List[Tool]) -> Dict[str, Any]:
        """Process the task using available tools."""
        try:
            # Create execution prompt
            execution_prompt = self._format_task_prompt(state, available_tools)
            execution_response = self.llm.invoke(execution_prompt)
            
            try:
                plan = json.loads(execution_response.content)
            except json.JSONDecodeError:
                content = execution_response.content
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    plan = json.loads(json_str)
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                    plan = json.loads(json_str)
                else:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        plan = json.loads(json_match.group())
                    else:
                        raise ValueError(f"Could not parse LLM response as JSON: {content}")
            
            # Validate required fields
            if "tool" not in plan or "input" not in plan:
                raise ValueError("Missing required fields in execution plan")
            
            tool_name = plan["tool"]
            if not tool_name:
                # If no tool specified, default to getTaskDetails
                tool_name = "getTaskDetails"
                plan["input"] = {"caseInstanceId": state["case_id"]}
            
            tool_input = plan["input"]
            
            # Find and execute the tool
            tool = next((t for t in available_tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            
            # Execute tool
            logger.info(f"Executing {tool_name} with input: {json.dumps(tool_input)}")
            result = await tool.execute(tool_input)
            
            # Handle null or empty results
            if result is None:
                result = {"status": "error", "message": "Tool returned null result"}
            elif isinstance(result, dict) and not result:
                result = {"status": "error", "message": "Tool returned empty result"}
            
            # Record tool execution
            tool_execution = {
                "tool": tool_name,
                "input": tool_input,
                "reasoning": plan.get("reasoning"),
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            state["workflow_metadata"]["tool_executions"].append(tool_execution)
            
            # Update state based on result
            state["last_response"] = result
            
            # Handle getTaskDetails result
            if tool_name == "getTaskDetails":
                if isinstance(result, list):
                    # Store the last task details for reference
                    self._last_task_details = result
                    # Log all tasks for debugging
                    logger.info(f"All tasks from server: {json.dumps(result, indent=2)}")
                    
                    # Filter for unprocessed tasks with Open status
                    open_tasks = [
                        t for t in result 
                        if t.get("status") == "Open" 
                        and not self._is_task_processed(t.get("taskId"))
                    ]
                    
                    # Log filtered tasks for debugging
                    logger.info(f"Filtered tasks with Open status: {json.dumps(open_tasks, indent=2)}")
                    logger.info(f"Processed tasks: {list(self._processed_tasks)}")
                    
                    state["current_task"] = open_tasks
                    state["processed_tasks"] = list(self._processed_tasks)
                    
                    # If we have open tasks, continue processing
                    if open_tasks:
                        state["next_agent"] = "task_handler"
                    else:
                        # Check if we have any unavailable tasks that might become available
                        unavailable_tasks = [t for t in result if t.get("status") == "Unavailable"]
                        if unavailable_tasks:
                            state["next_agent"] = "task_handler"
                        else:
                            state["next_agent"] = "end"
                    return state
                else:
                    state["error"] = "Invalid task details format"
                    state["next_agent"] = "end"
                    return state
            
            # Handle claimTask result
            if tool_name == "claimTask":
                if isinstance(result, dict) and result.get("status") == "success":
                    # Extract taskId from input
                    task_id = tool_input.get("claimRequest", {}).get("taskId")
                    if task_id:
                        logger.info(f"Successfully claimed task {task_id}")
                        # Force transition to completeTask
                        state["next_agent"] = "task_handler"
                        state["last_claimed_task"] = task_id
                        return state
                else:
                    state["error"] = "Failed to claim task"
                    state["next_agent"] = "end"
                    return state
            
            # Handle completeTask result
            if tool_name == "completeTask":
                if isinstance(result, dict) and result.get("status") == "success":
                    # Extract taskId from input
                    task_id = tool_input.get("completeRequest", {}).get("taskId")
                    if task_id:
                        self._processed_tasks.add(task_id)
                        logger.info(f"Successfully completed task {task_id}")
                        # Clear the claimed task
                        state.pop("last_claimed_task", None)
                        # Update state with processed tasks
                        state["processed_tasks"] = list(self._processed_tasks)
                        # Get task details again after completing
                        state["next_agent"] = "task_handler"
                        return state
                else:
                    state["error"] = "Failed to complete task"
                    state["next_agent"] = "end"
                    return state
            
            return state
            
        except Exception as e:
            logger.error(f"Error in TaskAgent: {str(e)}")
            state["error"] = str(e)
            state["next_agent"] = "end"
            return state

class Workflow:
    """Workflow that coordinates agents and tools."""
    
    def __init__(self, llm: ChatOpenAI):
        self.agents = {
            "supervisor": SupervisorAgent(llm),
            "task_handler": TaskAgent(llm, DEFAULT_WORKFLOW_CONFIG)
        }
        self.state = self._create_initial_state()
        self.workflow_start_time = None
        self.tool_registry = ToolRegistry()
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Initialize tools from MCP discovery."""
        try:
            # Get available tools from MCP
            available_tools = asyncio.run(MCP_CLIENT.call_tool("tools/list", {}))
            logger.info(f"Tool discovery response: {json.dumps(available_tools, indent=2)}")
            
            # Register default tools since tools/list only returns success status
            default_tools = [
                DynamicTool("getTaskDetails", "Get details of tasks for a case instance"),
                DynamicTool("claimTask", "Claim a task for processing"),
                DynamicTool("completeTask", "Complete a task")
            ]
            
            for tool in default_tools:
                self.tool_registry.register_tool(tool)
            logger.info(f"Registered {len(default_tools)} default tools")
                
        except Exception as e:
            logger.error(f"Error during tool discovery: {str(e)}")
            raise

    def _create_initial_state(self) -> Dict[str, Any]:
        return {
            "case_id": None,
            "current_task": {},
            "processed_tasks": [],
            "last_response": {},
            "error": None,
            "next_agent": "supervisor",
            "analysis": {},
            "execution_plan": {},
            "workflow_metadata": {
                "start_time": None,
                "current_agent": None,
                "previous_agent": None,
                "agent_transitions": [],
                "tool_executions": []
            }
        }
    
    async def run(self, case_id: str) -> Dict[str, Any]:
        """Run the workflow for a case."""
        self.workflow_start_time = datetime.now()
        self.state["case_id"] = case_id
        self.state["workflow_metadata"]["start_time"] = self.workflow_start_time.isoformat()
        
        logger.info(f"Starting workflow for case {case_id}")
        
        while True:
            current_agent_name = self.state["next_agent"]
            current_agent = self.agents.get(current_agent_name)
            
            if not current_agent:
                logger.error(f"Unknown agent: {current_agent_name}")
                break
            
            # Update workflow metadata
            self.state["workflow_metadata"]["previous_agent"] = self.state["workflow_metadata"]["current_agent"]
            self.state["workflow_metadata"]["current_agent"] = current_agent_name
            self.state["workflow_metadata"]["agent_transitions"].append({
                "from": self.state["workflow_metadata"]["previous_agent"],
                "to": current_agent_name,
                "timestamp": datetime.now().isoformat()
            })
            
            # Execute agent with available tools
            self.state = await current_agent.process(self.state, self.tool_registry.get_all_tools())
            
            if self.state.get("error"):
                logger.error(f"Workflow error in {current_agent_name}: {self.state['error']}")
                break
            
            if self.state["next_agent"] == "end":
                logger.info("Workflow completed successfully")
                break
        
        return self.state

if __name__ == "__main__":
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0
        )

        # Create workflow with LLM
        workflow = Workflow(llm)

        # Run workflow
        logger.info("Starting workflow...")
        final_state = asyncio.run(workflow.run("CAS-123"))
        logger.info("Workflow completed with final state:")
        logger.info(json.dumps(final_state, indent=2))

    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise
    finally:
        # Clean up
        asyncio.run(MCP_CLIENT.call_tool("tools/close", {})) 
