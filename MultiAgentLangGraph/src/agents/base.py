# src/agents/base.py
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from src.services.mcp_service import MCPService
from src.agents.prompts import AGENT_PROMPTS
from src.models.types import AgentState, AgentStateDict
from src.utils.interaction_logger import InteractionLogger
import json


class BaseAgent:
    def __init__(self, name: str, mcp_service: MCPService):
        self.name = name
        self.llm = ChatOpenAI(temperature=0)
        self.mcp_service = mcp_service
        self.prompt = AGENT_PROMPTS.get(name, "")
        self.logger = InteractionLogger()

    def _get_tool(self, name: str) -> Any:
        # Map tool names to MCP service methods
        tool_map = {
            "get_client_info": self.mcp_service.get_client_info,
            "update_client_info": self.mcp_service.update_client_info,
            "send_followup_email": self.mcp_service.send_followup_email,
            "verify_client_info": self.mcp_service.verify_client_info,
            "check_company_registration": self.mcp_service.check_company_registration,
            "validate_contact_info": self.mcp_service.validate_contact_info,
            "review_compliance": self.mcp_service.review_compliance,
            "check_risk_assessment": self.mcp_service.check_risk_assessment,
            "final_approval": self.mcp_service.final_approval
        }
        return tool_map.get(name)

    def process(self, state: AgentState) -> AgentState:
        raise NotImplementedError

    def _get_llm_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get response from LLM with context"""
        try:
            response = self.llm.invoke(prompt.format(**context))
            return response.content
        except Exception as e:
            self.logger.log_interaction(
                agent_name=self.name,
                step_type="LLM_ERROR",
                input_data={"prompt": prompt, "context": context},
                output_data={"error": str(e)},
                tools_used=[]
            )
            raise

    def _extract_actions(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response with enhanced validation"""
        actions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            try:
                # Try to parse the line as JSON
                action = json.loads(line)
                
                # Validate required fields
                if not isinstance(action, dict):
                    continue
                    
                tool_name = action.get("tool")
                params = action.get("input", {})
                
                if not tool_name or not isinstance(params, dict):
                    continue
                
                actions.append({
                    "tool": tool_name,
                    "input": params,
                    "reasoning": "Extracted from LLM response",
                    "original_line": line
                })
                
                # Log successful parsing
                self.logger.log_interaction(
                    agent_name=self.name,
                    step_type="ACTION_PARSED",
                    input_data={"line": line},
                    output_data={"tool": tool_name, "params": params},
                    tools_used=[]
                )
                
            except Exception as e:
                # Log parsing error with details
                self.logger.log_interaction(
                    agent_name=self.name,
                    step_type="ACTION_PARSE_ERROR",
                    input_data={
                        "line": line,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    output_data={},
                    tools_used=[]
                )
        
        return actions

    def _get_tool_map(self) -> Dict[str, Any]:
        """Get mapping of tool names to their functions"""
        return {
            "get_client_info": self.mcp_service.get_client_info,
            "update_client_info": self.mcp_service.update_client_info,
            "send_followup_email": self.mcp_service.send_followup_email,
            "verify_client_info": self.mcp_service.verify_client_info,
            "check_company_registration": self.mcp_service.check_company_registration,
            "validate_contact_info": self.mcp_service.validate_contact_info,
            "review_compliance": self.mcp_service.review_compliance,
            "check_risk_assessment": self.mcp_service.check_risk_assessment,
            "final_approval": self.mcp_service.final_approval
        }

    def _validate_tool_parameters(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Validate that required parameters are present for each tool"""
        required_params = {
            "get_client_info": ["client_id"],
            "update_client_info": ["client_id"],
            "send_followup_email": ["email"],
            "verify_client_info": ["client_id"],
            "check_company_registration": ["company"],
            "validate_contact_info": ["email", "phone"],
            "review_compliance": ["client_id"],
            "check_risk_assessment": ["client_id"],
            "final_approval": ["client_id"]
        }
        
        if tool_name in required_params:
            missing_params = [param for param in required_params[tool_name] if param not in params]
            if missing_params:
                raise ValueError(f"Missing required parameters for {tool_name}: {', '.join(missing_params)}")

    def _execute_planned_actions(self, actions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the actions planned by LLM"""
        results = []
        for action in actions:
            tool_name = action.get("tool")
            tool_input = action.get("input", {})
            
            # Log planned action
            self.logger.log_interaction(
                agent_name=self.name,
                step_type="ACTION_PLANNED",
                input_data={
                    "tool": tool_name,
                    "input": tool_input,
                    "reasoning": action.get("reasoning", "")
                },
                output_data={},
                tools_used=[]
            )
            
            # Execute the tool
            result = self._execute_tool(tool_name, tool_input)
            results.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result
            })
            
            # Log action result
            self.logger.log_interaction(
                agent_name=self.name,
                step_type="ACTION_COMPLETED",
                input_data={"tool": tool_name},
                output_data={"result": result},
                tools_used=[tool_name]
            )
        
        return results

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        tool = self._get_tool(tool_name)
        if not tool:
            return {"error": f"Tool {tool_name} not found"}
        
        # Log tool execution
        self.logger.log_interaction(
            agent_name=self.name,
            step_type="TOOL_EXECUTION",
            input_data={
                "tool": tool_name,
                "input": tool_input
            },
            output_data={},
            tools_used=[tool_name]
        )
        
        result = tool(tool_input)
        
        # Log tool result
        self.logger.log_interaction(
            agent_name=self.name,
            step_type="TOOL_RESULT",
            input_data={},
            output_data={"result": result},
            tools_used=[tool_name]
        )
        
        return result