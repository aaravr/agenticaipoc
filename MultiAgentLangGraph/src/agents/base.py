# src/agents/base.py
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from src.services.mock_service import MockService
from src.agents.prompts import AGENT_PROMPTS
from src.models.types import AgentState, AgentStateDict
from src.utils.interaction_logger import InteractionLogger


class BaseAgent:
    def __init__(self, name: str, mock_service: MockService):
        self.name = name
        self.llm = ChatOpenAI(temperature=0)
        self.mock_service = mock_service
        self.prompt = AGENT_PROMPTS.get(name, "")
        self.logger = InteractionLogger()

    def _get_tool(self, name: str) -> Any:
        # Map tool names to mock service methods
        tool_map = {
            "get_client_info": self.mock_service.get_client_info,
            "update_client_info": self.mock_service.update_client_info,
            "send_followup_email": self.mock_service.send_followup_email,
            "verify_client_info": self.mock_service.verify_client_info,
            "check_company_registration": self.mock_service.check_company_registration,
            "validate_contact_info": self.mock_service.validate_contact_info,
            "review_compliance": self.mock_service.review_compliance,
            "check_risk_assessment": self.mock_service.check_risk_assessment,
            "final_approval": self.mock_service.final_approval
        }
        return tool_map.get(name)

    def process(self, state: AgentState) -> AgentState:
        raise NotImplementedError

    def _get_llm_response(self, prompt: str, context: Dict[str, Any]) -> str:
        # Format the prompt with context
        formatted_prompt = prompt.format(
            client_id=context["client_id"],
            client_info=context["client_info"],
            status=context["status"]
        )
        
        # Log the prompt being sent to LLM
        self.logger.log_interaction(
            agent_name=self.name,
            step_type="LLM_REQUEST",
            input_data={"prompt": formatted_prompt},
            output_data={},
            tools_used=[]
        )
        
        # Get LLM response with structured thinking
        response = self.llm.invoke(formatted_prompt)
        
        # Parse the response to extract reasoning and actions
        reasoning = self._extract_reasoning(response.content)
        actions = self._extract_actions(response.content)
        
        # Log the LLM response, reasoning, and planned actions
        self.logger.log_interaction(
            agent_name=self.name,
            step_type="LLM_RESPONSE",
            input_data={},
            output_data={
                "raw_response": response.content,
                "structured_reasoning": reasoning,
                "planned_actions": actions
            },
            tools_used=[]
        )
        
        # Execute the actions extracted from LLM response
        results = self._execute_planned_actions(actions, context)
        
        return response.content

    def _extract_reasoning(self, response: str) -> Dict[str, Any]:
        """Extract structured reasoning from LLM response"""
        reasoning = {
            "analysis": "",
            "plan": "",
            "execution": "",
            "verification": ""
        }
        
        # Split response into lines
        lines = response.split('\n')
        current_section = None
        
        # Map keywords to reasoning sections
        section_keywords = {
            'ANALYZE:': 'analysis',
            'PLAN:': 'plan',
            'EXECUTE:': 'execution',
            'VERIFY:': 'verification'
        }
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new section
            for keyword, section in section_keywords.items():
                if line.startswith(keyword):
                    current_section = section
                    # Remove the keyword from the content
                    content = line.replace(keyword, '').strip()
                    reasoning[current_section] = content
                    break
            else:
                # If no new section started and we have a current section,
                # append this line to the current section
                if current_section and not line.startswith('EXECUTE:'):
                    reasoning[current_section] += f"\n{line}"
        
        # Log the extracted reasoning
        self.logger.log_interaction(
            agent_name=self.name,
            step_type="REASONING_EXTRACTED",
            input_data={"raw_response": response},
            output_data={"structured_reasoning": reasoning},
            tools_used=[]
        )
        
        return reasoning

    def _extract_actions(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response with enhanced validation"""
        actions = []
        lines = response.split('\n')
        
        # Track which tools are available
        available_tools = set(self._get_tool_map().keys())
        
        for line in lines:
            line = line.strip()
            if not line.startswith('EXECUTE:'):
                continue
            
            try:
                # Remove 'EXECUTE: use ' from the start
                action_part = line.replace('EXECUTE:', '').strip()
                
                # Validate format
                if 'use' not in action_part or 'with' not in action_part:
                    raise ValueError(f"Invalid EXECUTE format. Expected 'EXECUTE: use [tool] with [params]', got: {line}")
                
                # Extract tool name and parameters
                action_part = action_part.replace('use', '', 1).strip()
                if 'with' not in action_part:
                    raise ValueError(f"Missing 'with' keyword in EXECUTE statement: {line}")
                
                tool_part, param_part = action_part.split('with', 1)
                tool_name = tool_part.strip()
                
                # Validate tool exists
                if tool_name not in available_tools:
                    raise ValueError(f"Unknown tool '{tool_name}'. Available tools: {', '.join(available_tools)}")
                
                # Parse and validate JSON parameters
                try:
                    import json
                    params = json.loads(param_part.strip())
                    if not isinstance(params, dict):
                        raise ValueError("Parameters must be a JSON object")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON parameters: {str(e)}")
                
                # Validate required parameters for each tool
                self._validate_tool_parameters(tool_name, params)
                
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
            "get_client_info": self.mock_service.get_client_info,
            "update_client_info": self.mock_service.update_client_info,
            "send_followup_email": self.mock_service.send_followup_email,
            "verify_client_info": self.mock_service.verify_client_info,
            "check_company_registration": self.mock_service.check_company_registration,
            "validate_contact_info": self.mock_service.validate_contact_info,
            "review_compliance": self.mock_service.review_compliance,
            "check_risk_assessment": self.mock_service.check_risk_assessment,
            "final_approval": self.mock_service.final_approval
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