# src/agents/approval_agent.py
from src.agents.base import BaseAgent
from src.models.types import AgentState, ClientStatus
from src.services.mcp_service import MCPService


class ApprovalAgent(BaseAgent):
    def __init__(self, mcp_service: MCPService):
        super().__init__("approval_agent", mcp_service)

    def process(self, state: AgentState) -> AgentState:
        try:
            # Get LLM response with current state
            context = {
                "client_id": state.client_id,
                "client_info": state.client_info,
                "status": state.status.value
            }

            llm_response = self._get_llm_response(self.prompt, context)

            # Review compliance
            compliance_result = self._get_tool("review_compliance")(state.client_info)
            if "error" in compliance_result:
                state.status = ClientStatus.READY_FOR_QA
                state.memory.add_memory(
                    self.name,
                    "compliance_error",
                    {"error": compliance_result["error"]}
                )
                return state

            # Check risk assessment
            risk_result = self._get_tool("check_risk_assessment")(state.client_info)
            if "error" in risk_result:
                state.status = ClientStatus.READY_FOR_QA
                state.memory.add_memory(
                    self.name,
                    "risk_error",
                    {"error": risk_result["error"]}
                )
                return state

            # Get final approval
            approval_result = self._get_tool("final_approval")(state.client_info)
            if "error" not in approval_result and approval_result.get("approved", False):
                state.status = ClientStatus.COMPLETED
            else:
                state.status = ClientStatus.READY_FOR_QA

            state.memory.add_memory(
                self.name,
                "approval",
                {
                    "compliance": compliance_result,
                    "risk": risk_result,
                    "approval": approval_result,
                    "status": state.status.value,
                    "llm_response": llm_response
                }
            )

        except Exception as e:
            print(f"Approval Agent Error: {str(e)}")
            state.status = ClientStatus.READY_FOR_QA

        return state