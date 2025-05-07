# src/agents/qa_agent.py
from src.agents.base import BaseAgent
from src.models.types import AgentState, ClientStatus
from src.services.mcp_service import MCPService


class QAAgent(BaseAgent):
    def __init__(self, mcp_service: MCPService):
        super().__init__("qa_agent", mcp_service)

    def process(self, state: AgentState) -> AgentState:
        try:
            # Get LLM response with current state
            context = {
                "client_id": state.client_id,
                "client_info": state.client_info,
                "status": state.status.value
            }

            llm_response = self._get_llm_response(self.prompt, context)

            # Verify client info
            verification_result = self._get_tool("verify_client_info")(state.client_info)
            if "error" in verification_result:
                state.status = ClientStatus.INFORMATION_GATHERING
                state.memory.add_memory(
                    self.name,
                    "verification_error",
                    {"error": verification_result["error"]}
                )
                return state

            # Check company registration
            company_result = self._get_tool("check_company_registration")(state.client_info.get("company", ""))
            if "error" in company_result:
                state.status = ClientStatus.INFORMATION_GATHERING
                state.memory.add_memory(
                    self.name,
                    "company_error",
                    {"error": company_result["error"]}
                )
                return state

            # Validate contact info
            contact_result = self._get_tool("validate_contact_info")({
                "email": state.client_info.get("email", ""),
                "phone": state.client_info.get("phone", "")
            })

            if "error" not in contact_result:
                state.status = ClientStatus.READY_FOR_APPROVAL
            else:
                state.status = ClientStatus.INFORMATION_GATHERING

            state.memory.add_memory(
                self.name,
                "qa",
                {
                    "verification": verification_result,
                    "company": company_result,
                    "contact": contact_result,
                    "status": state.status.value,
                    "llm_response": llm_response
                }
            )

        except Exception as e:
            print(f"QA Agent Error: {str(e)}")
            state.status = ClientStatus.INFORMATION_GATHERING

        return state