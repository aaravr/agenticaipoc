# src/agents/outreach_agent.py
from src.agents.base import BaseAgent
from src.models.types import AgentState, ClientStatus
from src.services.mock_service import MockService


class OutreachAgent(BaseAgent):
    def __init__(self, mock_service: MockService):
        super().__init__("outreach_agent", mock_service)

    def process(self, state: AgentState) -> AgentState:
        try:
            # Get LLM response with current state
            context = {
                "client_id": state.client_id,
                "client_info": state.client_info,
                "status": state.status.value
            }

            llm_response = self._get_llm_response(self.prompt, context)

            # Process the response and update state
            if not state.client_info:
                # If no client info, try to get it
                lookup_result = self._get_tool("get_client_info")(state.client_id)
                if "error" in lookup_result:
                    state.status = ClientStatus.INFORMATION_GATHERING
                    state.memory.add_memory(
                        self.name,
                        "lookup_error",
                        {"error": lookup_result["error"]}
                    )
                    return state
                state.client_info = lookup_result

            # Update client info based on LLM analysis
            result = self._get_tool("update_client_info")(state.client_info)

            if "error" not in result:
                state.client_info.update(result)
                state.status = ClientStatus.READY_FOR_QA
            else:
                state.status = ClientStatus.INFORMATION_GATHERING

            state.memory.add_memory(
                self.name,
                "outreach",
                {
                    "result": result,
                    "status": state.status.value,
                    "llm_response": llm_response
                }
            )

        except Exception as e:
            print(f"Outreach Agent Error: {str(e)}")
            state.status = ClientStatus.INFORMATION_GATHERING

        return state