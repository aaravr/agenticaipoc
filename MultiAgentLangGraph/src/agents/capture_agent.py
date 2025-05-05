# src/agents/capture_agent.py
from src.agents.base import BaseAgent
from src.models.types import AgentState, ClientStatus


class InformationCaptureAgent(BaseAgent):
    def __init__(self):
        super().__init__("capture_agent")

    def process(self, state: AgentState) -> AgentState:
        try:
            result = self._get_tool("capture_client_info")(state.client_info)

            if "error" not in result:
                state.client_info.update(result)
                state.status = ClientStatus.READY_FOR_QA
            else:
                state.status = ClientStatus.INFORMATION_GATHERING

            state.memory.add_memory(
                self.name,
                "capture",
                {"result": result, "status": state.status.value}
            )

        except Exception as e:
            print(f"Capture Agent Error: {str(e)}")
            state.status = ClientStatus.INFORMATION_GATHERING

        return state