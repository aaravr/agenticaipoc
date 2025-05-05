# src/workflow/orchestrator.py
from typing import Dict, Any
import uuid
from langgraph.graph import StateGraph, Graph
from src.models.types import AgentState, ClientStatus, AgentStateDict
from src.services.memory import WorkflowMemory
from src.agents.outreach_agent import OutreachAgent
from src.agents.qa_agent import QAAgent
from src.agents.approval_agent import ApprovalAgent
from src.services.mock_service import MockService


class WorkflowOrchestrator:
    def __init__(self, mock_service: MockService = None):
        # Use provided mock service or create new one
        self.mock_service = mock_service or MockService()
        # Initialize agents with the same mock service instance
        self.outreach_agent = OutreachAgent(mock_service=self.mock_service)
        self.qa_agent = QAAgent(mock_service=self.mock_service)
        self.approval_agent = ApprovalAgent(mock_service=self.mock_service)

    def create_workflow(self) -> Graph:
        workflow = StateGraph(AgentStateDict)

        def route_next(state: Dict[str, Any]) -> str:
            return state["status"]

        # Wrap agent process methods to handle dict conversion
        def wrap_agent_process(agent_method):
            def wrapped(state_dict: Dict[str, Any]) -> Dict[str, Any]:
                state = AgentState.from_dict(state_dict)
                result = agent_method(state)
                return result.to_dict()

            return wrapped

        # Add nodes with wrapped process methods
        workflow.add_node("outreach", wrap_agent_process(self.outreach_agent.process))
        workflow.add_node("qa", wrap_agent_process(self.qa_agent.process))
        workflow.add_node("approval", wrap_agent_process(self.approval_agent.process))
        workflow.add_node("end", lambda x: x)

        # Add edges
        workflow.add_conditional_edges(
            "outreach",
            route_next,
            {
                ClientStatus.READY_FOR_QA.value: "qa",
                ClientStatus.INFORMATION_GATHERING.value: "outreach"
            }
        )

        workflow.add_conditional_edges(
            "qa",
            route_next,
            {
                ClientStatus.READY_FOR_APPROVAL.value: "approval",
                ClientStatus.INFORMATION_GATHERING.value: "outreach"
            }
        )

        workflow.add_conditional_edges(
            "approval",
            route_next,
            {
                ClientStatus.COMPLETED.value: "end",
                ClientStatus.READY_FOR_QA.value: "qa"
            }
        )

        workflow.set_entry_point("outreach")

        return workflow.compile()

    def execute_workflow(self, client_id: str):
        # Lookup client info before starting workflow
        lookup_result = self.mock_service.get_client_info(client_id)
        if "error" in lookup_result:
            raise ValueError(f"Failed to lookup client info: {lookup_result['error']}")

        workflow = self.create_workflow()
        memory = WorkflowMemory(str(uuid.uuid4()))

        initial_state = AgentState(
            client_id=client_id,
            client_info=lookup_result,
            status=ClientStatus.NEW,
            memory=memory
        )

        try:
            config = {"recursion_limit": 25}
            final_state_dict = workflow.invoke(initial_state.to_dict(), config=config)
            final_state = AgentState.from_dict(final_state_dict)

            print("Workflow completed!")
            print(f"Final Status: {final_state.status}")
            print(f"Final Client Info: {final_state.client_info}")

            print("\nWorkflow History:")
            for memory_entry in memory.get_memory():
                print(f"Agent: {memory_entry['agent']}")
                print(f"Action: {memory_entry['action']}")
                print(f"Data: {memory_entry['data']}")
                print("---")

        except Exception as e:
            print(f"Workflow error: {str(e)}")
            raise
        finally:
            memory.clear_memory()