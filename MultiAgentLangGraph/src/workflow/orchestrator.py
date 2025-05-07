# src/workflow/orchestrator.py
from typing import Dict, Any
import uuid
from langgraph.graph import StateGraph, Graph
from src.models.types import AgentState, ClientStatus, AgentStateDict
from src.services.memory import WorkflowMemory
from src.agents.outreach_agent import OutreachAgent
from src.agents.qa_agent import QAAgent
from src.agents.approval_agent import ApprovalAgent
from src.services.mcp_service import MCPService


class WorkflowOrchestrator:
    def __init__(self, mcp_service: MCPService = None):
        # Use provided MCP service or create new one
        self.mcp_service = mcp_service or MCPService()
        # Initialize agents with the same MCP service instance
        self.outreach_agent = OutreachAgent(mcp_service=self.mcp_service)
        self.qa_agent = QAAgent(mcp_service=self.mcp_service)
        self.approval_agent = ApprovalAgent(mcp_service=self.mcp_service)

    def create_workflow(self) -> Graph:
        # Create the workflow graph
        workflow = StateGraph(AgentStateDict)

        # Define the nodes
        workflow.add_node("outreach", self.outreach_agent.process)
        workflow.add_node("qa", self.qa_agent.process)
        workflow.add_node("approval", self.approval_agent.process)

        # Define the edges
        workflow.add_edge("outreach", "qa")
        workflow.add_edge("qa", "approval")
        workflow.add_edge("approval", "end")

        # Set the entry point
        workflow.set_entry_point("outreach")

        # Compile the workflow
        return workflow.compile()

    def execute_workflow(self, client_id: str):
        # Lookup client info before starting workflow
        lookup_result = self.mcp_service.get_client_info(client_id)
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