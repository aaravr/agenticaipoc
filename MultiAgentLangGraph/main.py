# main.py
from src.workflow.orchestrator import WorkflowOrchestrator
from src.services.mcp_service import MCPService
import os
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()

    # Verify OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Initialize MCP service and add sample client
    mcp_service = MCPService()
    sample_client = {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "123-456-7890",
        "company": "AI Agent Corp"
    }
    
    # Add the sample client and store its ID
    client_id = mcp_service.add_sample_client(sample_client)
    print(f"Created sample client with ID: {client_id}")

    try:
        # Pass the mcp_service instance to the orchestrator
        orchestrator = WorkflowOrchestrator(mcp_service=mcp_service)
        orchestrator.execute_workflow(client_id)
    except Exception as e:
        print(f"Main error: {str(e)}")


if __name__ == "__main__":
    main()