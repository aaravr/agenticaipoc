# main.py
from src.workflow.orchestrator import WorkflowOrchestrator
from src.services.mock_service import MockService
import os
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()

    # Verify OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Initialize mock service and add sample client
    mock_service = MockService()
    sample_client = {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "123-456-7890",
        "company": "AI Agent Corp"
    }
    
    # Add the sample client and store its ID
    client_id = mock_service.add_sample_client(sample_client)
    print(f"Created sample client with ID: {client_id}")

    try:
        # Pass the mock_service instance to the orchestrator
        orchestrator = WorkflowOrchestrator(mock_service=mock_service)
        orchestrator.execute_workflow(client_id)
    except Exception as e:
        print(f"Main error: {str(e)}")


if __name__ == "__main__":
    main()