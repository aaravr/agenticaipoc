from typing import Dict, Any, List, Optional
import uuid
import json
from FastMcpAgent import MCPAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServiceError(Exception):
    """Base exception for MCPService errors"""
    pass

class ToolNotAvailableError(MCPServiceError):
    """Exception raised when a requested tool is not available"""
    pass

class ToolExecutionError(MCPServiceError):
    """Exception raised when a tool execution fails"""
    pass

class MCPService:
    def __init__(self, mcp_url: str = "http://localhost:8080/mcp/message"):
        """Initialize MCP service with the given URL."""
        self.mcp_agent = MCPAgent(mcp_url=mcp_url)
        
        # Map our internal tool names to MCP server tool names
        self._tool_map = {
            "get_client_info": "getClientInfo",
            "update_client_info": "saveClientInfo",
            "send_followup_email": "sendFollowupEmail",
            "verify_client_info": "qaVerify",
            "check_company_registration": "checkCompanyRegistration",
            "validate_contact_info": "validateContactInfo",
            "review_compliance": "reviewCompliance",
            "check_risk_assessment": "checkRiskAssessment",
            "final_approval": "approve"
        }
        
        # Get available tools and validate mapping
        self._available_tools = self._get_available_tools()
        self._validate_tool_mapping()
        
    def _get_available_tools(self) -> List[str]:
        """Get list of available tools from MCP server."""
        try:
            tools = self.mcp_agent.list_tools()
            tool_names = [tool["name"] for tool in tools]
            logger.info("\n=== Available MCP Tools ===")
            for tool in tool_names:
                logger.info(f"  ✓ {tool}")
            logger.info("==========================\n")
            return tool_names
        except Exception as e:
            logger.error(f"Failed to get available tools: {str(e)}")
            return []

    def _validate_tool_mapping(self) -> None:
        """Validate that all mapped tools are available on the MCP server."""
        missing_tools = []
        for internal_name, mcp_name in self._tool_map.items():
            if mcp_name not in self._available_tools:
                missing_tools.append((internal_name, mcp_name))
        
        if missing_tools:
            logger.warning("\n=== Tool Availability Warning ===")
            logger.warning("The following tools are not available on the MCP server:")
            for internal_name, mcp_name in missing_tools:
                logger.warning(f"  - {internal_name} -> {mcp_name}")
            logger.warning("\nThese operations will be simulated with mock responses.")
            logger.warning("===============================\n")

    def _call_mcp_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with proper error handling and logging."""
        mcp_tool_name = self._tool_map.get(tool_name)
        if not mcp_tool_name:
            logger.error(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
            
        if mcp_tool_name not in self._available_tools:
            logger.warning(f"\n=== Simulating Tool: {tool_name} ===")
            logger.warning(f"Tool {mcp_tool_name} is not available on MCP server")
            logger.warning("Returning mock response")
            logger.warning("========================\n")
            return self._get_mock_response(tool_name, input_data)
            
        try:
            logger.info(f"\n=== Executing Tool: {mcp_tool_name} ===")
            logger.info(f"Input: {json.dumps(input_data, indent=2)}")
            result = self.mcp_agent.call_tool(mcp_tool_name, input_data)
            logger.info(f"Result: {json.dumps(result, indent=2)}")
            logger.info("==============================\n")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {mcp_tool_name}: {str(e)}")
            return {"error": str(e)}

    def _get_mock_response(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock responses for unavailable tools."""
        mock_responses = {
            "send_followup_email": {"status": "success", "message": "Mock email sent"},
            "check_company_registration": {"status": "success", "registered": True},
            "validate_contact_info": {"status": "success", "valid": True},
            "review_compliance": {"status": "success", "compliant": True},
            "check_risk_assessment": {"status": "success", "risk_level": "low"}
        }
        return mock_responses.get(tool_name, {"status": "error", "message": "No mock response available"})

    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """Get client information."""
        return self._call_mcp_tool("get_client_info", {"clientId": client_id})

    def update_client_info(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update client information."""
        return self._call_mcp_tool("update_client_info", {"clientInfo": client_data})

    def send_followup_email(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send follow-up email to client."""
        return self._call_mcp_tool("send_followup_email", {"clientInfo": client_data})

    def verify_client_info(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify client information."""
        return self._call_mcp_tool("verify_client_info", {"clientInfo": client_data})

    def check_company_registration(self, company_name: str) -> Dict[str, Any]:
        """Check company registration status."""
        return self._call_mcp_tool("check_company_registration", {"companyName": company_name})

    def validate_contact_info(self, contact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate contact information."""
        return self._call_mcp_tool("validate_contact_info", {"contactInfo": contact_data})

    def review_compliance(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review client compliance."""
        return self._call_mcp_tool("review_compliance", {"clientInfo": client_data})

    def check_risk_assessment(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check client risk assessment."""
        return self._call_mcp_tool("check_risk_assessment", {"clientInfo": client_data})

    def final_approval(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get final approval for client."""
        return self._call_mcp_tool("final_approval", {"clientInfo": client_data})

    def add_sample_client(self, client_data: Optional[Dict[str, Any]] = None) -> str:
        """Add a sample client and return the client ID.
        
        Args:
            client_data: Optional client data to use. If not provided, a default sample client will be used.
        """
        if client_data is None:
            client_data = {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1234567890",
                "company": "Acme Corp",
                "status": "pending"
            }
        
        logger.info("\n=== Adding Sample Client ===")
        logger.info(f"Client Data: {json.dumps(client_data, indent=2)}")
        
        try:
            result = self.update_client_info(client_data)
            
            # Handle different response formats
            if isinstance(result, dict):
                client_id = result.get("clientId")
                if client_id:
                    logger.info(f"Successfully added client with ID: {client_id}")
                    return client_id
                    
            # If we get here, either result is None or doesn't have clientId
            logger.warning("No client ID in response, generating UUID")
            client_id = str(uuid.uuid4())
            logger.info(f"Generated client ID: {client_id}")
            return client_id
            
        except Exception as e:
            logger.error(f"Error adding sample client: {str(e)}")
            client_id = str(uuid.uuid4())
            logger.info(f"Generated fallback client ID: {client_id}")
            return client_id
        finally:
            logger.info("==========================\n") 