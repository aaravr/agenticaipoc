# src/services/mcp_tools.py
from typing import Dict, Any
import requests
from src.config.settings import API_BASE_URL


def call_mcp_endpoint(endpoint: str, data: Dict[str, Any] = None, method: str = "GET") -> Dict:
    url = f"{API_BASE_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        else:
            response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        return {"error": str(e)}


mcp_tools = {
    "capture_client_info": lambda x: call_mcp_endpoint("capture", x, "POST"),
    "update_client_info": lambda x: call_mcp_endpoint("outreach", x, "POST"),
    "qa_verify": lambda x: call_mcp_endpoint("qa-verify", x, "POST"),
    "approve_client": lambda x: call_mcp_endpoint("approve", x, "POST"),
    "get_client_info": lambda x: call_mcp_endpoint(f"client/{x}"),
    "get_clients_by_status": lambda x: call_mcp_endpoint(f"clients/status/{x}")
}