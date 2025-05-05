# src/services/mock_service.py
from typing import Dict, Any
import uuid

class MockService:
    def __init__(self):
        self._client_data = {}
        self._company_registry = {
            "AI Agent Corp": {
                "registration_number": "REG123456",
                "status": "Active",
                "incorporation_date": "2020-01-01"
            },
            "Tech Solutions Inc": {
                "registration_number": "REG789012",
                "status": "Active",
                "incorporation_date": "2019-05-15"
            }
        }
        self._risk_scores = {}

    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        if client_id not in self._client_data:
            return {"error": "Client not found"}
        return self._client_data[client_id]

    def update_client_info(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        client_id = client_data.get("client_id", str(uuid.uuid4()))
        self._client_data[client_id] = client_data
        return {"status": "success", "client_id": client_id}

    def send_followup_email(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "message": f"Follow-up email sent to {client_data.get('email')}"
        }

    def verify_client_info(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        # Mock verification logic
        return {
            "status": "success",
            "verified_fields": ["name", "email", "phone", "company"],
            "issues": []
        }

    def check_company_registration(self, company_name: str) -> Dict[str, Any]:
        if company_name in self._company_registry:
            return {
                "status": "success",
                "company_info": self._company_registry[company_name]
            }
        return {
            "status": "error",
            "message": "Company not found in registry"
        }

    def validate_contact_info(self, contact_data: Dict[str, Any]) -> Dict[str, Any]:
        # Mock validation logic
        return {
            "status": "success",
            "validated_fields": ["email", "phone"],
            "issues": []
        }

    def review_compliance(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        # Mock compliance check
        return {
            "status": "success",
            "compliance_status": "Compliant",
            "requirements_met": ["KYC", "AML", "Data Protection"]
        }

    def check_risk_assessment(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        # Mock risk assessment
        risk_score = self._risk_scores.get(client_data.get("company", ""), 0.3)
        return {
            "status": "success",
            "risk_score": risk_score,
            "risk_level": "Low" if risk_score < 0.5 else "Medium"
        }

    def final_approval(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        # Mock approval process
        return {
            "status": "success",
            "approved": True,
            "approval_date": "2024-04-21",
            "approver": "System"
        }

    def add_sample_client(self, client_data: Dict[str, Any]) -> str:
        client_id = str(uuid.uuid4())
        client_data["client_id"] = client_id
        self._client_data[client_id] = client_data
        return client_id 