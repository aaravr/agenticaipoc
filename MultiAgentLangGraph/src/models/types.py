# src/models/types.py
from dataclasses import dataclass
from typing import Dict, Any, Protocol, TypedDict
from enum import Enum
from typing import List

class MemoryProtocol(Protocol):
    def add_memory(self, agent_name: str, action: str, data: Dict[str, Any]) -> None: ...
    def get_memory(self, agent_name: str = None, limit: int = 10) -> List[Dict[str, Any]]: ...
    def clear_memory(self) -> None: ...

class ClientStatus(str, Enum):
    NEW = "NEW"
    INFORMATION_GATHERING = "INFORMATION_GATHERING"
    READY_FOR_QA = "READY_FOR_QA"
    QA_IN_PROGRESS = "QA_IN_PROGRESS"
    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"

class AgentStateDict(TypedDict):
    client_id: str
    client_info: Dict[str, Any]
    status: str
    memory: MemoryProtocol

@dataclass
class AgentState:
    client_id: str
    client_info: Dict[str, Any]
    status: ClientStatus
    memory: MemoryProtocol

    def to_dict(self) -> AgentStateDict:
        return {
            "client_id": self.client_id,
            "client_info": self.client_info,
            "status": self.status.value,
            "memory": self.memory
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        return cls(
            client_id=data["client_id"],
            client_info=data["client_info"],
            status=ClientStatus(data["status"]),
            memory=data["memory"]
        )