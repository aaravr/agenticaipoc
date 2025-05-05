# src/services/memory.py
from typing import Dict, List, Any
from datetime import datetime
from src.models.types import MemoryProtocol


class WorkflowMemory(MemoryProtocol):
    def __init__(self, workflow_id: str):
        self.memories: List[Dict[str, Any]] = []
        self.workflow_id = workflow_id

    def add_memory(self, agent_name: str, action: str, data: Dict[str, Any]) -> None:
        memory_entry = {
            "workflow_id": self.workflow_id,
            "agent": agent_name,
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.memories.append(memory_entry)

    def get_memory(self, agent_name: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        if agent_name:
            filtered = [m for m in self.memories if m["agent"] == agent_name]
            return filtered[-limit:]
        return self.memories[-limit:]

    def clear_memory(self) -> None:
        self.memories.clear()