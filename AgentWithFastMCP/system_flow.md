# MCP Agent System Flow

## System Architecture

```mermaid
graph TD
    subgraph Python Client
        A[MCPAgent] --> B[OpenAI LLM]
        A --> C[Memory System]
        A --> D[SSE Client]
    end

    subgraph Spring Boot Server
        E[MCP Server] --> F[Tool Registry]
        E --> G[SSE Endpoint]
    end

    D --> G
    A --> E
```

## Detailed Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant MCP Server

    User->>Agent: Execute Task
    Agent->>MCP Server: List Tools (JSON-RPC)
    MCP Server-->>Agent: Tools List (SSE)
    Agent->>LLM: Generate Plan
    LLM-->>Agent: JSON Plan
    loop For Each Tool
        Agent->>MCP Server: Tool Call (JSON-RPC)
        MCP Server-->>Agent: Tool Result (SSE)
        Agent->>Agent: Update Memory
    end
    Agent-->>User: Task Complete
```

## Data Flow Example

### Tool Listing
```mermaid
sequenceDiagram
    participant Agent
    participant MCP Server

    Agent->>MCP Server: {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": "uuid",
        "params": {}
    }
    MCP Server-->>Agent: {
        "result": {
            "tools": [
                {"name": "tool1", "description": "desc1"},
                {"name": "tool2", "description": "desc2"}
            ]
        }
    }
```

### Tool Execution
```mermaid
sequenceDiagram
    participant Agent
    participant MCP Server

    Agent->>MCP Server: {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": "uuid",
        "params": {
            "name": "toolName",
            "arguments": {...}
        }
    }
    MCP Server-->>Agent: {
        "result": {
            "status": "success",
            "data": {...}
        }
    }
```

## Memory Flow

```mermaid
graph LR
    A[Tool Call] --> B[Memory]
    C[Tool Result] --> B
    B --> D[Next Tool]
    D --> A
```

## Error Handling Flow

```mermaid
graph TD
    A[Execute Tool] --> B{Success?}
    B -->|Yes| C[Save Result]
    B -->|No| D[Handle Error]
    D --> E[Log Error]
    E --> F[Continue/Stop]
    C --> G[Next Tool]
```

## Component Details

### MCPAgent
- Handles communication with MCP Server
- Manages tool execution flow
- Maintains conversation memory
- Coordinates with LLM for planning

### OpenAI LLM
- Generates execution plans
- Validates tool usage
- Ensures proper JSON formatting

### Memory System
- Stores tool execution history
- Maintains conversation context
- Supports future planning

### SSE Client
- Handles server-sent events
- Manages streaming responses
- Parses JSON-RPC messages

### MCP Server
- Exposes tool registry
- Handles JSON-RPC requests
- Streams responses via SSE
- Manages tool execution 