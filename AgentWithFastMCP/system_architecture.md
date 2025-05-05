# MCP Agent System Architecture

## Complete System Overview

```mermaid
graph TB
    subgraph Python Environment
        subgraph MCP Agent
            A1[MCPAgent Class] --> A2[OpenAI Integration]
            A1 --> A3[Memory Management]
            A1 --> A4[SSE Client Handler]
            
            A2 --> A5[LLM Planning]
            A3 --> A6[Conversation History]
            A4 --> A7[Stream Processing]
        end
        
        subgraph Dependencies
            B1[langchain] --> B2[OpenAI Client]
            B3[sseclient-py] --> B4[Stream Handler]
            B5[requests] --> B6[HTTP Client]
        end
    end

    subgraph Spring Boot Server
        subgraph MCP Server
            C1[JSON-RPC Endpoint] --> C2[Tool Registry]
            C3[SSE Endpoint] --> C4[Event Stream]
            
            C2 --> C5[Tool 1]
            C2 --> C6[Tool 2]
            C2 --> C7[Tool N]
        end
    end

    A1 --> C1
    A4 --> C3
```

## Detailed Communication Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant Memory
    participant SSE
    participant MCP Server

    User->>Agent: Execute Task
    activate Agent
    
    Agent->>MCP Server: GET /tools/list
    activate MCP Server
    MCP Server-->>Agent: SSE Stream: Tools List
    deactivate MCP Server
    
    Agent->>LLM: Generate Execution Plan
    activate LLM
    LLM-->>Agent: JSON Plan
    deactivate LLM
    
    loop For Each Tool
        Agent->>Memory: Save Context
        activate Memory
        Memory-->>Agent: Context Updated
        deactivate Memory
        
        Agent->>MCP Server: POST /tools/call
        activate MCP Server
        MCP Server-->>SSE: Stream Response
        activate SSE
        SSE-->>Agent: Tool Result
        deactivate SSE
        deactivate MCP Server
        
        Agent->>Memory: Update History
    end
    
    Agent-->>User: Task Complete
    deactivate Agent
```

## Data Flow with JSON Examples

```mermaid
graph LR
    subgraph Request Flow
        A[JSON-RPC Request] --> B[Tool Call]
        B --> C[SSE Stream]
    end
    
    subgraph Response Flow
        D[SSE Event] --> E[JSON Parse]
        E --> F[Result Processing]
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
```

### Example JSON Flows

```json
// Tool Listing Request
{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": "req-123",
    "params": {}
}

// Tool Call Request
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "id": "req-456",
    "params": {
        "name": "createClient",
        "arguments": {
            "name": "Alice",
            "address": "123 Main St"
        }
    }
}

// SSE Response
data: {
    "jsonrpc": "2.0",
    "id": "req-456",
    "result": {
        "status": "success",
        "clientId": "CL12345"
    }
}
```

## Error Handling and Recovery

```mermaid
graph TD
    A[Tool Execution] --> B{Success?}
    B -->|Yes| C[Process Result]
    B -->|No| D[Error Handler]
    
    D --> E{Recoverable?}
    E -->|Yes| F[Retry Logic]
    E -->|No| G[Fail Gracefully]
    
    F --> H[Max Retries?]
    H -->|No| A
    H -->|Yes| G
    
    C --> I[Update Memory]
    G --> J[Log Error]
    J --> K[Notify User]
```

## Memory Management Flow

```mermaid
graph LR
    subgraph Memory Operations
        A[Tool Call] --> B[Format Context]
        B --> C[Save to Memory]
        D[Tool Result] --> E[Format Result]
        E --> C
        C --> F[Next Operation]
    end
    
    style A fill:#f96,stroke:#333,stroke-width:2px
    style D fill:#f96,stroke:#333,stroke-width:2px
    style C fill:#9f9,stroke:#333,stroke-width:2px
``` 