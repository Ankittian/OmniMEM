# Why HydraDB Matters: Graph vs Flat Vector Search

This document illustrates exactly why a standard RAG pipeline fails on long-term memory tasks (like the BEAM benchmark) and how HydraDB's Graph + Vector architecture solves it through relational continuity.

```mermaid
flowchart TD
    %% Styling
    classDef query fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#000
    classDef flatChunk fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#000
    classDef graphNode fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#000
    classDef edgeLabel fill:#ffffff,stroke:#e5e7eb,color:#374151,font-size:12px
    classDef fail fill:#fef08a,stroke:#eab308,stroke-width:2px,color:#000
    classDef win fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#000

    subgraph Naive["❌ Naive RAG (Flat Vector Store)"]
        direction TB
        Q1("Query: 'What framework am I using?'"):::query
        
        C1["Session 1 Chunk:<br/>'I am using Flask'"]:::flatChunk
        C2["Session 15 Chunk:<br/>'I switched to FastAPI'"]:::flatChunk
        
        Q1 -. "1. High Semantic Similarity" .-> C1
        Q1 -. "2. Lower Semantic Match (Missed)" .-> C2
        
        R1("Result: Agent Hallucinates<br/>(Returns outdated fact: Flask)"):::fail
        C1 --> R1
    end

    subgraph Hydra["✅ HydraDB (Graph + Vector Store)"]
        direction TB
        Q2("Query: 'What framework am I using?'"):::query
        
        N1(("Node: Session 1<br/>Fact: 'Using Flask'")):::graphNode
        N2(("Node: Session 15<br/>Fact: 'Switched to FastAPI'")):::graphNode
        
        %% The Magic of the Graph
        N1 == "REPLACES_PREVIOUS" ==> N2
        N1 -. "TEMPORAL_NEXT" .-> N2
        
        Q2 -- "1. Vector Match" --> N1
        N1 -- "2. Graph Traversal<br/>(graph_context=True)" --> N2
        
        R2("Result: Agent Reasons Correctly<br/>(Returns updated fact: FastAPI)"):::win
        N2 --> R2
    end
```

### Key Concepts Highlighted

1. **The Semantic Trap**: In naive RAG, "I am using Flask" often has a higher cosine similarity to the question "What framework am I using?" than the update statement ("switched to FastAPI"). The retriever blindly pulls the old fact.
2. **Relational Continuity**: HydraDB doesn't just store isolated text chunks. It links them as nodes in a graph. When the vector engine hits Session 1, the graph traversal engine automatically follows the `TEMPORAL_NEXT` or `REPLACES_PREVIOUS` edges to pull the subsequent context from Session 15.
3. **Reasoning-Ready Payload**: Because HydraDB returns the *connected timeline* of facts, our category-aware LLM prompt can successfully apply `Knowledge Update` logic to output the correct, latest fact.
