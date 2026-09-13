# TripMate Architecture

## Component diagram

```mermaid
flowchart TB
    User(["User"])

    subgraph L1["L1 · Interface"]
        CLI["cli.py<br/>argparse · REPL · --verbose"]
    end

    subgraph L2["L2 · Orchestration"]
        ORCH["agent/orchestrator.py<br/><b>the tool-use loop</b><br/>iteration guard · error classify"]
        PROMPT["agent/prompts.py<br/>system prompt · scope rules"]
        TRACE["agent/trace.py<br/>reasoning trace"]
    end

    subgraph L2B["L2b · LLM provider layer"]
        BASE["llm/base.py<br/>neutral types:<br/>Turn · ToolCall · LLMResponse"]
        GP["llm/groq_provider.py<br/><i>default</i>"]
        AP["llm/anthropic_provider.py"]
    end

    GROQ["<b>Groq API</b><br/>llama-3.3-70b-versatile<br/>free tier"]
    ANTH["<b>Anthropic API</b><br/>claude-opus-5<br/>adaptive thinking"]

    subgraph L3["L3 · Tool layer"]
        REG["tools/registry.py<br/>schemas() · dispatch()<br/>validation · error capture"]
        GT["tools/guide_tool.py"]
        WT["tools/weather_tool.py"]
    end

    subgraph L4["L4 · Domain logic"]
        RET["rag/retriever.py<br/>city+country detection<br/>similarity floor"]
        SVC["forecast/service.py<br/>date normalisation"]
    end

    subgraph L5["L5 · Data / providers"]
        STORE["rag/store.py<br/>numpy cosine · 20 chunks"]
        EMB["rag/embedder.py<br/>MiniLM → TF-IDF fallback"]
        DATA[("data/*.txt<br/>4 guides")]
        PROV["forecast/providers.py<br/>mock climate table"]
    end

    CFG["config.py · logging_setup.py<br/><i>cross-cutting</i>"]

    User -->|"natural-language query"| CLI
    CLI --> ORCH
    PROMPT --> ORCH
    ORCH --> TRACE
    ORCH <-->|"complete(system, history, tools)<br/>← LLMResponse"| BASE
    BASE --- GP & AP
    GP <-->|"chat.completions<br/>role:tool messages"| GROQ
    AP <-->|"messages.create<br/>tool_result blocks"| ANTH
    ORCH -->|"dispatch(name, args)"| REG
    REG --> GT & WT
    GT --> RET
    WT --> SVC
    RET --> STORE
    STORE --> EMB
    DATA -->|"ingest at startup"| STORE
    SVC --> PROV
    TRACE -.->|"answer + trace"| CLI
    CLI -->|"answer"| User

    CFG -.-> ORCH
    CFG -.-> REG
    CFG -.-> RET
```

**Dependency rule:** arrows point downward only. L2 imports L3; L3 imports L4;
L4 imports L5. Nothing imports upward.

Two seams do the load-bearing work:

- **`registry.specs()` / `registry.dispatch()`** — the orchestrator's entire
  view of the tool layer. It has no travel knowledge, so adding a third tool
  requires no change to it.
- **`LLMProvider.complete()`** — the orchestrator's entire view of the vendor.
  It never sees an Anthropic content block or an OpenAI `tool_calls` array, so
  switching vendors is a one-file change and the 109 tests are unaffected.

## Request flow — the multi-tool case

`"What should I pack for Reykjavik in December?"`

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant O as Orchestrator
    participant P as Provider (Groq / Anthropic)
    participant M as LLM
    participant R as Registry
    participant G as RAG tool
    participant W as Weather tool

    U->>O: query
    O->>O: validate (non-empty, < 2000 chars)
    O->>P: complete(system, history, tools)
    P->>M: vendor-shaped request

    Note over M: reasons: "packing needs the guide<br/>AND actual December conditions"

    M-->>P: 2 parallel tool calls
    P-->>O: LLMResponse(stop_reason=tool_use,<br/>tool_calls=[...])

    par executed together
        O->>R: dispatch(search_destination_guide)
        R->>G: query="Reykjavik packing winter"
        G-->>R: reykjavik:packing_tips (0.43)
    and
        O->>R: dispatch(get_weather_forecast)
        R->>W: city=Reykjavik, month=December
        W-->>R: -2..3 °C, high precip, 4 h daylight
    end

    O->>P: ONE ToolResultsTurn carrying BOTH results
    Note right of P: Anthropic: one user message with<br/>both tool_result blocks — splitting them<br/>trains the model out of parallel calls.<br/>Groq: one role:"tool" message each.
    P->>M: vendor-shaped results

    M-->>P: finished answer
    P-->>O: LLMResponse(stop_reason=end_turn)
    O-->>U: single coherent answer + reasoning trace
```

## Where each failure is handled

Failures are pushed **down**, never up. A tool that raises would kill the loop;
a tool that returns `tool_result(is_error=True)` lets the model recover.

| Failure | Layer | Behaviour |
|---|---|---|
| Empty / malformed / over-long input | L1–L2 | Rejected before any API call |
| Unknown destination | L4 retriever | Zero vocabulary overlap → no results → model names its actual coverage |
| Missing weather data | L4 service | `{"available": false, "reason": "unsupported_city"}` |
| Season instead of a month | L4 service | `unparseable_date` → model asks which month |
| Tool raises / times out | L3 registry | Caught → `is_error` result → loop continues |
| Unknown tool name | L3 registry | Error result listing valid tools |
| Malformed tool-call JSON from the model | L2b provider | Parsed defensively → empty args → registry rejects with a usable message |
| Provider down / rate-limited | L2b → L2 | `describe_error()` classifies most-specific-first → graceful message |
| Unknown provider / missing SDK | L2b factory | `ProviderError` at startup, naming the fix |
| Model loops without settling | L2 | `max_iterations` guard |
| Out-of-scope request | L2 prompt | No tool exists → the model says so |
