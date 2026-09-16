# TripMate — an agentic AI travel assistant

TripMate answers natural-language travel questions about four destinations —
**Tokyo, Reykjavik, Bangkok, Barcelona** — by reasoning about what the question
needs and calling tools accordingly.

There is **no keyword routing anywhere in the codebase.** Tool selection is
performed by the LLM per query, driven entirely by the tool descriptions and
the system prompt. If routing misbehaves, the fix belongs in a description or
the prompt — never in an `if` statement in the loop.

```
$ python -m tripmate --verbose "What should I pack for Reykjavik in December?"

December in Reykjavik runs about -2 to 3 °C with high precipitation, strong
winds and only ~4 hours of daylight. Pack for wet and windy rather than merely
cold: waterproof and windproof outer layers, warm mid-layers, a hat and gloves,
and sturdy waterproof boots if you're heading out to waterfalls or glaciers.
```

---

## Contents

- [Setup](#setup) · [Running](#running)
- [Architecture](#architecture)
- [Tool schemas](#tool-schemas-as-given-to-the-llm)
- [Example runs](#example-runs)
- [Design decisions](#design-decisions)
- [Error handling](#error-handling) · [Logging](#logging)
- [Testing](#testing)
- [Scalability](#scalability-discussion)
- [Limitations](#known-limitations) · [Future work](#future-improvements)

---

## Setup

Requires **Python 3.10+** (developed on 3.11).

```bash
git clone <repo-url> && cd tripmate
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # then add your GROQ_API_KEY
```

**Getting a key — free, no credit card:** sign up at
[console.groq.com](https://console.groq.com) and create an API key. Groq is the
default provider because its free tier supports native *parallel* tool calling,
which is exactly what the multi-tool packing flow depends on.

Only one key is required. Everything else has a default and is overridable by
environment variable — see [`.env.example`](.env.example). No key, path, or model
id is hardcoded in logic.

### Switching provider

Two providers ship behind one interface:

| Provider | Default model | Key | Select with |
|---|---|---|---|
| **Groq** (default) | `openai/gpt-oss-120b` | `GROQ_API_KEY` | — it's the default |
| Anthropic | `claude-opus-5` | `ANTHROPIC_API_KEY` | `TRIPMATE_PROVIDER=anthropic` |

```bash
python -m tripmate --provider anthropic "Do I need a visa for Japan?"
```

Only the SDK you actually use needs installing — both are imported lazily.

> **On `sentence-transformers`:** it pulls in PyTorch (~2 GB). If you'd rather
> not, skip it — TripMate detects its absence at startup and falls back to a
> built-in TF-IDF embedder with no loss of functionality. See
> [embedding strategy](#3-embeddings-semantic-by-default-lexical-when-unavailable).

## Running

```bash
# one-shot
python -m tripmate "Do I need a visa for Japan?"

# with the reasoning trace (Module 1)
python -m tripmate --verbose "What should I pack for Reykjavik in December?"

# interactive -- the REPL carries the conversation, so follow-ups resolve
python -m tripmate --interactive

# write the structured trace to a file
python -m tripmate --trace-json out.json "Is July a good time for Barcelona?"
```

The answer goes to **stdout**; structured JSON logs go to **stderr**, so
`python -m tripmate "..." > answer.txt` does the obvious thing. Add
`--quiet-logs` to silence the logs entirely.

In the REPL each answer is carried into the next question, so `"How warm is
Tokyo in May?"` followed by `"And Bangkok?"` resolves. Type `new` to start a
fresh conversation, `exit` or Ctrl-D to quit. One-shot runs are unaffected:
`run()` still defaults to an empty history.

### Configuration

Everything is read from the environment; nothing is hardcoded in logic. See
[config.py](tripmate/config.py).

| Variable | Default | What it does |
| --- | --- | --- |
| `TRIPMATE_PROVIDER` | `groq` | `groq` or `anthropic` |
| `GROQ_API_KEY` / `ANTHROPIC_API_KEY` | — | key for the selected provider |
| `TRIPMATE_MODEL` | per provider | override the model id |
| `TRIPMATE_MAX_TOKENS` | `4096` | output cap per request |
| `TRIPMATE_TEMPERATURE` | `0.2` | low for routing determinism |
| `TRIPMATE_MAX_ITERATIONS` | `6` | hard ceiling on agent loop turns |
| `TRIPMATE_MAX_HISTORY_TURNS` | `6` | user turns the REPL carries forward |
| `TRIPMATE_TIMEOUT_S` | `60` | per-request timeout |
| `TRIPMATE_TOP_K` | `3` | chunks retrieved per search |
| `TRIPMATE_MIN_SIMILARITY` | per embedder | override the relevance floor |
| `TRIPMATE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `TRIPMATE_DATA_DIR` | `./data` | where the guides live |
| `TRIPMATE_LOG_LEVEL` | `INFO` | an unrecognised value warns and uses `INFO` |
| `TRIPMATE_LOG_FILE` | — | also append JSON logs to this path |

---

## Architecture

Full diagrams — component graph, multi-tool sequence diagram, and the
failure-handling table — are in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

```
L1   Interface       cli.py
L2   Orchestration   agent/orchestrator.py · prompts.py · trace.py
L2b  LLM providers   llm/base.py · groq_provider.py · anthropic_provider.py  ←→ vendor API
L3   Tool layer      tools/registry.py · guide_tool.py · weather_tool.py
L4   Domain logic    rag/retriever.py · forecast/service.py
L5   Data/providers  rag/store.py · embedder.py · data/*.txt · forecast/providers.py
```

The whole design rests on one rule:

> **L2 never knows what a tool does, and never knows which vendor it is talking to.**

The orchestrator's entire view of the tool layer is `registry.specs()` and
`registry.dispatch(name, args)`; its entire view of the LLM is
`provider.complete(system, history, tools)`. It contains no travel logic and no
vendor-specific message shaping, so it runs unchanged against a different tool
set *or* a different vendor. Conversely the tools know nothing about
conversations, retries, or wire formats, which is why they can be unit-tested
without an API key.

### The agent loop

```python
history = [*prior, UserTurn(query)]             # `prior` is [] for a one-shot

for iteration in range(1, config.max_iterations + 1):
    response = provider.complete(
        system=SYSTEM_PROMPT, history=history, tools=registry.specs()
    )
    if not response.tool_calls:
        history.append(AssistantTurn(response))
        return response.text, history           # the model is done

    history.append(AssistantTurn(response))
    results = [
        ToolResult(id=c.id, name=c.name, ...registry.dispatch(c.name, c.arguments))
        for c in response.tool_calls
    ]
    history.append(ToolResultsTurn(results))    # ← one turn, every result
```

Four details that matter more than they look:

1. **Every parallel result travels in one `ToolResultsTurn`.** How that reaches
   the wire is the provider's business, and the two vendors genuinely differ:
   Anthropic batches them into a single *user* message — splitting them silently
   teaches the model to stop issuing parallel calls — while OpenAI-shaped APIs
   emit one `role: "tool"` message per result. Both shapes are pinned by tests
   in `test_providers.py`.
2. **The assistant turn keeps its provider-native content** (`LLMResponse.raw`),
   so Anthropic thinking blocks and tool-call ids survive verbatim into the next
   request.
3. **The conversation is returned, not retained.** `run()` takes an optional
   `history` and hands back the one to use next, so the agent itself stays
   stateless: the REPL gets follow-ups, and one-shot callers keep the old
   behaviour by ignoring the field. Only a turn that ended with real assistant
   text is handed back — a turn that failed, was refused, or tripped the loop
   guard ends on an unanswered tool call, which both wire formats reject on
   replay, so it is dropped instead. Trimming likewise cuts only at `UserTurn`
   boundaries, the one place that cannot separate a tool call from its result.
4. **The loop imports no vendor SDK.** `orchestrator.py` references only the
   neutral types in `llm/base.py`, which is why swapping vendors touches one
   file and breaks no tests.

---

## Tool schemas as given to the LLM

These descriptions *are* the routing logic — they are the only thing the model
reads when deciding what applies. Both name their exact coverage and, crucially,
state what they **don't** do and when to combine them.

### `search_destination_guide`

```json
{
  "name": "search_destination_guide",
  "description": "Search TripMate's curated destination guides and return the most relevant excerpts. Covers exactly four cities: Tokyo, Reykjavik, Bangkok, and Barcelona.\n\nEach guide covers five topics: visa and entry requirements, the best time of year to visit, local customs and etiquette, packing tips, and safety and health.\n\nUse this for any question about what a destination is like, what to expect there, or how to behave there. Include the city name in your query so the search can target the right guide.\n\nThis tool returns reference text only. It does NOT provide temperatures or forecasts for a specific month -- use get_weather_forecast for that. For packing questions, call BOTH tools: the guide gives destination-specific advice, the forecast gives the actual conditions to pack for.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "A natural-language question or topic, including the city name. Example: 'Do I need a visa for Japan?' or 'Reykjavik winter packing'."
      }
    },
    "required": ["query"],
    "additionalProperties": false
  }
}
```

**Returns:** `{"results": [{city, section, text, relevance, citation}], "covered_cities": [...]}`.
`covered_cities` is always present — including on a miss — so the model can
state its real coverage instead of inventing some.

### `get_weather_forecast`

```json
{
  "name": "get_weather_forecast",
  "description": "Get typical weather conditions for a city in a given month: temperature range in Celsius, general conditions, precipitation level, and daylight hours. Covers exactly four cities: Tokyo, Reykjavik, Bangkok, and Barcelona.\n\nUse this whenever the answer depends on what the weather will actually be like -- including packing questions, 'is X a good time to visit', and any question mentioning a month or travel date.\n\nReturns monthly climate normals, not a live day-by-day forecast, so it cannot answer 'will it rain next Tuesday'. If the user names a season rather than a month, ask which month they mean rather than guessing.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name, e.g. 'Reykjavik'. One city per call."},
      "date_or_month": {"type": "string", "description": "A month name or a date. Accepts 'December', 'Dec', '2026-12-25', or '12/25'. Must resolve to a single month."}
    },
    "required": ["city", "date_or_month"],
    "additionalProperties": false
  }
}
```

**Returns (success):**

```json
{"available": true, "city": "Reykjavik", "month": "December",
 "conditions": "cold and dark, snow and strong winds", "temp_range_c": [-2, 3],
 "precipitation": "high", "daylight_hours": 4.0, "source": "mock-climate-table",
 "note": "Typical monthly climate normals, not a live forecast."}
```

**Returns (no data):** `{"available": false, "reason": "unsupported_city", "detail": "...", "supported_cities": [...]}`
— a *successful* call reporting a gap, never a fabricated forecast.

### The system prompt

Lives in [`tripmate/agent/prompts.py`](tripmate/agent/prompts.py). It carries
the four covered cities, the "answer from tool output, not memory" rule, the
instruction to combine both tools for packing questions, and the scope
boundaries that implement Module 5.

---

## Example runs

All traces below are **real, unedited output** from `openai/gpt-oss-120b` on
Groq's free tier. Regenerate every one with:

```bash
python scripts/generate_traces.py     # writes examples/<name>.{json,txt}
```

Seven scenarios are captured in [`examples/`](examples/); four are shown here.

### 1. Multi-tool — packing (Module 4)

```
====================================================================
REASONING TRACE  (groq · openai/gpt-oss-120b)
====================================================================
  QUERY: "What should I pack for Reykjavik in December?"

  [1] LLM TURN -> stop_reason=tool_use
      thinking aloud: We need packing advice for Reykjavik in December.
      According to instructions, packing questions need both tools: guide for
      packing tips and weather forecast for December. So we need to call
      search_destination_guide with query about Reykjavik packing tips, and
      get_weather_forecast for Reykjavik December...
      decided to call: search_destination_guide

      TOOL search_destination_guide  [ok, 0.43ms]
        args   : {"query": "Reykjavik packing tips"}
        result : {"results": [{"city": "Reykjavik", "section": "Packing Tips",
                 "text": "Weather can change rapidly regardless of season, so
                 waterproof and windproof outer layers are strongly recommended
                 year-round. Warm layers, a hat, and gloves are essential in
                 winter...", "relevance": 0.436}]}

  [2] LLM TURN -> stop_reason=tool_use
      thinking aloud: We need also weather forecast for Reykjavik in December.
      decided to call: get_weather_forecast

      TOOL get_weather_forecast  [ok, 0.6ms]
        args   : {"city": "Reykjavik", "date_or_month": "December"}
        result : {"available": true, "month": "December", "temp_range_c": [-2, 3],
                 "conditions": "cold and dark, snow and strong winds",
                 "precipitation": "high", "daylight_hours": 4.0}

  [3] LLM TURN -> stop_reason=end_turn
      decided no (further) tool is needed
--------------------------------------------------------------------
  SUMMARY: 3 LLM turn(s), tools used: search_destination_guide,
           get_weather_forecast, tokens in/out: 4079/445
====================================================================
```

> For a December trip to Reykjavík, focus on staying warm, dry, and prepared for
> rapidly changing weather. **Base layer:** thermal tops and leggings.
> **Mid-layer:** fleece or wool sweaters. **Outer layer:** a waterproof,
> wind-proof shell, because December is cold, snowy and often windy (average
> −2 °C to 3 °C, high precipitation)…

The answer is genuinely grounded in **both** sources: the waterproof/windproof
advice comes from the guide, the −2 to 3 °C and 4-hour daylight from the
forecast. Neither tool alone produces it.

**An observation worth recording.** The model's own reasoning in turn 1 states
it needs both tools — then it emits only one, and fetches the second in turn 2.
So `gpt-oss-120b` resolves this **sequentially in 3 turns**, not in parallel in
2, even with `parallel_tool_calls=True` set on the request.

The outcome is correct and Module 4 is satisfied, but it costs an extra round
trip. Parallel calling turns out to be a model capability, not just a request
flag. The architecture is already correct for both: results are batched into one
`ToolResultsTurn` regardless, so a model that *does* call in parallel gets the
2-turn path with no code change — as Claude does under `--provider anthropic`.

### 2. Single-tool — RAG only

```
  QUERY: "Do I need a visa to visit Japan, and what are the local customs I should know?"

  [1] LLM TURN -> stop_reason=tool_use
      decided to call: search_destination_guide
      TOOL search_destination_guide  [ok, 0.61ms]
        args   : {"query": "Tokyo visa and entry requirements and local customs"}
  [2] LLM TURN -> stop_reason=end_turn
  SUMMARY: 2 LLM turn(s), tools used: search_destination_guide,
           tokens in/out: 2563/344
```

The weather tool is **not** called — nothing in the question depends on
conditions. Note also that the model rewrote "Japan" into a Tokyo-targeted
query; the retriever resolves country names to cities independently, so both
layers get it right.

### 3. Single-tool — weather only

```
  QUERY: "How hot does Bangkok get in April?"

  [1] LLM TURN -> stop_reason=tool_use
      decided to call: get_weather_forecast
      TOOL get_weather_forecast  [ok, 0.36ms]
        args   : {"city": "Bangkok", "date_or_month": "April"}
  [2] LLM TURN -> stop_reason=end_turn
  SUMMARY: 2 LLM turn(s), tools used: get_weather_forecast, tokens in/out: 2415/131
```

The mirror image of the previous case: no guide lookup, because the question is
purely about numbers. This pair is the clearest evidence that routing is
genuinely dynamic rather than "always call everything".

### 4. Out-of-scope — booking (Module 5)

```
  QUERY: "Can you book me a flight to Tokyo next Tuesday?"

  [1] LLM TURN -> stop_reason=end_turn
      thinking aloud: The user asks to book a flight. According to policy, we
      cannot take actions like booking. Must respond that we cannot book, but
      can help with planning.
      decided no (further) tool is needed
  SUMMARY: 1 LLM turn(s), tools used: none, tokens in/out: 1150/117
```

> I'm not able to make bookings or reservations. However, I can help you plan
> your trip to Tokyo — whether you'd like tips on visa requirements, the best
> time to visit, what to pack, or anything else you need to know before you book
> your flight.

**Zero tool calls, one LLM turn, no fabricated action.** Module 5 is handled by
the system prompt, not a `refuse()` tool — the correct behaviour is the model
finding that no applicable tool exists and saying so. Adding a refusal tool
would be inventing routing where scope awareness should emerge.

### Also captured in `examples/`

- **`unknown-destination`** — *"What are the visa requirements for Cairo?"* →
  **zero tool calls**, 1 turn: *"I don't have a guide for Cairo. I can provide
  visa information for Tokyo, Reykjavik, Bangkok, or Barcelona."* The model
  recognised the destination was outside its stated coverage without needing to
  search first. (The retriever independently returns zero results for Cairo, so
  the honest answer is reached whichever path it takes.)
- **`ambiguous-query`** — *"What should I pack?"* → **zero tool calls**:
  *"Could you let me know which city you're traveling to and the month?"* It
  asks rather than guessing a destination.

---

## Design decisions

### 1. A hand-written tool loop, not a framework

Both SDKs ship a helper that writes this loop for you (Anthropic's
`client.beta.messages.tool_runner()`); LangChain and CrewAI would too. All were
rejected.

The orchestration *is* the deliverable here. Delegating it would mean the most
interesting ~60 lines of the project were library internals I couldn't explain,
and it would put a framework's abstractions between me and the per-turn
visibility the reasoning-trace requirement needs.

### 2. LLM provider: Groq, behind a provider interface

The assessment leaves the provider open, so the deciding constraint was the one
thing that isn't optional: **native function calling**. Without it, tool
selection degrades into parsing JSON out of prose, which is the keyword routing
the brief explicitly rules out.

Groq's free tier gives that on `openai/gpt-oss-120b`, with no credit card,
and its low latency keeps the agent responsive when a single query triggers two
tool calls plus a synthesis turn — which matters for a live demo.

Rather than hard-coding it, the vendor sits behind a small `LLMProvider`
interface (`llm/base.py`) with two implementations. That was worth the extra
~150 lines for three reasons:

- **The wire formats genuinely differ**, and the difference is subtle enough to
  be worth isolating. Anthropic returns tool calls as `tool_use` content blocks
  and takes results back as `tool_result` blocks inside **one user message**;
  OpenAI-shaped APIs put `tool_calls` on the assistant message and take each
  result back as its own `role: "tool"` message. Burying that in the loop would
  have made the loop unreadable and vendor-locked.
- **It proves the layering claim** rather than just asserting it. Swapping
  vendors changes one file; all 109 tests pass untouched, because they drive the
  neutral interface.
- **Free tiers rate-limit.** Being able to fall back with
  `--provider anthropic` is genuine operational insurance.

The trade-off: `gpt-oss-120b` is a smaller model than Claude Opus, so tool
selection on genuinely ambiguous queries is less reliable. See
[limitations](#known-limitations).

### 3. Embeddings: semantic by default, lexical when unavailable

`sentence-transformers/all-MiniLM-L6-v2` runs locally — no API key, no server,
no second vendor. It's the default.

But it pulls PyTorch (~2 GB), which is a real cost to impose on a reviewer who
just wants to run the tests. So `build_embedder()` tries it, and on any failure
(not installed, offline, download blocked) falls back to a **pure-numpy TF-IDF
embedder** with a logged warning. Everything still works. The test suite pins
the lexical path explicitly (`prefer_semantic=False`) so scores are identical on
every machine and the suite runs in ~1 second.

The fallback needed two fixes that are worth naming, both found by measurement:

- **Stemming.** Without it, "what should I **pack**" doesn't match the
  "**PACKING** TIPS" heading at all — the single most important query in the
  assignment silently retrieved the wrong section.
- **Field boosting** (`SECTION_TITLE_BOOST = 3`). A heading is one token inside
  a ~60-word body, so it loses to any chunk that merely mentions the same month.
  Repeating it three times moved `packing_tips` from rank 2 (0.145) to rank 1
  (0.232), with clear separation from `best_time_to_visit` (0.143).

### 4. Chunking: one chunk per section (20 chunks)

The guides are already authored as short, topically pure sections whose
headings map almost one-to-one onto the questions users ask. Splitting on a
boundary the author already drew beats any fixed window: no sentence is cut in
half, no chunk mixes visa rules with packing advice, and every chunk carries a
city and section label usable as a citation.

Each chunk is embedded as `"{city}, {country} - {HEADING x3}: {text}"`. The
city prefix is not cosmetic — without it the four `PACKING TIPS` chunks embed
almost identically and the retriever cannot tell Tokyo from Bangkok.

### 5. Vector store: numpy, not FAISS/Chroma

Twenty chunks. A brute-force cosine scan is one matrix-vector product —
microseconds. A vector database here would add a dependency, a build step and a
persistence story to solve a problem this corpus does not have. See
[Scalability](#scalability-discussion) for what changes at a few hundred cities.

### 6. Weather: mock table (Option B)

The brief states Options A and B are scored identically, so the mock table wins
on determinism: tests are exact and repeatable, demos never fail on someone
else's rate limit, and no time goes into HTTP plumbing. It sits behind a
`WeatherProvider` Protocol, so an Open-Meteo client is one new class implementing
`lookup()` — no change to the service, tool, registry, or agent.

### 7. Per-embedder similarity floors — and what a floor can't do

Cosine scores from sparse and dense models aren't on the same scale: 0.25 is a
strong TF-IDF match and a weak MiniLM one. So the floor travels with the
embedder (`LexicalEmbedder.default_min_similarity = 0.10`,
`SentenceTransformer... = 0.25`), overridable via `TRIPMATE_MIN_SIMILARITY`.

Measuring the actual distribution produced a finding worth recording: for the
lexical embedder the ranges **overlap**. Genuine questions score 0.15–0.35, but
`"book me a flight to Paris"` scores 0.175 — higher than a legitimate
`"temple etiquette in Thailand"` at 0.150. **No threshold separates in-scope
from out-of-scope queries.**

The conclusion is architectural, not numerical: relevance scoring cannot do
scope detection, so the floor was given the narrower job it can actually do —
suppressing the *zero-overlap* case (uncovered cities and gibberish both score
exactly 0.000). Deciding a request is out of scope belongs to the agent and its
prompt.

### 8. Reasoning surfaced where the vendor exposes it

`LLMResponse` carries a `thinking` field that each provider fills in from
whatever its vendor exposes. The Anthropic provider requests
`thinking={"type": "adaptive", "display": "summarized"}`, so the trace shows the
model's *actual* reasoning about tool choice rather than a rationalisation
reconstructed afterwards. The Groq provider reads the message's `reasoning`
field where the model supplies one, and leaves it empty otherwise.

Either way the trace is still complete and useful: it always records which tools
were chosen, with what arguments, and what came back. The reasoning text is a
bonus on top, not the mechanism — which is deliberate, since a trace that only
worked on one vendor would defeat the point of the provider layer.

---

## Error handling

Failures are pushed **down**, never up. A raising tool would kill the loop; a
tool returning `tool_result(is_error=True)` lets the model see the problem and
recover — by retrying with better arguments, trying the other tool, or telling
the user. `ToolRegistry.dispatch()` is therefore total: it never raises.

| Scenario | Behaviour |
|---|---|
| Empty / whitespace / `None` input | Rejected before any API call, with a prompt to ask something |
| Over-long input (>2000 chars) | Rejected with the length stated |
| Unknown destination | Zero results + `covered_cities` → model names its real coverage |
| Missing weather data | `{"available": false, "reason": "unsupported_city", "supported_cities": [...]}` |
| Season given where a month is needed | `unparseable_date` → the model asks which month, rather than guessing across a hemisphere |
| Ambiguous query | System prompt instructs one short clarifying question |
| Tool raises / times out | Caught in the registry → `is_error` result → loop continues |
| Unknown tool name | Error result listing the valid tools |
| Bad / missing / extra arguments | Validated against the schema before execution |
| Auth failure, rate limit, 5xx, connection error | Classified most-specific-first; user gets a graceful message, never a fabricated answer |
| Model loops without settling | `max_iterations` guard (default 6) ends the run cleanly |
| Response truncated by `max_tokens` | Disclosed in the answer |

## Logging

Structured JSON, one object per line, to stderr and optionally a file
(`TRIPMATE_LOG_FILE` or `--log-file`):

```json
{"ts":"2026-09-01T12:41:49","level":"INFO","logger":"tripmate.tools.registry","event":"tool.call","tool":"get_weather_forecast","arguments":{"city":"Reykjavik","date_or_month":"December"},"status":"ok","duration_ms":0.3,"result_preview":"{\"available\": true, ..."}
```

Logged events: `rag.index_built`, `embedder.selected`, `agent.start`,
`agent.llm_turn` (with the planned calls and their arguments), `tool.call`
(name, arguments, status, duration, result preview), `rag.search`,
`weather.lookup_ok` / `weather.no_data`, `agent.done`, and every error path.

The reasoning trace is a **first-class object** (`agent/trace.py`), not print
statements. The orchestrator appends to it; two renderers consume it —
`to_dict()`/`to_json()` for logs and `examples/`, `render()` for `--verbose`.
Same data, two audiences.

## Testing

```bash
pytest                 # 109 tests, <1s, no API key, no network
pytest -m live         # opt-in: real API calls (needs a provider key)
```

| File | Covers |
|---|---|
| `test_rag_tool.py` | Chunking, stemming, embedding, retrieval ranking, city/country detection, misses, malformed guides |
| `test_weather_tool.py` | Date parsing (10 formats + 9 rejections), provider coverage for all 4×12 cells, unavailable paths |
| `test_registry.py` | Schema shape, validation, unknown tool, crashing tool → error outcome |
| `test_agent.py` | Single-tool, multi-tool, no-tool routing; one-turn result batching; input validation; provider failure; loop guard; refusals; trace contents |
| `test_providers.py` | Wire-format translation both ways for Groq and Anthropic; finish-reason normalisation; malformed tool-call JSON; factory + key resolution |
| `test_live_selection.py` | Real tool **selection** (opt-in) |

**An honest note on what the default suite proves.** The first five files drive
a `ScriptedProvider` that replays fixed responses. That makes them fast and
deterministic, and it genuinely verifies the orchestrator handles every path —
but since the script decides which tools get "chosen", it does **not** verify
the model's judgement. Dynamic tool selection is the core requirement, so it
gets real coverage in `test_live_selection.py`, excluded by default only because
it needs network and a key.

---

## Scalability discussion

*(Discussion only, as specified — not implemented.)*

**RAG from 4 cities to several hundred.** ~500 cities × 5 sections = ~2,500
chunks, still only a few MB — brute-force numpy would honestly still work. The
first real change is **persistence**: re-embedding on every startup becomes
seconds of latency, so the index moves to a build step with an on-disk artifact
keyed by a content hash. Past ~50k chunks I'd move to an ANN index (FAISS
HNSW, or pgvector if the data is already in Postgres) and add **metadata
pre-filtering** on city so the vector search runs over one destination's chunks
rather than all of them — the `city_filter` in `store.search()` is already that
seam. At that scale I'd also add reranking: retrieve top-50 by vector, rerank to
top-3 with a cross-encoder.

**Avoiding redundant tool calls.** Three layers. (1) A per-process cache on the
tool layer — weather lookups are pure functions of `(city, month)` and the guide
corpus is static, so `@lru_cache` is nearly free and needs no invalidation.
(2) A semantic cache above the agent: embed the query, and if it's within a
similarity threshold of a recent one, return the cached answer — this is what
actually kills cost, since paraphrases of the same twenty questions dominate
real traffic. (3) Within a conversation, the system prompt already tells the
model not to re-fetch what it has retrieved.

**Reducing LLM cost at volume.** The provider layer already helps here: cost is
a per-vendor property, so routing cheap/common queries to a small model and hard
ones to a large one is a factory change, not a rewrite. Beyond that, in order of
impact: **prompt caching** first —
the system prompt and tool schemas are byte-identical on every request and sit
in the stable prefix, so marking them cached turns the largest fixed input cost
into a cache read. Then **`output_config.effort`**: this is routing plus
summarising, not deep reasoning, so `low`/`medium` likely holds quality —
measured per route, not globally. Then the semantic cache above. Only then
consider a cheaper model, and I'd measure Opus at low effort before assuming a
smaller model is cheaper per *completed* task. Batch API (50% off) suits any
offline evaluation runs.

**Keeping tool-selection latency low as tools grow.** Two tools is trivially
fine; the cost is that every tool's description sits in every request's prefix.
At ~20+ tools I'd use the API's **tool search** server tool with
`defer_loading: true` on the long tail, so only relevant schemas are pulled into
context. Structurally, I'd group tools into a small set of coarse ones with a
discriminated `input_schema` rather than proliferating fine-grained tools, and —
if a specific high-volume intent dominated — put a fast classifier in front to
skip the selection turn entirely for that path, keeping the general agent as
the fallback.

---

## Known limitations

- **`gpt-oss-120b` calls tools sequentially, not in parallel.** Verified in the
  packing trace: its reasoning correctly identifies that *both* tools are
  needed, but it emits one call per turn, so the multi-tool flow costs 3 LLM
  turns instead of 2 — even with `parallel_tool_calls=True` on the request.
  Correctness is unaffected; latency and token cost are roughly 1.5×. Parallel
  tool calling turns out to be a model capability, not merely a request flag.
  No code change is needed to benefit from a model that has it: results are
  already batched into one `ToolResultsTurn` either way.
- **Groq rotates its model catalogue.** `llama-3.3-70b-versatile` had already
  been retired by the time this was built. On `404 model_not_found`, list what
  your account can actually see and set `TRIPMATE_MODEL` — the one-line command
  is in a comment in [config.py](tripmate/config.py). The free tier also
  rate-limits under repeated runs.
- **Tool selection is only as good as the model.** `gpt-oss-120b` handles the
  clear cases reliably, but it is smaller than Claude Opus and weaker on
  genuinely ambiguous queries. `--provider anthropic` swaps in a stronger model
  with no other change.
- **Retrieval quality degrades on the fallback embedder.** With TF-IDF,
  `"What should I wear in Bangkok?"` retrieves `safety_health` instead of
  `packing_tips` — "wear" has no lexical overlap with "clothing"/"packing".
  The semantic embedder handles it; this is exactly the gap it exists to close.
- **Weather is climate normals, not a forecast.** It cannot answer "will it rain
  next Tuesday". The tool description says so, and the payload carries a `note`.
- **Conversational memory is bounded and in-process.** The REPL carries the
  last `TRIPMATE_MAX_HISTORY_TURNS` exchanges (default 6) and nothing is
  persisted between processes. History is trimmed only at user-turn
  boundaries, because slicing mid-exchange would orphan a tool call from its
  result and both wire formats reject that. A turn that failed or hit the loop
  guard is dropped rather than replayed, since it ends on an unanswered tool
  call.
- **Four cities.** Everything outside them is correctly declined, not answered.
- **Single-city weather calls.** Comparing two cities takes two calls; the model
  handles that with parallel calls, but a `cities: list[str]` parameter would be
  cheaper.
- **The knowledge base is simplified assessment data**, not authoritative travel
  or visa advice. The prompt makes the model say so for visa questions.
- **Scope enforcement is prompt-based**, so it inherits prompt-injection risk —
  content in a guide instructing the model to ignore its rules would not be
  filtered.

## Future improvements

1. **Persistent conversations** — in-process history landed (see the REPL);
   writing it to disk would let a session survive a restart.
2. **Citations in the answer** — the retriever already returns a `citation` per
   chunk; surfacing them inline would make the RAG grounding checkable by the user.
3. **Prompt caching** on the stable tools+system prefix — the single highest-value
   production change.
4. **A real weather provider** — the Protocol is already there; Open-Meteo is one class.
5. **An eval set** — ~40 query/expected-tools pairs run as a scored suite, so
   prompt or description edits show their effect on routing accuracy instead of
   being judged by eyeball.
6. **Streaming** — first token far sooner on the synthesis turn.
7. **Retry with backoff** on 429/5xx; the errors are already classified as
   retryable vs fatal, but nothing acts on it yet.
