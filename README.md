# Build Your First AI Chatbot → AI Agent → RAG System

A step-by-step guide from "Hello World" chatbot to a tool-calling AI agent with MCP and RAG.

---

## Setup (Do This First!)

### 1. Install the packages

```bash
pip install gradio groq fastapi uvicorn requests fastmcp sentence-transformers
```

### 2. Get your Groq API key

1. Go to https://console.groq.com
2. Sign up (it's free)
3. Click "API Keys" → "Create API Key"
4. Copy your key

### 3. Set your API key

```bash
export GROQ_API_KEY=your_key_here
```

---

## Part 1: Build a Chatbot (Steps 1-4)

### Step 1: Hello Gradio

**File:** `step1_simple_chat.py`

This creates a simple chat interface that echoes your messages.

```bash
python step1_simple_chat.py
```

Open http://127.0.0.1:7860 and try typing something!

**What you learned:** Gradio makes it easy to create chat interfaces.

---

### Step 2: Add AI

**File:** `step2_gradio_with_groq.py`

Now your chatbot actually thinks! It connects to Groq's AI.

```bash
python step2_gradio_with_groq.py
```

Try asking it questions!

**What you learned:** You can connect to AI APIs with just a few lines of code.

---

### Step 3: Create Your Own API

**File:** `step3_fastapi_backend.py`

Turn your chatbot into an API that other apps can use.

```bash
python step3_fastapi_backend.py
```

Open http://localhost:8000/docs to see your API!

**What you learned:** FastAPI lets you create APIs easily.

---

### Step 4: Connect Frontend to Backend

**Files:** `step3_fastapi_backend.py` + `step4_gradio_frontend.py`

Run both together to see how real apps work.

**Terminal 1:**
```bash
python step3_fastapi_backend.py
```

**Terminal 2:**
```bash
python step4_gradio_frontend.py
```

**What you learned:** Real apps have separate frontend and backend.

---

## Part 2: Tool Calling — From Hallucination to Agents (Steps 5-9)

### Step 5: See the Problem — Hallucination

**File:** `step5_see_the_problem.py`

Ask the LLM real-time questions it CANNOT know. Watch it confidently make things up.

```bash
python step5_see_the_problem.py
```

**What you learned:** LLMs predict tokens, they don't verify truth. Hallucination = Uncertainty x Forced Response.

---

### Step 6: Your First Tool Call (The Manual Way)

**File:** `step6_manual_tool_call.py`

Connect the LLM to a weather function by forcing JSON output through the system prompt.

```bash
python step6_manual_tool_call.py
```

**The flow:**
```
User asks about weather
    → LLM outputs JSON (structured)
    → We parse JSON and call get_weather() (deterministic)
    → We send result back to LLM
    → LLM gives natural language answer (grounded in real data)
```

**What you learned:** You can bridge probabilistic LLMs and deterministic code using structured output. But the manual approach is fragile.

---

### Step 7: Proper Tool Calling with the API

**File:** `step7_tool_calling_api.py`

Use Groq's built-in tool calling — no manual JSON hacks. Define tools as JSON Schema, let the API handle everything.

```bash
python step7_tool_calling_api.py
```

**What you learned:** Modern APIs support structured tool calling with JSON Schema. This is reliable, standard, and what production systems use.

---

### Step 8: Multi-Tool Agent — The LLM as a Router

**File:** `step8_multi_tool_agent.py`

Give the LLM THREE tools (weather, calculator, contacts). It decides which tool to call based on the question — or answers directly when no tool is needed.

```bash
python step8_multi_tool_agent.py
```

**What you learned:** The LLM acts as an intelligent router. It reads tool descriptions and picks the right one. This is the "routing decision" from the lecture.

---

### Step 9: The Agentic Loop — Foundation of AI Agents

**File:** `step9_agentic_loop.py`

The core loop behind ALL AI agents: call LLM → execute tools → send results back → repeat until done.

```bash
python step9_agentic_loop.py
```

**What you learned:** The agentic loop is the foundation of AI Agents, RAG, and MCP. The LLM keeps calling tools until it has enough information to answer.

---

## Part 3: MCP — Model Context Protocol (Steps 10-13)

### Step 10: Your First MCP Server

**File:** `step10_mcp_server.py`

Build a tool server using FastMCP. One decorator replaces 15 lines of JSON Schema from step 7.

```bash
python step10_mcp_server.py
```

Inspect your server in the browser:
```bash
fastmcp dev step10_mcp_server.py
```

**What you learned:** MCP is a standard protocol for exposing tools. FastMCP auto-generates JSON Schemas from Python type hints.

---

### Step 11: Your First MCP Client

**File:** `step11_mcp_client.py`

Connect to the MCP server, discover available tools, and call them through the protocol.

```bash
python step11_mcp_client.py
```

**What you learned:** MCP clients discover tools at runtime. The client never sees the server's code — it uses the protocol. Any client can talk to any server.

---

### Step 12: Multi-Tool MCP Server

**File:** `step12_mcp_multi_tool.py`

Three tools (weather, calculator, contacts) in one MCP server. Compare with step 8: same tools, 1/10th the boilerplate.

```bash
python step12_mcp_multi_tool.py
```

Inspect all tools:
```bash
fastmcp dev step12_mcp_multi_tool.py
```

**What you learned:** MCP scales effortlessly. Add a tool = add a function with `@mcp.tool`. No schemas, no dispatch dicts, no registration code.

---

### Step 13: MCP + LLM — The Complete Picture

**File:** `step13_mcp_with_llm.py`

The capstone: MCP server provides tools, MCP client discovers them, LLM uses them in an agentic loop. This is how Claude Desktop, Cursor, and production AI systems work.

```bash
python step13_mcp_with_llm.py
```

**The flow:**
```
MCP server exposes tools (auto-generated schemas)
    → MCP client discovers tools (list_tools)
    → Convert to Groq format (MCP schema → Groq tool format)
    → LLM decides which tool to call
    → Execute via MCP (call_tool)
    → Send result back to LLM
    → Repeat until final answer
```

**What you learned:** MCP decouples tools from the agent. The agentic loop is identical to step 9 — only the tool source changed. This is the production pattern.

---

## Part 4: RAG — Retrieval-Augmented Generation (Steps 14-20)

### Step 14: The Scaling Problem — Why Tool Calling Isn't Enough

**File:** `step14_rag_the_problem.py`

You already built RAG with tool calling (R: call API, A: inject result, G: LLM answers). But that worked for 10 words. What about 10,000 pages? Stuff all data into the prompt and watch it break.

```bash
python step14_rag_the_problem.py
```

**The three forcing functions:**
```
1. Window Limits   → Data doesn't fit in context
2. Cost Explosion  → Every query sends ALL data ($$$$)
3. Attention Dilution → Answer gets lost in the noise
```

**What you learned:** Context stuffing doesn't scale. You need to send ONLY what's relevant. That's what RAG solves.

---

### Step 15: Chunking — Breaking Documents into Pieces

**File:** `step15_chunking.py`

The first step in building an index: break large documents into small, searchable pieces. Implement four strategies from scratch.

```bash
python step15_chunking.py
```

No API key needed — this is pure Python computation.

**What you learned:** Fixed-size chunking is simple but splits sentences. Sentence/paragraph chunking preserves meaning. Overlapping chunks prevent boundary information loss. Chunk size is a critical parameter.

---

### Step 16: Embeddings — Turning Text into Numbers

**File:** `step16_embeddings.py`

Why keyword search fails and how embeddings capture meaning. Build cosine similarity from scratch, then use real neural embeddings.

```bash
python step16_embeddings.py
```

**The key insight:**
```
Student asks: "Has anyone had trouble getting data from the weather API?"
Semantically = Tool Calling question. But "tool calling" never appears.

Keyword search: MISSES it (no matching words)
Semantic search: FINDS it (understands the meaning)
```

**What you learned:** Embeddings convert text to vectors where similar meaning = similar direction. Cosine similarity measures alignment. Semantic search finds relevant results even with different words.

---

### Step 17: Build Your Own Vector Store

**File:** `step17_vector_store.py`

Build the "index" from the textbook analogy. An in-memory vector store with add, search, and metadata filtering — the same thing Pinecone, Weaviate, and ChromaDB do internally.

```bash
python step17_vector_store.py
```

**What you learned:** A vector store = embeddings + metadata + similarity search. Metadata filtering (module, week, content_type) narrows the search space for better precision.

---

### Step 18: The Complete RAG Pipeline

**File:** `step18_rag_pipeline.py`

Wire everything together: chunking + embeddings + vector store + LLM. The full pipeline that turns your data into grounded answers.

```bash
python step18_rag_pipeline.py
```

**The pipeline:**
```
INDEX (offline, one-time)
    → Chunk documents → Embed chunks → Store with metadata

QUERY (per question)
    → Retrieve top-k chunks → Augment the prompt → Generate answer
```

**What you learned:** RAG uses 3 relevant chunks instead of 25 (or 25,000) documents. The augmented prompt grounds the LLM in real data. Metadata filtering makes retrieval precise.

---

### Step 19: RAGAS Evaluation — Measure, Don't Vibe Check

**File:** `step19_evaluation.py`

"I tried a few queries and it seemed okay" is not evaluation. Implement the four RAGAS metrics to diagnose exactly where your pipeline is failing.

```bash
python step19_evaluation.py
```

**The four metrics:**
```
RETRIEVAL METRICS (measure the retriever):
  Context Precision → Of retrieved, how much was relevant?  (Low = too much noise)
  Context Recall    → Of relevant, how much was found?      (Low = missing content)

GENERATION METRICS (measure the LLM):
  Faithfulness      → Answer supported by context?          (Low = hallucinating)
  Answer Relevancy  → Answer addresses the question?        (Low = off-target)
```

**What you learned:** Each metric points to a specific fix. Low precision = fix retriever filtering. Low faithfulness = fix system prompt. Metrics, not vibes.

---

### Step 20: RAG + Agentic Loop — The Full Picture

**File:** `step20_rag_with_agent.py`

The capstone: RAG as a tool in the agentic loop from step 9. The agent autonomously decides whether to search the knowledge base, call the weather API, or do a calculation.

```bash
python step20_rag_with_agent.py
```

**The decision framework:**
```
TOOL CALLING          RAG                    BOTH
─────────────         ────────────           ──────────────
Small data            Large data             Act AND retrieve
API-shaped            Document search        KB + actions
Real-time             Precision matters      MCP makes this clean
"What's the weather?" "When do we learn RAG?" "Search KB + create ticket"
```

**What you learned:** RAG is just another tool in the agentic loop. The loop from step 9 is identical — only the tools changed. Start with tool calling, graduate to RAG when you hit a wall.

---

## What You Built

```
Part 1: Chatbot
  Step 1:  Gradio (echo bot)
  Step 2:  Gradio → Groq AI
  Step 3:  FastAPI → Groq AI
  Step 4:  Gradio → FastAPI → Groq AI

Part 2: Tool Calling
  Step 5:  See hallucination (the problem)
  Step 6:  Manual tool call (JSON parsing — fragile)
  Step 7:  API tool calling (JSON Schema — reliable)
  Step 8:  Multi-tool routing (LLM picks the right tool)
  Step 9:  Agentic loop (autonomous multi-step reasoning)

Part 3: MCP (Model Context Protocol)
  Step 10: MCP server (standardized tool exposure)
  Step 11: MCP client (tool discovery + calling)
  Step 12: Multi-tool MCP server (scalable, minimal code)
  Step 13: MCP + LLM agentic loop (production-ready)

Part 4: RAG (Retrieval-Augmented Generation)
  Step 14: Context stuffing fails (the scaling problem)
  Step 15: Chunking (break documents into pieces)
  Step 16: Embeddings (text → vectors, semantic search)
  Step 17: Vector store (searchable index with metadata)
  Step 18: Full RAG pipeline (index → retrieve → augment → generate)
  Step 19: RAGAS evaluation (metrics, not vibes)
  Step 20: RAG + agentic loop (the complete system)
```

---

## Quick Fixes

**"Module not found"** → Run: `pip install gradio groq fastapi uvicorn requests fastmcp sentence-transformers`

**"API key error"** → Run: `export GROQ_API_KEY=your_key_here`

**"Connection refused"** → Make sure step3 is running before step4

**MCP client can't connect** → Make sure the server file (e.g., step10) is in the same directory

**sentence-transformers slow to load** → First run downloads the model (~80MB). Subsequent runs use the cached version.
