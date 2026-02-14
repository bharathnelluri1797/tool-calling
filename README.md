# Build Your First AI Chatbot → AI Agent

A step-by-step guide from "Hello World" chatbot to a tool-calling AI agent with MCP.

---

## Setup (Do This First!)

### 1. Install the packages

```bash
pip install gradio groq fastapi uvicorn requests fastmcp
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
```

---

## Quick Fixes

**"Module not found"** → Run: `pip install gradio groq fastapi uvicorn requests fastmcp`

**"API key error"** → Run: `export GROQ_API_KEY=your_key_here`

**"Connection refused"** → Make sure step3 is running before step4

**MCP client can't connect** → Make sure the server file (e.g., step10) is in the same directory
