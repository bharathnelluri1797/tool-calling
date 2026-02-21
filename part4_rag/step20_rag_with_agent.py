# ============================================
# STEP 20: RAG + Agentic Loop — The Full Picture
# ============================================
#
# WHAT YOU'LL LEARN:
# - How RAG fits as a TOOL in the agentic loop (from step 9)
# - When to use tool calling vs RAG vs BOTH
# - The agent autonomously decides: search knowledge base OR call API
#
# KEY IDEA (from lecture):
#   TOOL CALLING: Small data, API-shaped, real-time
#     "What's the weather?" → call weather API
#
#   RAG: Large data, document search, precision matters
#     "What's the refund policy for Q3?" → search knowledge base
#
#   BOTH: Act AND retrieve
#     "Search KB for the user's issue + create a support ticket"
#     MCP makes this clean: resources + tools
#
#   Don't over-engineer. Start with tool calling.
#   Graduate to RAG when you hit a wall.
#
# THIS IS THE CAPSTONE:
#   Step 9:  Agentic loop with hardcoded tools (weather, calculator)
#   Step 13: Agentic loop with tools from MCP
#   Step 20: Agentic loop with RAG as a tool
#
#   The loop is IDENTICAL. Only the tools changed.
#   RAG is just another tool the LLM can decide to use.
#
# RUN THIS FILE:
#   python step20_rag_with_agent.py
# ============================================

import os
import json
import math
from groq import Groq
from sentence_transformers import SentenceTransformer
from rag_documents import CURRICULUM_DOCUMENTS

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
MODEL = "openai/gpt-oss-20b"


# ============================================
# PART 1: Build the RAG Knowledge Base
# ============================================
# Same vector store from step 17, now used as a tool.

class KnowledgeBase:
    """Vector store wrapped as a searchable knowledge base."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def index(self, documents):
        texts = [d["content"] for d in documents]
        self.embeddings = embedding_model.encode(texts).tolist()
        self.documents = documents

    def search(self, query, top_k=3):
        query_emb = embedding_model.encode(query).tolist()
        scored = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = self._cosine_sim(query_emb, doc_emb)
            scored.append({
                "content": self.documents[i]["content"],
                "metadata": self.documents[i].get("metadata", {}),
                "similarity": round(sim, 4),
            })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def _cosine_sim(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        ma = math.sqrt(sum(x*x for x in a))
        mb = math.sqrt(sum(x*x for x in b))
        return dot / (ma * mb) if ma and mb else 0.0


# Index the curriculum
print("Indexing knowledge base...")
kb = KnowledgeBase()
kb.index([{"content": d["content"], "metadata": d["metadata"]} for d in CURRICULUM_DOCUMENTS])
print(f"Knowledge base ready: {len(CURRICULUM_DOCUMENTS)} documents indexed.\n")


# ============================================
# PART 2: Define Tools (RAG + Traditional)
# ============================================
# The agent now has THREE tools:
#   1. search_knowledge_base — RAG retrieval (searches curriculum)
#   2. get_weather — API-style tool call (from step 9)
#   3. calculate — deterministic computation (from step 9)
#
# The LLM decides which to use based on the question.

def search_knowledge_base(query, top_k=3):
    """Search the 100xEngineers curriculum knowledge base."""
    results = kb.search(query, top_k=int(top_k))
    return {
        "results": [
            {
                "content": r["content"],
                "week": r["metadata"].get("week"),
                "module": r["metadata"].get("module"),
                "type": r["metadata"].get("content_type"),
                "relevance": r["similarity"],
            }
            for r in results
        ]
    }


def get_weather(city):
    """Get current weather for a city."""
    data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy", "humidity": 65},
        "delhi": {"temp_c": 42, "condition": "Extreme Heat", "humidity": 25},
        "mumbai": {"temp_c": 32, "condition": "Humid", "humidity": 80},
        "london": {"temp_c": 15, "condition": "Rainy", "humidity": 90},
    }
    return data.get(city.lower(), {"error": f"No weather data for {city}"})


def calculate(expression):
    """Evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception:
        return {"error": f"Cannot calculate: {expression}"}


available_tools = {
    "search_knowledge_base": search_knowledge_base,
    "get_weather": get_weather,
    "calculate": calculate,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the 100xEngineers curriculum knowledge base. Use this for ANY question about the course, lectures, assignments, coding exercises, or student discussions. Returns the most relevant documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query — describe what you're looking for"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default: 3)", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Use this for real-time weather questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use this for calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '42 - 28'"},
                },
                "required": ["expression"],
            },
        },
    },
]


# ============================================
# PART 3: The Agentic Loop (identical to step 9!)
# ============================================
# The loop doesn't change. At all.
# We just added a new tool (search_knowledge_base).
# The LLM figures out when to use it.

def run_agent(user_message):
    """Run the agentic loop with RAG + traditional tools."""

    print(f"  User: {user_message}")
    print()

    messages = [
        {
            "role": "system",
            "content": (
                "You are Sage, a helpful AI assistant for the 100xEngineers program. "
                "You have access to three tools:\n"
                "1. search_knowledge_base — Search course curriculum, lectures, and forum threads\n"
                "2. get_weather — Get real-time weather data\n"
                "3. calculate — Perform calculations\n\n"
                "For questions about the course, ALWAYS use search_knowledge_base first. "
                "Base your answers on the retrieved content. "
                "If the knowledge base doesn't have the answer, say so."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    step = 1

    while True:
        print(f"    [Step {step}] Calling LLM...")

        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # EXIT: No tool calls = final answer
        if not message.tool_calls:
            print(f"    [Step {step}] Final answer.")
            print()
            print(f"  Sage: {message.content}")
            return message.content

        # TOOL CALLS: Execute each one
        messages.append(message)

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            # Show which tool is being called
            if fn_name == "search_knowledge_base":
                print(f"    [Step {step}] 📚 RAG search: \"{fn_args.get('query', '')}\"")
            else:
                print(f"    [Step {step}] 🔧 Tool call: {fn_name}({fn_args})")

            result = available_tools[fn_name](**fn_args)
            result_str = json.dumps(result)

            # Show result summary
            if fn_name == "search_knowledge_base":
                num_results = len(result.get("results", []))
                print(f"    [Step {step}] Retrieved {num_results} documents")
            else:
                print(f"    [Step {step}] Result: {result_str[:100]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })

        step += 1

        if step > 5:
            print("    [Max steps reached]")
            return "Sorry, I couldn't complete the task."


# ============================================
# PART 4: Test — RAG Questions (Knowledge Base)
# ============================================

print("=" * 60)
print("TEST 1: RAG — Course Questions (uses knowledge base)")
print("=" * 60)
print()

run_agent("When will we learn about RAG in the 100xEngineers cohort?")

print()
print("=" * 60)
print("TEST 2: RAG — Debugging Question (finds forum thread)")
print("=" * 60)
print()

run_agent("My agent keeps calling the same tool over and over. How do I fix this?")


# ============================================
# PART 5: Test — Tool Calling Questions (APIs)
# ============================================

print()
print("=" * 60)
print("TEST 3: Tool Calling — Weather (uses API)")
print("=" * 60)
print()

run_agent("What's the weather in Bengaluru right now?")


# ============================================
# PART 6: Test — BOTH (RAG + Tool Calling)
# ============================================

print()
print("=" * 60)
print("TEST 4: BOTH — RAG + Tool Calling Combined")
print("=" * 60)
print()

run_agent("What week do we cover tool calling in the course, and what is the current temperature in Delhi?")

print()
print("=" * 60)
print("TEST 5: No Tools Needed — General Knowledge")
print("=" * 60)
print()

run_agent("What is the capital of France?")


# ============================================
# PART 7: The Decision Framework
# ============================================

print()
print("=" * 60)
print("THE DECISION FRAMEWORK (from lecture)")
print("=" * 60)

print("""
  ┌───────────────────┬────────────────────┬────────────────────────┐
  │ TOOL CALLING      │ RAG                │ BOTH                   │
  ├───────────────────┼────────────────────┼────────────────────────┤
  │ Small data        │ Large data         │ Act AND retrieve       │
  │ Simple queries    │ Document search    │ KB + actions           │
  │ API-shaped        │ Precision matters  │ MCP makes this clean   │
  │ Real-time data    │ Cost per query     │                        │
  ├───────────────────┼────────────────────┼────────────────────────┤
  │ "What's the       │ "What's the refund │ "Search KB for the     │
  │  weather?"        │  policy for Q3?"   │  issue + create ticket"│
  └───────────────────┴────────────────────┴────────────────────────┘

  Don't over-engineer. Start with tool calling.
  Graduate to RAG when you hit a wall.

  RAG is NOT search + LLM.
  RAG is the core of Context Engineering.
""")

print("=" * 60)

# ============================================
# THE COMPLETE JOURNEY (Steps 1-20):
#
#   Part 1: Chatbot (Steps 1-4)
#     Build UI → Add AI → Create API → Connect frontend to backend
#
#   Part 2: Tool Calling (Steps 5-9)
#     See hallucination → Manual fix → API tool calling →
#     Multi-tool routing → Agentic loop (FOUNDATION)
#
#   Part 3: MCP (Steps 10-13)
#     MCP server → MCP client → Multi-tool MCP →
#     MCP + LLM integration (PRODUCTION PATTERN)
#
#   Part 4: RAG (Steps 14-20)
#     Context doesn't scale → Chunking → Embeddings →
#     Vector store → Full pipeline → RAGAS evaluation →
#     RAG + Agentic loop (COMPLETE SYSTEM)
#
# EXERCISE 1:
# Add a fourth tool: "create_support_ticket" that logs a
# student's issue. Test the "BOTH" pattern from the lecture:
#   "I'm getting a 422 error with the weather API. Search the
#    knowledge base for a fix and create a support ticket."
# The agent should search KB AND create a ticket.
#
# EXERCISE 2:
# Add metadata filtering to the search tool:
#   - Modify search_knowledge_base to accept an optional "module" param
#   - The LLM can now say: search_knowledge_base(query="...", module="rag")
#   - This gives the agent more precise control over retrieval
#
# EXERCISE 3:
# Build a Gradio chat interface for Sage:
#   import gradio as gr
#   def respond(message, history):
#       return run_agent(message)
#   gr.ChatInterface(respond, title="Sage - 100xEngineers Assistant").launch()
# Now you have a web-based AI assistant with RAG + tools!
#
# EXERCISE 4 (Advanced):
# Replace the in-memory knowledge base with an MCP resource:
#   - Create an MCP server that exposes search_knowledge_base as a tool
#   - Connect via MCP client (like step 13)
#   - Now the knowledge base is decoupled from the agent
#   - Any agent can connect to your KB through MCP
#
# EXERCISE 5 (Advanced):
# Implement streaming responses. Instead of waiting for the
# full answer, stream tokens as they arrive:
#   response = groq_client.chat.completions.create(..., stream=True)
#   for chunk in response:
#       print(chunk.choices[0].delta.content, end="")
# Combine with Gradio streaming for a real-time chat experience.
# ============================================
