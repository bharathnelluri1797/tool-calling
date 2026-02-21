# ============================================
# RAG Documents — Shared Dataset for Steps 14-20
# ============================================
#
# This file contains the mock 100xEngineers curriculum
# used across all RAG exercises.
#
# WHY A SHARED FILE?
#   Each RAG step builds on the same dataset.
#   Rather than copy-pasting 25 documents into every file,
#   we import from here. This is also good engineering practice.
#
# THE DATASET:
#   25 curriculum items spanning 7 weeks of learning:
#   - Lectures, code walkthroughs, forum threads, assignments
#   - Topics: LLM basics, prompt engineering, API dev, tool calling, RAG, MCP, agents
#   - Rich metadata: week, module, content_type, resolved (for threads)
#
# USED BY:
#   step14_rag_the_problem.py  (sees the scaling problem)
#   step15_chunking.py         (breaks documents into chunks)
#   step16_embeddings.py       (turns text into vectors)
#   step17_vector_store.py     (builds searchable index)
#   step18_rag_pipeline.py     (full RAG pipeline)
#   step19_evaluation.py       (measures RAG quality)
#   step20_rag_with_agent.py   (RAG + agentic loop)
# ============================================

CURRICULUM_DOCUMENTS = [
    {
        "id": "doc_01",
        "content": "Week 1 - Introduction to LLMs: Large Language Models are neural networks trained on massive text datasets. They predict the next token in a sequence. Key concepts: tokens, context window, temperature, top-p sampling. The model doesn't 'understand' — it predicts statistical patterns. GPT, LLaMA, and Mixtral are popular LLM families. Each has different strengths in reasoning, speed, and cost.",
        "metadata": {"week": 1, "module": "llm_basics", "content_type": "lecture"},
    },
    {
        "id": "doc_02",
        "content": "Week 1 - Setting Up Your Environment: Install Python 3.11+, create a virtual environment with python -m venv myenv, activate it, and install the groq library with pip install groq. Get your API key from console.groq.com. Test with a simple chat completion call. Always store API keys in environment variables, never hardcode them in your scripts.",
        "metadata": {"week": 1, "module": "llm_basics", "content_type": "code_walkthrough"},
    },
    {
        "id": "doc_03",
        "content": "Week 2 - Prompt Engineering: System prompts set the AI's behavior and personality. User prompts provide the actual input. Few-shot examples dramatically improve output quality by showing the model what you expect. Chain-of-thought prompting helps with complex reasoning tasks — ask the model to 'think step by step'. Temperature controls creativity: 0 is deterministic, 1 is creative.",
        "metadata": {"week": 2, "module": "prompt_engineering", "content_type": "lecture"},
    },
    {
        "id": "doc_04",
        "content": "Week 2 - Building a Chatbot UI with Gradio: Use gr.ChatInterface to create a web-based chat interface in 5 lines of code. Gradio handles message history automatically. Connect it to Groq API for AI-powered responses. The UI runs locally on port 7860. You can share it publicly with share=True for demos.",
        "metadata": {"week": 2, "module": "prompt_engineering", "content_type": "code_walkthrough"},
    },
    {
        "id": "doc_05",
        "content": "Week 3 - API Development with FastAPI: Build REST APIs with FastAPI. Define endpoints with @app.post and @app.get decorators. Use Pydantic models for automatic request/response validation. Run with uvicorn. FastAPI auto-generates interactive API docs at /docs endpoint. This is how you separate your frontend from your backend in production applications.",
        "metadata": {"week": 3, "module": "api_development", "content_type": "lecture"},
    },
    {
        "id": "doc_06",
        "content": "Week 3 - Forum Thread: 'CORS error when connecting Gradio frontend to FastAPI backend' — Solution: Add CORS middleware to FastAPI. Import CORSMiddleware from fastapi.middleware.cors. Set allow_origins=['*'] for development. In production, restrict to your actual frontend domain. This fixed cross-origin request issues for multiple students.",
        "metadata": {"week": 3, "module": "api_development", "content_type": "forum_thread", "resolved": True},
    },
    {
        "id": "doc_07",
        "content": "Week 4 - Tool Calling Fundamentals: LLMs hallucinate when they lack real-time data. The solution: give them tools. Define functions as JSON Schema with name, description, and parameters. The LLM decides WHEN to call a tool and WHAT arguments to pass. You execute the function deterministically and return the results. The LLM then generates a grounded response.",
        "metadata": {"week": 4, "module": "tool_calling", "content_type": "lecture"},
    },
    {
        "id": "doc_08",
        "content": "Week 4 - Building a Weather Agent: Create a get_weather tool that calls the weather API. Define the tool schema with a city parameter (type: string). When the user asks 'What's the weather in Bengaluru?', the LLM calls get_weather('Bengaluru'), you execute the API call, return temp=28°C with condition='Partly Cloudy', and the LLM responds with grounded data instead of hallucinating.",
        "metadata": {"week": 4, "module": "tool_calling", "content_type": "code_walkthrough"},
    },
    {
        "id": "doc_09",
        "content": "Week 4 - The Agentic Loop Pattern: The core loop behind ALL AI agents. Architecture: send message to LLM with tool definitions → if LLM returns tool_calls, execute them and send results back → if LLM returns text, that's the final answer → repeat. This handles multi-step reasoning automatically. Example: 'Compare weather in Delhi and Mumbai' requires two tool calls before the LLM can compare.",
        "metadata": {"week": 4, "module": "tool_calling", "content_type": "lecture"},
    },
    {
        "id": "doc_10",
        "content": "Week 4 - Forum Thread: 'My agent keeps calling the same tool in an infinite loop' — Solution: Add a max_steps counter to your agentic loop. Set if step > 5: break as a safety limit. Also verify you're appending tool results correctly — use role='tool' with the matching tool_call_id. Without the tool_call_id, the LLM doesn't know which call the result belongs to.",
        "metadata": {"week": 4, "module": "tool_calling", "content_type": "forum_thread", "resolved": True},
    },
    {
        "id": "doc_11",
        "content": "Week 5 - RAG from First Principles: RAG = Retrieval + Augmentation + Generation. You already built this with tool calling! Step 1: Called weather API (retrieval). Step 2: Injected result into prompt (augmentation). Step 3: LLM answered with real data (generation). But tool calling returns ~10 words. What happens when the answer is buried in 10,000 pages of documentation?",
        "metadata": {"week": 5, "module": "rag", "content_type": "lecture"},
    },
    {
        "id": "doc_12",
        "content": "Week 5 - Why Context Stuffing Fails: Three forcing functions make naive approaches break. 1) Window Limits: 6 months of curriculum literally doesn't fit in any context window. 2) Cost Explosion: 300 students × multiple queries per day at maximum context = business-killing token costs. 3) Attention Dilution: Even if it fits, the model loses the relevant answer buried in the middle of thousands of irrelevant lines.",
        "metadata": {"week": 5, "module": "rag", "content_type": "lecture"},
    },
    {
        "id": "doc_13",
        "content": "Week 5 - Chunking Strategies for RAG: Break documents into smaller pieces for retrieval. Fixed-size chunks (e.g., 200 words) are simple but may split mid-sentence. Sentence-based chunking preserves meaning boundaries. Overlapping chunks (e.g., 50-word overlap) maintain context across chunk boundaries. The right chunk size depends on your data — too small loses context, too large adds noise.",
        "metadata": {"week": 5, "module": "rag", "content_type": "code_walkthrough"},
    },
    {
        "id": "doc_14",
        "content": "Week 5 - Embeddings and Vector Search: Convert text to high-dimensional vectors using embedding models like all-MiniLM-L6-v2. Similar meaning = similar vectors (close in vector space). Cosine similarity measures how aligned two vectors are (1.0 = identical meaning, 0.0 = unrelated). Store embeddings in a vector store. At query time: embed the question, find the closest document vectors, return top-k results.",
        "metadata": {"week": 5, "module": "rag", "content_type": "lecture"},
    },
    {
        "id": "doc_15",
        "content": "Week 5 - Forum Thread: 'Has anyone had trouble getting their data from the weather API?' — I was getting 422 errors when calling the weather tool. The issue was my city parameter had extra whitespace from user input. Fixed it with city.strip() before passing to the API. Also make sure your API key is correctly set in your .env file and loaded with os.environ.get().",
        "metadata": {"week": 5, "module": "tool_calling", "content_type": "forum_thread", "resolved": True},
    },
    {
        "id": "doc_16",
        "content": "Week 5 - RAGAS Evaluation Framework: Stop evaluating RAG with vibes. Use four metrics: Context Precision — of what was retrieved, how much was actually relevant? Context Recall — of what should have been retrieved, how much was found? Faithfulness — is the answer actually supported by the retrieved context? Answer Relevancy — does the answer address what was actually asked?",
        "metadata": {"week": 5, "module": "rag", "content_type": "lecture"},
    },
    {
        "id": "doc_17",
        "content": "Week 6 - Model Context Protocol (MCP): MCP is an open standard that decouples tool definitions from tool usage. Server defines tools with @mcp.tool decorator — FastMCP auto-generates JSON schemas from Python type hints and docstrings. Client discovers tools via list_tools() at runtime. The LLM doesn't know or care where tools come from — MCP abstracts the connection.",
        "metadata": {"week": 6, "module": "mcp", "content_type": "lecture"},
    },
    {
        "id": "doc_18",
        "content": "Week 6 - Building an MCP Server with FastMCP: Create a server with mcp = FastMCP('My Tools'). Add tools as decorated Python functions: @mcp.tool. Type hints become parameter schemas. Docstrings become tool descriptions. Run directly or inspect with fastmcp dev server.py. This replaces 15+ lines of manual JSON Schema from step 7 with a single decorator.",
        "metadata": {"week": 6, "module": "mcp", "content_type": "code_walkthrough"},
    },
    {
        "id": "doc_19",
        "content": "Week 6 - MCP + LLM Integration: Connect MCP tools to the agentic loop. 1) Discover tools from MCP server via client.list_tools(). 2) Convert MCP schemas to Groq format. 3) LLM decides which tool to call. 4) Execute via mcp_client.call_tool(name, args). 5) Return result to LLM. The agentic loop is identical to step 9 — only the tool source changed from hardcoded to MCP.",
        "metadata": {"week": 6, "module": "mcp", "content_type": "code_walkthrough"},
    },
    {
        "id": "doc_20",
        "content": "Week 6 - Forum Thread: 'MCP server not connecting from Claude Desktop' — Make sure your server is running before connecting. Check that the transport matches (stdio vs SSE). For Claude Desktop, add the server config to claude_desktop_config.json with the correct command and args. Restart Claude Desktop completely after config changes. Check logs at ~/Library/Logs/Claude/ for errors.",
        "metadata": {"week": 6, "module": "mcp", "content_type": "forum_thread", "resolved": True},
    },
    {
        "id": "doc_21",
        "content": "Assignment 1 - Build a Menu Chatbot: Create a chatbot that answers questions about a restaurant menu. Use Gradio for the UI and Groq for the LLM. The chatbot should handle greetings, menu questions, and price inquiries. Include at least 10 menu items. System prompt should restrict responses to menu-related topics only. Due: End of Week 2.",
        "metadata": {"week": 2, "module": "prompt_engineering", "content_type": "assignment"},
    },
    {
        "id": "doc_22",
        "content": "Assignment 2 - Multi-Tool Agent: Build an agent with at least 3 tools: one for data retrieval, one for calculation, and one of your choice. Implement the full agentic loop from step 9. The agent must handle multi-step questions that require chaining tool calls. Test with at least 5 different queries. Due: End of Week 4.",
        "metadata": {"week": 4, "module": "tool_calling", "content_type": "assignment"},
    },
    {
        "id": "doc_23",
        "content": "Assignment 3 - Build a RAG System: Build a RAG pipeline for a custom dataset of your choice (PDF, website content, documentation). Implement chunking with at least 2 strategies. Use embeddings for semantic search. Connect to an LLM for answer generation. Evaluate with at least 2 RAGAS metrics (context precision + faithfulness minimum). Due: End of Week 5.",
        "metadata": {"week": 5, "module": "rag", "content_type": "assignment"},
    },
    {
        "id": "doc_24",
        "content": "Week 7 - AI Agents in Production: Deploy agents with proper error handling, rate limiting, and monitoring. Use structured logging to track every tool call and LLM response. Implement fallback strategies when tools fail. Cache frequent queries to reduce API costs. Monitor token usage per user. The gap between a working demo and a production system is operational excellence.",
        "metadata": {"week": 7, "module": "agents", "content_type": "lecture"},
    },
    {
        "id": "doc_25",
        "content": "Week 5 - The Textbook Analogy: You don't read all 600 pages to answer one question. You use the INDEX: find the topic → navigate to the chapter → locate the page → read the paragraph → get the answer. RAG builds this index for ANY dataset. Without an index, retrieval is impossible. The quality of your index determines the quality of your retrieval, which determines the quality of your answers.",
        "metadata": {"week": 5, "module": "rag", "content_type": "lecture"},
    },
]


# ============================================
# Helper: Get all content as a single string
# (Used in step14 to show the scaling problem)
# ============================================
def get_all_content():
    """Return all curriculum content as one big string."""
    return "\n\n".join(doc["content"] for doc in CURRICULUM_DOCUMENTS)


def get_all_content_with_metadata():
    """Return all curriculum content with metadata as formatted text."""
    lines = []
    for doc in CURRICULUM_DOCUMENTS:
        meta = doc["metadata"]
        header = f"[Week {meta['week']} | {meta['module']} | {meta['content_type']}]"
        lines.append(f"{header}\n{doc['content']}")
    return "\n\n".join(lines)
