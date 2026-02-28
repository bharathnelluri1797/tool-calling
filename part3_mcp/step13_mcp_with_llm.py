# ============================================
# STEP 13: MCP + LLM — The Complete Picture
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to wire MCP tools to an LLM (the production pattern)
# - How to convert MCP tool definitions to Groq's format
# - The full flow: MCP server -> MCP client -> LLM -> tool execution -> repeat
#
# KEY IDEA:
#   This is where EVERYTHING connects.
#
#   Step 9:  Agentic loop with hardcoded tools
#   Step 13: Agentic loop with tools DISCOVERED from MCP
#
#   The LLM doesn't know or care where the tools come from.
#   MCP provides them through a standard protocol.
#   The LLM uses them through the same tool calling API.
#
#   THE FLOW:
#     1. MCP server exposes tools (with auto-generated schemas)
#     2. MCP client discovers tools (list_tools)
#     3. We convert MCP schemas -> Groq tool format
#     4. LLM receives tools and decides which to call
#     5. We execute the call via MCP (call_tool)
#     6. Result goes back to LLM
#     7. Repeat until LLM has a final answer
#
#   This is how production AI systems work.
#   This is the foundation of Claude Desktop, Cursor, Windsurf, etc.
#
# RUN THIS FILE:
#   python step13_mcp_with_llm.py
# ============================================

import os
import json
import asyncio
from groq import Groq
from fastmcp import FastMCP, Client

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


# ============================================
# PART 1: Define Tools as an MCP Server
# ============================================
# Same tools as step 9, but now registered via MCP.
# The LLM will discover these through the protocol,
# not through hardcoded JSON schemas.

mcp = FastMCP("Agent Tools")


@mcp.tool
def get_weather(city: str) -> dict:
    """Get current weather for a city. Returns temperature, condition, and humidity.

    Args:
        city: The name of the city (e.g., 'Bengaluru', 'Delhi', 'Mumbai', 'London')
    """
    data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy", "humidity": 65},
        "delhi": {"temp_c": 42, "condition": "Extreme Heat", "humidity": 25},
        "mumbai": {"temp_c": 32, "condition": "Humid", "humidity": 80},
        "london": {"temp_c": 15, "condition": "Rainy", "humidity": 90},
    }
    return data.get(city.lower(), {"error": f"No weather data for {city}"})


@mcp.tool
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A math expression, e.g. '42 - 28' or '15 * 3.14'
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception:
        return {"error": f"Cannot calculate: {expression}"}


# ============================================
# PART 2: Convert MCP Tools to Groq Format
# ============================================
# MCP tools have: name, description, inputSchema
# Groq expects: {"type": "function", "function": {"name", "description", "parameters"}}
#
# This bridge function translates between the two formats.
# It's simple because MCP's inputSchema IS a JSON Schema —
# exactly what Groq's parameters field expects.

def mcp_tools_to_groq_format(mcp_tools):
    """Convert MCP tool definitions to Groq/OpenAI tool format."""
    groq_tools = []
    for tool in mcp_tools:
        groq_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            }
        })
    return groq_tools


# ============================================
# PART 3: The Agentic Loop (now powered by MCP)
# ============================================
# Same loop as step 9, but tool calls go through MCP.
# The LLM decides WHAT to call. MCP handles HOW to call it.

async def run_agent(user_message, mcp_client, groq_tools):
    """Run the agentic loop with MCP-provided tools."""

    print(f"User: {user_message}")
    print()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when needed to get real data."
        },
        {"role": "user", "content": user_message},
    ]

    step = 1

    while True:
        print(f"  [Step {step}] Calling LLM...")

        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # EXIT: No tool calls = final answer
        if not message.tool_calls:
            print(f"  [Step {step}] Final answer.")
            print()
            print(f"Agent: {message.content}")
            return

        # TOOL CALLS: Execute each one via MCP
        messages.append(message)

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"  [Step {step}] MCP call: {fn_name}({fn_args})")

            # Call the tool through MCP protocol (not a direct Python call!)
            result = await mcp_client.call_tool(fn_name, fn_args)

            # Extract the result data
            # .data gives us the Python object (dict, str, int, etc.)
            result_str = json.dumps(result.data) if result.data is not None else str(result)

            print(f"  [Step {step}] Result:   {result_str}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })

        step += 1

        if step > 5:
            print("  [Max steps reached]")
            return


# ============================================
# PART 4: Wire It All Together
# ============================================

async def main():
    # Connect to MCP server (in-process for this demo)
    async with Client(mcp) as mcp_client:

        # Step 1: DISCOVER tools from MCP
        mcp_tools = await mcp_client.list_tools()
        groq_tools = mcp_tools_to_groq_format(mcp_tools)

        print("=" * 60)
        print("TOOLS DISCOVERED VIA MCP:")
        for t in groq_tools:
            print(f"  - {t['function']['name']}: {t['function']['description']}")
        print("=" * 60)
        print()

        # Step 2: Run agent — tools come from MCP, intelligence from LLM
        await run_agent(
            "Compare the weather in Delhi and London. Which is hotter and by how much?",
            mcp_client,
            groq_tools,
        )

        print()
        print("=" * 60)
        print()

        await run_agent(
            "What is the temperature difference between Mumbai and Bengaluru? Calculate it.",
            mcp_client,
            groq_tools,
        )


if __name__ == "__main__":
    asyncio.run(main())

# ============================================
# THE COMPLETE JOURNEY (Steps 5-13):
#
#   Step 5:  LLM hallucinates (the problem)
#   Step 6:  Manual tool call via JSON parsing (fragile)
#   Step 7:  API tool calling with JSON Schema (reliable)
#   Step 8:  Multiple tools + routing (intelligent)
#   Step 9:  Agentic loop (autonomous)
#   Step 10: MCP server (standardized tool exposure)
#   Step 11: MCP client (standardized tool discovery)
#   Step 12: Multi-tool MCP server (scalable)
#   Step 13: MCP + LLM agentic loop (production-ready)
#
# COMPARE step 9 vs step 13:
#   Step 9:  Tools hardcoded as JSON Schema + Python dict
#   Step 13: Tools discovered from MCP server at runtime
#
#   The agentic loop is IDENTICAL. Only the tool SOURCE changed.
#   That's the whole point of MCP — decouple tools from the agent.
#
# THIS IS HOW REAL AI PRODUCTS WORK:
#   - Claude Desktop uses MCP to connect to local tools
#   - Cursor/Windsurf use MCP for code tools
#   - Enterprise agents use MCP for internal APIs
#   - You can publish MCP servers for others to use
#
# EXERCISE 1:
# Add a third MCP tool (e.g., search_contacts from step 12).
# You don't need to change run_agent() at all!
# The agent discovers and uses it automatically.
#
# EXERCISE 2:
# Change Client(mcp) to Client("step12_mcp_multi_tool.py").
# Now the agent uses tools from a SEPARATE server process.
# The code doesn't change — only the connection target.
#
# EXERCISE 3 (Advanced):
# Connect to TWO MCP servers at once:
#   async with Client("server1.py") as client1, Client("server2.py") as client2:
#       tools1 = await client1.list_tools()
#       tools2 = await client2.list_tools()
#       all_tools = mcp_tools_to_groq_format(tools1 + tools2)
# Now the agent has tools from multiple providers!
#
# EXERCISE 4 (Advanced):
# Wrap this in a Gradio chat interface (combine with step 2).
# You'll have a web UI where the agent uses MCP tools.
# ============================================
