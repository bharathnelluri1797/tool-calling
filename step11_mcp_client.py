# ============================================
# STEP 11: Your First MCP Client
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to connect to an MCP server
# - How to DISCOVER tools (list what's available)
# - How to CALL tools through the MCP protocol
# - Why this is more powerful than direct function calls
#
# KEY IDEA:
#   In step 8, you had to manually maintain a dictionary:
#     available_tools = {"get_weather": get_weather, ...}
#
#   With MCP, the CLIENT doesn't need to know the tools in advance.
#   It DISCOVERS them at runtime by asking the server.
#
#   This means:
#   - Server can add/remove tools without changing the client
#   - Client can connect to ANY MCP server and use its tools
#   - Tools are truly decoupled from the code that uses them
#
# BEFORE RUNNING:
#   Make sure step10_mcp_server.py exists in the same directory.
#   The client will automatically start it as a subprocess.
#
# RUN THIS FILE:
#   python step11_mcp_client.py
# ============================================

import asyncio
import json
from fastmcp import Client


# MCP clients are async — they use Python's asyncio.
# If you're new to async: just think of 'await' as "wait for this to finish".

async def main():
    # ---- PART 1: Connect to the MCP Server ----
    # We pass the server FILE PATH. FastMCP automatically:
    #   1. Starts step10_mcp_server.py as a subprocess
    #   2. Connects to it via stdio (stdin/stdout)
    #   3. Handles all MCP protocol communication

    async with Client("step10_mcp_server.py") as client:
        print("Connected to MCP server!")
        print()

        # ---- PART 2: Discover Tools ----
        # This is the magic of MCP. We don't hardcode what tools exist.
        # We ASK the server: "What tools do you have?"

        tools = await client.list_tools()

        print(f"Server has {len(tools)} tool(s):")
        print()
        for tool in tools:
            print(f"  Name:        {tool.name}")
            print(f"  Description: {tool.description}")
            print(f"  Schema:      {json.dumps(tool.inputSchema, indent=4)}")
            print()

        # Notice: the schema was auto-generated from Python type hints!
        # You never wrote any JSON Schema — FastMCP did it for you.

        # ---- PART 3: Call a Tool ----
        # Now use the tool through the MCP protocol.
        # The client sends a request, the server executes the function,
        # and the result comes back through the protocol.

        print("Calling get_weather('Bengaluru')...")
        result = await client.call_tool("get_weather", {"city": "Bengaluru"})
        print(f"  Result: {result.data}")
        print()

        print("Calling get_weather('Delhi')...")
        result = await client.call_tool("get_weather", {"city": "Delhi"})
        print(f"  Result: {result.data}")
        print()

        print("Calling get_weather('Tokyo')...")
        result = await client.call_tool("get_weather", {"city": "Tokyo"})
        print(f"  Result: {result.data}")
        print()


# Run the async function
asyncio.run(main())

# ============================================
# WHAT JUST HAPPENED:
#   1. The client started the server as a subprocess
#   2. Asked "what tools do you have?" (list_tools)
#   3. Got back the tool name, description, and JSON Schema
#   4. Called the tool with arguments (call_tool)
#   5. Got the result back through the protocol
#
# THE BIG INSIGHT:
#   The client never imported get_weather().
#   It never saw the Python code.
#   It discovered and used the tool purely through the MCP PROTOCOL.
#
#   This means the server could be:
#   - Written in a different language (TypeScript, Go, Rust)
#   - Running on a different machine
#   - Maintained by a completely different team
#
#   And the client would still work. That's the power of a standard.
#
# EXERCISE 1:
# Go back to step10_mcp_server.py and add a new tool (e.g., get_time).
# Run this client again WITHOUT changing it.
# Does it discover the new tool automatically?
#
# EXERCISE 2:
# Try connecting to the server via HTTP instead of stdio.
# In step10, change: mcp.run(transport="sse", port=8080)
# In this file, change: Client("http://localhost:8080/sse")
# Now the server runs independently, and clients connect over the network.
#
# EXERCISE 3:
# Use the in-process connection (no subprocess):
#   from step10_mcp_server import mcp
#   async with Client(mcp) as client:
# This is useful for testing. Same protocol, no process overhead.
# ============================================
