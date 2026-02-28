# ============================================
# STEP 12: Multi-Tool MCP Server
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to register multiple tools in one MCP server
# - How MCP auto-generates schemas for all of them
# - How a client discovers and uses all tools
#
# KEY IDEA:
#   Compare this file with step8_multi_tool_agent.py.
#
#   Step 8: You wrote 3 functions + 40 lines of JSON Schemas + a dispatch dict
#   Step 12: You write 3 functions with @mcp.tool. That's it.
#
#   The JSON Schemas, tool registration, and dispatch are ALL automatic.
#
# RUN THIS FILE:
#   python step12_mcp_multi_tool.py
#
# OR inspect all tools:
#   fastmcp dev step12_mcp_multi_tool.py
# ============================================

import asyncio
import json
from fastmcp import FastMCP, Client

mcp = FastMCP("Multi-Tool Server")


# ---- Tool 1: Weather ----
@mcp.tool
def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: City name (e.g., 'Bengaluru', 'Delhi', 'Mumbai')
    """
    data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy"},
        "delhi": {"temp_c": 35, "condition": "Sunny"},
        "mumbai": {"temp_c": 32, "condition": "Humid"},
    }
    return data.get(city.lower(), {"error": f"No weather data for {city}"})


# ---- Tool 2: Calculator ----
@mcp.tool
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A math expression like '245 * 38 + 17'
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception:
        return {"error": f"Cannot calculate: {expression}"}


# ---- Tool 3: Contacts ----
@mcp.tool
def search_contacts(name: str) -> dict:
    """Search for a contact by name to get their phone number and email.

    Args:
        name: The name of the contact to search for
    """
    contacts = {
        "alice": {"phone": "+91-9876543210", "email": "alice@example.com"},
        "bob": {"phone": "+91-9123456789", "email": "bob@example.com"},
        "charlie": {"phone": "+91-9988776655", "email": "charlie@example.com"},
    }
    return contacts.get(name.lower(), {"error": f"Contact '{name}' not found"})


# ---- Demo: Connect as client and test all tools ----
# Using in-process connection (Client(mcp)) for simplicity.
# No subprocess needed — perfect for testing.

async def demo():
    async with Client(mcp) as client:
        # Discover all tools
        tools = await client.list_tools()
        print(f"MCP server has {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        print()

        # Test each tool
        tests = [
            ("get_weather", {"city": "Mumbai"}),
            ("calculate", {"expression": "245 * 38 + 17"}),
            ("search_contacts", {"name": "Alice"}),
        ]

        for tool_name, args in tests:
            print(f"Calling {tool_name}({args})")
            result = await client.call_tool(tool_name, args)
            print(f"  Result: {result.data}")
            print()


if __name__ == "__main__":
    asyncio.run(demo())

# ============================================
# COMPARE step8 vs step12:
#
#   Step 8: 3 functions + 40 lines of JSON Schema + dispatch dict
#   Step 12: 3 functions with @mcp.tool + 0 lines of schema
#
#   Both do the same thing. MCP generates the schema from type hints.
#
#   But MCP gives you MORE:
#   - Any MCP client can discover these tools automatically
#   - You can run this server independently and connect remotely
#   - Other people's tools work with your client (and vice versa)
#   - It's a standard, not custom code
#
# EXERCISE 1:
# Add a 4th tool: get_news(topic: str) -> dict
# Mock it with a few topics. Run the file to see it auto-discovered.
#
# EXERCISE 2:
# Try using Annotated types for richer parameter descriptions:
#   from typing import Annotated
#   from pydantic import Field
#
#   @mcp.tool
#   def get_weather(
#       city: Annotated[str, Field(description="City name")],
#       units: Annotated[str, Field(description="celsius or fahrenheit")] = "celsius"
#   ) -> dict:
#
# Run fastmcp dev to see how the schema changes.
#
# EXERCISE 3 (Advanced):
# Separate this into two files:
#   - step12_server.py (just the server, run with: python step12_server.py)
#   - step12_client.py (connects via: Client("step12_server.py"))
# This mirrors real production setups where server and client are separate.
# ============================================
