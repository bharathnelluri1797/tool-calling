# ============================================
# STEP 10: Your First MCP Server with FastMCP
# ============================================
#
# WHAT YOU'LL LEARN:
# - What MCP (Model Context Protocol) is and why it exists
# - How to create an MCP server using FastMCP
# - How @mcp.tool replaces manual JSON Schema definitions
#
# KEY IDEA:
#   In steps 7-9, you manually wrote JSON Schemas for every tool.
#   That's tedious and error-prone.
#
#   MCP is a STANDARD PROTOCOL that solves this:
#   - Tool providers (servers) EXPOSE tools in a universal format
#   - Tool consumers (clients/LLMs) DISCOVER and USE tools automatically
#   - Any MCP client can talk to any MCP server
#
#   Think of MCP like USB — a universal standard for connecting
#   AI models to tools, regardless of who built them.
#
#   FastMCP is a Python library that makes building MCP servers easy.
#   You just write normal Python functions + type hints.
#   FastMCP generates the JSON Schema automatically.
#
# INSTALL:
#   pip install fastmcp
#
# RUN THIS FILE:
#   python step10_mcp_server.py
#
# OR inspect it in the MCP Inspector:
#   fastmcp dev step10_mcp_server.py
# ============================================

from fastmcp import FastMCP

# Create an MCP server with a name
mcp = FastMCP("Weather Server")


# ---- Define a tool using the @mcp.tool decorator ----
#
# Compare this with step7 where you wrote 15 lines of JSON Schema:
#   {"type": "function", "function": {"name": "get_weather", ...}}
#
# With FastMCP, you just write a normal Python function.
# The decorator + type hints do all the work.

@mcp.tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city (e.g., 'Bengaluru', 'Delhi', 'Mumbai')
    """
    weather_data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy", "humidity": 65},
        "delhi": {"temp_c": 35, "condition": "Sunny", "humidity": 40},
        "mumbai": {"temp_c": 32, "condition": "Humid", "humidity": 80},
    }
    return weather_data.get(city.lower(), {"error": f"No weather data for {city}"})


# That's it. One decorator. No JSON Schema. No manual registration.
# FastMCP reads the function name, docstring, and type hints
# to generate everything the MCP protocol needs.


# Run the MCP server
if __name__ == "__main__":
    mcp.run()

# ============================================
# WHAT JUST HAPPENED:
#   mcp.run() starts an MCP server using stdio transport.
#   It waits for an MCP client to connect and request tools.
#
#   On its own, the server just sits there waiting.
#   In the next step, we'll build a CLIENT to connect to it.
#
# TO INSPECT YOUR SERVER:
#   Run: fastmcp dev step10_mcp_server.py
#   This opens the MCP Inspector in your browser where you can:
#   - See all registered tools
#   - See the auto-generated JSON Schema
#   - Test tools interactively
#
# COMPARE step7 vs step10:
#
#   Step 7 (manual — 15 lines per tool):
#     tools = [{
#         "type": "function",
#         "function": {
#             "name": "get_weather",
#             "description": "Get the current weather...",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "city": {"type": "string", "description": "..."}
#                 },
#                 "required": ["city"]
#             }
#         }
#     }]
#
#   Step 10 (FastMCP — just a decorator):
#     @mcp.tool
#     def get_weather(city: str) -> dict:
#         """Get the current weather..."""
#
#   Same result. 1/10th the code.
#
# EXERCISE 1:
# Add a second tool to this server:
#   @mcp.tool
#   def get_forecast(city: str, days: int) -> dict:
#       """Get weather forecast for upcoming days."""
# Run fastmcp dev again to see both tools in the Inspector.
#
# EXERCISE 2:
# Try adding type hints with defaults:
#   def get_weather(city: str, units: str = "celsius") -> dict:
# Check the Inspector — notice how 'units' becomes optional in the schema.
#
# EXERCISE 3:
# What happens if you remove the type hint from 'city'?
#   def get_weather(city):
# Does the schema still generate correctly?
# ============================================
