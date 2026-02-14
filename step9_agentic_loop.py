# ============================================
# STEP 9: The Agentic Loop — Foundation of AI Agents
# ============================================
#
# WHAT YOU'LL LEARN:
# - The core loop behind ALL AI agents
# - How to handle MULTI-STEP reasoning
# - How the LLM calls tools repeatedly until it has enough info
#
# KEY IDEA (from lecture):
#   An AI agent is a LOOP:
#     1. Send message to LLM (with tool definitions)
#     2. If LLM returns tool_calls -> execute them, send results back, REPEAT
#     3. If LLM returns text -> that's the final answer, STOP
#
#   This is the foundation of Agents, RAG, and MCP.
#
# WHY A LOOP?
#   Some questions need MULTIPLE tool calls:
#   "Compare weather in Delhi and Bengaluru" needs TWO get_weather calls.
#   "What is 15% of the temperature in Mumbai?" needs get_weather THEN calculate.
#
# RUN THIS FILE:
#   python step9_agentic_loop.py
# ============================================

import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


# ---- Tools ----

def get_weather(city):
    data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy", "humidity": 65},
        "delhi": {"temp_c": 42, "condition": "Extreme Heat", "humidity": 25},
        "mumbai": {"temp_c": 32, "condition": "Humid", "humidity": 80},
        "london": {"temp_c": 15, "condition": "Rainy", "humidity": 90},
    }
    return data.get(city.lower(), {"error": f"No weather data for {city}"})


def calculate(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception:
        return {"error": f"Cannot calculate: {expression}"}


available_tools = {
    "get_weather": get_weather,
    "calculate": calculate,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temp_c, condition, humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '42 - 28'"}
                },
                "required": ["expression"]
            }
        }
    },
]


# ============================================
# THE AGENTIC LOOP
# This is the key pattern. Everything else is built on this.
# ============================================

def run_agent(user_message):
    """Run the agent loop until the LLM gives a final text answer."""

    print(f"User: {user_message}")
    print()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when needed to get real data. You can call multiple tools to gather all the information before answering."
        },
        {"role": "user", "content": user_message},
    ]

    step = 1

    while True:
        print(f"  [Step {step}] Calling LLM...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # EXIT CONDITION: No tool calls means the LLM has its final answer
        if not message.tool_calls:
            print(f"  [Step {step}] LLM responded with final answer.")
            print()
            print(f"Agent: {message.content}")
            return message.content

        # TOOL CALLS: The LLM needs more information
        # Note: The LLM can request MULTIPLE tool calls at once!
        messages.append(message)

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"  [Step {step}] Tool call: {fn_name}({fn_args})")

            # Execute the deterministic function
            result = available_tools[fn_name](**fn_args)
            print(f"  [Step {step}] Result:    {json.dumps(result)}")

            # Send result back using the proper "tool" role
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        step += 1

        # Safety: prevent infinite loops
        if step > 5:
            print("  [Max steps reached — stopping]")
            return "Sorry, I couldn't complete the task."


# ---- TEST 1: Multi-Tool Comparison ----
# This needs TWO get_weather calls, then comparison logic
print("=" * 60)
print("TEST 1: Requires multiple tool calls")
print("=" * 60)
run_agent("Compare the weather in Delhi and Bengaluru. Which city is hotter and by how much?")

print()

# ---- TEST 2: Chained Tool Calls ----
# This needs get_weather FIRST, then calculate with the result
print("=" * 60)
print("TEST 2: Requires chained tool calls")
print("=" * 60)
run_agent("What is the temperature difference between London and Delhi? Show the calculation.")

print()

# ---- TEST 3: No Tools Needed ----
# The LLM should answer directly without calling any tool
print("=" * 60)
print("TEST 3: No tools needed")
print("=" * 60)
run_agent("What is the capital of France?")

# ============================================
# THE BIG PICTURE (from lecture):
#
#   What we built across steps 5-9:
#
#   Step 5: Saw the problem (hallucination)
#   Step 6: Bridged it manually (JSON parsing — fragile)
#   Step 7: Used proper API tool calling (reliable)
#   Step 8: Added multiple tools + routing (intelligent)
#   Step 9: Built the agentic loop (autonomous)
#
#   This loop is the foundation of:
#     - AI Agents (autonomous task completion)
#     - RAG (retrieve data, then answer)
#     - MCP (Model Context Protocol)
#     - Every reliable AI system in production
#
# EXERCISE 1:
# Add a "search_web" mock tool that takes a "query" parameter.
# Mock it to return results for a few predefined queries.
# Test: "Search for the latest news about AI and summarize it"
#
# EXERCISE 2:
# Add conversation memory. After run_agent() finishes, take the
# full messages list and pass it into the next run_agent() call.
# This way the agent remembers previous conversations.
# Test a multi-turn conversation:
#   Turn 1: "What's the weather in Mumbai?"
#   Turn 2: "How about Delhi?"
#   Turn 3: "Which one was hotter?"
#
# EXERCISE 3 (Advanced):
# Integrate this agentic loop into a Gradio chat interface.
# Hint: Replace the respond() function from step2 with run_agent().
# Now you have a UI-based agent that can use tools!
#
# EXERCISE 4 (Advanced):
# Replace the mock get_weather() with a REAL weather API.
# Free option: Open-Meteo API (no key needed)
#   import requests
#   def get_weather(city):
#       # Use Open-Meteo geocoding + weather API
#       geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()
#       lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
#       weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true").json()
#       return weather["current_weather"]
# ============================================
