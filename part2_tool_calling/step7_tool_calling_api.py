# ============================================
# STEP 7: Proper Tool Calling with the API
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to use Groq's built-in tool calling (no manual JSON hacks)
# - How to define tools using JSON Schema
# - How to handle tool_calls in the API response
# - The proper "tool" role message to send results back
#
# KEY IDEA (from lecture):
#   Instead of asking the LLM to generate JSON via prompting:
#   - Pre-write deterministic code in backend
#   - Describe tools as JSON Schema (input optimization)
#   - Let the API handle structured output (constrained decoding)
#   - Use the proper tool message format
#
#   This was not possible 3 years ago. It's the result of
#   fine-tuning and system-level engineering.
#
# RUN THIS FILE:
#   python step7_tool_calling_api.py
# ============================================

import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# We need a model that supports tool calling on Groq
MODEL="openai/gpt-oss-20b"


# ---- PART 1: Same Tool, Now Described as JSON Schema ----
# The function code stays the same.
# But now we ALSO describe it as a schema for the LLM to read.

def get_weather(city):
    """Our deterministic tool — same input always gives same output."""
    weather_data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy", "humidity": 65},
        "delhi": {"temp_c": 35, "condition": "Sunny", "humidity": 40},
        "mumbai": {"temp_c": 32, "condition": "Humid", "humidity": 80},
    }
    return weather_data.get(city.lower(), {"error": f"No weather data for {city}"})


# This JSON Schema tells the LLM what our tool does.
# The LLM reads THIS (not the Python code!) to decide when/how to use it.
# This is "input optimization" from the lecture — structured input prompt, narrow scope, maximum clarity.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city. Returns temperature, condition, and humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city (e.g., 'Bengaluru', 'Delhi', 'Mumbai')"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


# ---- PART 2: Send Message WITH Tools ----
messages = [
    {"role": "user", "content": "What's the weather in Bengaluru?"}
]

print(f"User: {messages[0]['content']}")
print()

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    # tool_choice options:
    #   "auto"     -> LLM decides whether to use a tool (default)
    #   "required" -> LLM MUST use a tool
    #   "none"     -> LLM cannot use any tool
    tool_choice="auto",
)

message = response.choices[0].message


# ---- PART 3: Check If the LLM Wants to Call a Tool ----
# With the API approach, we don't parse raw text.
# The response has a structured tool_calls field.

if message.tool_calls:
    tool_call = message.tool_calls[0]

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    print(f"LLM wants to call: {function_name}({arguments})")

    # Execute the deterministic function
    result = get_weather(**arguments)
    print(f"Tool result: {json.dumps(result)}")
    print()

    # ---- PART 4: Send Result Back (The Proper Way) ----
    # Notice: we use role "tool" (not "user") and include tool_call_id.
    # This is the standard format — much cleaner than the manual approach.
    messages.append(message)  # Add the assistant's message (with tool call info)
    messages.append({
        "role": "tool",                  # Special role for tool results
        "tool_call_id": tool_call.id,    # Links result to the specific tool call
        "content": json.dumps(result),   # The actual data from our function
    })

    # Now the LLM can give a natural language answer using REAL data
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    print(f"Final answer: {final_response.choices[0].message.content}")

else:
    # No tool call needed — LLM answered directly
    print(f"Direct answer: {message.content}")

# ============================================
# COMPARE STEP 6 vs STEP 7:
#
# Step 6 (Manual):
#   - System prompt forces JSON -> fragile, can break
#   - We parse raw text with json.loads() -> error-prone
#   - Send result back as "user" message -> hacky
#
# Step 7 (API):
#   - Tools defined as JSON Schema -> clean, standard
#   - API returns structured tool_calls -> reliable
#   - Send result back as "tool" role -> proper protocol
#
# The API approach is what all modern AI systems use.
#
# EXERCISE 1:
# Change tool_choice from "auto" to "none".
# What happens when you ask about weather?
# (The LLM can't use the tool — does it hallucinate?)
#
# EXERCISE 2:
# Add a "units" parameter to the tool schema:
#   "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
# Update get_weather() to handle the units parameter.
# Test: "What's the weather in Delhi in fahrenheit?"
#
# EXERCISE 3:
# Ask a non-weather question: "Who is the PM of India?"
# Does the LLM use the tool or answer directly?
# This is the "routing decision" from the lecture.
# ============================================
