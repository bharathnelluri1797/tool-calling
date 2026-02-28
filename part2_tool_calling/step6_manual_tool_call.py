# ============================================
# STEP 6: Your First Tool Call (The Manual Way)
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to connect an LLM to a real function
# - Why structured output (JSON) matters
# - The "deterministic vs probabilistic" problem
#
# KEY IDEA (from lecture):
#   Software is deterministic. LLMs are probabilistic.
#   APIs need exact inputs. LLMs generate flexible outputs.
#   To bridge this gap, we force the LLM to output structured JSON.
#
# THE FLOW:
#   User asks question
#       -> LLM outputs JSON (structured)
#       -> We parse the JSON
#       -> We call our Python function (deterministic)
#       -> We send the result back to LLM
#       -> LLM gives a natural language answer
#
# RUN THIS FILE:
#   python step6_manual_tool_call.py
# ============================================

import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---- PART 1: The Deterministic Tool ----
# This is just regular Python. Same input = same output. Always.
# In real life, this would call a weather API.
# We use mock data so you can focus on the CONCEPT, not the API.

def get_weather(city):
    weather_data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy", "humidity": 65},
        "delhi": {"temp_c": 35, "condition": "Sunny", "humidity": 40},
        "mumbai": {"temp_c": 32, "condition": "Humid", "humidity": 80},
    }
    return weather_data.get(city.lower(), {"error": f"No weather data for {city}"})


# ---- PART 2: Tell the LLM About Our Tool ----
# We use the system prompt to instruct the LLM to output JSON.
# This is the "manual" approach — forcing structure through prompting.

system_prompt = """You are a helpful assistant with access to a weather tool.

RULES:
1. If the user asks about weather in a city, respond with ONLY this JSON (no extra text):
   {"function": "get_weather", "arguments": {"city": "city_name_here"}}
2. For any other question, respond normally in plain text.
3. NEVER make up weather data. ALWAYS use the tool."""


# ---- PART 3: Send a Weather Question ----
user_message = "What's the weather like in Bengaluru?"

print(f"User: {user_message}")
print()

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    model="openai/gpt-oss-20b",
)

ai_output = response.choices[0].message.content.strip()
print(f"LLM raw output: {ai_output}")
print()


# ---- PART 4: Parse and Execute ----
# Try to parse as JSON. If it works -> it's a tool call.
# If it fails -> the LLM answered directly.

try:
    tool_call = json.loads(ai_output)

    if tool_call.get("function") == "get_weather":
        city = tool_call["arguments"]["city"]
        result = get_weather(city)
        print(f"Tool executed: get_weather('{city}')")
        print(f"Tool result:   {json.dumps(result)}")
        print()

        # ---- PART 5: Send Result Back to LLM ----
        # Now the LLM can use REAL data to form its answer.
        # No more hallucination — the data came from our tool.
        final = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ai_output},
                {"role": "user", "content": f"Here is the tool result: {json.dumps(result)}. Now respond naturally to the user."},
            ],
            model="openai/gpt-oss-20b",
        )
        print(f"Final answer: {final.choices[0].message.content}")
    else:
        print(f"Unknown function: {tool_call.get('function')}")

except json.JSONDecodeError:
    # Not JSON — the LLM just answered directly (no tool needed)
    print(f"Direct answer: {ai_output}")

# ============================================
# WHAT JUST HAPPENED:
# 1. We wrote a deterministic function (get_weather)
# 2. We told the LLM to output JSON when it needs weather data
# 3. We parsed the JSON and called our function
# 4. We sent the real result back to the LLM
# 5. The LLM gave an answer based on REAL data, not prediction
#
# THE PROBLEM WITH THIS APPROACH:
# - What if the LLM doesn't output valid JSON?
# - What if it adds extra text around the JSON?
# - What if it uses a different format than we expected?
#
# This manual approach is FRAGILE.
# That's why APIs now have built-in tool calling support.
# We'll use that in the next step.
#
# EXERCISE 1:
# Change user_message to "Tell me a joke" (not a weather question).
# What happens? Does the LLM output JSON or plain text?
#
# EXERCISE 2:
# Add a second tool: get_time(timezone)
# Mock data: {"IST": "14:30", "PST": "02:00", "EST": "05:00"}
# Update the system prompt to handle both tools.
# Test with: "What time is it in IST?"
#
# EXERCISE 3 (Think about it):
# What happens if the user asks "What's the weather in Tokyo?"
# (Tokyo is not in our mock data)
# How would you handle this gracefully?
# ============================================
