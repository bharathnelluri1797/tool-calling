# ============================================
# STEP 8: Multi-Tool Agent — The LLM as a Router
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to give the LLM MULTIPLE tools
# - How the LLM decides WHICH tool to call (routing)
# - When the LLM answers WITHOUT any tool
#
# KEY IDEA (from lecture):
#   "When should the model call a tool? Which tool should it call?"
#   This is the ROUTING problem.
#
#   The LLM acts as an intelligent router:
#   - Weather question  -> get_weather tool
#   - Math question     -> calculate tool
#   - Contact lookup    -> search_contacts tool
#   - General question  -> no tool (answer directly)
#
# RUN THIS FILE:
#   python step8_multi_tool_agent.py
# ============================================

import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


# ---- PART 1: Define Multiple Tools (Python Functions) ----

def get_weather(city):
    data = {
        "bengaluru": {"temp_c": 28, "condition": "Partly Cloudy"},
        "delhi": {"temp_c": 35, "condition": "Sunny"},
        "mumbai": {"temp_c": 32, "condition": "Humid"},
    }
    return data.get(city.lower(), {"error": f"No weather data for {city}"})


def calculate(expression):
    """Simple calculator. Evaluates a math expression."""
    try:
        # NOTE: eval() is used here for simplicity in this learning exercise.
        # In production, use a safe math parser library instead.
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Cannot calculate: {expression}"}


def search_contacts(name):
    contacts = {
        "alice": {"phone": "+91-9876543210", "email": "alice@example.com"},
        "bob": {"phone": "+91-9123456789", "email": "bob@example.com"},
        "charlie": {"phone": "+91-9988776655", "email": "charlie@example.com"},
    }
    return contacts.get(name.lower(), {"error": f"Contact '{name}' not found"})


# Map function names to actual Python functions
# This lets us look up and call the right function by name
available_tools = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_contacts": search_contacts,
}


# ---- PART 2: Register All Tools as JSON Schemas ----
# Each tool has a name, description, and parameter schema.
# The LLM reads these schemas to decide which tool fits the question.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
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
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '245 * 38 + 17'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_contacts",
            "description": "Search for a contact by name to get their phone number and email",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the contact to search for"}
                },
                "required": ["name"]
            }
        }
    },
]


# ---- PART 3: Test Routing with Different Questions ----
# Each question should trigger a DIFFERENT tool (or no tool at all).

test_questions = [
    "What's the weather in Mumbai?",          # -> get_weather
    "What is 245 * 38 + 17?",                 # -> calculate
    "What's Bob's phone number?",             # -> search_contacts
    "Who wrote the book Harry Potter?",       # -> no tool (direct answer)
]

for question in test_questions:
    print(f"\nUser: {question}")
    print("-" * 50)

    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        print(f"  Routed to: {fn_name}({fn_args})")

        # Execute the right function
        result = available_tools[fn_name](**fn_args)
        print(f"  Result:    {json.dumps(result)}")

        # Send result back for natural language answer
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

        final = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        print(f"  Answer:    {final.choices[0].message.content}")
    else:
        print(f"  No tool needed.")
        print(f"  Answer:    {message.content[:200]}")

# ============================================
# WHAT JUST HAPPENED:
# The LLM looked at each question and DECIDED:
#   - Weather question?  -> Call get_weather
#   - Math question?     -> Call calculate
#   - Contact question?  -> Call search_contacts
#   - General knowledge? -> Answer directly, no tool
#
# The LLM is acting as an INTELLIGENT ROUTER.
# It reads the tool descriptions and picks the best match.
# This is possible because of the JSON Schema we provided.
#
# EXERCISE 1:
# Add a 4th tool: get_news(topic)
# Mock data: {"tech": "AI startup raises $100M", "sports": "India wins cricket series"}
# Test with: "What's the latest tech news?"
#
# EXERCISE 2:
# Ask an ambiguous question: "Is it hot in Delhi?"
# Does it use get_weather? Or answer from its training data?
# Try making the question more explicit to control routing.
#
# EXERCISE 3:
# What happens if you ask: "Calculate the weather in Mumbai"?
# (A confusing question that mixes two tools)
# Which tool does it pick? Why?
#
# EXERCISE 4 (Advanced):
# Look at the test for "Harry Potter" — no tool was used.
# Now change tool_choice from "auto" to "required".
# What happens? The LLM is FORCED to use a tool even when it doesn't need one.
# This shows why "auto" is usually the right choice.
# ============================================
