# ============================================
# STEP 2: Connect Your Chatbot to AI (GroqCloud)
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to connect to GroqCloud's API
# - How to send messages and get AI responses
#
# BEFORE RUNNING:
# 1. Get your free API key from: https://console.groq.com
# 2. Set it in your terminal:
#    export GROQ_API_KEY=your_key_here
#
# RUN THIS FILE:
#   python step2_gradio_with_groq.py
# ============================================

import gradio as gr
import os
from groq import Groq

# Connect to Groq using your API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def respond(message, history):
    # Build the conversation history for the AI
    messages = []

    # Add all previous messages
    for msg in history:
        messages.append(msg)

    # Add the new message from user
    messages.append({"role": "user", "content": message})

    # Send to AI and get response
    response = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-20b",
    )

    # Return the AI's reply
    return response.choices[0].message.content

# Create and launch the chat interface
demo = gr.ChatInterface(fn=respond)
demo.launch()

# ============================================
# TRY THIS:
# 1. Ask "What is Python?"
# 2. Then ask "Why is it popular?"
#    (The AI remembers the context!)
#
# HOW IT WORKS:
# User types message → We send it to Groq AI → AI responds
# ============================================
