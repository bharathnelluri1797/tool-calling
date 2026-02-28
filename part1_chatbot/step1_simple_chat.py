# ============================================
# STEP 1: Your First Chatbot with Gradio
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to create a chat interface with Gradio
# - How the respond() function works
#
# RUN THIS FILE:
#   python step1_simple_chat.py
#
# Then open: http://127.0.0.1:7860
# ============================================

import gradio as gr

# This function gets called every time you send a message
# - message: what the user just typed
# - history: all previous messages (we'll use this later)
def respond(message, history):
    return f"You said: {message}"

# Create the chat interface
demo = gr.ChatInterface(fn=respond)

# Start the app
demo.launch()

# ============================================
# TRY THIS:
# 1. Run the file
# 2. Type "Hello" and press Enter
# 3. You should see "You said: Hello"
#
# CHALLENGE:
# Change the respond() function to say something different!
# Example: return f"Echo: {message}"
# ============================================
