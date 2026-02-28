# ============================================
# STEP 4: Connect Gradio to Your API
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to connect a frontend to a backend API
# - How to send HTTP requests from Python
#
# BEFORE RUNNING:
# 1. First start the backend in another terminal:
#    python step3_fastapi_backend.py
#
# 2. Then run this file:
#    python step4_gradio_frontend.py
# ============================================

import gradio as gr
import requests

def respond(message, history):
    # Send message to our FastAPI backend
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": message}
    )

    # Get the reply from the response
    return response.json()["reply"]

# Create and launch the chat interface
demo = gr.ChatInterface(fn=respond)
demo.launch()

# ============================================
# THE COMPLETE PICTURE:
#
# User types message
#       ↓
# Gradio (this file) sends it to FastAPI
#       ↓
# FastAPI (step3) sends it to Groq AI
#       ↓
# AI responds back through the chain
#
# This is how real apps are built!
# Frontend (Gradio) + Backend (FastAPI) + AI (Groq)
# ============================================
