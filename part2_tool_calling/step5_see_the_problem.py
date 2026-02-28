# ============================================
# STEP 5: See the Problem — LLMs Hallucinate
# ============================================
#
# WHAT YOU'LL LEARN:
# - Why LLMs confidently give wrong answers
# - What "hallucination" really means from first principles
#
# KEY IDEA (from lecture):
#   Hallucination = Uncertainty x Forced Response
#
#   The LLM has NO real-time data. But it MUST generate a response.
#   So it predicts the most likely answer — which may be completely wrong.
#
# RUN THIS FILE:
#   python step5_see_the_problem.py
# ============================================

import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# These are questions the LLM CANNOT answer truthfully.
# It has no access to live weather, stock prices, or 100xEngineers data.
# But watch what happens...

questions = [
    "What is the exact temperature in Bengaluru right now?",
    "What is the current price of Bitcoin in USD?",
    "Which courses are available in the 100xEngineers program right now?",
]

print("=" * 60)
print("HALLUCINATION DEMO")
print("Watch the LLM confidently answer questions it CANNOT know")
print("=" * 60)

for question in questions:
    print(f"\nQuestion: {question}")
    print("-" * 50)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": question}],
        model="llama-3.3-70b-versatile",
    )

    answer = response.choices[0].message.content
    # Show first 300 chars to keep output readable
    print(f"AI: {answer[:300]}")
    if len(answer) > 300:
        print("... (truncated)")

print("\n" + "=" * 60)
print("WHAT JUST HAPPENED:")
print("The AI answered ALL questions with confidence.")
print("But it has NO access to live weather, prices, or 100x data!")
print("It just PREDICTED the most likely response based on training data.")
print()
print("This is NOT a bug. The LLM is doing exactly what it was trained to do:")
print("  Predict the next token.")
print()
print("To fix this, we need to REDUCE UNCERTAINTY by connecting")
print("the LLM to real tools and data sources.")
print("That's what we'll build in the next steps.")
print("=" * 60)

# ============================================
# THINK ABOUT THIS:
# 1. Were any of the answers actually correct?
# 2. How would a user know if an answer is made up?
# 3. What if this was a customer-facing chatbot?
#    (Remember: Air Canada lost a lawsuit over this!)
#
# EXERCISE:
# Add your own real-time questions to the list above and run again:
#   - "What's trending on Twitter right now?"
#   - "What's the score of today's IPL match?"
#   - "How many students are enrolled in 100x Engineers?"
#
# NEXT STEP: We'll FIX this by connecting the LLM to real tools.
# ============================================
