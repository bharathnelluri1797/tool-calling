# ============================================
# STEP 14: The Scaling Problem — Why Tool Calling Isn't Enough
# ============================================
#
# WHAT YOU'LL LEARN:
# - Why stuffing all data into the prompt breaks
# - The three forcing functions: window limits, cost explosion, attention dilution
# - Why we need a smarter approach (RAG)
#
# KEY IDEA (from lecture):
#   You already built RAG with tool calling!
#     R: Called weather API (retrieval)
#     A: Injected result into prompt (augmentation)
#     G: LLM answered with real data (generation)
#
#   But that worked because the API returned ~10 words.
#   What happens when the answer is buried in 10,000 pages?
#
# RUN THIS FILE:
#   python step14_rag_the_problem.py
# ============================================

import os
from groq import Groq
from rag_documents import CURRICULUM_DOCUMENTS, get_all_content, get_all_content_with_metadata

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


# ============================================
# PART 1: The Naive Approach — Stuff Everything In
# ============================================
# Let's try the "tool calling" approach: put ALL curriculum data
# into the prompt and ask a question.
# With 25 documents, this actually "works"... but watch the cost.

all_content = get_all_content_with_metadata()

# Count approximate tokens (1 token ≈ 4 characters)
char_count = len(all_content)
approx_tokens = char_count // 4

print("=" * 60)
print("PART 1: Context Stuffing — The Naive Approach")
print("=" * 60)
print(f"\nDataset: 100xEngineers Curriculum")
print(f"Documents: {len(CURRICULUM_DOCUMENTS)}")
print(f"Characters: {char_count:,}")
print(f"Approximate tokens: {approx_tokens:,}")
print()

question = "When will we learn about RAG in this cohort?"

print(f"Question: {question}")
print("-" * 50)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": f"You are a helpful assistant for the 100xEngineers program. Answer questions using ONLY the curriculum data below. If the answer isn't in the data, say so.\n\nCURRICULUM DATA:\n{all_content}"
        },
        {"role": "user", "content": question},
    ],
)

answer = response.choices[0].message.content
usage = response.usage

print(f"Answer: {answer[:500]}")
print()
print(f"Tokens used:")
print(f"  Prompt (input):     {usage.prompt_tokens:,} tokens")
print(f"  Completion (output): {usage.completion_tokens:,} tokens")
print(f"  Total:               {usage.total_tokens:,} tokens")


# ============================================
# PART 2: The Math — Why This Doesn't Scale
# ============================================

print()
print("=" * 60)
print("PART 2: The Math — Why This Breaks at Scale")
print("=" * 60)

# Our tiny dataset
tokens_per_query = usage.prompt_tokens

# Real-world numbers
students = 300
queries_per_student_per_day = 5
queries_per_day = students * queries_per_student_per_day
tokens_per_day = queries_per_day * tokens_per_query

# Groq pricing (approximate: $0.05 per 1M input tokens for Llama 3.3)
cost_per_million_tokens = 0.05
daily_cost = (tokens_per_day / 1_000_000) * cost_per_million_tokens
monthly_cost = daily_cost * 30

print(f"\nWith our TINY dataset ({len(CURRICULUM_DOCUMENTS)} docs, {tokens_per_query:,} tokens/query):")
print(f"  {students} students × {queries_per_student_per_day} queries/day = {queries_per_day:,} queries/day")
print(f"  {queries_per_day:,} queries × {tokens_per_query:,} tokens = {tokens_per_day:,} tokens/day")
print(f"  Daily cost:  ${daily_cost:.2f}")
print(f"  Monthly cost: ${monthly_cost:.2f}")

# Now scale to realistic data
scale_factor = 50  # Real curriculum would be 50x larger
print(f"\nWith REAL data ({len(CURRICULUM_DOCUMENTS) * scale_factor:,} docs, ~{tokens_per_query * scale_factor:,} tokens/query):")
real_daily_cost = daily_cost * scale_factor
real_monthly_cost = monthly_cost * scale_factor
print(f"  Daily cost:  ${real_daily_cost:.2f}")
print(f"  Monthly cost: ${real_monthly_cost:.2f}")
print(f"  That's ${real_monthly_cost * 12:.2f}/year — and this is just ONE feature!")


# ============================================
# PART 3: The Attention Problem
# ============================================

print()
print("=" * 60)
print("PART 3: Attention Dilution — The Hidden Failure")
print("=" * 60)

# Ask a question where the answer is buried deep in the data
buried_question = "What error code do students get when the city parameter has whitespace?"

print(f"\nQuestion: {buried_question}")
print("(The answer is in doc_15, buried among 24 other documents)")
print("-" * 50)

response2 = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": f"You are a helpful assistant. Answer using ONLY the data below. Be specific — cite the exact details.\n\nDATA:\n{all_content}"
        },
        {"role": "user", "content": buried_question},
    ],
)

print(f"Answer: {response2.choices[0].message.content[:500]}")
print()
print("Did the LLM find the specific detail (422 error, city.strip() fix)?")
print("With only 25 docs, it might. With 1,250 docs? The answer gets lost.")


# ============================================
# PART 4: The Three Forcing Functions
# ============================================

print()
print("=" * 60)
print("PART 4: The Three Forcing Functions (from lecture)")
print("=" * 60)

print("""
WHY CONTEXT STUFFING FAILS:

1. WINDOW LIMITS
   Our tiny dataset: ~{tokens} tokens
   Real curriculum (6 months): ~{scaled_tokens} tokens
   Most models max out at 128K tokens.
   The data simply DOESN'T FIT.

2. COST EXPLOSION
   Every query sends ALL the data, even if the answer
   is in just 1 paragraph. You're paying for 24 irrelevant
   documents every single time.
   Monthly cost at scale: ${monthly}

3. ATTENTION DILUTION
   LLMs have a "lost in the middle" problem.
   Important information buried in the middle of a long
   context gets lower attention than info at the start/end.
   More context = MORE noise = WORSE answers.

THE SOLUTION:
   Don't send everything. Send only what's RELEVANT.
   But how do you find what's relevant?

   That's what RAG solves. Next step: we learn to chunk,
   embed, and search — so we send 3-5 relevant pieces
   instead of 25 (or 25,000) irrelevant ones.
""".format(
    tokens=approx_tokens,
    scaled_tokens=f"{approx_tokens * scale_factor:,}",
    monthly=f"{real_monthly_cost:.2f}"
))

print("=" * 60)

# ============================================
# WHAT YOU LEARNED:
# 1. Context stuffing "works" on small data but doesn't scale
# 2. Three forcing functions: window limits, cost, attention dilution
# 3. The answer is to retrieve ONLY relevant data (= RAG)
#
# THINK ABOUT THIS:
# 1. If you had 50,000 lines of data, how many lines
#    actually contain the answer to a typical question? (3-5)
# 2. What if you could send ONLY those 3-5 relevant pieces?
# 3. How much would that save in tokens and cost?
#
# EXERCISE 1:
# Add more documents to CURRICULUM_DOCUMENTS in rag_documents.py.
# Double the dataset and re-run. Notice how tokens and cost scale.
#
# EXERCISE 2:
# Try a question that spans multiple documents:
#   "How does the agentic loop from Week 4 connect to MCP in Week 6?"
# Does the LLM connect the dots across documents?
#
# EXERCISE 3:
# Calculate the cost for YOUR use case:
#   - How many documents would your dataset have?
#   - How many users × queries per day?
#   - What's the monthly cost with context stuffing?
#
# NEXT STEP: We'll learn to break documents into chunks
# so we can retrieve just the relevant pieces.
# ============================================
