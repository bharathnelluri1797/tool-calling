# ============================================
# STEP 19: RAGAS Evaluation — Measure, Don't Vibe Check
# ============================================
#
# WHAT YOU'LL LEARN:
# - Why "it seems to work" isn't good enough
# - The four RAGAS metrics and what each one diagnoses
# - How to implement evaluation from scratch
# - How to pinpoint whether retrieval or generation is the problem
#
# KEY IDEA (from lecture):
#   VIBES-BASED:   "I tried a few queries and it seemed okay"
#   METRICS-BASED: "Context precision: 82%. Faithfulness: 91%."
#
#   Don't tell me "it works."
#   Tell me your context precision. Tell me your faithfulness score.
#   That's how you go from demo to production.
#
# THE FOUR METRICS:
#   Context Precision — Of what was retrieved, how much was relevant?
#   Context Recall    — Of what should've been retrieved, how much was?
#   Faithfulness      — Is the answer supported by the retrieved context?
#   Answer Relevancy  — Does the answer address what was asked?
#
# RUN THIS FILE:
#   python step19_evaluation.py
# ============================================

import os
import json
import math
from groq import Groq
from sentence_transformers import SentenceTransformer
from rag_documents import CURRICULUM_DOCUMENTS

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
LLM_MODEL = "openai/gpt-oss-20b"


# ============================================
# PART 1: Build a Minimal RAG Pipeline (from step 18)
# ============================================
# We need a working RAG to evaluate it.

class SimpleRAG:
    """Minimal RAG for evaluation purposes."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def index(self, documents):
        texts = [d["content"] for d in documents]
        self.embeddings = embedding_model.encode(texts).tolist()
        self.documents = documents

    def retrieve(self, query, top_k=3):
        query_emb = embedding_model.encode(query).tolist()
        scored = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = self._cosine_sim(query_emb, doc_emb)
            scored.append({"document": self.documents[i], "similarity": sim})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def query(self, question, top_k=3):
        retrieved = self.retrieve(question, top_k)
        context = "\n\n".join(
            f"[Source {i+1}]: {r['document']['content']}"
            for i, r in enumerate(retrieved)
        )
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": f"Answer using ONLY the context below. If the context doesn't contain the answer, say so.\n\nCONTEXT:\n{context}"},
                {"role": "user", "content": question},
            ],
        )
        return {
            "answer": response.choices[0].message.content,
            "retrieved": retrieved,
            "context": context,
        }

    def _cosine_sim(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        ma = math.sqrt(sum(x*x for x in a))
        mb = math.sqrt(sum(x*x for x in b))
        return dot / (ma * mb) if ma and mb else 0.0


# Index curriculum
rag = SimpleRAG()
docs = [{"content": d["content"], "metadata": d["metadata"], "id": d["id"]} for d in CURRICULUM_DOCUMENTS]
print("Indexing documents...")
rag.index(docs)
print(f"Indexed {len(docs)} documents.\n")


# ============================================
# PART 2: The Test Suite — Known Questions + Known Answers
# ============================================
# To evaluate, you need questions where YOU know the right answer
# and which documents SHOULD be retrieved.
#
# This is your "ground truth" — the gold standard to measure against.

EVAL_DATASET = [
    {
        "question": "When will we learn about RAG?",
        "expected_answer": "RAG is covered in Week 5",
        "relevant_doc_ids": ["doc_11", "doc_12", "doc_25"],
        "description": "Direct factual question about curriculum timing",
    },
    {
        "question": "How do I fix the infinite loop in my agent?",
        "expected_answer": "Add a max_steps counter (if step > 5: break) and make sure tool results use role='tool' with matching tool_call_id",
        "relevant_doc_ids": ["doc_10"],
        "description": "Debugging question — should find the forum thread",
    },
    {
        "question": "Has anyone had trouble getting their data from the weather API?",
        "expected_answer": "422 error due to whitespace in city parameter. Fix with city.strip(). Also check .env file for API key.",
        "relevant_doc_ids": ["doc_15"],
        "description": "Semantic search challenge — 'tool calling' never mentioned",
    },
    {
        "question": "What is the agentic loop?",
        "expected_answer": "A loop: send message to LLM with tools → if tool_calls, execute and loop back → if text, that's the answer. Foundation of AI agents.",
        "relevant_doc_ids": ["doc_09", "doc_10"],
        "description": "Conceptual question about a core pattern",
    },
    {
        "question": "What assignments do I need to complete for the RAG module?",
        "expected_answer": "Assignment 3: Build a RAG system for a custom dataset with chunking, embeddings, retrieval, and RAGAS evaluation. Due end of Week 5.",
        "relevant_doc_ids": ["doc_23"],
        "description": "Specific question about assignments",
    },
]


# ============================================
# PART 3: Retrieval Metrics
# ============================================
# These measure the RETRIEVER — before the LLM even sees the query.

def context_precision(retrieved_docs, relevant_doc_ids):
    """Of what was retrieved, how much was actually relevant?

    High precision = retriever returns mostly relevant docs (low noise)
    Low precision = retriever returns too much irrelevant content

    Fix low precision: better metadata filtering, tighter chunking.
    """
    if not retrieved_docs:
        return 0.0
    relevant_count = sum(
        1 for r in retrieved_docs
        if r["document"].get("id") in relevant_doc_ids
    )
    return relevant_count / len(retrieved_docs)


def context_recall(retrieved_docs, relevant_doc_ids):
    """Of what should have been retrieved, how much was found?

    High recall = retriever finds most of the relevant documents
    Low recall = retriever misses important content

    Fix low recall: better embeddings, broader search, chunk overlap.
    """
    if not relevant_doc_ids:
        return 0.0
    found_count = sum(
        1 for doc_id in relevant_doc_ids
        if any(r["document"].get("id") == doc_id for r in retrieved_docs)
    )
    return found_count / len(relevant_doc_ids)


# ============================================
# PART 4: Generation Metrics (LLM-as-Judge)
# ============================================
# These measure the GENERATOR — the LLM's response quality.
# We use the LLM itself to judge (a standard evaluation technique).

def faithfulness(answer, context):
    """Is the answer actually supported by the retrieved context?

    High faithfulness = answer sticks to what's in the context
    Low faithfulness = LLM hallucinated information not in context

    This is GROUNDED HALLUCINATION — the model makes up info
    even though it has real context available.

    Fix: stricter system prompts, explicit grounding instructions.
    """
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an evaluation judge. Given a CONTEXT and an ANSWER, determine what fraction of the claims in the answer are supported by the context.

Respond with ONLY a JSON object:
{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}

Score guide:
- 1.0 = every claim in the answer is directly supported by the context
- 0.5 = about half the claims are supported, half are not in context
- 0.0 = the answer contains no information from the context"""
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
            },
        ],
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("score", 0.0), result.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        return 0.0, "Could not parse judge response"


def answer_relevancy(question, answer):
    """Does the answer actually address what was asked?

    High relevancy = answer directly addresses the question
    Low relevancy = answer is off-target (even if context was right)

    If relevancy is low + precision is high = GENERATION is the problem.

    Fix: better system prompts, more specific instructions.
    """
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an evaluation judge. Given a QUESTION and an ANSWER, determine how well the answer addresses the question.

Respond with ONLY a JSON object:
{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}

Score guide:
- 1.0 = answer directly and completely addresses the question
- 0.5 = answer partially addresses the question or is too vague
- 0.0 = answer is completely off-topic or doesn't address the question"""
            },
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
            },
        ],
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("score", 0.0), result.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        return 0.0, "Could not parse judge response"


# ============================================
# PART 5: Run the Evaluation
# ============================================

print("=" * 60)
print("RAGAS EVALUATION — Running on test suite")
print("=" * 60)

all_precision = []
all_recall = []
all_faithfulness = []
all_relevancy = []

for i, test_case in enumerate(EVAL_DATASET):
    print(f"\n{'─' * 60}")
    print(f"  Test {i+1}: {test_case['description']}")
    print(f"  Q: {test_case['question']}")
    print(f"{'─' * 60}")

    # Run RAG pipeline
    result = rag.query(test_case["question"], top_k=3)

    # Show answer (truncated)
    answer_preview = result["answer"][:200].replace("\n", " ")
    print(f"\n  Answer: {answer_preview}...")

    # Show retrieved doc IDs
    retrieved_ids = [r["document"].get("id", "?") for r in result["retrieved"]]
    expected_ids = test_case["relevant_doc_ids"]
    print(f"  Retrieved: {retrieved_ids}")
    print(f"  Expected:  {expected_ids}")

    # Calculate retrieval metrics (fast, no LLM call needed)
    precision = context_precision(result["retrieved"], expected_ids)
    recall = context_recall(result["retrieved"], expected_ids)

    # Calculate generation metrics (uses LLM-as-judge)
    faith_score, faith_reason = faithfulness(result["answer"], result["context"])
    relev_score, relev_reason = answer_relevancy(test_case["question"], result["answer"])

    print(f"\n  METRICS:")
    print(f"    Context Precision:  {precision:.2f}  (of retrieved, how much was relevant?)")
    print(f"    Context Recall:     {recall:.2f}  (of relevant, how much was found?)")
    print(f"    Faithfulness:       {faith_score:.2f}  ({faith_reason})")
    print(f"    Answer Relevancy:   {relev_score:.2f}  ({relev_reason})")

    all_precision.append(precision)
    all_recall.append(recall)
    all_faithfulness.append(faith_score)
    all_relevancy.append(relev_score)


# ============================================
# PART 6: The Diagnostic Report
# ============================================

print()
print("=" * 60)
print("DIAGNOSTIC REPORT")
print("=" * 60)

avg_precision = sum(all_precision) / len(all_precision)
avg_recall = sum(all_recall) / len(all_recall)
avg_faithfulness = sum(all_faithfulness) / len(all_faithfulness)
avg_relevancy = sum(all_relevancy) / len(all_relevancy)

print(f"""
  ┌──────────────────────┬───────┬────────────────────────────────────┐
  │ Metric               │ Score │ What It Tells You                  │
  ├──────────────────────┼───────┼────────────────────────────────────┤
  │ Context Precision    │ {avg_precision:.2f}  │ {'Good' if avg_precision >= 0.6 else 'Fix'}: retriever noise level          │
  │ Context Recall       │ {avg_recall:.2f}  │ {'Good' if avg_recall >= 0.6 else 'Fix'}: retriever coverage              │
  │ Faithfulness         │ {avg_faithfulness:.2f}  │ {'Good' if avg_faithfulness >= 0.7 else 'Fix'}: grounding quality             │
  │ Answer Relevancy     │ {avg_relevancy:.2f}  │ {'Good' if avg_relevancy >= 0.7 else 'Fix'}: answer targeting              │
  └──────────────────────┴───────┴────────────────────────────────────┘
""")

# Diagnostic logic from the lecture
print("  DIAGNOSIS:")
if avg_precision < 0.6:
    print("  → Low precision: retriever returns too much noise.")
    print("    FIX: Better metadata filtering, tighter chunking.")
if avg_recall < 0.6:
    print("  → Low recall: retriever misses relevant documents.")
    print("    FIX: Better embeddings, broader search, chunk overlap.")
if avg_faithfulness < 0.7:
    print("  → Low faithfulness: LLM hallucinating beyond context.")
    print("    FIX: Stricter system prompts, explicit grounding instructions.")
if avg_relevancy < 0.7:
    print("  → Low relevancy: answers don't address the question.")
    print("    FIX: Better system prompts, more specific instructions.")
if all(x >= 0.6 for x in [avg_precision, avg_recall]) and all(x >= 0.7 for x in [avg_faithfulness, avg_relevancy]):
    print("  → All metrics look healthy! Your RAG pipeline is working well.")

print(f"""
  THE DIAGNOSTIC MAP (from lecture):
  ┌──────────────────────┬───────────────────────┬────────────────────────┐
  │ Metric               │ Measures              │ When Low, Fix The...   │
  ├──────────────────────┼───────────────────────┼────────────────────────┤
  │ Context Precision    │ Retrieved → % relevant│ Retriever (too noisy)  │
  │ Context Recall       │ Relevant → % found    │ Retriever (missing)    │
  │ Faithfulness         │ Answer → supported?   │ Generator (hallucinate)│
  │ Answer Relevancy     │ Answer → on target?   │ Generator (off-topic)  │
  └──────────────────────┴───────────────────────┴────────────────────────┘
""")

print("=" * 60)

# ============================================
# WHAT YOU LEARNED:
# 1. Evaluation needs a test suite with known answers + relevant docs
# 2. Retrieval metrics (precision, recall) don't need LLM calls
# 3. Generation metrics (faithfulness, relevancy) use LLM-as-judge
# 4. Each metric points to a specific part of the pipeline to fix
# 5. "It works" is not a metric. Precision scores are.
#
# EXERCISE 1:
# Add 5 more test cases to EVAL_DATASET. Include:
#   - A multi-hop question: "How does tool calling relate to MCP?"
#   - An unanswerable question: "What is the instructor's email?"
#   - A trick question: something that SOUNDS like it's in the data but isn't
# How do the metrics change?
#
# EXERCISE 2:
# Vary top_k and measure the impact on precision vs recall:
#   for k in [1, 3, 5, 10]:
#       # run eval with top_k=k
# What's the optimal k? (Hint: precision usually drops as k increases,
# recall usually increases. Find the sweet spot.)
#
# EXERCISE 3:
# Improve faithfulness by modifying the system prompt:
#   - Add: "ONLY use information from the context above."
#   - Add: "If you're unsure, say 'I don't have enough information.'"
#   - Add: "Cite the source number for each claim you make."
# Measure faithfulness before and after the prompt change.
#
# EXERCISE 4 (Advanced):
# Implement "Semantic Answer Similarity" — compare the RAG answer
# to the expected answer using embedding cosine similarity.
# This gives you a numeric quality score without LLM-as-judge:
#   expected_emb = embedding_model.encode(expected_answer)
#   actual_emb = embedding_model.encode(rag_answer)
#   score = cosine_similarity(expected_emb, actual_emb)
#
# NEXT STEP: We'll integrate RAG as a tool in the agentic loop,
# combining retrieval with action — the full picture.
# ============================================
