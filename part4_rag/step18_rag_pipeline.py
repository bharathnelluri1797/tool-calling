# ============================================
# STEP 18: The Complete RAG Pipeline
# ============================================
#
# WHAT YOU'LL LEARN:
# - How to wire chunking + embeddings + vector store + LLM
# - The full pipeline: Index → Retrieve → Augment → Generate
# - How augmented prompts ground the LLM in real data
# - Before/after comparison: hallucination vs RAG
#
# KEY IDEA (from lecture):
#   The Full RAG Pipeline:
#     1. INDEX:     Chunk → Embed → Store with metadata (offline)
#     2. RETRIEVE:  Filter by metadata → Semantic search → Top-k (query time)
#     3. AUGMENT:   Inject relevant chunks into the prompt
#     4. GENERATE:  LLM produces an answer grounded in retrieved context
#
# INSTALL (one-time):
#   pip install sentence-transformers groq
#
# RUN THIS FILE:
#   python step18_rag_pipeline.py
# ============================================

import os
import math
from groq import Groq
from sentence_transformers import SentenceTransformer
from rag_documents import CURRICULUM_DOCUMENTS

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
LLM_MODEL = "openai/gpt-oss-20b"


# ============================================
# PART 1: The RAG Pipeline Class
# ============================================
# This brings together everything from steps 15-17:
#   - Chunking (step 15)
#   - Embeddings (step 16)
#   - Vector store + search (step 17)
#   - LLM generation (step 2)

class RAGPipeline:
    """A complete RAG pipeline from first principles."""

    def __init__(self, embedding_model, llm_client, llm_model):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.documents = []
        self.embeddings = []

    # ---- STEP 1: INDEX (offline, one-time) ----

    def index(self, documents):
        """Index a list of documents with content and metadata.

        Each document: {"content": "...", "metadata": {...}}
        """
        print(f"  Indexing {len(documents)} documents...")

        texts = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()

        for doc, emb in zip(documents, embeddings):
            self.documents.append(doc)
            self.embeddings.append(emb)

        print(f"  Done! {len(self.documents)} documents indexed.")

    # ---- STEP 2: RETRIEVE (at query time) ----

    def retrieve(self, query, top_k=3, metadata_filter=None):
        """Find the most relevant documents for a query."""
        query_embedding = self.embedding_model.encode(query).tolist()

        scored_results = []
        for i, doc_emb in enumerate(self.embeddings):
            doc = self.documents[i]

            # Apply metadata filter
            if metadata_filter:
                skip = False
                for key, value in metadata_filter.items():
                    if doc.get("metadata", {}).get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            similarity = self._cosine_similarity(query_embedding, doc_emb)
            scored_results.append({"document": doc, "similarity": similarity})

        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_results[:top_k]

    # ---- STEP 3: AUGMENT (build the prompt) ----

    def augment(self, query, retrieved_docs):
        """Build an augmented prompt with retrieved context."""
        context_parts = []
        for i, result in enumerate(retrieved_docs):
            doc = result["document"]
            meta = doc.get("metadata", {})
            header = f"[Source {i+1}: Week {meta.get('week', '?')}, {meta.get('module', '?')}, {meta.get('content_type', '?')}]"
            context_parts.append(f"{header}\n{doc['content']}")

        context = "\n\n".join(context_parts)

        system_prompt = f"""You are a helpful assistant for the 100xEngineers program.
Answer the student's question using ONLY the context provided below.
If the context doesn't contain the answer, say "I don't have enough information to answer that."
Be specific and cite which source you're using.

CONTEXT:
{context}"""

        return system_prompt

    # ---- STEP 4: GENERATE (LLM produces the answer) ----

    def generate(self, query, system_prompt):
        """Generate an answer using the LLM with augmented context."""
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content, response.usage

    # ---- FULL PIPELINE: query() ----

    def query(self, question, top_k=3, metadata_filter=None):
        """Run the full RAG pipeline: retrieve → augment → generate."""
        # Step 1: Retrieve relevant documents
        retrieved = self.retrieve(question, top_k=top_k, metadata_filter=metadata_filter)

        # Step 2: Augment the prompt
        system_prompt = self.augment(question, retrieved)

        # Step 3: Generate the answer
        answer, usage = self.generate(question, system_prompt)

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved,
            "tokens_used": usage.total_tokens,
        }

    def _cosine_similarity(self, vec_a, vec_b):
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# ============================================
# PART 2: Index the Curriculum
# ============================================

print("=" * 60)
print("STEP 18: The Complete RAG Pipeline")
print("=" * 60)

rag = RAGPipeline(
    embedding_model=embedding_model,
    llm_client=groq_client,
    llm_model=LLM_MODEL,
)

# Index all curriculum documents
docs_to_index = [
    {"content": doc["content"], "metadata": doc["metadata"]}
    for doc in CURRICULUM_DOCUMENTS
]
rag.index(docs_to_index)


# ============================================
# PART 3: Test the Pipeline
# ============================================

print()
print("=" * 60)
print("PART 3: RAG in Action")
print("=" * 60)

test_questions = [
    "When will we learn about RAG in this cohort?",
    "How do I fix the infinite loop in my agent?",
    "What are the assignments I need to complete?",
]

for question in test_questions:
    print(f"\n{'─' * 60}")
    print(f"  Question: {question}")
    print(f"{'─' * 60}")

    result = rag.query(question, top_k=3)

    # Show retrieved documents
    print(f"\n  RETRIEVED ({len(result['retrieved_docs'])} docs):")
    for r in result["retrieved_docs"]:
        doc = r["document"]
        sim = r["similarity"]
        meta = doc.get("metadata", {})
        preview = doc["content"][:60].replace("\n", " ")
        print(f"    {sim:.4f}  [W{meta.get('week', '?')}/{meta.get('module', '?')}] {preview}...")

    # Show the generated answer
    print(f"\n  ANSWER:")
    # Wrap answer text for readability
    answer = result["answer"]
    for line in answer.split("\n"):
        print(f"    {line}")

    print(f"\n  Tokens used: {result['tokens_used']:,}")


# ============================================
# PART 4: Before vs After — The Power of RAG
# ============================================

print()
print("=" * 60)
print("PART 4: Without RAG vs With RAG")
print("=" * 60)

comparison_question = "What specific error code do students get when calling the weather API with extra whitespace?"

# WITHOUT RAG — raw LLM, no context
print(f"\n  Question: {comparison_question}")
print(f"\n  WITHOUT RAG (raw LLM):")
print(f"  " + "-" * 50)

response_no_rag = groq_client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant for the 100xEngineers program."},
        {"role": "user", "content": comparison_question},
    ],
)
no_rag_answer = response_no_rag.choices[0].message.content
for line in no_rag_answer[:300].split("\n"):
    print(f"    {line}")
print(f"    Tokens: {response_no_rag.usage.total_tokens:,}")

# WITH RAG — grounded in retrieved context
print(f"\n  WITH RAG (grounded in retrieved context):")
print(f"  " + "-" * 50)

rag_result = rag.query(comparison_question, top_k=3)
for line in rag_result["answer"][:300].split("\n"):
    print(f"    {line}")
print(f"    Tokens: {rag_result['tokens_used']:,}")

print(f"""
  The RAG answer should mention:
  - 422 error code (from doc_15)
  - city.strip() as the fix
  - .env file check

  The raw LLM either hallucinated an answer or said it doesn't know.
  RAG grounds the response in YOUR actual data.
""")


# ============================================
# PART 5: Metadata-Filtered RAG
# ============================================

print("=" * 60)
print("PART 5: Metadata-Filtered RAG")
print("=" * 60)

meta_question = "What coding exercises do I need to do?"

# Search only assignments
print(f"\n  Question: {meta_question}")
print(f"\n  Filter: content_type='assignment'")
print(f"  " + "-" * 50)

result_filtered = rag.query(
    meta_question,
    top_k=3,
    metadata_filter={"content_type": "assignment"},
)

print(f"\n  RETRIEVED (only assignments):")
for r in result_filtered["retrieved_docs"]:
    doc = r["document"]
    sim = r["similarity"]
    preview = doc["content"][:60].replace("\n", " ")
    print(f"    {sim:.4f}  {preview}...")

print(f"\n  ANSWER:")
for line in result_filtered["answer"][:500].split("\n"):
    print(f"    {line}")

print(f"""
  Without the metadata filter, the search might return lectures
  or forum threads that mention "exercises." With the filter,
  we get ONLY actual assignments — exactly what the student needs.
""")

print("=" * 60)

# ============================================
# WHAT YOU LEARNED:
# 1. The RAG pipeline has 4 stages: Index → Retrieve → Augment → Generate
# 2. Indexing is offline (one-time). Retrieval + augmentation + generation is per-query
# 3. The augmented prompt grounds the LLM in retrieved context
# 4. RAG uses far fewer tokens than context stuffing (3 docs vs 25)
# 5. Metadata filtering makes retrieval more precise
#
# THE PIPELINE DIAGRAM (from lecture):
#   ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
#   │  INDEX  │ →  │ RETRIEVE │ →  │ AUGMENT │ →  │ GENERATE │
#   │ (once)  │    │ (top-k)  │    │ (prompt) │    │  (LLM)   │
#   └─────────┘    └──────────┘    └─────────┘    └──────────┘
#
# EXERCISE 1:
# Add a "chat with history" feature. After the first answer,
# ask a follow-up question that references the previous answer:
#   result1 = rag.query("What is RAG?")
#   # Now ask: "Can you show me the code for that?"
#   # You'll need to pass previous messages to the LLM.
#
# EXERCISE 2:
# Experiment with top_k values (1, 3, 5, 10).
# For the same question, compare:
#   - Answer quality (is more context always better?)
#   - Token usage (how does cost change?)
#   - Which top_k gives the best quality-to-cost ratio?
#
# EXERCISE 3:
# Build a Gradio chat interface for the RAG pipeline:
#   import gradio as gr
#   def respond(message, history):
#       result = rag.query(message, top_k=3)
#       return result["answer"]
#   gr.ChatInterface(respond).launch()
#
# EXERCISE 4 (Advanced):
# Add "source citations" to the answer:
#   Modify the system prompt to instruct the LLM to cite sources
#   like [Source 1], [Source 2]. Then display which documents
#   were cited in the response. This is how production RAG
#   systems provide transparency.
#
# NEXT STEP: We built it, but how do we know it WORKS?
# We need metrics, not vibes. That's RAGAS evaluation.
# ============================================
