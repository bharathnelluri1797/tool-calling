# ============================================
# STEP 17: Build Your Own Vector Store
# ============================================
#
# WHAT YOU'LL LEARN:
# - How vector stores work internally
# - Build an in-memory vector store from scratch
# - Add, search, and filter with metadata
# - The "index" from the textbook analogy (lecture)
#
# KEY IDEA (from lecture):
#   "Can we build this index for any dataset?"
#
#   A vector store IS that index:
#   - Each document chunk → embedded as a vector
#   - Each vector → stored with its metadata
#   - At query time → embed the question, find closest vectors
#   - Metadata → filter by week, module, content_type
#
# INSTALL (one-time):
#   pip install sentence-transformers
#
# RUN THIS FILE:
#   python step17_vector_store.py
# ============================================

import math
from sentence_transformers import SentenceTransformer
from rag_documents import CURRICULUM_DOCUMENTS

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================
# PART 1: The SimpleVectorStore Class
# ============================================
# This is the core data structure behind every vector database
# (Pinecone, Weaviate, ChromaDB, pgvector, etc.)
#
# Under the hood, they all do the same thing:
#   1. Store vectors with associated data
#   2. Find the most similar vectors to a query
#   3. Optionally filter by metadata
#
# We'll build this from scratch — no vector DB needed.

class SimpleVectorStore:
    """A minimal vector store built from first principles."""

    def __init__(self, embedding_model):
        self.model = embedding_model
        self.documents = []      # Original text + metadata
        self.embeddings = []     # Corresponding vectors

    def add(self, text, metadata=None, doc_id=None):
        """Add a document to the store."""
        embedding = self.model.encode(text).tolist()
        self.documents.append({
            "id": doc_id or f"doc_{len(self.documents)}",
            "text": text,
            "metadata": metadata or {},
        })
        self.embeddings.append(embedding)

    def add_batch(self, texts, metadatas=None, doc_ids=None):
        """Add multiple documents at once (faster — batched encoding)."""
        embeddings = self.model.encode(texts).tolist()
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            self.documents.append({
                "id": doc_ids[i] if doc_ids else f"doc_{len(self.documents)}",
                "text": text,
                "metadata": metadatas[i] if metadatas else {},
            })
            self.embeddings.append(emb)

    def search(self, query, top_k=3, metadata_filter=None):
        """Find the most similar documents to a query.

        Args:
            query: The search query (text)
            top_k: Number of results to return
            metadata_filter: Dict of metadata key-value pairs to filter by
                             e.g., {"module": "rag", "content_type": "lecture"}
        """
        query_embedding = self.model.encode(query).tolist()

        results = []
        for i, doc_emb in enumerate(self.embeddings):
            doc = self.documents[i]

            # Apply metadata filter (if provided)
            if metadata_filter:
                skip = False
                for key, value in metadata_filter.items():
                    if doc["metadata"].get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_emb)
            results.append({
                "document": doc,
                "similarity": similarity,
            })

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec_a, vec_b):
        """Cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def __len__(self):
        return len(self.documents)


# ============================================
# PART 2: Index the Curriculum
# ============================================
# This is the OFFLINE step — you do this once, not per query.
# In production, you'd save these embeddings to disk/database.

print("=" * 60)
print("PART 2: Indexing the Curriculum")
print("=" * 60)

store = SimpleVectorStore(embedding_model=model)

# Add all curriculum documents with their metadata
texts = [doc["content"] for doc in CURRICULUM_DOCUMENTS]
metadatas = [doc["metadata"] for doc in CURRICULUM_DOCUMENTS]
doc_ids = [doc["id"] for doc in CURRICULUM_DOCUMENTS]

print(f"\n  Encoding {len(texts)} documents...")
store.add_batch(texts, metadatas, doc_ids)
print(f"  Done! Vector store has {len(store)} documents.")
print(f"  Each document is now a {len(store.embeddings[0])}-dimensional vector.")


# ============================================
# PART 3: Semantic Search
# ============================================

print()
print("=" * 60)
print("PART 3: Semantic Search")
print("=" * 60)

queries = [
    "When will we learn about RAG in this cohort?",
    "How do I fix the infinite loop in my agent?",
    "Has anyone had trouble getting their data from the weather API?",
]

for query in queries:
    print(f"\n  Query: \"{query}\"")
    print(f"  " + "-" * 50)

    results = store.search(query, top_k=3)

    for r in results:
        doc = r["document"]
        sim = r["similarity"]
        module = doc["metadata"].get("module", "?")
        ctype = doc["metadata"].get("content_type", "?")
        preview = doc["text"][:70].replace("\n", " ")
        print(f"    {sim:.4f}  [{module}/{ctype}] {preview}...")

    print()


# ============================================
# PART 4: Metadata Filtering — The Precision Multiplier
# ============================================
# From the lecture exercise: Design Sage's Index
#
# Metadata fields like module, content_type, and week let you
# NARROW the search space before doing semantic search.
# This dramatically improves precision.

print("=" * 60)
print("PART 4: Metadata Filtering")
print("=" * 60)

# Without filter: search everything
print(f"\n  Query: \"How to build tools\"")
print(f"  WITHOUT metadata filter (searching all {len(store)} docs):")

results_all = store.search("How to build tools", top_k=3)
for r in results_all:
    doc = r["document"]
    sim = r["similarity"]
    module = doc["metadata"].get("module", "?")
    week = doc["metadata"].get("week", "?")
    print(f"    {sim:.4f}  [week {week}, {module}] {doc['text'][:60]}...")

# With filter: only search MCP module
print(f"\n  WITH metadata filter (module='mcp' only):")

results_mcp = store.search("How to build tools", top_k=3, metadata_filter={"module": "mcp"})
for r in results_mcp:
    doc = r["document"]
    sim = r["similarity"]
    week = doc["metadata"].get("week", "?")
    ctype = doc["metadata"].get("content_type", "?")
    print(f"    {sim:.4f}  [week {week}, {ctype}] {doc['text'][:60]}...")

# With filter: only forum threads
print(f"\n  WITH metadata filter (content_type='forum_thread' only):")

results_forum = store.search("How to build tools", top_k=3, metadata_filter={"content_type": "forum_thread"})
for r in results_forum:
    doc = r["document"]
    sim = r["similarity"]
    module = doc["metadata"].get("module", "?")
    resolved = doc["metadata"].get("resolved", "?")
    print(f"    {sim:.4f}  [{module}, resolved={resolved}] {doc['text'][:60]}...")

print(f"""
  METADATA FILTERING from the lecture:
  ┌──────────────┬───────────────────────────┬──────────────────────────────┐
  │ Field        │ Example Values            │ Why It Matters               │
  ├──────────────┼───────────────────────────┼──────────────────────────────┤
  │ module       │ rag, tool_calling, mcp    │ Filters to right subject     │
  │ content_type │ lecture, code_walkthrough  │ Debugging needs code         │
  │ week         │ 1, 2, 3, 4, 5, 6, 7      │ Students reference weeks     │
  │ resolved     │ True / False              │ Prioritize solved threads    │
  └──────────────┴───────────────────────────┴──────────────────────────────┘

  Filter FIRST, then search. This is how you go from
  "sort of relevant" to "exactly what the student needs."
""")

print("=" * 60)

# ============================================
# WHAT YOU LEARNED:
# 1. A vector store = embeddings + metadata + similarity search
# 2. add() embeds text and stores it with metadata
# 3. search() embeds the query and finds closest vectors
# 4. Metadata filtering narrows the search space for better precision
# 5. This is what Pinecone, Weaviate, and ChromaDB do internally
#
# EXERCISE 1:
# Add a delete() method to SimpleVectorStore:
#   def delete(self, doc_id):
#       # Remove document and its embedding by ID
# Test it: add a document, search for it, delete it, search again.
#
# EXERCISE 2:
# Add a "resolved only" filter for forum threads:
#   results = store.search(
#       "API error", top_k=3,
#       metadata_filter={"content_type": "forum_thread", "resolved": True}
#   )
# This simulates Sage showing only SOLVED problems.
#
# EXERCISE 3:
# Implement range filtering for the "week" field:
#   def search(..., week_range=(4, 6)):
#       # Only return docs from weeks 4-6
# This lets students ask "What did we cover in weeks 4-6?"
#
# EXERCISE 4 (Advanced):
# Save and load the vector store to/from disk:
#   import json
#   def save(self, path):
#       data = {"documents": self.documents, "embeddings": self.embeddings}
#       json.dump(data, open(path, "w"))
#   def load(self, path):
#       data = json.load(open(path))
#       self.documents = data["documents"]
#       self.embeddings = data["embeddings"]
# Why is this important? Embedding 25 docs takes seconds.
# Embedding 25,000 docs takes minutes. You don't want to re-embed every time.
#
# NEXT STEP: We'll connect the vector store to an LLM
# to build the complete RAG pipeline.
# ============================================
