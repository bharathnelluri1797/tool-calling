# ============================================
# STEP 16: Embeddings — Turning Text into Numbers
# ============================================
#
# WHAT YOU'LL LEARN:
# - Why keyword search fails for natural language
# - What embeddings are (text → vectors)
# - How cosine similarity works (built from scratch)
# - How to use real embedding models (sentence-transformers)
# - Semantic search: "trouble with weather data" → finds tool calling content
#
# KEY IDEA (from lecture):
#   Student asks: "Has anyone had trouble getting their data from weather API?"
#   Semantically = Tool Calling question.
#   But the word "Tool Call" never appears in the question.
#
#   Keyword search would MISS this. Semantic search finds it.
#   Embeddings are what make semantic search possible.
#
# INSTALL (one-time):
#   pip install sentence-transformers
#
# RUN THIS FILE:
#   python step16_embeddings.py
# ============================================

import math
from rag_documents import CURRICULUM_DOCUMENTS


# ============================================
# PART 1: Why Keyword Search Fails
# ============================================
# Let's try the simplest approach: search by word overlap.

def keyword_search(query, documents, top_k=3):
    """Search documents by counting matching words."""
    query_words = set(query.lower().split())
    results = []

    for doc in documents:
        doc_words = set(doc["content"].lower().split())
        # Count how many query words appear in the document
        overlap = len(query_words & doc_words)
        results.append((doc, overlap))

    # Sort by overlap (most matching words first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


print("=" * 60)
print("PART 1: Keyword Search — The Limitation")
print("=" * 60)

# This question is SEMANTICALLY about tool calling
# but doesn't use the words "tool calling"
query = "Has anyone had trouble getting their data from the weather API?"

print(f"\nQuery: {query}")
print(f"(This is about tool calling, but doesn't say 'tool calling')")
print("-" * 50)

results = keyword_search(query, CURRICULUM_DOCUMENTS)
for doc, score in results:
    preview = doc["content"][:80].replace("\n", " ")
    print(f"  [{score} words match] {doc['id']}: {preview}...")

print(f"\n  The CORRECT answer is doc_15 (forum thread about weather API 422 error).")
print(f"  Did keyword search find it? Maybe — but only because 'weather' appears.")
print(f"  It has no understanding of MEANING.")


# ============================================
# PART 2: Cosine Similarity — From Scratch
# ============================================
# Before using a library, let's understand the math.
#
# Cosine similarity measures the angle between two vectors:
#   cos(θ) = (A · B) / (|A| × |B|)
#
# - 1.0 = vectors point same direction (identical meaning)
# - 0.0 = vectors are perpendicular (unrelated)
# - -1.0 = vectors point opposite directions (opposite meaning)

def dot_product(vec_a, vec_b):
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def magnitude(vec):
    """Compute the magnitude (length) of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = dot_product(vec_a, vec_b)
    mag_a = magnitude(vec_a)
    mag_b = magnitude(vec_b)

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


print()
print("=" * 60)
print("PART 2: Cosine Similarity — The Math")
print("=" * 60)

# Simple example: 2D vectors
vec_same = [1, 0]
vec_similar = [0.9, 0.1]
vec_different = [0, 1]
vec_opposite = [-1, 0]

print(f"\n  Same direction:     cosine({vec_same}, {vec_similar})   = {cosine_similarity(vec_same, vec_similar):.4f}")
print(f"  Perpendicular:      cosine({vec_same}, {vec_different})     = {cosine_similarity(vec_same, vec_different):.4f}")
print(f"  Opposite:           cosine({vec_same}, {vec_opposite})    = {cosine_similarity(vec_same, vec_opposite):.4f}")
print(f"  Identical:          cosine({vec_same}, {vec_same})     = {cosine_similarity(vec_same, vec_same):.4f}")

print("""
  KEY INSIGHT:
  If we can convert text into vectors where similar MEANING
  produces similar DIRECTIONS, then cosine similarity
  gives us semantic search for free.

  That's exactly what embedding models do.
""")


# ============================================
# PART 3: Bag-of-Words Embeddings (From Scratch)
# ============================================
# The simplest "embedding": count word occurrences.
# Each dimension = one unique word.
# Each value = how many times that word appears.
#
# This captures some meaning but misses synonyms and context.

def bag_of_words_embed(text, vocabulary):
    """Convert text to a bag-of-words vector."""
    words = text.lower().split()
    # Count occurrences of each vocabulary word
    return [words.count(word) for word in vocabulary]


print("=" * 60)
print("PART 3: Bag-of-Words Embeddings (Simple but Limited)")
print("=" * 60)

# Build a small vocabulary from sample sentences
sentences = [
    "RAG retrieves relevant documents for the LLM",
    "Retrieval augmented generation helps the model answer",
    "The weather in Delhi is very hot today",
]

# Create vocabulary from all unique words
all_words = set()
for s in sentences:
    all_words.update(s.lower().split())
vocabulary = sorted(all_words)

print(f"\n  Vocabulary size: {len(vocabulary)} words")
print(f"  (Each word becomes one dimension in the vector)")
print()

embeddings = []
for s in sentences:
    emb = bag_of_words_embed(s, vocabulary)
    embeddings.append(emb)
    # Show only non-zero dimensions
    nonzero = [(vocabulary[i], v) for i, v in enumerate(emb) if v > 0]
    print(f"  \"{s[:50]}...\"")
    print(f"    Non-zero: {nonzero}")
    print()

# Compare similarities
sim_01 = cosine_similarity(embeddings[0], embeddings[1])
sim_02 = cosine_similarity(embeddings[0], embeddings[2])
sim_12 = cosine_similarity(embeddings[1], embeddings[2])

print(f"  Similarity (RAG sentence 1 vs RAG sentence 2): {sim_01:.4f}")
print(f"  Similarity (RAG sentence 1 vs Weather sentence): {sim_02:.4f}")
print(f"  Similarity (RAG sentence 2 vs Weather sentence): {sim_12:.4f}")
print()
print(f"  The two RAG sentences ARE more similar! But only because")
print(f"  they share common words like 'the', not because of meaning.")
print(f"  Real embeddings do much better.")


# ============================================
# PART 4: Real Embeddings with sentence-transformers
# ============================================
# sentence-transformers uses neural networks trained specifically
# to produce vectors where similar MEANINGS are close together.
#
# Model: all-MiniLM-L6-v2
# - 384 dimensions (vs our vocabulary-sized BoW vectors)
# - Trained on 1B+ sentence pairs
# - Captures synonyms, paraphrasing, and semantic relationships

print()
print("=" * 60)
print("PART 4: Real Embeddings with sentence-transformers")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # These sentences mean the same thing but use different words
    test_sentences = [
        "How do I fix tool calling errors?",                  # Original question
        "My function calling isn't working properly",          # Synonym: tool → function
        "Has anyone had trouble getting data from the API?",   # The lecture example!
        "What is the best restaurant in Bengaluru?",           # Completely unrelated
        "Debugging issues with LLM tool use",                  # Technical synonym
    ]

    print(f"\n  Model: all-MiniLM-L6-v2")
    print(f"  Embedding dimensions: 384")
    print()

    # Generate embeddings
    embeddings = model.encode(test_sentences)

    # Compare the first sentence against all others
    print(f"  Base: \"{test_sentences[0]}\"")
    print(f"  " + "-" * 50)

    for i in range(1, len(test_sentences)):
        sim = cosine_similarity(embeddings[0].tolist(), embeddings[i].tolist())
        print(f"  {sim:.4f}  \"{test_sentences[i]}\"")

    print(f"""
  NOTICE:
  - "function calling isn't working" scores HIGH (synonym!)
  - "trouble getting data from API" scores HIGH (same concept!)
  - "Debugging issues with LLM tool use" scores HIGH (related topic!)
  - "best restaurant in Bengaluru" scores LOW (unrelated)

  Embeddings understand MEANING, not just words.
  This is what makes semantic search possible.
""")


    # ============================================
    # PART 5: Semantic Search on Curriculum Data
    # ============================================
    # Now let's search our actual curriculum!

    print("=" * 60)
    print("PART 5: Semantic Search on the Curriculum")
    print("=" * 60)

    # Embed all curriculum documents
    doc_texts = [doc["content"] for doc in CURRICULUM_DOCUMENTS]
    doc_embeddings = model.encode(doc_texts)

    # Search for the tricky query from the lecture
    search_query = "Has anyone had trouble getting their data from the weather API?"
    query_embedding = model.encode([search_query])[0]

    print(f"\n  Query: \"{search_query}\"")
    print(f"  " + "-" * 50)

    # Calculate similarity with every document
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding.tolist(), doc_emb.tolist())
        similarities.append((CURRICULUM_DOCUMENTS[i], sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 5 results (semantic search):")
    for doc, sim in similarities[:5]:
        module = doc["metadata"]["module"]
        ctype = doc["metadata"]["content_type"]
        preview = doc["content"][:70].replace("\n", " ")
        print(f"  {sim:.4f}  [{module}/{ctype}] {preview}...")

    print(f"""
  doc_15 (the weather API forum thread) should rank near the top!
  The embedding model understands that "trouble getting data from
  the weather API" is semantically related to "422 errors when
  calling the weather tool."

  Compare this with keyword search from Part 1.
  Semantic search captures MEANING, not just words.
""")

except ImportError:
    print("""
  sentence-transformers is not installed.
  Install it with: pip install sentence-transformers

  Parts 4-5 use real neural network embeddings to demonstrate
  semantic search. The concepts from Parts 1-3 (cosine similarity,
  bag-of-words) work without any extra dependencies.

  After installing, re-run this file to see the full demo.
""")

print("=" * 60)

# ============================================
# WHAT YOU LEARNED:
# 1. Keyword search matches words, not meaning
# 2. Cosine similarity measures direction alignment (angle between vectors)
# 3. Bag-of-words is a simple embedding (word counts as vectors)
# 4. Neural embeddings (sentence-transformers) capture semantic meaning
# 5. Semantic search finds relevant results even with different words
#
# EXERCISE 1:
# Write a query that keyword search gets WRONG but semantic search
# gets RIGHT. Hint: use synonyms or rephrase a concept.
#   - "How to make the AI stop making things up" (= hallucination)
#   - "Connecting different software components" (= API development)
#
# EXERCISE 2:
# Compare embedding similarity for these pairs:
#   - "RAG" vs "retrieval augmented generation" (abbreviation)
#   - "The model hallucinates" vs "The AI makes things up" (synonyms)
#   - "Python code" vs "JavaScript code" (related but different)
# Which pairs score highest? Does it match your intuition?
#
# EXERCISE 3:
# Implement TF-IDF from scratch (a better bag-of-words):
#   TF(word, doc) = count(word in doc) / total_words(doc)
#   IDF(word) = log(total_docs / docs_containing_word)
#   TF-IDF(word, doc) = TF × IDF
# This weights rare, distinctive words higher than common ones.
# Compare TF-IDF search vs bag-of-words vs semantic search.
#
# EXERCISE 4 (Advanced):
# Measure the SPEED difference between keyword, BoW, and semantic search:
#   import time
#   start = time.time()
#   # ... run search ...
#   elapsed = time.time() - start
# How do they compare? Which would you use for 100K documents?
#
# NEXT STEP: We'll build a vector store that stores embeddings
# and provides fast, metadata-filtered search.
# ============================================
