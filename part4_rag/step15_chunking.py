# ============================================
# STEP 15: Chunking — Breaking Documents into Pieces
# ============================================
#
# WHAT YOU'LL LEARN:
# - Why we need chunking (documents are too long for retrieval)
# - Three chunking strategies: fixed-size, sentence-based, overlapping
# - How chunk size affects what you retrieve
#
# KEY IDEA (from lecture):
#   You don't read all 600 pages to answer one question.
#   You use the INDEX to find the right chapter → page → paragraph.
#
#   Chunking is the first step in building that index:
#   Break large documents into small, searchable pieces.
#
# NO API KEY NEEDED — this step is pure Python computation.
#
# RUN THIS FILE:
#   python step15_chunking.py
# ============================================


# ============================================
# PART 1: Why Chunking?
# ============================================
# Our curriculum documents are already short (~50-80 words each).
# In the real world, a single document could be a 10-page PDF,
# a 2000-word blog post, or an entire codebase.
#
# We need to break them into "chunks" that are:
#   - Small enough to embed and search efficiently
#   - Large enough to contain meaningful information
#   - Aligned with natural content boundaries

# A long document to demonstrate chunking
LONG_DOCUMENT = """Week 5 - Complete RAG Tutorial

Introduction to RAG:
RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation. The idea is simple: instead of relying on the LLM's training data alone, you first retrieve relevant documents from your own data, then use those documents as context for the LLM to generate an answer.

Why RAG Matters:
LLMs are trained on data up to a cutoff date. They don't know about your company's internal documents, recent events, or specialized domain knowledge. RAG bridges this gap by giving the LLM access to external knowledge at query time. This is different from fine-tuning, which permanently changes the model's weights.

The RAG Pipeline:
Step 1 - Indexing: Before you can retrieve, you need to prepare your data. This involves chunking documents into smaller pieces, converting each chunk into a vector embedding, and storing these embeddings in a vector database. This is a one-time offline process.

Step 2 - Retrieval: When a user asks a question, convert the question into a vector embedding using the same model. Search the vector database for the most similar chunks. Return the top-k results (typically 3-5 chunks).

Step 3 - Augmentation: Take the retrieved chunks and inject them into the LLM's prompt. This is the "augmentation" part — you're augmenting the prompt with retrieved context. The system prompt typically says "Answer based on the following context."

Step 4 - Generation: The LLM reads the augmented prompt and generates an answer grounded in the retrieved context. Because the answer is based on your actual data, it's far less likely to hallucinate.

Common Pitfalls:
Chunk size matters enormously. Too small and you lose context. Too large and you add noise. Overlapping chunks help maintain context across boundaries. Metadata filtering can dramatically improve precision by narrowing the search space before semantic search.

Evaluation:
Don't evaluate with vibes. Use RAGAS metrics: context precision, context recall, faithfulness, and answer relevancy. Each metric tells you something different about where your pipeline is failing."""


# ============================================
# PART 2: Fixed-Size Chunking
# ============================================
# The simplest approach: split by character/word count.
# Fast, predictable, but dumb — may split mid-sentence.

def chunk_by_size(text, chunk_size=200, unit="words"):
    """Split text into fixed-size chunks by word count."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


print("=" * 60)
print("STRATEGY 1: Fixed-Size Chunking (200 words)")
print("=" * 60)

fixed_chunks = chunk_by_size(LONG_DOCUMENT, chunk_size=200)

for i, chunk in enumerate(fixed_chunks):
    word_count = len(chunk.split())
    # Show first 100 chars
    preview = chunk[:100].replace("\n", " ")
    print(f"\n  Chunk {i + 1} ({word_count} words): {preview}...")

print(f"\n  Total chunks: {len(fixed_chunks)}")
print(f"\n  PROBLEM: Look at where chunks split.")
print(f"  Chunk boundaries don't respect sentence boundaries!")


# ============================================
# PART 3: Sentence-Based Chunking
# ============================================
# Split on sentence boundaries. Preserves meaning better,
# but chunks may vary widely in size.

def chunk_by_sentences(text, max_sentences=3):
    """Split text into chunks of N sentences each."""
    # Simple sentence splitting (handles ., !, ?)
    # In production, use nltk or spaCy for better sentence detection
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks


print()
print("=" * 60)
print("STRATEGY 2: Sentence-Based Chunking (3 sentences per chunk)")
print("=" * 60)

sentence_chunks = chunk_by_sentences(LONG_DOCUMENT, max_sentences=3)

for i, chunk in enumerate(sentence_chunks):
    word_count = len(chunk.split())
    preview = chunk[:100].replace("\n", " ")
    print(f"\n  Chunk {i + 1} ({word_count} words): {preview}...")

print(f"\n  Total chunks: {len(sentence_chunks)}")
print(f"\n  BETTER: Each chunk contains complete sentences.")
print(f"  But some chunks are much longer than others.")


# ============================================
# PART 4: Paragraph-Based Chunking
# ============================================
# Split on double newlines (paragraph boundaries).
# Best for structured documents with clear sections.

def chunk_by_paragraphs(text):
    """Split text into chunks by paragraph (double newline)."""
    paragraphs = text.strip().split("\n\n")
    # Filter out very short paragraphs (headers, etc.)
    chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    return chunks


print()
print("=" * 60)
print("STRATEGY 3: Paragraph-Based Chunking")
print("=" * 60)

para_chunks = chunk_by_paragraphs(LONG_DOCUMENT)

for i, chunk in enumerate(para_chunks):
    word_count = len(chunk.split())
    preview = chunk[:100].replace("\n", " ")
    print(f"\n  Chunk {i + 1} ({word_count} words): {preview}...")

print(f"\n  Total chunks: {len(para_chunks)}")
print(f"\n  BEST FOR STRUCTURED TEXT: Each chunk is a coherent section.")


# ============================================
# PART 5: Overlapping Chunks
# ============================================
# The key insight: when you split text, information at the
# boundary gets lost. Overlapping chunks solve this.
#
# Example: "The answer is at the end of chunk 1 and
#           the start of chunk 2" — without overlap,
#           neither chunk has the complete answer.

def chunk_with_overlap(text, chunk_size=150, overlap=50, unit="words"):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # How far to advance each time

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        # Stop if we've covered everything
        if i + chunk_size >= len(words):
            break

    return chunks


print()
print("=" * 60)
print("STRATEGY 4: Overlapping Chunks (150 words, 50-word overlap)")
print("=" * 60)

overlap_chunks = chunk_with_overlap(LONG_DOCUMENT, chunk_size=150, overlap=50)

for i, chunk in enumerate(overlap_chunks):
    word_count = len(chunk.split())
    preview = chunk[:100].replace("\n", " ")
    print(f"\n  Chunk {i + 1} ({word_count} words): {preview}...")

print(f"\n  Total chunks: {len(overlap_chunks)}")
print(f"\n  Each chunk overlaps with its neighbor by ~50 words.")
print(f"  This means boundary information appears in BOTH chunks!")


# ============================================
# PART 6: Comparison
# ============================================

print()
print("=" * 60)
print("COMPARISON: Which Strategy When?")
print("=" * 60)

print(f"""
  Strategy          Chunks   Best For
  ─────────────────────────────────────────────────────
  Fixed-size         {len(fixed_chunks)}       Simple data, consistent sizes needed
  Sentence-based     {len(sentence_chunks)}      Natural language, meaning preservation
  Paragraph-based    {len(para_chunks)}       Structured docs with clear sections
  Overlapping        {len(overlap_chunks)}       Any data — preserves boundary context

  IN PRACTICE:
  - Start with paragraph/section-based chunking
  - Add overlap (10-20% of chunk size)
  - Experiment with chunk size for YOUR data
  - Too small (< 50 words) = chunks lack context
  - Too large (> 500 words) = chunks add noise to retrieval
""")

print("=" * 60)

# ============================================
# WHAT YOU LEARNED:
# 1. Chunking breaks documents into searchable pieces
# 2. Fixed-size is simple but may split sentences
# 3. Sentence/paragraph chunking preserves meaning
# 4. Overlapping chunks prevent boundary information loss
# 5. Chunk size is a critical parameter — experiment!
#
# EXERCISE 1:
# Try different chunk sizes (50, 100, 200, 500 words).
# For each, search for the word "RAGAS" — which chunk
# size gives you the most useful context around it?
#
# EXERCISE 2:
# Chunk the curriculum documents from rag_documents.py
# using paragraph-based chunking with overlap:
#   from rag_documents import CURRICULUM_DOCUMENTS
#   for doc in CURRICULUM_DOCUMENTS:
#       chunks = chunk_with_overlap(doc["content"], chunk_size=50, overlap=15)
#       print(f"{doc['id']}: {len(chunks)} chunks")
#
# EXERCISE 3:
# Implement a "semantic chunking" strategy:
# Split on section headers (lines starting with "Week",
# "Step", "Assignment", etc.). This produces chunks that
# are topically coherent regardless of length.
#
# EXERCISE 4 (Advanced):
# Build a chunking strategy that PRESERVES metadata.
# Each chunk should inherit the metadata from its parent
# document (week, module, content_type). You'll need this
# for metadata filtering in step 17.
# Hint:
#   def chunk_document(doc, chunk_size=100, overlap=30):
#       chunks = chunk_with_overlap(doc["content"], chunk_size, overlap)
#       return [{"content": c, "metadata": doc["metadata"]} for c in chunks]
#
# NEXT STEP: Now that we have chunks, how do we find which
# ones are RELEVANT to a question? We need embeddings.
# ============================================
