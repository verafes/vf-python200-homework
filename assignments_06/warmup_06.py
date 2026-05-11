# warmup_06.py — Week 6 Warmup Exercises

import os
import string

from dotenv import load_dotenv

load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded env variables from .env file")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---
# Concepts Q1

# Scenario A:
# Best approach: RAG.
# Reason: The assistant needs to answer questions about a large, frequently updated set of internal PDFs,
# so the model should not memorize them.
# RAG lets the assistant retrieve the most relevant, up-to-date policy chunks from a document index at query time,
# without retraining the model every quarter.

# Scenario B:
# Best approach: Fine-tuning.
# Reason: The goal is to consistently write in a very specific writing style that isn’t common online,
# and they have 3,000 examples.
# Fine-tuning on 3,000 high-quality, in-house examples teaches the model that exact style directly,
# so it can generalize to new copy.

# Scenario C:
# Best approach: Prompt engineering.
# Reason: The analyst only needs to query a single short, two-page report, and does not need a reusable system.
# The simplest solution is to include the report text directly in the prompt and ask questions,
# without building a full RAG pipeline or fine-tuning.


# Concepts Q2

# A confidently wrong answer is more harmful than "I am not sure" because it encourages the user
# to trust incorrect information and act on it without double-checking. Uncertainty signals that
# the user should verify or seek another source.

# Example: In a medical triage chatbot, a model might confidently state that a user’s symptoms are
# "definitely not serious" and recommend home care, when in fact they match signs of a heart attack.
# The confident tone can delay seeking emergency care and cause real physical harm.

# The way the model expresses its answer matters because humans use tone and certainty as cues for
# credibility. A calm, authoritative, detailed explanation feels trustworthy, so we are less likely
# to question it—even when the content is wrong.


# Concepts Q3
# Correct RAG pipeline order:
steps = [
    "Receive the user's query",
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
]

# So the correct workflow is:
# 1. The system gets the natural-language question or request from the user.
# 2. Raw text is pulled out of PDFs, web pages, or other document formats so it can be processed.
# 3. The long document text is broken into smaller, manageable pieces that can be indexed and retrieved efficiently.
# 4. Each chunk is turned into a vector representation that captures its semantic meaning.
# 5. The user’s question is also converted into an embedding so it can be compared to the document chunks.
# 6. The system finds the chunks whose embeddings are most similar to the query embedding (e.g., via cosine similarity).
# 7. The selected chunks are added to the LLM prompt as context, alongside the user’s question.
# 8. The LLM uses the injected context plus the query to produce a grounded, context-aware answer.


# Helper
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# --- Keyword RAG  ---

# Keyword RAG Q1:
print("\n--- Keyword Q1 ---")

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)
best_name, best_content = results[0]
print("Selected document:", best_name)

# Q1: The selected document was "loyalty.txt". Keyword retrieval got this wrong.
# The correct answer should be "hours.txt", but keyword RAG only matches exact tokens.
# The query contained "your", and both hiring.txt and loyalty.txt contain "your",
# while hours.txt does not contain exact matches for "hours" or "weekend".
# Because of this, the algorithm incorrectly selected loyalty.txt.
# his shows a limitation of simple keyword matching.


# Keyword RAG - Q2
print("\n--- Keyword Q2 ---")

query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)
best_name, best_content = results[0]
print("Selected document:", best_name)

# Keyword retrieval returned "None found" because none of the documents contain
# the exact words "caffeine" or "without".
# Keyword RAG fails here because it cannot understand the concept of "caffeine-free" or "decaf".
# A semantic / embedding-based retrieval system would do better because it understands meaning,
# not just exact word matches.


# Keyword RAG - Q3
print("\n--- Keyword Q3 ---")

query = "How do I sign up for rewards?"

# Prediction (before running):
# I predict the selected document will be "loyalty.txt" because it talks about a loyalty program,
# earning points, and redeeming rewards, which is conceptually closest to "sign up for rewards".

results = simple_keyword_retrieval(query, documents, verbose=True)
best_name, best_content = results[0]
print("Selected document:", best_name)

# After running:
# I expected "loyalty.txt" to be selected. Actual result: "None found".
# My prediction was incorrect. Keyword RAG only matches exact tokens, and none of the documents contain
# "rewards", "sign", or "up". Even though loyalty.txt is the correct semantic match,
# keyword retrieval cannot understand that "rewards" relates to "loyalty program".
# This demonstrates another limitation of keyword-based retrieval.


# --- Semantic RAG Concepts ---

#  Semantic Question 1
# What is a vector embedding?
# A vector embedding is a list of numbers that represents the meaning of a piece of text.
# so that similar ideas end up close together in vector space even if they use different words.

# If one chunk has cosine similarity of 0.85 and another has 0.30, the 0.85 chunk is more relevant.
# Higher cosine score means the two pieces of text point in a similar direction, so their meaning is closer.
#
# Semantic search can find the right chunk even without matching words because it compares meaning,
# not exact text. It understands that different words can still express the same idea and that
# "loyalty program" can match a query about "rewards" even if the words differ.


# Semantic Question 2
# | Feature                    | Keyword RAG                     | Semantic RAG                               |
# |----------------------------|---------------------------------|--------------------------------------------|
# | What is compared?          | Exact word overlap              | Vector embeddings (semantic meaning)       |
# | What is retrieved?         | Full document                   | Most relevant (similar) chunks             |
# | Can it handle synonyms?    | No                              | Yes                                        |
# | Storage format             | Plain text dictionary           | Embedding vectors stored in an index       |
# | Relevance score            | Number of overlapping keywords  | Cosine similarity (how close meanings are) |

