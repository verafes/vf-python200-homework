# project_06.py — Week 6 Mini Project

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from warmup_06 import simple_keyword_retrieval


# --- Step 1: Setup ---

load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded env variables from .env file")
else:
    print("Warning: could not load API key. Check your .env file.")

BASE_DIR = Path(__file__).parent

# assignments_06/resources/groundwork_docs
docs_dir = BASE_DIR / "resources" / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Document directory found: {docs_dir}")


# --- Step 2: Load the Documents ---
print(f"\nStep 2: Loading documents from: {docs_dir}")
reader = SimpleDirectoryReader(input_dir=str(docs_dir))
documents = reader.load_data()

print(f"Total documents loaded: {len(documents)}")
for i, doc in enumerate(documents, start=1):
    file_name = doc.metadata.get("file_name", "UNKNOWN_FILE")
    print(f"  {i}. {file_name}")


# --- Step 3:  Build the Index and Query Engine ---
print("\n--- Step 3 ---")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.\n")


# --- Step 4: Query the Assistant ---
print("\n--- Step 4 ---")
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

print("Groundwork Coffee Q&A")
for q in questions:
    print("-" * 50)
    print(f"Question: {q}\n")

    response = query_engine.query(q)
    print("Answer:")
    #  answer text
    print(response.responce)

    # Top retrieved source node
    if response.source_nodes:
        top_node = response.source_nodes[0]
        node = top_node.node
        score = top_node.score

        file_name = node.metadata.get("file_name", "UNKNOWN_FILE")
        chunk_preview = " ".join(node.text[:200].split())

        print("Top retrieved source node:")
        print(f"  Document: {file_name}")
        print(f"  Similarity score: {score:.4f}")
        print(f"  Text preview:\n{chunk_preview}...")
    else:
        print("No source nodes returned for this query.")

    print()

# Q4: Reflection on the Five Queries

# All five queries returned accurate, grounded answers, and each one retrieved the correct source document
# with strong similarity scores (0.75–0.89). The assistant sounded confident and the responses matched
# the content of the source files: FAQ for hours and loyalty, seasonal_specials for dairy-free options,
# our_story for the origin story, and wholesale_catering for catering/wholesale.
# I didn’t see any hallucinations or mismatched sources and each answer remained grounded in the retrieved context.
# This shows that the RAG pipeline works reliably and produced faithful, relevant responses.


# --- Step 5: Find a Failure ---
print("\n--- Step 5 ---")
# Choose a question you expect to be hard. Example:
failure_question = (
    "If I host a monthly book club, can Groundwork create a custom package "
    "with tiered discounts based on attendance?"
)

print("=" * 80)
print("FAILURE CASE INVESTIGATION")
print(f"Question: {failure_question}\n")

failure_response = query_engine.query(failure_question)

print("Answer:")
print(failure_response.response)
print()

print("Retrieved source nodes (up to 3):")
for i, sn in enumerate(failure_response.source_nodes[:3], start=1):
    node = sn.node
    score = sn.score
    file_name = node.metadata.get("file_name", "UNKNOWN_FILE")
    chunk_preview = node.text[:200]

    print(f"\nSource node {i}:")
    print(f"  Document: {file_name}")
    print(f"  Similarity score: {score:.4f}")
    print(f"  Text preview: {chunk_preview}...")

# --- Step 5 Reflection ---
# I asked about custom tiered discounts for a monthly book club because this information
# does not exist in the Groundwork documents. I expected the system to struggle, and it did:
# the retrieval pulled the closest matches (catering, story, and menu),
# but none of them actually contain anything about custom pricing or tiered discounts.
# Despite this, the model still produced confident answer that blended real catering details with assumptions.
# The tone did not become uncertain even when the evidence was weak,
# which shows how easy it is for an AI to sound authoritative while being wrong.
# To improve this, I would add a verification step that checks whether the answer is explicitly supported
# by the retrieved text and instruct the model to say “I don’t know” when the documents
# do not contain the required information.


# --- Step 6: Final Reflection ---
# 1. The manual semantic RAG pipeline in the lesson required many lines of code for chunking, embedding,
# storing vectors, and performing similarity search. In contrast, the LlamaIndex version in this project
# only needed a few lines to load build a VectorStoreIndex, and create a query engine.
# This shows the value of using a framework: it abstracts away the boilerplate
# and lets me focus on retrieval quality and evaluation rather than low-level implementation details.

# 2. A strong real-world use case for this approach would be an internal policy
# or compliance assistant for a hospital, law firm, or university.
# Staff could query large collections of PDFs and manuals without manually searching through them,
# and the system would return grounded answers based on approved documents.

# 3. One failure mode that RAG cannot fully prevent is confident hallucination.
# Even when retrieval works correctly, the model may still infer details that are not present
# in the retrieved text or present guesses as facts. RAG reduces this risk but cannot eliminate it entirely,
# which is why checking source nodes and designing for uncertainty are important.


# --- Extension A: Side-by-Side Comparison (Moderate) ---
# No new setup required. use existing settings and helpers

# read all text files
documents_text = {f.name: f.read_text() for f in docs_dir.glob("*.txt")}

def compare_keyword_vs_semantic_rag():
    print("\n--- Extension A: Side-by-Side Comparison ---")
    for q in questions:
        # --- Keyword RAG ---
        kw_result = simple_keyword_retrieval(q, documents_text, verbose=False)
        kw_doc, kw_content = kw_result[0]
        kw_preview = " ".join(kw_content[:300].split())

        print("KEYWORD RAG")
        print(f"Retrieved Document: {kw_doc}")
        print(f"Answer Preview: {kw_preview}\n")

        # --- Semantic RAG ---
        sem_response = query_engine.query(q)
        sem_answer = sem_response.response
        sem_doc = sem_response.source_nodes[0].node.metadata.get("file_name")

        print("SEMANTIC RAG (LlamaIndex)")
        print(f"Retrieved Document: {sem_doc}")
        print(f"Answer: {sem_answer}\n")

# --- Extension A Reflections (Concise Version) ---
# - Did keyword RAG retrieve the right document?
# - How did the answer quality differ?
# - Did keyword RAG match semantic RAG or fail?

# 1. Hours on weekends: Keyword RAG failed (retrieved wholesale_catering.txt),
# while Semantic RAG correctly retrieved faq.txt.
# This shows keyword RAG cannot interpret meaning and may match irrelevant tokens.

# 2. Dairy-free options: Keyword RAG partially relevant (retrieved menu.txt),
# Semantic RAG retrieved seasonal_specials.txt with a more accurate answer.
# Keyword RAG cannot distinguish dairy vs non-dairy; semantic RAG understands intent.

# 3. Loyalty program: both systems retrieved faq.txt.
# Keyword RAG works because the query used exact words found in the document.

# 4. Origin story: both retrieved our_story.txt.
# Strong keyword overlap helped keyword retrieval succeed.

# 5. Catering/wholesale — both retrieved wholesale_catering.txt.
# Exact keyword matches made retrieval straightforward for both approaches.

# Summary:
# Keyword RAG only works when the query contains exact words from the document.
# Semantic RAG performs better overall because it retrieves based on meaning, not only exact token overlap.
