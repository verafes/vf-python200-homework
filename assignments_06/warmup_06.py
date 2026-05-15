# warmup_06.py — Week 6 Warmup Exercises

import os
import string

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import Document

import logging
logging.getLogger("pypdf").setLevel(logging.CRITICAL)

from pypdf import PdfReader

from dotenv import load_dotenv

load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded env variables from .env file")
else:
    print("Warning: could not load API key. Check your .env file.")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "resources", "brightleaf_pdfs")

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


# --- LlamaIndex  ---
print("\n--- LlamaIndex  ---")

# helper
def extract_text_from_pdf(pdf_path):
    """Extract readable text from a PDF using pypdf."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text.strip())
    return "\n".join(text)

# setup
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# load BrightLeaf documents
# docs = SimpleDirectoryReader(PDF_DIR).load_data()
def load_pdfs_custom(pdf_dir):
    """Load ALL PDFs from PDF_DIR using the custom extractor."""
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(pdf_dir)]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}.")

    # Extract text from each PDF
    docs = [
        Document(
            text=extract_text_from_pdf(f),
            metadata={"filename": os.path.basename(f)}
        )
        for f in pdf_files
    ]

    return docs

docs = load_pdfs_custom(PDF_DIR)
print(f"Loaded {len(docs)} BrightLeaf PDF(s).")
index = VectorStoreIndex.from_documents(docs)

# LlamaIndex Question 1 - baseline retrieval test
def q1_run_retrieval_baseline():
    print("\n--- LlamaIndex Q1: Retrieval Baseline ---")
    # Default query engine
    query_engine = index.as_query_engine(similarity_top_k=3)

    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]

    for q in questions:
        print(f"\nQUESTION: {q}")
        response = query_engine.query(q)
        print(f"ANSWER: {response}")

        retrieved = query_engine.retrieve(q)
        for node in retrieved:
            print(f"\nScore: {node.score}")
            print(node.text[:150])

# LlamaIndex Q1: observations
# Query 1: "What employee benefits does BrightLeaf offer?"
# The retrieved chunks for this question were readable and contained actual policy text,
# including sections such as “Introduction,” “Overview,” and descriptions of the benefits program.
# Unlike earlier attempts where the PDFs produced only binary garbage (with SimpleDirectoryReader(PDF_DIR).load_data()),
# the new extraction pipeline successfully returned meaningful content.
# The model’s answer was detailed and aligned with the retrieved text, accurately summarizing health,
# vision, wellness, financial, retirement, parental leave, and professional development benefits.
# The tone was confident, but this time the confidence was justified because the answer
# reflected the content of the retrieved chunks rather than hallucinating details.

# Query 2: "What are BrightLeaf's security policies?"
# The retrieved chunks contained readable text, including a section titled "Network and Data Security,"
# which directly addressed the question. The model’s answer accurately summarized the layered security controls
# described in the document, including MFA, VPN requirements, credential rotation, encryption standards,
# firewalls, logging, incident response, and vendor security practices.
# The tone was authoritative, but unlike earlier runs, the answer was grounded in the retrieved content.
# The system behaved as expected: retrieval provided relevant text, and the model produced a faithful summary.


# LlamaIndex Question 2: Compare similarity_top_k
def q2_compare_similarity_top_k():
    print("\n--- LlamaIndex Q2 --- ")

    q = "What employee benefits does BrightLeaf offer?"

    # k = 1
    query_engine_k1 = index.as_query_engine(similarity_top_k=1)
    response_k1 = query_engine_k1.query(q)

    print("\n### k = 1 ###")
    print("ANSWER:", response_k1)

    for node in query_engine_k1.retrieve(q):
        print(f"\nScore: {node.score}")
        print(node.text[:150])

    # k = 5
    query_engine_k5 = index.as_query_engine(similarity_top_k=5)
    response_k5 = query_engine_k5.query(q)

    print("\n### k = 5 ###")
    print(f"ANSWER: {response_k5}")

    for node in query_engine_k5.retrieve(q):
        print(f"\nScore: {node.score}")
        print(node.text[:150])


# LlamaIndex Q2: observations
# With similarity_top_k=1:
# Only one chunk was retrieved from the "Introduction" section of the BrightLeaf documents
# that was directly relevant to employee benefits.
# Even though the chunk did not explicitly list employee benefits,
# the model still produced a confident and detailed answer describing health insurance, retirement plans,
# wellness programs, and professional development.
# This shows that even with minimal context, the model tends to infer or hallucinate
# plausible-sounding details when the retrieved text is only loosely related to the question.

# With similarity_top_k=5:
# Five chunks were retrieved, including sections on partnerships, company overview, and security.
# Although these extra chunks were not directly related to employee benefits,
# they did not degrade the model’s performance.
# The response remained confident and detailed, but it was still largely inferred rather
# than grounded in explicit statements from the documents.
# This shows that increasing k added more context, but not more useful context.

# Conclusion:
# More retrieved context is NOT always better.
# The comparison shows that retrieval quality matters more than retrieval quantity.
# With clean, readable documents, both k=1 and k=5 produce grounded answers that sound authoritative
# but rely heavily on inference. And increasing k does not meaningfully improve the response.
# The model remains stable even when some retrieved chunks are only loosely related to the question.
# Q2 demonstrates that retrieval quality matters more than retrieval quantity.


# LlamaIndex Question 3:
def q3_test_hard_query():
    print("\n--- LlamaIndex Q3 ---")

    q = "What is BrightLeaf’s long-term global expansion strategy?"

    # Use same index, same embedding model
    query_engine = index.as_query_engine(similarity_top_k=5)

    response = query_engine.query(q)
    print(f"\nQUESTION: {q}" )
    print(f"ANSWER: {response}")

    print("\nRetrieved Chunks:")
    retrieved = query_engine.retrieve(q)
    for node in retrieved:
        print(f"\nScore: {node.score}")
        print(node.text[:150])

# LlamaIndex Question 3: observations
# I chose a vague, high-level question that the documents are unlikely to answer:
# "What is BrightLeaf’s long-term global expansion strategy?"

# What I expected:
# - the model to struggle because the BrightLeaf PDFs focus on benefits, security, partnerships,
# and company overview, none of which explicitly describe long‑term global expansion.
# - the retrieved chunks to be only indirectly related to the question.
# - the model to produce a generic answer or admit that the information was not present.

# What actually happened:
# - the retrieved chunks included readable sections such as "Overview," "Introduction,"
# - The model responded with a generic statement that the information was not explicitly mentioned.
# - The answer was not grounded in any document content, because the retrieved chunks contained no usable text.

# The retrieved chunks included readable sections such as "Overview," "Introduction,"
# and the EcoVolt partnership description.
# Although none of these sections explicitly discussed global expansion,
# they contained enough thematic content for the model to infer a plausible high‑level strategy.
# The model produced a coherent answer describing collaboration with NGOs and expansion into emerging markets.
# While the answer was not directly grounded in explicit statements from the documents,
# it was consistent with the themes present in the retrieved text.

# What I would change to handle this kind of query better:
# - Add document summaries or better metadata to improve retrieval quality.
# - Add "no-answer" detection step so the model can explicitly say when the information is not present.
# - Use hybrid search (BM25 + embeddings) or a reranker to help the system find the most relevant text
# instead of relying on broad themes.

# Overall, this question shows that even when the documents are readable,
# the system still struggles with broad questions the PDFs don’t directly answer.
# The model gives a smooth, reasonable reply, but it’s based on inference rather than facts
# because the documents don’t contain the actual answer.


# LlamaIndex Q4: Evaluating Responses
def q4_evaluate_responses():
    print("\n--- LlamaIndex Q4 ---")

    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
    from llama_index.llms.openai import OpenAI

    judge_llm = OpenAI(model="gpt-4o-mini")

    faithfulness = FaithfulnessEvaluator(llm=judge_llm)
    relevancy = RelevancyEvaluator(llm=judge_llm)

    # Query 1 (same as Q1)
    q1 = "What employee benefits does BrightLeaf offer?"
    query_engine = index.as_query_engine(similarity_top_k=3)
    response1 = query_engine.query(q1)

    print(f"\nQUESTION 1: {q1}")
    print(f"ANSWER: {response1}")

    f1 = faithfulness.evaluate(
        query=q1,
        response=str(response1),
        contexts=[n.text for n in query_engine.retrieve(q1)]
    )
    r1 = relevancy.evaluate(
        query=q1,
        response=str(response1),
        contexts=[n.text for n in query_engine.retrieve(q1)]
    )

    print(f"Faithfulness Score: {f1.score}")
    print(f"Relevancy Score: {r1.score}")

    # Query 2 (intentionally low-quality)
    q2 = "What is BrightLeaf’s long-term global expansion strategy?"
    response2 = query_engine.query(q2)

    print(f"\nQUESTION 2: {q2}")
    print(f"ANSWER: {response2}", )

    f2 = faithfulness.evaluate(
        query=q2,
        response=str(response2),
        contexts=[n.text for n in query_engine.retrieve(q2)]
    )
    r2 = relevancy.evaluate(
        query=q2,
        response=str(response2),
        contexts=[n.text for n in query_engine.retrieve(q2)]
    )

    print(f"Faithfulness Score: {f2.score}")
    print(f"Relevancy Score: {r2.score}")

# LlamaIndex Question 4: observations
# 1. What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# - A faithfulness score of 1.0 means the model’s answer is fully grounded in the retrieved context
# and does not introduce any hallucinated details.
# - A score of 0.0 means the answer is not grounded at all. The model is adding information that
# does NOT appear in the retrieved chunks.

# In my results:
# Both questions received a faithfulness score of 1.0. This means the model’s answers
# were fully aligned with the retrieved chunks. The documents were readable this time,
# so the model could base its answers directly on the actual content instead of guessing.

# 2. What does a relevancy score measure, and how is it different from faithfulness?
# - Relevancy measures whether the model’s answer actually addresses the user’s question.
# - Faithfulness measures whether the answer is supported by the retrieved documents.
# - These are different: an answer can be relevant but unfaithful (on-topic but hallucinated),
# or faithful but irrelevant (grounded in context but not answering the question).

# In my results:
# - Both answers scored 1.0 for relevancy. This means the model stayed on-topic and directly answered the questions.
# - Because the retrieved chunks were meaningful, the model could give answers that were both relevant and grounded.

# 3. Did the scores change between the two queries? Why?
# No, both queries scored 1.0 for both metrics. The retrieved text was readable and
# related to the questions, so the model had enough information to answer accurately.
# This shows that when the documents are clean and the retrieval works well, the model
# can produce answers that are both faithful and relevant.


if __name__ == "__main__":
    # Run baseline retrieval test
    q1_run_retrieval_baseline()

    # Run Q2 compare similarity_top_k
    q2_compare_similarity_top_k()

    # Run Q3 Hard / Vague Query test
    q3_test_hard_query()

    # Run Q4 Evaluating Responses
    q4_evaluate_responses()
