# --- LLMs as Transform ---
# Q1

# Parse "Jan 5th, 2024" into ISO date like "2024-01-05":
# Use deterministic code: date parsing follows strict, rule‑based patterns.
# And standard Python libraries (`datetime` or `dateutil`) can parse this reliably, quickly, and at zero token cost.

# Classify support ticket ("my card was charged twice"):
# Use LLM: classification of natural language requires understanding language context,
# and LLM is good at semantic understanding.

# Calculate the average of a list of numbers:
# Use deterministic code: mathematical operations must be 100% exact and Python can compute averages reliably,
# whereas an LLM might hallucinate or approximate instead of performing exact arithmetic.

# Extract company name from "Sr. Data Eng @ Acme Corp (contract)":
# Use an LLM: people write job titles in all sorts of messy ways, and an LLM is much better
# at spotting context clues than rigid, complicated text-matching rules.

# Determine whether a review is more than 100 words:
# Use deterministic code: counting words is a basic task that Python can do instantly with a single line of code;
# a simple (`len(review.split()) > 100`) is instantaneous, so using an AI is unnecessary.


# --- Q2 ---
# Problem:
# The system prompt "Summarize this product review in a few sentences." produces unpredictable, conversational text.
# In data pipeline, this is hard to parse, store, or validate because the output has no structure,
# no guaranteed fields, and no consistent format.

# Rewritten Prompt Strategy:
# Force the model to return structured data (like JSON) with exact keys.

def get_improved_prompt():
    """
    Returns a system prompt engineered for structured downstream parsing.
    """
    system_prompt = """
    You are a transformation step in a data pipeline.
    Summarize the provided product review. 
    Return a JSON object with exactly two keys:
    - "summary": a 2–3 sentence summary of the review
    - "sentiment": one of ["positive", "neutral", "negative"]
    
    Do not include markdown formatting or backticks; respond ONLY with valid JSON.
    """
    return system_prompt


# --- Q3 ---
# If each call takes 1 second and there are 50,000 records:
# Sequential time = 50,000 seconds - this equals roughly 13.9 hours.

# One practical scaling strategy for efficiency:
# Use asynchronous programming (such as Python's asyncio) to send many requests at the same time
# instead of waiting for each one to finish sequentially.
# This greatly reduces idle network wait time while still respecting API rate limits (RPM/TPM).
# Another option is to use the OpenAI Batch API, which processes large datasets in bulk overnight
# at a lower cost and without requiring people to manage concurrency manually.


# --- Azure OpenAI ---

# --- Q1 ----
# Two reasons organizations choose Azure OpenAI:
# 1. Enterprise security and compliance:
# All data processed remains strictly within the organization's private Azure tenant, keeping it isolated
# from the public cloud.
# Microsoft guarantees that customer prompts and completions are never used to train public base models.
# This setup allows companies to meet strict legal and regulatory standards like HIPAA and GDPR.

# 2. Native Network Security and Access Control:
# Azure OpenAI can run inside private networks using Azure Virtual Networks (VNets) and Private Endpoints,
# keeping AI traffic off the public internet.
# It also uses Azure Active Directory (RBAC) to handle user permissions, so companies can control
# who can access the service using the same security tools they already use.


# --- Q2 ---
# AzureOpenAI requires three Azure‑specific parameters:
# 1. azure_endpoint: the specific URL of your Azure OpenAI resource.
# 2. api_version: the specific date‑based API version string that Azure requires (e.g., "2026-06-01", "2026-05-01-preview").
# 3. azure_deployment: the custom name of the deployed model inside your Azure OpenAI resource.


# --- Q3 ---
# It does not accept general names like "gpt-4o-mini".
# Instead, it takes the custom "Deployment Name" we created when launching the model.

# Where to find it:
# Azure Portal -> Resource Manager -> look under "Deployments" section.
