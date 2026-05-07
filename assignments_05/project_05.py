import os

from dotenv import load_dotenv
from openai import OpenAI
import json


load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded env variables from .env file")
else:
    print("Warning: .env file not found. API key is missing")

client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


# --- Task 1: Setup and System Prompt ---
print("\n--- Task 1 ---")

system_prompt = """
You are a job application coach. You help people who are changing careers or applying
for new roles. Your job is to make their resume bullet points and cover letters clearer,
stronger, and easier to understand.

Stay focused only on job application materials. Help with:
  - resume bullet points
  - short professional summaries
  - cover letters
When rewriting resume bullets, make them:
  - specific and outcome-focused
  - written in clear, professional language
  - tailored toward the target role or industry when the user provides it
- When drafting cover letters, keep them:
  - concise (1 page or less)
  - structured (intro, relevant experience, closing)
  - aligned with the job description details the user shares

Do not answer questions that are not about job applications or career materials.

Always remind the user to review, edit, and personalize your writing before they send it
to any company, recruiter, or job portal.

You do not fully know the rules, norms, or expectations of every industry, company, or
country. When it matters, say that clearly and tell the user to use their own judgment
and adjust the text to fit their field and local norms.

Write in simple, clear language. Be supportive, practical, and respectful of the user's
experience, even if they are changing careers.
"""

# Deliberate choice:
# I clearly limited the assistant to ONLY job application materials and added a reminder
# for the user to review and edit the output. This helps keep the model focused and makes
# it clear that its answers are drafts, not final documents to send as-is.


# --- Task 2: Bullet Point Rewriter ---
print("\n--- Task 2 ---")

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    response_text = get_completion(messages)

    # Parsing JSON
    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError:
        print("JSON parsing failed. Here is the raw model output:")
        print(f"Raw response: {response_text}")
        raise

    # Print both versions side by side
    for item in response_data:
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}\n")

    return response_data

# Test Data
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

