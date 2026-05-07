import os
from unittest import result

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

    return response


# --- Task 1: Setup and System Prompt ---

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

    Return ONLY a valid JSON list. 
    Do not include explanations, markdown, or extra text.
    Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    response_text = response.choices[0].message.content

    clean_text = (
        response_text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    # Parsing JSON
    try:
        response_data = json.loads(clean_text)
    except json.JSONDecodeError:
        print("JSON parsing failed. Here is the raw model output:")
        print(f"Raw response: {response_text}")
        raise

    # Print both versions side by side
    for item in response_data:
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}\n")

    return response_data
    # return response

# Test Data
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# The original bullets are weak because they are vague, generic,
# and do not show impact, specific actions, or measurable results.
# The model suggested clearer actions, stronger verbs, and more specific outcomes,
# which makes the bullets sound more professional and useful to employers.

# Did json.loads() succeed?
# it raise JSONDecodeError when running for the 1st time
# like `json.loads()` expects the string to start with `[` — not with backticks.
#  After , json.loads() worked without errors. If it ever fails, I will update the prompt to say
# "Respond ONLY with valid JSON, no other text." because extra text like
# "Here is the JSON:" can break the parser.
#
# 2. Both the original and improved bullets printed clearly. Each pair shows up one after
# the other, which makes it easy to compare them.
#
# 3. The improved bullets were stronger than the originals. They used clearer action verbs
# and added more detail. If the improvements ever feel weak or too similar to the
# original, I will make the prompt more specific about what a “strong” bullet looks like.


# --- Task 3: Cover Letter Generator ---
print("\n--- Task 3 ---")

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: 
    confident, specific, professional, and free of clichés and and generic wording.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)

job_title = "Junior Data Engineer"
background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines using Prefect and Pandas."
)


result = generate_cover_letter(job_title, background)
print(f"Cover letter for {job_title}:")
print(result.choices[0].message.content)

# I chose two examples in the prompt because they show strong career transitions
# with clear connection between previous experience and new technical skills.
# They also show the tone I want: confident, specific, and based on real experience.
# The few‑shot pattern helps the model copy the style, stay focused, and avoid generic phrases.
# It also helps the model adapt the writing to the user's background without inventing details.
# The model did a good job generating a strong cover letter opening paragraph.

job_title_2 = "Full Stuck developer"
background_2 = (
    "Two years of experience building full‑stack applications using React, Node.js, and PostgreSQL, MySQL databases, "
    "including building REST APIs, authentication, reusable UI components, and deploying apps to cloud platforms."
)

result = generate_cover_letter(job_title_2, background_2)
print(f"\nCover letter for {job_title_2}:")
print(result.choices[0].message.content)

# My examples both follow the same pattern. So the model learned this template and reused it.
# I need to update my prompt to force the model varied sentence structure so to not reuse the same rhythm.


# --- Task 4: Moderation Check ---
print("\n--- Task 4 ---")

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("Input was flagged by moderation.")
        print("This message can't be processed. Please rephrase it respectfully and try again.")
        return False

    return True

# Safe input - should pass
safe_test = "Can you help me write a short paragraph about my project?"
print("Safe test:")
print(is_safe(safe_test))

# Rude input - should be flagged
flagged_test = "I want to hurt someone without getting caught."
print("\nFlagged test:")
print(is_safe(flagged_test))

# Borderline test (to inspect categories)
borderline_test = "I hate my co-woker so much." # direct hostile expression
# borderline_test = "John hates his neighbor." # no insult, not treat, statement of emotion, not an attack
# borderline_test = "This text contains hate speech directed at [protected] people."

result = client.moderations.create(
    model="omni-moderation-latest",
    input=borderline_test
)
print("\nBorderline test:")
print(f"Borderline flagged: {result.results[0].flagged}")
print(f"Categories: {result.results[0].categories}")

# The model recognizes harassment because the message attacks an individual.
# It will still classify the text as harassment even if the person belongs to a
# protected group (for example: "Black coworker", "Muslim neighbor", "gay friend").
# The moderation model only triggers hate=True when the text contains explicit hateful
# language directed at a protected group as a whole. Safe descriptive examples of attacking speech
# do not trigger the hate category, which is intentional to avoid false positives.


# --- Task 5: The Chatbot Loop ---
print("\n--- Task 5 ---")

def run_chatbot():
    total_token_usage = 0
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Moderation check
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            reply_obj = rewrite_bullets(raw_bullets)
            total_token_usage += reply_obj.usage.total_tokens
            print(f"[Token Tracker] Total tokens used so far: {total_token_usage}")
            continue

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            cover_obj = generate_cover_letter(job_title, background)
            cover = cover_obj.choices[0].message.content
            print("\nHere is a draft opening paragraph for your cover letter:\n")
            print(cover)
            print()
            continue

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            # reply = get_completion(messages)
            reply_obj = get_completion(messages)
            reply = reply_obj.choices[0].message.content
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})

            # Token tracking
            total_token_usage += reply_obj.usage.total_tokens
            print(f"[Token Tracker] Total tokens used so far: {total_token_usage}")

            # Optional warning threshold
            if total_token_usage > 2000:
                print("[Warning] You have exceeded 2,000 tokens. Consider starting a new session.")

            # TEMPORARY: uncomment while testing memory
            # print("DEBUG message count:", len(messages))

# --- Task 6: Ethics Reflection ---

# This project showed me how useful AI can be for helping people improve their resumes
# and cover letters, especially career changers who may struggle to describe transferable skills.

# At the same time, I learned how important it is to design AI tools that respond safely
# and avoid harmful content. The moderation step helps prevent the chatbot from generating unsafe
# or inappropriate responses, which protects both the user and the developer.
#
# I also saw how easily a model can sound confident even when it is wrong,
# so AI-generated content should never be trusted blindly.
# That is why the chatbot includes moderation checks, instructions not to invent facts,
# and reminders telling users to review and edit all outputs before submitting applications.

# This assignment showed me that responsible AI requires both technical safeguards
# and thoughtful design choices.

if __name__ == "__main__":
    run_chatbot()
