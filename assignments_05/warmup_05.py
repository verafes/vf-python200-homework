# warmup_05.py — Week 5 Warmup Exercises
import os

from dotenv import load_dotenv
from openai import OpenAI
import json


load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded env variables from .env file")
else:
    print("Warning: .env file not found. API key is missing")

# Initialize OpenAI client
client = OpenAI()


# --- Completions API ---

# Q1: First Chat Completion
print("\n--- API Q1 ---")

prompt_api_q1 = "What is one thing that makes Python a good language for beginners?"

response_q1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{ "role": "user", "content": prompt_api_q1 }]
)

model_response = response_q1.choices[0]

print(f"Response text: {response_q1.choices[0].message.content}")
print(f"Model: {response_q1.model}")
print(f"Total tokens: {response_q1.usage.total_tokens}")


# Q2: Temperature Comparison
print("\n--- API Q2 ---")

prompt_api_q2 = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response_q2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{ "role": "user", "content": prompt_api_q2 }],
        temperature=temp
    )
    print(f"\nTemperature {temp}:")
    print(response_q2.choices[0].message.content)

# Lower temperature (0) gives consistent, predictable output.
# Higher temperature (1.5) gives more random and more varied responses.
# For reproducible output, using temperature=0 is preferred.


# API Q3: n=3 completions
print("\n--- API Q3 ---")

prompt_api_q3 = "Give me a one-sentence fun fact about pandas (the animal, not the library)."
response_q3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{ "role": "user", "content": prompt_api_q3 }],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response_q3.choices, start=1):
    print(f"Completion {i}: {choice.message.content}")


# API Q4: max_tokens
print("\n--- API Q4 ---")

prompt_api_q4 = "Explain how neural networks work."
response_q4 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{ "role": "user", "content": prompt_api_q4 }],
    max_tokens=15
)

print("Truncated response:", response_q4.choices[0].message.content)

# The model stops early because max_tokens limits how long the output can be.
# Useful for control cost, speed, or when you need short responses like in real applications.


# --- System Messages & Personas ---
# Q1: Personas
print("\n--- System Question 1  ---")

# Persona 1
user_prompt_q1 = "I don't understand what a list comprehension is."
messages_persona1 = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {
        "role": "user",
        "content": user_prompt_q1
    }
]

response_p1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_persona1
)

print(f"Tutor Personality: {response_p1.choices[0].message.content}")


# Persona 2
messages_persona2 = [
    {
        "role": "system",
        "content": "You are a grumpy senior engineer who explains things bluntly and with zero patience."
    },
    {
        "role": "user",
        "content": user_prompt_q1
    }
]

response_p2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_persona2
)

print(f"Strict Instructor Personality: {response_p2.choices[0].message.content}")

# The tone changes completely because the system message defines the model's personality.

# Q2: Conversation Memory
print("\n--- System Question 2 ---")
messages_q2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": (
            "Nice to meet you, Jordan! Python is a great choice. "
            "What would you like to work on?"
        )
    },
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response_q2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_q2
)

print(f"Conversation Response: {response_q2.choices[0].message.content}")

# The model knows Jordan's name because it is included the entire conversation in the messages list.
# Even though the API is stateless and the full conversation history,
# model only knows what is sent in the current request.


# ---- Prompt Engineering ----

# Q1: Zero-Shot Sentiment Classification
print("\n--- Prompt Engineering Q1  ---")

# input data
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

def classify_reviews(reviews):
    """
    Performs zero-shot sentiment classification on each review.
    """
    results = []

    for i, review in enumerate(reviews, start=1):
        # instructions
        prompt = f"Classify the sentiment (positive, negative, or mixed): {review}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        sentiment = response.choices[0].message.content.strip()
        results.append(f"Review {i}: {sentiment}")

    return "\n".join(results)

print(classify_reviews(reviews))


# Q2: One-Shot Sentiment Classification
print("\n--- Prompt Engineering Q2  ---")

def one_shot_classification(reviews):
    """
    Performs one-shot sentiment classification using a single example
    to guide the model's expected format and reasoning.
    """
    example = (
        'Example:\n'
        'Review: "Great price, but the documentation is nearly impossible to follow."\n'
        'Sentiment: mixed\n\n'
    )

    prompt = example + "Now classify these reviews:\n" + "\n".join(reviews)

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

# Execute using input data from previous task
print(one_shot_classification(reviews))

# Adding one example usually makes the output format more consistent.


# Q3: Few-Shot Sentiment Classification
print("\n--- Prompt Engineering Q3  ---")

def few_shot_classification(reviews):
    """Classify sentiment using several example reviews as guidance."""
    few_shot_examples = (
        'Example 1:\n'
        'Review: "The UI is clean and everything works perfectly."\n'
        'Sentiment: positive - the reviewer expresses strong satisfaction.\n\n'
        
        'Example 2:\n'
        'Review: "Nothing loads and the app freezes every time."\n'
        'Sentiment: negative - the reviewer reports repeated failures.\n\n'
        
        'Example 3:\n'
        'Review: "Good features, but the setup was confusing."\n'
        'Sentiment: mixed - both praise and criticism are present.\n\n'
    )

    prompt_q3 = few_shot_examples + "Now classify these reviews:\n" + "\n".join(reviews)

    messages_q3 = [{"role": "user", "content": prompt_q3}]

    response_q3 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_q3
    )

    return response_q3.choices[0].message.content

# Execute using input data from previous task
print(few_shot_classification(reviews))

# Zero-shot is fast, but shorter, less structured, sometimes inconsistent.
# The model chose its own format, can use bold or plain text.

# One-shot is more consistent formatting. Adding one example made the output more consistent and closer to the example’s format.

# Few-shot - the model follows the style exactly, uses the same tone, spacing;
# it is most reliable, when you need strict formatting or a specific structure.


# Q4: Chain of Thought
print("\n--- Prompt Engineering Q4 ---")
def chain_of_thought():
    """
    Ask the model to solve a salary word problem with step-by-step reasoning
    and a clearly labeled final answer.
    """
    prompt = (
        "Solve this step by step and show your reasoning before giving the final answer.\n\n"
        "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
        "takes a new job that pays $7,500 more per year than her post-raise salary. "
        "What is her final annual salary?\n\n"
        "Label the final answer clearly."
    )

    messages_q4 = [{"role": "user", "content": prompt}]

    response_q4 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_q4
    )

    return response_q4.choices[0].message.content

print(chain_of_thought())

# Asking for step-by-step reasoning improves accuracy because the model
# breaks the problem into smaller logical steps instead of guessing.


# Q5: Structured JSON Output
print("\n--- Prompt Q5 ---")

review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

def structured_output(review):
    """
    Request a JSON-only sentiment analysis of a review,
    then parse and print sentiment, confidence, and reason.
    """
    json_prompt = (
        "Analyze the review below and return ONLY valid JSON with keys:\n"
        "sentiment, confidence (0 to 1 float), and reason (one sentence).\n\n"
        f"Review: {review}"
    )

    messages = [{"role": "user", "content": json_prompt}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw_json = response.choices[0].message.content
    print("Raw response:", raw_json)

    try:
        parsed_json = json.loads(raw_json)
        print("Sentiment:", parsed_json["sentiment"])
        print("Confidence:", parsed_json["confidence"])
        print("Reason:", parsed_json["reason"])
    except Exception:
        print("JSON parsing failed. Raw response shown above.")

structured_output(review)

# Q6: Delimiters
print("\n--- Prompt Engineering Q6 ---")

def delimiter_test(text):
    """ Test model behavior using triple‑backtick input. """
    prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{text}```
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# First prompt: instructions present
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the pasta. "
    "Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

print(f"Prompt with instructions:\n{delimiter_test(user_text)}")

# Second prompt: no instructions in text
non_instruction_text = "The weather today is mild with scattered clouds, and I enjoyed a quiet walk in the park."

print(f"\nPrompt without instructions:\n{delimiter_test(non_instruction_text)}")

# Delimiters help prevent the model from confusing the user's text with the instructions.
# They clearly separate what the model should analyze from the instructions about how to analyze it.


# --- Local Models with Ollama ---
print("\n--- Ollama Q1: OpenAI API Response ---")

# In terminal, I ran:
# ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
# Result:
"""
--- OLLAMA OUTPUT ---

Thinking...
Okay, the user wants me to explain what a large language model is in two sentences. Let me start by breaking down
the key elements. First, I know that large language models are AI systems designed to understand and generate
human language. They need to be concise, so I should mention their ability to process and generate text.

Next, I should highlight their purpose. Maybe talk about tasks like writing, answering questions, or summarizing.
Then, include that they are trained on massive datasets to improve accuracy. Wait, but the user asked for two
sentences. Let me make sure I don't repeat. Also, check if there's any jargon I need to simplify. Alright, putting
it all together: First sentence introduces the model's function, second sentence explains their training and
purpose. That should cover it in two sentences.
...done thinking.

A large language model is an AI system designed to understand and generate human language, enabling tasks like
writing, answering questions, or summarizing text. It learns from vast datasets to improve accuracy and adapt to
various contexts, making it powerful for tasks requiring natural language processing.
"""

# Running the same prompt using OpenAI API
prompt = "Explain what a large language model is in two sentences."
messages_ollama_q1 = [
    {"role": "user", "content": prompt}
]

response_ollama_q1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_ollama_q1
)

print(f"OpenAI OUTPUT:\n{response_ollama_q1.choices[0].message.content}")

# Differences:
# The OpenAI response is more concise and polished, directly answering in two sentences as requested.
# The Ollama response included extra "thinking" steps with a long explanation
# before giving the final answer, making it longer and less direct.


# Advantage of running a model locally (Ollama):
# Local models does not require internet access and keeps data private on your machine,
# and no cost nothing after installation.

# Disadvantage:
# Local models are usually smaller and less powerful, so their responses may be less accurate
# or less refined compared to OpenAI models.
