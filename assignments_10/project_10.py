# Reflection:
# Classifying hourly weather conditions with an LLM works, but it is not the most efficient or necessary approach for this task.
# The model can interpret borderline cases and handle ambiguous combinations of temperature and precipitation,
# but this flexibility comes with higher cost, slower runtime, and potential inconsistency.
# A deterministic rule-based system (e.g., temperature > 10  and precipitation < 1 → good) would be faster, cheaper,
# and fully predictable. By switching to rules, we lose the LLM’s ability to generalize nuanced conditions,
# but we gain full control, transparency, and predictable output.

# Project 10 - video link: https://youtu.be/V6_B5Hf0LK0

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from openai import OpenAI

from dotenv import load_dotenv


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

API_KEY=os.getenv("OPENAI_API_KEY")

ACCOUNT_NAME = os.getenv("ACCOUNT_NAME")
if not ACCOUNT_NAME:
    print("Warning: missing ACCOUNT_NAME variable. Check your .env file.")
ACCOUNT_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
# ACCOUNT_URL = "https://veractd2026sa.blob.core.windows.net"

CONTAINER = "pipeline-data"

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}
MODEL="gpt-4o-mini"

# project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# assignments/resources
DATA_DIR = ROOT_DIR / "assignments" / "resources"

# assignments_10/outputs
OUTPUT_DIR = ROOT_DIR / "assignments_10" / "outputs"
OUTPUT = OUTPUT_DIR / "first_10_records.json"


def get_blob_clients():
    """Return BlobServiceClient and ContainerClient."""
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container = blob_service.get_container_client(CONTAINER)
    return blob_service, container


def download_and_reshape(container, today, blob_path, fallback_path):
    """Load raw weather JSON from Blob Storage."""
    try:
        print(f"Trying to download blob: {blob_path}")
        raw = container.download_blob(blob_path).readall()
        data = json.loads(raw.decode("utf-8"))
        print("Loaded raw weather data from Blob Storage.")
    except Exception as e:
        print(f"No blob found for today's date ({today}). Scanning for earlier Week 9 uploads...")

        # look for any previously uploaded raw blobs
        uploaded_blobs = [b.name for b in container.list_blobs() if
                          b.name.startswith("raw/") and b.name.endswith(".json")]

        if uploaded_blobs:
            actual_blob_path = uploaded_blobs[0]
            print(f"Found your historical upload at: {actual_blob_path}")
            raw = container.download_blob(actual_blob_path).readall()
            data = json.loads(raw.decode("utf-8"))
        else:
            print("No historical blobs found. Using fallback file...")
            try:
                with open(fallback_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print("Loaded fallback weather data from local file.")
            except Exception:
                raise FileNotFoundError(
                    "ERROR: No weather data found in Azure or fallback file."
                )

    # Reshape hourly parallel lists into a list of records
    hourly = data["hourly"]
    records = []
    for i in range(len(hourly["time"])):
        record = {
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }
        records.append(record)

    print(f"Reshaped into {len(records)} hourly records.")
    return records


def make_user_message(record):
    """Make user message for each record"""
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )


def classify_record(client, record, valid_labels=None):
    """Call OpenAI to classify a single weather record."""
    if valid_labels is None:
        valid_labels = VALID_LABELS

    user_msg = make_user_message(record)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0
        )
        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in valid_labels else "unknown"

        return label
    except Exception as e:
        print(f"API Error processing record: {e}")
        return "unknown"


def transform_conditions(client, records):
    """Process first 24 hourly records using Azure OpenAI."""
    enriched_records = []
    subset_records = records[:24]
    print(f"Starting LLM transformation for {len(subset_records)} records...")
    for i, record in enumerate(subset_records):
        conditions = classify_record(client, record)
        enriched = {**record, "conditions": conditions}
        enriched_records.append(enriched)

        if (i + 1) % 6 == 0:
            print(f"Processed {i + 1} records...")

    print(f"Finished processing {len(enriched_records)} records.")
    return enriched_records


def upload_processed_blob(container, data, blob_path):
    """Upload structured JSON data back to processed storage path."""
    payload = json.dumps(data).encode("utf-8")
    blob_client = container.get_blob_client(blob_path)
    blob_client.upload_blob(payload, overwrite=True)

    print(f"Uploaded {len(data)} classified dataset to: {blob_path}")
    return blob_path


def spot_check(container, blob_path: str):
    """Download processed dataset, load into pandas, and output check stats."""
    raw = container.download_blob(blob_path).readall()
    records = json.loads(raw.decode("utf-8"))

    df = pd.DataFrame(records)
    print("\nSPOT CHECK: value_counts(): ")
    print(df["conditions"].value_counts())
    print("\nSPOT CHECK: First 5 rows:")
    print(df.head())

    return records


def save_records(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved first 10 enriched records to {OUTPUT}.")


def transform_pipeline():
    # --- Setup ---
    blob_service, container = get_blob_clients()

    # Step 1: Read and Reshape
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"
    fallback_path = DATA_DIR / "weather_raw.json"

    records = download_and_reshape(container, today, blob_path, fallback_path)

    # Step 2: Transform
    client = OpenAI(api_key=API_KEY)
    enriched_records = transform_conditions(client, records)

    # Step 3: Write
    raw_blob_path = f"processed/{today}/weather_classified.json" # input
    processed_blob_path = upload_processed_blob(container, enriched_records, raw_blob_path)

    # Step 4: Spot-Check
    processed_records = spot_check(container, processed_blob_path)

    # Step 5: Save Output
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_records(processed_records[:10], OUTPUT)


if __name__ == "__main__":
    transform_pipeline()
