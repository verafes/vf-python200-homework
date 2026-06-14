"""
etl_pipeline.py

Week 11 Cloud ETL Capstone

Extract:
    - Download 7 days of hourly weather data from Open-Meteo.

Transform:
    - Convert parallel weather arrays into records.
    - Classify the first 24 hourly records using OpenAI.

Load:
    - Upload enriched records to Azure Blob Storage.

Container:
    pipeline-data

Blob Path:
    final/YYYY-MM-DD/weather_etl.json

ETL_pipline video : https://youtu.be/pRXhVcDFuIw
"""

import os
import json
from datetime import date

import requests
from dotenv import load_dotenv
from prefect import task, flow, get_run_logger

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from openai import OpenAI


# ENV + CONSTANTS
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

API_KEY=os.getenv("OPENAI_API_KEY")

ACCOUNT_NAME = os.getenv("ACCOUNT_NAME")
if not ACCOUNT_NAME:
    print("Warning: missing ACCOUNT_NAME variable. Check your .env file.")
    raise ValueError(
        "Missing ACCOUNT_NAME environment variable. "
        "Set it in your .env file (ACCOUNT_NAME=<your-storage-account>)."
    )
ACCOUNT_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
CONTAINER = "pipeline-data"

LATITUDE = 38.684830
LONGITUDE = -121.456917

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}
MODEL = "gpt-4o-mini"
MAX_RECORDS = 24

client = OpenAI(api_key=API_KEY)


# --- HELPERS ---

def get_blob_clients():
    """ Create and return an Azure BlobServiceClient. """
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container = blob_service.get_container_client(CONTAINER)
    return blob_service, container

def verify_blobs(container, logger):
    """Print all blob names and sizes."""
    logger.info("Verifying blobs in container...")
    for blob in container.list_blobs():
        logger.info(f"- {blob.name} ({blob.size} bytes)")

# --- Transform helpers ---
def make_user_message(record, logger):
    """Make user message for each record"""
    if "temperature_2m" not in record or "precipitation" not in record:
        logger.error("Record missing required keys for LLM classification")
        return None
    if record["temperature_2m"] is None or record["precipitation"] is None:
        logger.warning("Record contains None values; classification may be unreliable")
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

def classify_record(client, record, valid_labels=None):
    """Call OpenAI to classify a single weather record."""
    logger = get_run_logger()
    if valid_labels is None:
        valid_labels = VALID_LABELS

    user_msg = make_user_message(record, logger)
    if user_msg is None:
        return "unknown"

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
    except Exception:
        return "unknown"

# --- load helpers ---

def upload_json_blob(container, payload, blob_path, logger):
    """Upload JSON bytes to Blob Storage."""
    blob_client = container.get_blob_client(blob_path)
    blob_client.upload_blob(payload, overwrite=True)
    logger.info(f"Uploaded: {blob_path} ({len(payload)} bytes)")
    return blob_path


# --- Extract Task ---

@task(name="Extract Weather Data", retries=2, retry_delay_seconds=10)
def extract_weather(lat: float = LATITUDE, lon: float = LONGITUDE) -> dict:
    """
    Extract 7 days of hourly temperature_2m and precipitation data
    from the Open-Meteo API for a chosen city.
    """
    logger = get_run_logger()

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation&forecast_days=7"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise RuntimeError(f"Extract step failed: {e}")

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Extract step failed: invalid JSON returned by API")
    if "hourly" not in data:
        logger.error("Extract step failed: 'hourly' key missing in API response")
        raise RuntimeError("Extract step failed: missing 'hourly' key")

    logger.info(f"Extracted raw forecast for ({lat}, {lon}) from Open-Meteo.")

    return data


#--- Transform Task ---

@task(retries=2,  cache_policy=None)
def transform_weather(client: OpenAI, records: dict, max_records: int) -> list[dict]:
    """
    Reshape hourly parallel lists into per-hour records and classify
    the first MAX_RECORDS (24) records using the OpenAI API.
    """
    logger = get_run_logger()

    hourly = records["hourly"]
    records = []
    for i in range(len(hourly["time"])):
        record = {
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }
        records.append(record)

    logger.info(f"Reshaped into {len(records)} hourly records.")
    enriched_records = []

    subset_records = records[:max_records]
    logger.info(f"Starting LLM transformation for {len(subset_records)} records...")

    for i, record in enumerate(subset_records):
        conditions = classify_record(client, record)
        enriched = {**record, "conditions": conditions}
        enriched_records.append(enriched)

        if (i + 1) % 6 == 0:
            print(f"Processed {i + 1} records...")

    logger.info(f"Finished processing {len(enriched_records)} records.")
    logger.info("Transform step completed.")

    return enriched_records


# --- Load Task ---

@task(name="Load Enriched Weather Data")
def load_weather(records: list[dict], date: str, blob_path) -> str:
    """
    Upload enriched records as JSON to final/<today>/weather_etl.json
    in the pipeline-data container with overwrite=True.
    """
    logger = get_run_logger()

    _, container = get_blob_clients()
    # serialize
    payload = json.dumps(records).encode("utf-8")

    try:
        upload_json_blob(container, payload, blob_path, logger)
    except Exception as e:
        logger.error(f"Blob upload failed: {e}")
        raise RuntimeError(f"Load step failed: {e}")

    logger.info(f"Uploaded {len(payload)} bytes to {blob_path} in container {CONTAINER}.")

    return blob_path


# --- Flow ---

@flow(name="Weather ETL Pipeline", log_prints=True)
def weather_etl_flow():
    """Full ETL pipeline: extract -> transform -> load."""
    logger = get_run_logger()

    # --- Setup ---
    _, container = get_blob_clients()

    # --- Extract ---
    today = date.today().isoformat()
    payload = extract_weather()

    # --- Transform ---
    enriched = transform_weather(client, payload, MAX_RECORDS)

    # --- Load ---
    blob_path_etl = f"final/{today}/weather_etl.json"
    blob_path = load_weather(enriched, today, blob_path_etl)

    # --- Verify ---
    verify_blobs(container, logger)

    logger.info(f"Flow completed successfully. Final blob path: {blob_path}")
    return blob_path


if __name__ == "__main__":
    weather_etl_flow()
