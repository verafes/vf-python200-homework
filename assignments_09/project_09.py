# Project 09 -video link - https://youtu.be/vflE5yuZxBs

import os

import json
from datetime import date
import requests
from pathlib import Path
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from dotenv import load_dotenv

if load_dotenv():
    print("Env variables loaded successfully.")
else:
    print("Warning: could not load variables. Check your .env file.")

ACCOUNT_NAME = os.getenv("ACCOUNT_NAME")
if not ACCOUNT_NAME:
    print("Warning: missing ACCOUNT_NAME variable. Check your .env file.")
ACCOUNT_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
# ACCOUNT_URL = "https://veractd2026sa.blob.core.windows.net"

CONTAINER = "pipeline-data"

# my city - Sacramento, CA
LATITUDE = 38.684830
LONGITUDE = -121.456917


def get_blob_clients():
    """Return BlobServiceClient and ContainerClient."""
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container = blob_service.get_container_client(CONTAINER)
    return blob_service, container


def extract_weather(lat=LATITUDE, lon=LONGITUDE):
    """Fetch 7‑day hourly weather data from Open‑Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation&forecast_days=7"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def serialize_json(data):
    """Convert dict to UTF‑8 JSON bytes."""
    return json.dumps(data).encode("utf-8")


def upload_blob(container, payload):
    """Upload JSON bytes to raw/<today>/weather.json."""
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"
    blob_client = container.get_blob_client(blob_path)
    blob_client.upload_blob(payload, overwrite=True)
    print(f"Uploaded: {blob_path} ({len(payload)} bytes)")
    return blob_path


def verify_blobs(container):
    """Print all blob names and sizes."""
    print("\nBlobs in container:")
    for blob in container.list_blobs():
        print(f"- {blob.name} ({blob.size} bytes)")


def read_back(container, blob_path, output_path=None):
    """Download blob, load DataFrame, save JSON to outputs."""
    blob_client = container.get_blob_client(blob_path)
    try:
        downloaded = blob_client.download_blob().readall()
    except Exception as e:
        print(f"Download failed: {e}")
    parsed = json.loads(downloaded)

    df = pd.DataFrame(parsed["hourly"])
    print("\nFirst 5 rows:")
    print(df.head())

    with open(output_path, "w") as f:
        json.dump(parsed, f, indent=2)


def project_pipeline():
    """Run full pipeline."""

    # --- Setup ---
    blob_service, container = get_blob_clients()

    # --- STEP 1: Extract ---
    data = extract_weather()

    # --- STEP 2: Serialize ---
    payload = serialize_json(data)

    # --- STEP 3: Load ---
    blob_path = upload_blob(container, payload)

    # --- STEP 4: Verify ---
    verify_blobs(container)

    # --- STEP 5: Read Back ---
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    output_path = outputs_dir / "weather_raw.json"

    read_back(container, blob_path, output_path)


if __name__ == "__main__":
    project_pipeline()
