# Python 200 Projects & Cloud ETL Workspace

This repository contains my collection of Python projects, exercises, and end‑to‑end workflows. The work spans data analysis, API integrations, machine learning basics, LLM‑powered transformations, and cloud‑based ETL pipelines. Each folder represents a self‑contained project or practice module, progressing from foundational Python tasks to full cloud automation.

Each folder contains Python scripts, warmups, and small projects that build toward more advanced workflows.
The final folder (assignments_11/) includes a complete cloud ETL pipeline.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Prefect](https://img.shields.io/badge/Prefect-Orchestration-0A0A0A)
![Azure](https://img.shields.io/badge/Azure-Blob%20Storage-0089D6)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)

## Setup Instructions
1. Clone the repository
```bash
git clone <your-repo-url>
cd python200-homework
```

2. Create and activate a virtual environment
```bash
uv venv
# Linux
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```
3. Install dependencies
```bash
uv pip install -r requirements
```

4. Add environment variables: copy .env.example to .env file:
```OPENAI_API_KEY=your_key_here
ACCOUNT_NAME=your_azure_storage_account
```

5. Log in to Azure: `az login`

6. Start Prefect Server: `prefect server start`

7. Run the ETL pipeline: `python assignments_11/etl_pipeline.py`

**Output Location:**  
The ETL pipeline writes results to: 
`pipeline-data/final/<YYYY-MM-DD>/weather_etl.json`


## Features
- End‑to‑end ETL pipelines using Prefect for orchestration
- API integrations including Open‑Meteo and OpenAI
- Cloud storage workflows using Azure Blob Storage
- Data transformation utilities for reshaping and enriching datasets
- LLM‑powered classification for augmenting raw data
- Warmups and practice modules covering analysis, ML, AI, and cloud patterns
- Clear project structure from foundational exercises to full cloud automation
- Reproducible environment setup with .env support and Azure authentication


## Tech Stack
### Languages & Core Tools

- Python 3
- Prefect (workflow orchestration)
- Azure CLI + Azure Blob Storage
- OpenAI API
- LlamaIndex (vector indexing experiments)

### Python Libraries
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn, scipy — ML + statistics
- requests — API calls
- python-dotenv — environment management
- azure-storage-blob, azure-identity — cloud storage
- openai — LLM integration

## ETL Pipeline Overview
The main pipeline in this repo:
- Extracts 7 days of hourly weather data from the Open‑Meteo API
- Transforms the data into per‑hour records
- Classifies the first 24 hours using an OpenAI model
- Uploads the enriched JSON output to Azure Blob Storage
- Runs as an orchestrated Prefect flow with clear logs and observability

## License

This project is licensed under the MIT License.
