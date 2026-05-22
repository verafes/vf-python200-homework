# project 07.py

import os
import pandas as pd
from scipy.stats import pearsonr
from smolagents import tool, CodeAgent, OpenAIServerModel
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv

load_dotenv()
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Global DataFrame
DATA_PATH = PROJECT_ROOT / "assignments_01" / "outputs" / "merged_happiness.csv"
FALLBACK_DIR = PROJECT_ROOT / "assignments" / "resources" / "happiness_project"
# outputs folder
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Task 1 — Define Tools ---

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the CSV from DATA_PATH. If that file does not exist,
    loads and merges all yearly CSVs from assignments/resources/happiness_project/.
    Stores the result in the global df and returns a dict with shape and columns.

    Returns:
        dict: A dictionary with dataset shape, columns, and the dataframe as a dict.
    """
    global df

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        frames = []
        for year in range(2015, 2025):
            path = os.path.join(FALLBACK_DIR, f"{year}.csv")
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
        if not frames:
            return {"error": "No happiness data found."}
        df = pd.concat(frames, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "df": df.to_dict(orient="list")
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column.

    Uses df[column].describe().to_dict().
    Returns {"error": "..."} if no data is loaded or column is missing.

    Args:
        column (str): The name of the column to summarize.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation and p-value between two numeric columns.

    Args:
        col1 (str): The first column name.
        col2 (str): The second column name.

    Returns:
        dict: Pearson r, p-value, or an error message.
    """
    global df
    if df is None:
        return {"error": "No data loaded."}
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found."}

    try:
        r, p = pearsonr(df[col1], df[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p, 4)
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column (str): The name of the column to rank countries by.
        year (int): The year to filter the dataset on.
        n (int): The number of top countries to return

    Returns:
        dict: A dictionary containing the dataset shape and column names.
    """
    global df
    if df is None:
        return {"error": "No data loaded."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if "year" not in df.columns:
        return {"error": "Dataset has no 'year' column."}

    subset = df[df["year"] == year]
    if subset.empty:
        return {"error": f"No data for year {year}."}

    top = subset.sort_values(column, ascending=False).head(n)
    return {
        "results": [
            {"country": row["country"], column: row[column]}
            for _, row in top.iterrows()
        ]
    }


# --- Task 2 — Build the Agent ---

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

TOOLS = [
load_happiness_data, summarize_column, compute_correlation, get_top_n_countries
]

def build_agent(api_key):
    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

    agent = CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
        max_steps=8,
    )
    return agent


if __name__ == "__main__":

    agent = build_agent(api_key)

    # Task 3 — Guided Queries
    QUERIES = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for q in QUERIES:
        print(f"\n--- Query: {q} ---")
        print(agent.run(q, reset=True))

    # Task 4 — My Queries
    my_query_1 = "Plot the distribution of happiness_score as a histogram and save it to outputs/hist.png."
    print(agent.run(my_query_1, reset=False))

    my_query_2 = "Which region had the highest average happiness_score in 2019?"
    print(agent.run(my_query_2, reset=False))


