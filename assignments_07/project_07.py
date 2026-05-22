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
        # "df": df.to_dict(orient="list")
        "df": df.to_json()
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
    return [
        {"country": row["country"], column: row[column]}
        for _, row in top.iterrows()
    ]


# --- Task 2 — Build the Agent ---

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations, and ranking countries. 
Write Python code directly only when tools are insufficient.
For multi-line regional plots, ALWAYS write Python code instead of calling tools.
When using the output of load_happiness_data(), ALWAYS reconstruct the DataFrame using:
df = pd.read_json(happiness_data["df"])
Do NOT treat happiness_data as a DataFrame. It is a dict containing a JSON string.
Do NOT print the entire DataFrame.
When generating plots, place the legend on the right side using exactly:
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
Also ALWAYS include bbox_inches='tight' in plt.savefig(...) so the legend is not cut off.
Be concise and student-friendly.
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

def check_plot(filename: str):
    path = OUTPUT_DIR / filename
    print(f"{filename}: {'Exists' if path.exists() else 'Missing'}")

if __name__ == "__main__":

    agent = build_agent(api_key)

    plot_happiness = "happiness_by_region.png"
    plot_hist = "hist.png"
    # Task 3 — Guided Queries
    QUERIES = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        f"Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/{plot_happiness}.",
    ]

    for q in QUERIES:
        print(f"\n--- Query: {q} ---")
        print(agent.run(q, reset=False))

    # Task 4 — My Queries
    my_query_1 = f"Plot the distribution of happiness_score as a histogram and save it to outputs/{plot_hist}."
    print(agent.run(my_query_1, reset=False))
    # This triggered CODE GENERATION because no tool exists for histograms.
    # The agent wrote matplotlib code to generate the plot.

    my_query_2 = "Which region had the highest average happiness_score in 2019?"
    print(agent.run(my_query_2, reset=False))
    # This triggered TOOL USE (summarize_column + load_happiness_data)
    # and some and some light CODE GENERATION to compute the groupby.

    print("\n--- Verifying Plot Files ---")
    check_plot(plot_happiness)
    check_plot(plot_hist)


# --- Task 5: Reflection ---
# 1. In Query 3, agent correctly computed the Pearson correlation between gdp_per_capita and happiness_score.
# It also reported the correlation and then marked the p-value as statistically significant,
# because the p-value was 0.0, which is below the common threshold of 0.05.

# 2. Query 5 revealed that the agent’s behavior could vary slightly across runs.
# Sometimes the agent sometimes failed to access the correct column ('region') and sometimes succeeded.
# These differences helped me identify weaknesses in my tool output and SYSTEM_PROMPT. After
# refining the prompt and clarifying how the DataFrame should be reconstructed, the agent consistently
# used the correct column ('regional_indicator') and generated the plot reliably.
# This showed good debugging behavior and adaptability.

# 3. Early runs also showed that returning the full DataFrame as a large dict caused the agent to print
# thousands of rows and occasionally exceed the context window.
# Switching the dataset output to JSON prevented this, and adding a SYSTEM_PROMPT rule to reconstruct
# the DataFrame with pd.read_json(...) ensured that all later queries worked correctly.

# 4. Adjusting the SYSTEM_PROMPT was an important part of the debugging process and stabilizing the pipeline.
# Once I clarified how the agent should rebuild the DataFrame and how it should handle plotting,
#  the results became consistent.

# 5. A useful future improvement would be adding a "filter_rows" tool that lets the agent
# filter the dataset by conditions (e.g., region == 'Europe', year >= 2019).
# This would reduce the need for custom code and make more complex analytical queries
# easier for the agent to handle.
