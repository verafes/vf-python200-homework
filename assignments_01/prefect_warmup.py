import numpy as np
import pandas as pd
from prefect import task, flow

# data
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    """Convert NumPy array to a pandas Series named 'values'."""
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    """Remove NaN values."""
    return series.dropna()

@task
def summarize_data(series):
    """Return summary statistics as a dictionary."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow(name="Data Summary Pipeline")
def pipeline_flow():
    """Prefect pipeline."""
    s = create_series(arr)
    clean_s = clean_data(s)
    summary = summarize_data(clean_s)

    return summary

if __name__ == "__main__":
    result_q2_pipeline = pipeline_flow()

    from warmup_01 import data_pipeline, arr

    result_q1_pipeline = data_pipeline(arr)
    print("Q1 Plain Python pipeline result:")
    print("\n".join(f"{key}: {value}" for key, value in result_q1_pipeline.items()))

    print("\nQ2 Perfect pipeline result:")
    print("\n".join(f"{k}: {v}" for k, v in result_q2_pipeline.items()))

    print("\nMatch:", result_q1_pipeline == result_q2_pipeline)

"""
Why might Prefect be more overhead than it is worth here?
This pipeline is tiny: only three simple functions running on a small array.
There is no scheduling, retries, caching, parallelism, or orchestration needed.
Prefect adds setup, decorators, and runtime overhead that doesn't provide extra value for such a small task.

When is Prefect still useful even for simple pipelines?
Prefect is helpful for real-world pipelines with large datasets; it also useful:
- when the pipeline needs to run regularly (hourly, daily, weekly);
- when you need logging, monitoring, or alerting, for example an automated Slack alert the moment the pipeline fails;
- when tasks depend on external systems (APIs, databases, cloud storage);
- when you want retries or caching: e.g. 'create_series' pulled data from a flaky web API that often fails.
- when you want observability: simple dashboard (the Prefect UI) to prove to a stakeholder 
  that the data was processed successfully.
"""
