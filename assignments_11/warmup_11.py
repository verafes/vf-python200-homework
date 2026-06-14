# warmup_11.py

from prefect import task, get_run_logger

# --- Prefect Orchestration ---

# Question 1

# @task is a single unit of work that Prefect can retry, cache, log, and observe separately.
# @flow is the orchestration layer. It defines the pipeline and controls task execution.

# A pure helper function, that converts Celsius to Fahrenheit, does not need @task because:
# - it is simple in-memory calculation with  no I/O
# - it is deterministic and fast
# - it does not benefit from task features such as retries, caching, or logging.

# So I would leave it as a normal Python function, not a @task.
# That reduces orchestration overhead and keeps the code simpler.


# Question 2
# @task(name="call_api", retries=3, retry_delay_seconds=30)


# Question 3

# In the Prefect UI, I would open the failed flow run and click on the 'transform' task run to inspect it.

# There I expect to see:
# - the task state (Failed)
# - the full traceback of the error
# - the exception message (e.g., KeyError, JSONDecodeError, API error, etc.)
# - the inputs passed into the task or task run metadata
# - task logs leading up to the error
# - logs printed before the failure

# This tells me exactly why transform failed and why load never ran.


# --- Production Patterns ---

#  Question 1

# raise_for_status() automatically throws an HTTPError exception if the response is 4xx or 5xx.
# This stops the task immediately and marks it as Failed.
#
# If I only check:
#     if response.status_code != 200:
#       print("error")
# the task still returns normally, so Prefect thinks it succeeded.
# Downstream tasks will run with bad or empty data.

# With raise_for_status():
# - the task fails immediately
# - Prefect marks the task as Failed, apply retries if configured
# - downstream tasks are skipped
# - the pipeline stops safely instead of continuing with invalid data


# Question 2
# overwrite=True protects the pipeline from failing when a file already exists at the destination path.
# And in case of re-run the pipeline after a crash,
# the new run will replace the old partial or corrupted file automatically.

# Without overwrite=True:
# - Azure would block the upload
# - the pipeline would fail again
# - I would be stuck with a broken file from the previous run

# overwrite=True protects me from leftover partial outputs.


# Question 3

@task
def log_loaded_records(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")

