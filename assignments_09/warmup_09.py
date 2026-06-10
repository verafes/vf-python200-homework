# --- Azure Authentication ---

# --- Azure Authentication Q1 ---

# When I run a Python script locally that uses DefaultAzureCredential,
# it relies on our *local Azure CLI login session* for authentication.
#
# Before running the script, I must run: `az login`
# After login, Azure CLI stores a cached authentication token on my machine.
#
# DefaultAzureCredential automatically checks credentials.
# When it detects that I am already logged in through `az login`,
# it silently uses that token to authenticate my Python code.
# The command `az account show` prints JSON, my Azure CLI session is now persistent.


# --- Azure Authentication Q2 ---

# A deployed pipeline (running on an Azure VM, container, App Service, etc.) cannot use `az login`
# because `az login` is an *interactive* command that requires a human to open a browser and sign in.
# Cloud resources cannot perform interactive authentication.
# Pipelines and cloud resources do not have user ot interactively sign in.
# Instead, deployed resources use a Managed Identity that Azure automatically provides to the VM/container.
# It requires no secrets, no tokens, and no login steps.
#
# DefaultAzureCredential works without code changes because it automatically detects the environment:
# - Locally, it uses the Azure CLI token from `az login`.
# - In Azure, it uses the Managed Identity.
# Because of this automatic switching, the same Python code works both locally and in the cloud without any changes.


# --- Azure Authentication Q3 ---

# If DefaultAzureCredential fails immediately with an AuthenticationError, it usually means
# either we are not logged into Azure CLI or you are logged into the wrong tenant/subscription.
# To diagnose the first case, run `az account show`; if it errors or returns nothing,
# we need to run `az login` again to create a valid local token.
# To diagnose the second case, we need to check the active tenant and subscription with `az account show`,
# then list all available accounts with `az account list --output table`.
# If the wrong one is active, switch to the correct subscription using:
# `az account set --subscription "<SUBSCRIPTION_ID>"`
# After fixing the login or subscription, we can rerun the script.


# --- Blob Storage ---

# --- Blob Storage Q1 ---
# Azure Blob Storage has a three-level hierarchy:
# 1. Storage Account  – the top-level container (like a hard drive)
# 2. Container        – folders inside the storage account
# 3. Blob             – the actual files (text, CSV, images, JSON, etc.)

# An analogy is a filing cabinet: the storage account is the entire cabinet,
# each container is one of the drawers, and each blob is a document stored inside a drawer.


#  --- Blob Storage Q2 ---
# Scenario 1: A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
# I'd use Blob Storage because hourly JSON responses are raw files that do not need relational queries.

# Scenario 2: Your pipeline produces a table of 50 million customer transactions
# that your analytics team queries by date range and customer ID every day.

# I'd use a relational database because 50 million customer transactions require fast searches.
# SQL indexing makes it quick to filter by date or customer ID, which Blob Storage can't do.

# Scenario 3: A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.

# I'd use Blob Storage because NumPy embeddings are large binary files,
# and Blob Storage is made for saving raw data like arrays between pipeline runs.


# --- Blob Storage Q3 ---
def list_container(container_client):
    """
    Prints the name and size (bytes) of every blob in the container.
    """
    for blob in container_client.list_blobs():
        print(blob.name, blob.size)


# --- Blob Storage Q4 ---
def upload_text(container_client, blob_name, text):
    """
    Uploads a UTF-8 encoded string as a blob, overwriting if it exists.
    """
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(text.encode("utf-8"), overwrite=True)
