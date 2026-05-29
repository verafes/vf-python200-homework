# Project 08 — Cloud Cost Analysis + Cloud Shell Script

## Part 1: Portal Walkthrough Summary

In the Azure Portal, I confirmed that I am working inside the **Code the Dream** subscription. I showed my personal resource group (`p200-2026-vera-rg`) and confirmed that it contains my storage account. 
I opened Cloud Shell and showed that my `test.txt` file is still present in `~/clouddrive`. I also listed my SSH keys inside `~/.ssh` and ran `az group list` to display all resource groups in the tenant. 
This confirmed that my environment is configured correctly.

---

## Part 2: Cost Analysis

## Infrastructure Scenarios Summary

| Scenario | Service / Component | Configuration Details | Hourly Rate | Estimated Monthly Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | Azure Virtual Machine | Standard_B1s (1 vCPU, 1 GB RAM) - 160 hours/mo | ~$0.0104 | **$1.66** |
| **Scenario B** | Azure Virtual Machine | Standard_NC6s_v3 (6 vCPU, 1 V100 GPU) - 730 hours/mo | ~$3.06 | $2,233.80 |
| | Azure SQL Database | General Purpose tier, 4 vCores, compute only | N/A | ~$380.00 |
| | Azure Blob Storage | Hot Tier, 1 TB Data, LRS (Locally Redundant) | N/A | ~$20.00 |
| **Scenario B Total**| | **Full Infrastructure Stack** | **N/A** | **~$2,633.80** |

### Scenario
We need to run a small Python script once per day. It runs for about 10 minutes and only needs a small machine (1 vCPU, 1–2 GB RAM). 
The goal is to compare the cost of running it in different environments.

### Cost Comparison

**Local laptop**
- No cloud cost.
- You maintain everything yourself.
- Not reliable for automation.

**Azure Virtual Machine**
- Billed per hour even when idle.
- A small VM (B1s) costs around $8–$12/month if running 24/7.
- Overkill for a 10‑minute daily script.

**Azure Container Instances (ACI)**
- You pay only for the container runtime.
- A 10‑minute run per day costs only a few cents per month.
- Good for scheduled jobs.

**Azure Functions (serverless)**
- You pay only for execution time + memory.
- Also, just a few cents per month for this workload.
- Easiest and cheapest option for automation.

**Scenario A** used a small Standard_B1s VM running 160 hours per month and cost about **$1.66**, which was surprisingly low for a virtual machine. 
Scenario B was much more expensive: the NC6s_v3 GPU VM alone cost over **$2,200/month**, and adding SQL Database and 1 TB of Blob Storage brought the total to roughly **$2,633/month**. 
The difference between the two scenarios was larger than I expected, especially how dramatically GPU compute increases the monthly cost.

### Conclusion
For a short daily script, **Azure Functions** or **Azure Container Instances** are the most cost‑efficient. 
A VM is the most expensive because it runs 24/7 even though the script only needs 10 minutes.

While exploring the Pricing Calculator, I noticed how much prices change when switching regions, how reservations and savings plans can reduce VM costs, and how storage and SQL pricing scale independently from compute. It was also interesting to see how Azure recommends certain VM sizes based on workload.

---

## Part 3 — Cloud Shell Script

I ran my script in Cloud Shell using command: `python3 project_08.py` 
This script prints the monthly cost for both scenarios using the hourly rates I entered.  
The calculated values **matched the Pricing Calculator exactly**, confirming that the script’s math aligned with Azure’s pricing.