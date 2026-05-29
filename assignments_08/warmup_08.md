# Warmup 08 — Cloud Computing

## Cloud Concepts Question 1
The core economic model of cloud computing is **pay‑as‑you‑go**, where we only pay for the resources we actually use.  This differs from owning servers because on‑prem hardware requires large upfront costs, dedicated data‑center space, and ongoing maintenance, and we pay for all of it whether the servers are busy or sitting idle.

---

## Cloud Concepts Question 2

**Vertical scaling** ("scaling up") means upgrading a single existing machine (more CPU, RAM, GPU, faster storage).  
**Horizontal scaling** ("scaling out") means adding more machines to share the workload.

### Scenarios
- **GIS or scientific simulation that runs as one giant process**: Vertical scaling — the model runs as a single big task, so it can’t be split across machines and needs a bigger, more powerful one.
- **Web app jumps to 100,000 users:** Horizontal scaling — adding more identical web‑server instances behind a load balancer to handle a massive spike in concurrent traffic.
- **Relational database struggling with complex, single‑threaded queries:** Vertical scaling — upgrading the database server’s CPU/RAM because the workload cannot be parallelized.
- **Pipeline grows from 10 → 10,000 files and can run in parallel:** Horizontal scaling — the work can be split across many machines.

---

## Cloud Concepts Question 3

### Classifications
**Gmail (SaaS)** -- fully managed application we simply use.
**Azure Virtual Machines (IaaS)** -- cloud‑based virtual computers.
**Azure App Service (PaaS)** -- platform for running web apps in the cloud.
**AWS S3 (IaaS)** -- cloud storage service.
**GitHub Codespaces (PaaS)** -- cloud‑based managed development environment.
**Snowflake (SaaS)** -- fully managed data platform.

### Definitions
**IaaS:** -- cloud services that give raw infrastructure (virtual machines, networks, and storage). User rents infrastructure and manages everything.
*Examples:* 
- **Azure Virtual Machines** : provides compute infrastructure where user must choose the OS, configure the machine, manage updates and runtime.
- **AWS S3** : storage infrastructure where user manages buckets and permissions, while AWS handles durability and hardware.

**PaaS:** -- cloud platform for running web apps. User deploys their code and the platform handles servers, OS, scaling, and runtime.  
*Examples:* 
- **Azure App Service** : user pushes their app and Azure runs it.
- **GitHub Codespaces** : GitHub provides ready‑to‑use cloud development environment (the editor, container, and compute) so user can code without setting up a local machine.

**SaaS:** -- complete software delivered over the internet where the provider runs everything.  
*Examples:* 
- **Gmail** : fully functional, end-user app, user just uses accessing via browser; Google handles all servers, updates, and security.
**Snowflake (SaaS)** : fully managed data platform, where the user works only with data and SQL; Snowflake handles compute, storage, scaling, and performance tuning for the user.

---

## Cloud Concepts Question 4
A managed data platform like Databricks or Snowflake provides a fully managed environment for big data, SQL, 
and ML without needing to manage infrastructure or tuning. User gains easier setup, automatic optimization, and collaboration tools. 
User gives up low‑level control and sometimes pay more for the convenience.

---

## Cloud Concepts Question 5
Two situations where the cloud is not the right choice:
1. When regulations or security rules prevent storing data off‑premises.
2. When company already own hardware and run predictable workloads that are cheaper on‑prem.

---

# Azure Basics

## Azure Basics Question 1
Difference: 
An Azure **subscription** is the billing account that owns all the resources in an organization, while a **resource group** is a logical container inside that subscription used to organize and deploy related Azure resources. 
CTD has a single subscription, and it owns (pays for) and manages that container for all students.

Ownership: 
The resource group is an individual workspace that belongs only to the student inside that shared subscription and is used as their personal sandbox for all Azure work, while the Azure subscription is shared at the CTD organization level.

---

## Azure Basics Question 2

Azure Cloud Shell is **ephemeral**, which means temporary. 
In practice, this means the Cloud Shell container is destroyed every time the session closes or times out from inactivity, 
so anything installed or saved outside your home directory disappears.

To make Cloud Shell **persistent**, the course setup automatically attaches a **permanent** Azure Storage Account (a file share) to each student’s Cloud Shell environment, usually mounted as the `clouddrive` folder. 
Anything saved there is stored on Azure’s disks and remains available every time the student logs back in.


---

## Azure Basics Question 3
The **private key** stays on your machine and must never be shared.  
The **public key** is uploaded to remote systems so they can verify your identity.  
It’s safe because the public key cannot be used to derive or guess the private key — only the private key can unlock the connection, so the server can verify you without ever seeing your private key..

---

## Azure Basics Question 4

### Output of `az account show`
*(Note: the real ID values and email are intentionally masked with `xxxx` or a generic placeholder to avoid exposing sensitive information on GitHub. The full, unmasked output is shown in the video submission.)*

{
    "environmentName": "AzureCloud",
    "homeTenantId": "0f040ddd-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "id": "4e07c58c-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "isDefault": true,
    "managedByTenants": [],
    "name": "CTD Student Subscription",
    "state": "Enabled",
    "tenantId": "0f040ddd-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "user": {
        "name": "live.com#xxxxxx@gmail.com",
        "type": "user"
    }
}


### What changes with `--output table`
Using `--output table` reformats the JSON into a clean, easy readable table with column headers instead of raw JSON.

EnvironmentName    HomeTenantId                          IsDefault    Name                       State    TenantId
-----------------  ------------------------------------  -----------  -------------------------  -------  ------------------------------------
AzureCloud         0f040ddd-xxxx-xxxx-xxxx-xxxxxxxxxxxx  True         CTD Nonprofit Sponsorship  Enabled  0f040ddd-xxxx-xxxx-xxxx-xxxxxxxxxxxx
