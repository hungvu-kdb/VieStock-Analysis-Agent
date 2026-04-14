## Setup

### 1. Self-host Langfuse via Docker Compose

Follow this [link](https://langfuse.com/self-hosting/deployment/docker-compose): https://langfuse.com/self-hosting/deployment/docker-compose

### 2. Create Langfuse API Key

- Log in to Langfuse: Access your Langfuse account on the official website.
- Navigate to Project: If you haven't already, create a new project or select an existing one.
- Access Project Settings: Within your chosen project, locate and click on "Settings" in the left-hand sidebar.
- Create API Keys: In the Settings section, find the "API Keys" or "Create API Keys" option and click on it.
- Copy and Save Credentials: Langfuse will generate a set of API credentials, typically including a Secret Key, Public Key, and the Host URL. Copy these values and store them securely, as the Secret Key is often only viewable once.

Reference [link](https://langfuse.com/faq/all/where-are-langfuse-api-keys): https://langfuse.com/faq/all/where-are-langfuse-api-keys

### 3. Setup Python Virtual Environment

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Setup Credentials
```sh
export LANGFUSE_SECRET_KEY=""
export LANGFUSE_PUBLIC_KEY=""
export LANGFUSE_HOST=""

export AWS_PROFILE=''
export AWS_DEFAULT_REGION='us-east-1'
export AWS_REGION='us-east-1'
```

### 5. Prepare Knowledge Bases and Redshift configurations

From `main.py`, enter your configurations:
```json
        "kb_agent": {
            "market_information": {
                "id": "<enter KB ID here>",
                "name": "market_information",
                ...
            },
            "company_information": {
                "id": "<enter KB ID here>",
                "name": "company_information",
                ...
            }
        },
        "sql_agent": {
            "technical_analysis": {
                "database_name": "<enter database name>",
                "workgroup_name": "<enter workgroup name>",
                "secret_arn": "<enter secret ARN>",
                "target_schema": "<enter target schema>",
                "db_region": "us-east-1",
                ...
            },
            "financial_report": {
                "database_name": "<enter database name>",
                "workgroup_name": "<enter workgroup name>",
                "secret_arn": "<enter secret ARN>",
                "target_schema": "<enter target schema>",
                "db_region": "us-east-1",
                ...
            }
```

### 6. Enter user query

From `main.py`, enter the user's query:
```python
    query = "Your query here"
```

### 7. Execute Agentic Bot
```sh
python -m app.main
```
