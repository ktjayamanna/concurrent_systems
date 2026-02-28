# OPACA LLM UI WITH SPECULATIVE ENGINE
##  Get Started

- Clone the [OPACA-Core Repository](https://github.com/GT-ARC/opaca-core) to the root folder and follow the **Getting Started** guide to build and launch an OPACA runtime platform. ```Commit SHA: 3f5fc9811022ad8a7db6b23e5eb1d450034b461c```
- Clone the [OPACA-LLM-Benchmark-Containers Repository](https://github.com/RobertStrehlow/opaca-llm-benchmark-containers) to the root folder and follow the **Deployment** guide to deploy the benchmark containers to the OPACA runtime platform. ```Commit SHA: 3579d358699f01992c2e3fe6245e564939874a86```


## VSCode Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js & npm

### Quick Start
1. Install dependencies (first time only):
   ```bash
   cd src/Backend && pip3 install -r requirements.txt
   cd ../Frontend && npm install
   ```

2. Press `F5` or select **"run-sage"** from the Run and Debug panel

This automatically:
- Starts Docker services (DB + backend container)
- Installs dependencies if needed
- Runs backend server (port 3001)
- Runs frontend dev server (port 5173)

### Manual Start
```bash
cd src                               # Terminal 1
docker compose up --build -d         # Terminal 1
cd Backend && python3 -m src.server  # Terminal 2
cd Frontend && npm run dev           # Terminal 3
```