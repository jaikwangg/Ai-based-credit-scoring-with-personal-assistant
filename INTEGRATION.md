# Production Integration Runbook

This repository runs as one connected production path:

1. `frontend` (`Next.js`) calls `POST /api/predict` and `POST /api/rag/query`
2. `frontend` proxy forwards to `backend` (`FastAPI bridge`)
3. `backend` runs local ML + SHAP and calls `Ai-Credit-Scoring` planner/RAG
4. `Ai-Credit-Scoring` returns Thai guidance from model + RAG pipeline

## One Command Start (Recommended)

From repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_up.ps1 -Clean
```

Optional: startup + smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_up.ps1 -Clean -WithSmoke
```

Stop all services:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_down.ps1
```

## Service Ports

- `frontend`: `3000`
- `backend` bridge: `8000`
- `Ai-Credit-Scoring`: `8001`
- `Ollama`: `11434`

## 1) Configure Environment

### `Ai-Credit-Scoring/.env`

```env
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
EMBED_MODEL=BAAI/bge-m3
CHROMA_PERSIST_DIR=./storage/chroma
CHROMA_COLLECTION=cimb_loans_bge_m3
SIMILARITY_CUTOFF=0.45

# Production speed controls
PLANNER_ENABLE_LLM_SYNTHESIS=false
PLANNER_MAX_ACTION_DRIVERS=3
PLANNER_MAX_RAG_QUERIES_PER_DRIVER=1
PLANNER_APPROVED_MAX_RAG_QUERIES=2
```

### `backend/.env`

```env
MODEL_PATH=model\lgbm_model.pkl
DEFAULT_LOAN_TERM=26
PLANNER_API_BASE_URL=http://127.0.0.1:8001
PLANNER_EXTERNAL_PLAN_PATH=/api/v1/plan/external
PLANNER_RAG_QUERY_PATH=/api/v1/rag/query
PLANNER_PLAN_TIMEOUT_SECONDS=75
PLANNER_RAG_TIMEOUT_SECONDS=90
```

### `frontend/.env.local`

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
BACKEND_URL=http://localhost:8000
BACKEND_TIMEOUT_MS=120000
```

## 2) Start Services (Order Matters)

### Start planner/RAG API

```powershell
cd Ai-Credit-Scoring
python -m uvicorn src.api.main:app --reload --port 8001
```

### Start backend bridge

```powershell
cd backend
fastapi run app/main.py --host 0.0.0.0 --port 8000
```

### Start frontend

```powershell
cd frontend
npm run dev
```

## 3) Verify End-to-End Workflow

Run smoke test from repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prod_smoke.ps1
```

The script checks:

- health of all 3 services
- backend required routes (`/predict`, `/rag/query`)
- predict flow via backend and frontend proxy
- RAG query flow via frontend proxy

## 4) Manual Health Endpoints

- Planner: `GET http://localhost:8001/health`
- Backend bridge: `GET http://localhost:8000/health`
- Frontend proxy health: `GET http://localhost:3000/api/health`

## 5) If Frontend Returns 500/webpack Runtime Errors

Clear stale Next build cache and restart frontend:

```powershell
cd frontend
Remove-Item -Recurse -Force .next
npm run dev
```

## 6) If `/api/rag/query` returns 404

Backend is running an old process. Restart backend from `backend/` with:

```powershell
fastapi run app/main.py --host 0.0.0.0 --port 8000
```

Then verify `http://localhost:8000/openapi.json` includes `/rag/query`.
