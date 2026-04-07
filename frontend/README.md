# Credit Scoring Frontend

Next.js frontend for the integrated credit-scoring workflow.

## Runtime Flow

1. User submits form on `/`
2. Frontend calls `POST /api/predict`
3. Next.js route proxies to backend bridge (`/predict`)
4. Backend returns model score, SHAP factors, planner text, and optional RAG sources
5. Assistant panel calls `POST /api/rag/query` for follow-up questions

## Required Environment

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
BACKEND_URL=http://localhost:8000
BACKEND_TIMEOUT_MS=120000
```

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## API Proxy Routes

- `GET /api/health`
- `POST /api/predict`
- `POST /api/rag/query`

## Notes

- This frontend does not use local mock scoring.
- All score/advice outputs are sourced from backend services.
