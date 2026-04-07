from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import httpx
import joblib
import shap

import explain as explain_module
import predict as predict_module

load_dotenv()

MODEL = None
EXPLAINER = None

DEFAULT_MODEL_PATH = "model/lgbm_model.pkl"
DEFAULT_LOAN_TERM = int(os.getenv("DEFAULT_LOAN_TERM", "26"))

PLANNER_API_BASE_URL = os.getenv("PLANNER_API_BASE_URL", "http://localhost:8001").rstrip("/")
PLANNER_EXTERNAL_PLAN_PATH = os.getenv("PLANNER_EXTERNAL_PLAN_PATH", "/api/v1/plan/external")
PLANNER_RAG_QUERY_PATH = os.getenv("PLANNER_RAG_QUERY_PATH", "/api/v1/rag/query")
_PLANNER_TIMEOUT_FALLBACK = os.getenv("PLANNER_TIMEOUT_SECONDS")
PLANNER_PLAN_TIMEOUT_SECONDS = float(os.getenv("PLANNER_PLAN_TIMEOUT_SECONDS", _PLANNER_TIMEOUT_FALLBACK or "75"))
PLANNER_RAG_TIMEOUT_SECONDS = float(os.getenv("PLANNER_RAG_TIMEOUT_SECONDS", _PLANNER_TIMEOUT_FALLBACK or "90"))


def load_model() -> None:
    global MODEL, EXPLAINER

    if MODEL is not None and EXPLAINER is not None:
        return

    model_path_env = os.getenv("MODEL_PATH") or os.getenv("model_path") or DEFAULT_MODEL_PATH
    model_path = (Path(__file__).parent.parent / model_path_env).resolve()
    print("Loading model from:", model_path)

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    MODEL = joblib.load(model_path)
    lgbm_model = MODEL.named_steps["lgbm"]
    EXPLAINER = shap.TreeExplainer(lgbm_model)
    print("Model loaded successfully")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Backend Bridge (Model + Ollama RAG Planner)", version="3.0.0", lifespan=lifespan)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_coapplicant_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 1 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1"}:
            return 1
        if normalized in {"no", "n", "false", "0", ""}:
            return 0
    return 0


def _normalize_credit_grade(value: Any) -> str:
    if value is None:
        return "CC"

    raw = str(value).strip()
    if not raw:
        return "CC"

    grade_map = {
        "excellent": "AA",
        "good": "BB",
        "fair": "CC",
        "poor": "FF",
    }
    return grade_map.get(raw.lower(), raw)


def _normalize_credit_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = payload.get("extra_features", payload)
    if not isinstance(source, dict):
        raise ValueError("Payload must be a JSON object")

    normalized = {
        "Sex": source.get("Sex", "Unknown"),
        "Occupation": source.get("Occupation", "Unknown"),
        "Salary": _to_float(source.get("Salary")),
        "Marriage_Status": source.get("Marriage_Status") or source.get("Marital_status") or "Unknown",
        "credit_score": _to_int(source.get("credit_score"), 0),
        "credit_grade": _normalize_credit_grade(source.get("credit_grade")),
        "outstanding": _to_float(source.get("outstanding")),
        "overdue": _to_float(source.get("overdue")),
        "Coapplicant": _to_coapplicant_int(source.get("Coapplicant")),
        "loan_amount": _to_float(source.get("loan_amount")),
        "loan_term": _to_int(source.get("loan_term"), DEFAULT_LOAN_TERM),
        "Interest_rate": _to_float(source.get("Interest_rate")),
    }

    required_keys = ["Salary", "outstanding", "overdue", "loan_amount", "Interest_rate"]
    missing = [k for k in required_keys if normalized[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    return normalized


def _extract_approve_confidence(prediction_result: Dict[str, Any], shap_result: Dict[str, Any]) -> float:
    probabilities = prediction_result.get("probabilities")
    if isinstance(probabilities, dict):
        approve_raw = probabilities.get("1")
        if approve_raw is None:
            approve_raw = probabilities.get(1)
        if approve_raw is not None:
            try:
                return max(0.0, min(1.0, float(approve_raw)))
            except (TypeError, ValueError):
                pass

    fallback_risk = _to_float(shap_result.get("probability"), 0.5)
    return max(0.0, min(1.0, 1.0 - fallback_risk))


def _build_fallback_explanation(prediction: Any, confidence: float, shap_values: Dict[str, float]) -> str:
    top_factors = sorted(shap_values.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    if not top_factors:
        return (
            f"The model predicts class {prediction} with approval confidence {confidence:.1%}. "
            "RAG planner is unavailable, so this is a local fallback explanation."
        )

    parts = []
    for name, value in top_factors:
        trend = "helps" if value > 0 else "hurts"
        parts.append(f"{name} ({value:+.3f}, {trend} approval)")

    return (
        f"The model predicts class {prediction} with approval confidence {confidence:.1%}. "
        f"Top SHAP drivers: {', '.join(parts)}."
    )


def _build_external_plan_payload(
    data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    shap_result: Dict[str, Any],
) -> Dict[str, Any]:
    probabilities = prediction_result.get("probabilities", {}) or {}
    p0 = _to_float(probabilities.get("0", probabilities.get(0)), 0.0)
    p1 = _to_float(probabilities.get("1", probabilities.get(1)), 0.0)
    if p0 == 0.0 and p1 == 0.0:
        pred = _to_int(prediction_result.get("prediction"), 0)
        p1 = 1.0 if pred == 1 else 0.0
        p0 = 1.0 - p1

    shap_values = shap_result.get("shap_values", {}) or {}
    shap_payload = {str(k): float(v) for k, v in shap_values.items()}

    return {
        "request_id": f"bridge-{uuid4().hex[:12]}",
        "user_input": {
            "Salary": float(data["Salary"]),
            "Occupation": data["Occupation"] or "Unknown",
            "Marriage_Status": data["Marriage_Status"] or "Unknown",
            "credit_score": float(data["credit_score"]),
            "credit_grade": data["credit_grade"] or "CC",
            "outstanding": float(data["outstanding"]),
            "overdue": float(data["overdue"]),
            "Coapplicant": bool(data["Coapplicant"]),
            "loan_amount": float(data["loan_amount"]),
            "loan_term": float(data["loan_term"]),
            "Interest_rate": float(data["Interest_rate"]),
        },
        "model_output": {
            "prediction": int(_to_int(prediction_result.get("prediction"), 0)),
            "probabilities": {
                "0": round(float(p0), 6),
                "1": round(float(p1), 6),
            },
        },
        "shap_json": {
            "base_value": _to_float(shap_result.get("base_value"), 0.5),
            "values": shap_payload,
        },
    }


async def _request_ollama_planner(external_plan_payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    planner_url = f"{PLANNER_API_BASE_URL}{PLANNER_EXTERNAL_PLAN_PATH}"

    try:
        timeout = httpx.Timeout(PLANNER_PLAN_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(planner_url, json=external_plan_payload)
    except httpx.TimeoutException:
        return None, "Planner timed out"
    except httpx.RequestError as exc:
        return None, f"Planner unreachable: {exc}"

    if response.status_code != 200:
        return None, f"Planner returned {response.status_code}"

    try:
        payload = response.json()
    except ValueError:
        return None, "Planner returned invalid JSON"

    if not isinstance(payload, dict):
        return None, "Planner response has invalid format"

    return payload, None


async def _request_planner_rag(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    rag_url = f"{PLANNER_API_BASE_URL}{PLANNER_RAG_QUERY_PATH}"

    try:
        timeout = httpx.Timeout(PLANNER_RAG_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(rag_url, json=payload)
    except httpx.TimeoutException:
        return None, "Planner RAG query timed out"
    except httpx.RequestError as exc:
        return None, f"Planner RAG unreachable: {exc}"

    if response.status_code != 200:
        return None, f"Planner RAG returned {response.status_code}"

    try:
        data = response.json()
    except ValueError:
        return None, "Planner RAG returned invalid JSON"

    if not isinstance(data, dict):
        return None, "Planner RAG response has invalid format"

    return data, None


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Backend bridge is running. Use /docs."}


@app.get("/health")
async def health() -> Dict[str, Any]:
    planner_health_url = f"{PLANNER_API_BASE_URL}/health"
    planner_status = "unknown"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(planner_health_url)
            planner_status = "ok" if response.status_code == 200 else f"error:{response.status_code}"
    except Exception:
        planner_status = "unreachable"

    return {
        "status": "healthy",
        "service": "backend-bridge",
        "planner_api": {
            "base_url": PLANNER_API_BASE_URL,
            "status": planner_status,
            "plan_timeout_seconds": PLANNER_PLAN_TIMEOUT_SECONDS,
            "rag_timeout_seconds": PLANNER_RAG_TIMEOUT_SECONDS,
        },
    }


@app.post("/predict")
async def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = _normalize_credit_payload(payload)
        prediction_result = predict_module.run_prediction(data, MODEL)
        shap_result = explain_module.compute_shap(data, MODEL, EXPLAINER)

        prediction = _to_int(prediction_result.get("prediction"), 0)
        confidence = _extract_approve_confidence(prediction_result, shap_result)
        shap_values = shap_result.get("shap_values", {}) or {}
        model_explanation = _build_fallback_explanation(prediction, confidence, shap_values)

        external_plan_payload = _build_external_plan_payload(data, prediction_result, shap_result)
        planner_result, planner_error = await _request_ollama_planner(external_plan_payload)

        planner_explanation: Optional[str] = None
        if planner_result:
            planner_explanation = str(planner_result.get("result_th", "")).strip() or None
            rag_sources = planner_result.get("rag_sources", [])
        else:
            rag_sources = []

        return {
            "prediction": prediction,
            "confidence": confidence,
            "shap_values": shap_values,
            # Keep "explanation" for backward compatibility, but force it to be model-based.
            "explanation": model_explanation,
            "model_explanation": model_explanation,
            "planner_explanation": planner_explanation,
            "probabilities": prediction_result.get("probabilities", {}),
            "explanation_details": shap_result,
            "planner": planner_result,
            "planner_error": planner_error,
            "rag_sources": rag_sources,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/rag/query")
async def rag_query(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=422, detail="question is required")

    request_payload: Dict[str, Any] = {"question": question.strip()}
    top_k = payload.get("top_k")
    if isinstance(top_k, int) and top_k > 0:
        request_payload["top_k"] = top_k

    result, err = await _request_planner_rag(request_payload)
    if err:
        raise HTTPException(status_code=503, detail=err)

    return result or {"question": question.strip(), "answer": "", "sources": []}
