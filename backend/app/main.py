from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import joblib
import shap

# Patch asyncio so sync wrappers (e.g. llama_index QueryEngine.query) can call
# the async GoogleGenAI / Gemini SDK from inside FastAPI's running event loop.
import nest_asyncio
nest_asyncio.apply()

import explain as explain_module
import predict as predict_module

# Local RAG imports (merged from Ai-Credit-Scoring)
from app.planner.planning import generate_response
from app.planner.rag_bridge import (
    build_shap_json,
    extract_rag_sources,
    get_rag_manager,
    make_rag_lookup,
)
from app.rag.advisor import run_advisor
from app.rag.cache import get_cache
from app.rag.self_rag import SelfRAGOrchestrator
from app.schemas.payload import AdvisorProfile
from app.planner.scoring import compute_plan_inputs
from app.routes import scoring as scoring_routes, rag as rag_routes

# DB setup (merged from Ai-Credit-Scoring)
from app.db.database import engine
from app.db import models as db_models

load_dotenv()

MODEL = None
EXPLAINER = None

DEFAULT_MODEL_PATH = "model/lgbm_model.pkl"
DEFAULT_LOAN_TERM = int(os.getenv("DEFAULT_LOAN_TERM", "26"))


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
    # Create DB tables for the scoring service
    db_models.Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Credit Scoring Backend (Model + RAG Planner)",
    version="4.0.0",
    lifespan=lifespan,
)

# Include RAG/planner routers from merged Ai-Credit-Scoring
app.include_router(scoring_routes.router, prefix="/api/v1", tags=["Decisioning"])
app.include_router(rag_routes.router, prefix="/api/v1", tags=["RAG"])


# ── Helper functions ────────────────────────────────────────────────────────

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


_VALID_GRADES = {"AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH"}
_LEGACY_GRADE_MAP = {
    "excellent": "AA",
    "good": "BB",
    "fair": "CC",
    "poor": "FF",
}


def _normalize_credit_grade(value: Any) -> str:
    if value is None:
        return "CC"
    raw = str(value).strip()
    if not raw:
        return "CC"
    upper = raw.upper()
    if upper in _VALID_GRADES:
        return upper
    return _LEGACY_GRADE_MAP.get(raw.lower(), "CC")


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

    warnings: list[str] = []
    salary = normalized["Salary"] or 0.0
    if salary > 0 and (salary < 20000 or salary > 270000):
        warnings.append(f"Salary {salary:,.0f} อยู่นอกช่วง 20,000-270,000 ที่โมเดลเรียนรู้")
    loan = normalized["loan_amount"] or 0.0
    if loan > 0 and (loan < 500000 or loan > 7400000):
        warnings.append(f"loan_amount {loan:,.0f} อยู่นอกช่วง 500k-7.4M ที่โมเดลเรียนรู้")
    term = normalized["loan_term"] or 0
    if term and (term < 20 or term > 30):
        warnings.append(f"loan_term {term} อยู่นอกช่วง 20-30 ปีที่โมเดลเรียนรู้")
    rate = normalized["Interest_rate"] or 0.0
    if rate > 0 and (rate < 5.5 or rate > 6.0):
        warnings.append(f"Interest_rate {rate}% อยู่นอกช่วง 5.5-6.0% ที่โมเดลเรียนรู้")
    grade = normalized["credit_grade"]
    if grade and grade not in {"AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH"}:
        warnings.append(f"credit_grade {grade!r} ไม่ใช่หนึ่งใน 8 grades ของชุดข้อมูล")
    occ = normalized["Occupation"]
    valid_occs = {
        "Salaried_Employee",
        "Government_or_State_Enterprise",
        "SME_Owner",
        "Professional_Specialist",
        "Freelancer_or_Self_Employed",
    }
    if occ and occ not in valid_occs:
        warnings.append(f"Occupation {occ!r} ไม่ใช่หนึ่งใน 5 หมวดที่โมเดลเรียนรู้")
    overdue = normalized["overdue"]
    if overdue not in (None, "") and int(overdue) not in {0, 15, 60, 120}:
        warnings.append(f"overdue {overdue} ไม่ตรงกับ training bucket {{0,15,60,120}}")

    normalized["_distribution_warnings"] = warnings
    return normalized


def _extract_approve_confidence(prediction_result: Dict[str, Any], shap_result: Dict[str, Any]) -> float:
    probabilities = prediction_result.get("probabilities")
    if isinstance(probabilities, dict):
        paid_raw = probabilities.get("0")
        if paid_raw is None:
            paid_raw = probabilities.get(0)
        if paid_raw is not None:
            try:
                return max(0.0, min(1.0, float(paid_raw)))
            except (TypeError, ValueError):
                pass

        default_raw = probabilities.get("1")
        if default_raw is None:
            default_raw = probabilities.get(1)
        if default_raw is not None:
            try:
                return max(0.0, min(1.0, 1.0 - float(default_raw)))
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
    """Build payload for planner generate_response (local call, no HTTP)."""
    probabilities = prediction_result.get("probabilities", {}) or {}
    p_paid = _to_float(probabilities.get("0", probabilities.get(0)), 0.0) or 0.0
    p_default = _to_float(probabilities.get("1", probabilities.get(1)), 0.0) or 0.0
    if p_paid == 0.0 and p_default == 0.0:
        raw_pred = _to_int(prediction_result.get("prediction"), 0)
        p_default = 1.0 if raw_pred == 1 else 0.0
        p_paid = 1.0 - p_default

    p_approve = p_paid
    p_reject = p_default

    raw_prediction = _to_int(prediction_result.get("prediction"), 0)
    inverted_prediction = 1 - raw_prediction if raw_prediction in (0, 1) else 0

    shap_values = shap_result.get("shap_values", {}) or {}
    shap_payload = {str(k): -float(v) for k, v in shap_values.items()}

    return {
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
            "prediction": inverted_prediction,
            "probabilities": {
                "0": round(float(p_reject), 6),
                "1": round(float(p_approve), 6),
            },
        },
        "shap_json": {
            "base_value": _to_float(shap_result.get("base_value"), 0.5),
            "values": shap_payload,
        },
    }


def _call_local_planner(plan_payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call planner locally instead of via HTTP."""
    try:
        manager = get_rag_manager()
        rag_lookup = make_rag_lookup(manager.query) if manager else None

        plan_result = generate_response(
            user_input=plan_payload["user_input"],
            model_output=plan_payload["model_output"],
            shap_json=plan_payload["shap_json"],
            rag_lookup=rag_lookup,
        )
        return plan_result, None
    except Exception as exc:
        return None, f"Planner failed: {exc}"


def _call_local_rag(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call RAG query locally instead of via HTTP."""
    try:
        manager = get_rag_manager()
        if manager is None:
            return None, "RAG index unavailable"

        cache = get_cache()
        question = payload["question"]
        top_k = payload.get("top_k")

        cached = cache.get(question, top_k=top_k)
        if cached is not None:
            result = cached
        else:
            kwargs: Dict[str, Any] = {
                "question": question,
                "include_sources": True,
            }
            if isinstance(top_k, int) and top_k > 0:
                kwargs["similarity_top_k"] = top_k
            result = manager.query(**kwargs)
            if result.get("answer"):
                cache.set(question, result, top_k=top_k)

        return result, None
    except Exception as exc:
        return None, f"RAG query failed: {exc}"


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "AI Credit Scoring Backend is running. Use /docs."}


@app.get("/health")
async def health() -> Dict[str, Any]:
    rag_manager = get_rag_manager()
    return {
        "status": "healthy",
        "service": "backend-unified",
        "rag_available": rag_manager is not None,
    }


@app.post("/predict")
async def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = _normalize_credit_payload(payload)
        distribution_warnings: list[str] = data.pop("_distribution_warnings", []) or []
        prediction_result = predict_module.run_prediction(data, MODEL)
        shap_result = explain_module.compute_shap(data, MODEL, EXPLAINER)

        # ── Label inversion ──────────────────────────────────────────────
        raw_prediction = _to_int(prediction_result.get("prediction"), 0)
        prediction = 1 - raw_prediction if raw_prediction in (0, 1) else 0
        confidence = _extract_approve_confidence(prediction_result, shap_result)

        raw_shap = shap_result.get("shap_values", {}) or {}
        shap_values = {k: -float(v) for k, v in raw_shap.items()}

        raw_probs = prediction_result.get("probabilities", {}) or {}
        p_paid = _to_float(raw_probs.get("0", raw_probs.get(0)), 0.0) or 0.0
        p_default = _to_float(raw_probs.get("1", raw_probs.get(1)), 0.0) or 0.0
        approval_probabilities = {
            "0": round(p_default, 6),
            "1": round(p_paid, 6),
        }

        model_explanation = _build_fallback_explanation(prediction, confidence, shap_values)

        # ── Call planner LOCALLY instead of via HTTP ─────────────────────
        plan_payload = _build_external_plan_payload(data, prediction_result, shap_result)
        planner_result, planner_error = _call_local_planner(plan_payload)

        planner_explanation: Optional[str] = None
        if planner_result:
            planner_explanation = str(planner_result.get("result_th", "")).strip() or None
            rag_sources = planner_result.get("rag_sources", [])
            if not rag_sources:
                rag_sources = extract_rag_sources(planner_result)
        else:
            rag_sources = []

        return {
            "prediction": prediction,
            "confidence": confidence,
            "shap_values": shap_values,
            "explanation": model_explanation,
            "model_explanation": model_explanation,
            "planner_explanation": planner_explanation,
            "probabilities": approval_probabilities,
            "explanation_details": shap_result,
            "planner": planner_result,
            "planner_error": planner_error,
            "rag_sources": rag_sources,
            "distribution_warnings": distribution_warnings,
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

    result, err = _call_local_rag(request_payload)
    if err:
        raise HTTPException(status_code=503, detail=err)

    return result or {"question": question.strip(), "answer": "", "sources": []}


@app.post("/rag/advisor")
async def rag_advisor(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Profile-conditioned advisory — calls RAG advisor locally."""
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=422, detail="question is required")

    profile = payload.get("profile") or {}
    if not isinstance(profile, dict):
        raise HTTPException(status_code=422, detail="profile must be an object")

    try:
        advisor_profile = AdvisorProfile(**profile)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid profile: {exc}")

    manager = get_rag_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="RAG index unavailable")

    try:
        result = run_advisor(
            question=question.strip(),
            profile=advisor_profile,
            rag_manager=manager,
            top_k=payload.get("top_k") or 6,
            use_multihop=bool(payload.get("use_multihop")),
            use_self_rag=bool(payload.get("use_self_rag")),
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Advisor failed: {exc}")
