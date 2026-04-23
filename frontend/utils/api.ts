/**
 * API client for backend communication
 */

const API_BASE_URL = '/api';

export interface PredictRequest {
  input_text: string;
  extra_features?: Record<string, any>;
}

export interface PredictResponse {
  prediction: any;
  confidence: number;
  shap_values: Record<string, number>;
  explanation: string;
  model_explanation?: string;
  planner_explanation?: string | null;
  planner?: {
    mode?: string;
    approved?: boolean;
    p_approve?: number;
    p_reject?: number;
    result_th?: string;
    rag_sources?: Array<Record<string, any>>;
    issup_score?: number;
    issup_passed?: boolean;
  } | null;
  planner_error?: string | null;
  rag_sources?: Array<Record<string, any>>;
  distribution_warnings?: string[];
}

export interface RagQueryResponse {
  question: string;
  answer: string;
  router_label?: string;
  sources?: Array<{
    title?: string;
    category?: string;
    institution?: string;
    score?: number;
  }>;
}

// === Profile-conditioned Advisor (Approach 1) =================================

export interface AdvisorProfile {
  salary_per_month?: number;
  occupation?: string;
  employment_tenure_months?: number;
  marriage_status?: string;
  has_coapplicant?: boolean;
  coapplicant_income?: number;
  credit_score?: number;
  credit_grade?: string;
  outstanding_debt?: number;
  overdue_days_max?: number;
  loan_amount_requested?: number;
  loan_term_years?: number;
  interest_rate?: number;
}

export interface AdvisorRequirementCheck {
  requirement: string;
  user_value: string;
  status: 'pass' | 'fail' | 'unknown' | 'not_applicable';
  explanation: string;
}

export interface AdvisorResponse {
  question: string;
  verdict: 'eligible' | 'partially_eligible' | 'ineligible' | 'needs_more_info';
  verdict_summary: string;
  requirement_checks: AdvisorRequirementCheck[];
  recommended_actions: string[];
  sources: Array<{
    title?: string;
    category?: string;
    institution?: string;
    score?: number;
  }>;
  raw_answer?: string;
}

export interface ApiError {
  detail?: string;
  error?: string;
  message?: string;
}

async function getErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const error = (await response.json()) as ApiError;
    return error.detail || error.error || error.message || fallback;
  } catch {
    return fallback;
  }
}

/**
 * Call ML prediction endpoint
 */
export async function predictCredit(
  inputText: string,
  extraFeatures?: Record<string, any>
): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input_text: inputText,
      extra_features: extraFeatures,
    }),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Failed to get prediction'));
  }

  return response.json();
}

/**
 * Check backend health
 */
export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Backend is not healthy'));
  }

  return response.json();
}

export async function queryAssistantRag(question: string, topK?: number): Promise<RagQueryResponse> {
  const response = await fetch(`${API_BASE_URL}/rag/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Failed to query assistant'));
  }

  return response.json();
}

/**
 * Profile-conditioned advisor — runs structured eligibility reasoning over
 * the RAG index instead of paraphrasing chunks (Approach 1).
 */
export async function queryAdvisor(
  question: string,
  profile: AdvisorProfile,
  topK?: number
): Promise<AdvisorResponse> {
  const response = await fetch(`${API_BASE_URL}/rag/advisor`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, profile, top_k: topK }),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Failed to query advisor'));
  }

  return response.json();
}
