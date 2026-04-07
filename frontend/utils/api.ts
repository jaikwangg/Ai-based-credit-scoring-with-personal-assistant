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
