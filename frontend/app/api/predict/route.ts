

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const DEV_BACKEND_URL = 'http://localhost:8000';
const BACKEND_TIMEOUT_MS = Number(process.env.BACKEND_TIMEOUT_MS || '120000');

function resolveBackendUrl(): string | null {
  const candidates = [
    process.env.BACKEND_URL,
    process.env.NEXT_PUBLIC_API_URL,
    process.env.API_BASE_URL,
  ].filter(Boolean) as string[];

  if (candidates.length > 0) {
    return candidates[0].replace(/\/$/, '');
  }

  if (process.env.NODE_ENV === 'development') {
    return DEV_BACKEND_URL;
  }

  return null;
}

async function tryParseJson(response: Response): Promise<Record<string, unknown> | null> {
  try {
    return (await response.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export async function POST(request: NextRequest) {
  const backendUrl = resolveBackendUrl();
  if (!backendUrl) {
    const detail = 'BACKEND_URL is not configured on the server';
    return NextResponse.json({ detail, error: detail }, { status: 500 });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    const detail = 'Invalid JSON request body';
    return NextResponse.json({ detail, error: detail }, { status: 400 });
  }

  try {
    // Forward request to FastAPI backend
    const response = await fetch(`${backendUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      cache: 'no-store',
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });

    const payload = await tryParseJson(response);

    if (!response.ok) {
      const detail =
        (payload?.detail as string | undefined) ||
        (payload?.error as string | undefined) ||
        'Prediction failed';

      return NextResponse.json(
        { detail, error: detail },
        { status: response.status }
      );
    }

    if (!payload) {
      const detail = 'Backend returned an invalid response payload';
      return NextResponse.json({ detail, error: detail }, { status: 502 });
    }

    return NextResponse.json(payload);
  } catch (error) {
    console.error('Prediction API error:', error);

    const isTimeout =
      error instanceof Error &&
      (error.name === 'TimeoutError' || error.name === 'AbortError');
    const status = isTimeout ? 504 : 500;
    const detail = isTimeout ? 'Backend request timed out' : 'Internal server error';

    return NextResponse.json(
      { detail, error: detail },
      { status }
    );
  }
}
