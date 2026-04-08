'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { CreditInput, ChecklistItem } from '@/types/credit';
import { predictCredit, PredictResponse, AdvisorProfile } from '@/utils/api';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import CreditForm from '@/components/CreditForm';
import ResultCard from '@/components/ResultCard';
import AssistantPanel from '@/components/AssistantPanel';
import AssistantChat from '@/components/AssistantChat';

// Validation schema
const creditSchema = z.object({
  Sex: z.string().min(1, 'กรุณาเลือกเพศ'),
  Occupation: z.string().min(1, 'กรุณาเลือกอาชีพ'),
  Salary: z.string().min(1, 'กรุณาระบุรายได้').regex(/^\d+(\.\d{1,2})?$/, 'กรุณากรอกตัวเลขให้ถูกต้อง'),
  Marital_status: z.string().min(1, 'กรุณาเลือกสถานภาพสมรส'),
  credit_score: z.string().regex(/^\d*$/, 'กรุณากรอกคะแนนที่ถูกต้อง').optional().or(z.literal('')),
  credit_grade: z.string().optional(),
  outstanding: z.string().min(1, 'กรุณาระบุหนี้คงค้าง').regex(/^\d+(\.\d{1,2})?$/, 'กรุณากรอกตัวเลขให้ถูกต้อง'),
  overdue: z.string().min(1, 'กรุณาระบุยอดค้างชำระ').regex(/^\d+(\.\d{1,2})?$/, 'กรุณากรอกตัวเลขให้ถูกต้อง'),
  loan_amount: z.string().min(1, 'กรุณาระบุวงเงินขอกู้').regex(/^\d+(\.\d{1,2})?$/, 'กรุณากรอกตัวเลขให้ถูกต้อง'),
  Coapplicant: z.string().min(1, 'กรุณาเลือกสถานะผู้กู้ร่วม'),
  Interest_rate: z.string().min(1, 'กรุณาระบุอัตราดอกเบี้ย').regex(/^\d+(\.\d{1,2})?$/, 'กรุณากรอกเปอร์เซ็นต์ให้ถูกต้อง'),
});

type View = 'form' | 'result' | 'assistant';

/**
 * Extract action items from planner Thai text.
 * The Ai-Credit-Scoring planner uses varying bullet styles: 1), 1., -, *, •,
 * Thai numerals ๑., or section heads like "มาตรการ:". Accept all of them.
 */
/**
 * Extract action titles from the planner's Gemini-generated Thai report.
 *
 * Gemini wraps each action with a consistent header like:
 *   **มาตรการที่ 1: ฟื้นฟูวินัยเครดิตอย่างต่อเนื่อง**
 *   **มาตรการที่ 2: ทบทวนความพร้อมทางการเงินก่อนยื่นใหม่**
 *
 * We pull only those titles. Everything else in the report (profile data,
 * SHAP values, probabilities, paragraphs under each measure) is NOT an action.
 *
 * Returns an empty array if no `มาตรการที่` headers are found — the caller
 * should then hide the checklist entirely rather than show garbage bullets.
 */
function extractActionsFromText(text?: string): string[] {
  if (!text) return [];

  const measurePattern = /\*\*\s*มาตรการที่\s*\d+\s*[:：]\s*([^*]+?)\s*\*\*/g;
  const titles: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = measurePattern.exec(text)) !== null) {
    const title = match[1]
      .trim()
      .replace(/\s{2,}/g, ' ')
      .replace(/[*]+$/, '');
    if (title) titles.push(title);
  }

  return Array.from(new Set(titles)).slice(0, 6);
}

function buildChecklist(actions: string[]): ChecklistItem[] {
  return actions.map((task, index) => ({
    id: String(index + 1),
    task,
    completed: false,
  }));
}

/**
 * Convert raw form input into the advisor profile schema (numeric fields).
 * Empty strings become undefined so the LLM treats them as "not specified".
 */
function buildAdvisorProfile(data: CreditInput): AdvisorProfile {
  const num = (s: string): number | undefined => {
    if (!s) return undefined;
    const n = parseFloat(s);
    return Number.isFinite(n) ? n : undefined;
  };
  const int = (s: string): number | undefined => {
    if (!s) return undefined;
    const n = parseInt(s, 10);
    return Number.isFinite(n) ? n : undefined;
  };
  return {
    salary_per_month: num(data.Salary),
    occupation: data.Occupation || undefined,
    marriage_status: data.Marital_status || undefined,
    has_coapplicant: data.Coapplicant ? data.Coapplicant === 'Yes' : undefined,
    credit_score: int(data.credit_score),
    credit_grade: data.credit_grade || undefined,
    outstanding_debt: num(data.outstanding),
    overdue_amount: num(data.overdue),
    loan_amount_requested: num(data.loan_amount),
    interest_rate: num(data.Interest_rate),
  };
}

function buildProfileSummary(data: CreditInput, prediction: PredictResponse): string {
  const decisionTh = prediction.prediction === 1 ? 'น่าจะอนุมัติ' : 'น่าจะถูกปฏิเสธ';
  return [
    `อาชีพ ${data.Occupation || '-'}`,
    `รายได้ ${data.Salary || '-'} บาท/เดือน`,
    `หนี้คงค้าง ${data.outstanding || '-'}`,
    `ค้างชำระ ${data.overdue || '-'}`,
    `ขอกู้ ${data.loan_amount || '-'} ดอกเบี้ย ${data.Interest_rate || '-'}%`,
    `เครดิต ${data.credit_score || '-'} เกรด ${data.credit_grade || '-'}`,
    `ผล ${decisionTh} (P=${(prediction.confidence * 100).toFixed(1)}%)`,
  ].join(', ');
}

export default function Home() {
  const [view, setView] = useState<View>('form');
  const [isCalculating, setIsCalculating] = useState(false);
  const [creditInput, setCreditInput] = useState<CreditInput | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [modelExplanation, setModelExplanation] = useState<string>('');
  const [plannerExplanation, setPlannerExplanation] = useState<string>('');
  const [ragSources, setRagSources] = useState<Array<Record<string, any>>>([]);
  const [plannerError, setPlannerError] = useState<string | null>(null);
  const [checklist, setChecklist] = useState<ChecklistItem[]>([]);
  const [userProfileSummary, setUserProfileSummary] = useState<string>('');
  const [advisorProfile, setAdvisorProfile] = useState<AdvisorProfile | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<CreditInput>({
    resolver: zodResolver(creditSchema),
    mode: 'onChange',
    defaultValues: {
      Sex: '',
      Occupation: '',
      Salary: '',
      Marital_status: '',
      credit_score: '',
      credit_grade: '',
      outstanding: '',
      overdue: '',
      loan_amount: '',
      Coapplicant: '',
      Interest_rate: '',
    },
  });

  const onSubmit = async (data: CreditInput) => {
    setIsCalculating(true);
    setView('result');

    try {
      const inputText = JSON.stringify(data);
      const prediction = await predictCredit(inputText, data as unknown as Record<string, any>);

      const modelText = prediction.model_explanation || prediction.explanation || '';
      // RAG + planner Thai output is the authoritative guidance text.
      const plannerText = (
        prediction.planner?.result_th ||
        prediction.planner_explanation ||
        ''
      ).trim();
      const sources = prediction.planner?.rag_sources || prediction.rag_sources || [];

      const actions = extractActionsFromText(plannerText);

      setCreditInput(data);
      setPredictResult(prediction);
      setModelExplanation(modelText);
      setPlannerExplanation(plannerText);
      setRagSources(sources);
      setPlannerError(prediction.planner_error || null);
      setChecklist(buildChecklist(actions));
      setUserProfileSummary(buildProfileSummary(data, prediction));
      setAdvisorProfile(buildAdvisorProfile(data));
    } catch (error) {
      console.error('Error calculating credit score:', error);
      const detail =
        error instanceof Error && error.message
          ? error.message
          : 'ไม่สามารถประเมินได้ กรุณาตรวจสอบว่า backend ทำงานอยู่แล้วลองใหม่อีกครั้ง';
      alert(detail);
      setView('form');
    } finally {
      setIsCalculating(false);
    }
  };

  const handleTalkToAssistant = () => setView('assistant');
  const handleBackToResult = () => setView('result');
  const handleBackToForm = () => setView('form');

  const handleToggleChecklist = (id: string) => {
    setChecklist((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, completed: !item.completed } : item
      )
    );
  };

  const approved = predictResult?.prediction === 1;
  const confidence = predictResult?.confidence ?? 0;

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-50 to-gray-100">
      <Header />

      <main className="flex-1 py-6 md:py-8">
        <div className="max-w-7xl mx-auto px-4">
          {view === 'form' && (
            <div className="max-w-4xl mx-auto">
              <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
                <form onSubmit={handleSubmit(onSubmit)} noValidate>
                  <CreditForm register={register} errors={errors} />

                  <div className="mt-8 pt-6 border-t border-gray-200">
                    <button
                      type="submit"
                      disabled={isCalculating}
                      className="w-full bg-primary-600 text-white py-4 px-6 rounded-lg font-semibold text-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md"
                    >
                      ประเมินคะแนนเครดิต
                    </button>
                  </div>
                </form>

                {!isValid && (
                  <p className="mt-3 text-sm text-gray-500">
                    กรุณากรอกข้อมูลที่จำเป็นให้ครบก่อนส่งแบบฟอร์ม
                  </p>
                )}
              </div>
            </div>
          )}

          {view === 'result' && (
            <div className="max-w-5xl mx-auto">
              {isCalculating ? (
                <div
                  className="bg-white rounded-2xl shadow-lg p-12 text-center"
                  role="status"
                  aria-live="polite"
                >
                  <div
                    className="w-16 h-16 border-4 border-primary-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"
                    aria-hidden="true"
                  />
                  <p className="text-lg font-semibold text-gray-700">
                    กำลังประมวลผล...
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    Model + RAG กำลังวิเคราะห์ (อาจใช้เวลา 30-60 วินาที)
                  </p>
                </div>
              ) : predictResult && creditInput ? (
                <ResultCard
                  prediction={predictResult.prediction}
                  confidence={confidence}
                  shapValues={predictResult.shap_values}
                  modelExplanation={modelExplanation}
                  plannerExplanation={plannerExplanation}
                  plannerError={plannerError}
                  userInput={creditInput}
                  onTalkToAssistant={handleTalkToAssistant}
                  onBack={handleBackToForm}
                />
              ) : null}
            </div>
          )}

          {view === 'assistant' && predictResult && (
            <div className="max-w-5xl mx-auto">
              <AssistantPanel
                approved={approved}
                confidence={confidence}
                plannerExplanation={plannerExplanation}
                plannerMode={predictResult.planner?.mode}
                ragSources={ragSources}
                plannerError={plannerError}
                checklist={checklist}
                onToggleChecklist={handleToggleChecklist}
                onBack={handleBackToResult}
              />

              <div className="mt-6">
                <AssistantChat
                  userProfileSummary={userProfileSummary}
                  advisorProfile={advisorProfile || undefined}
                />
              </div>
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
