'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { CreditInput, ChecklistItem } from '@/types/credit';
import { predictCredit, PredictResponse } from '@/utils/api';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import CreditForm from '@/components/CreditForm';
import ResultCard from '@/components/ResultCard';
import AssistantPanel from '@/components/AssistantPanel';
import AssistantChat from '@/components/AssistantChat';

// Validation schema
const creditSchema = z.object({
  Sex: z.string().min(1, 'Sex is required'),
  Occupation: z.string().min(1, 'Occupation is required'),
  Salary: z.string().min(1, 'Salary is required').regex(/^\d+(\.\d{1,2})?$/, 'Please enter a valid number'),
  Marital_status: z.string().min(1, 'Marital status is required'),
  credit_score: z.string().regex(/^\d*$/, 'Please enter a valid score').optional().or(z.literal('')),
  credit_grade: z.string().optional(),
  outstanding: z.string().min(1, 'Outstanding debt is required').regex(/^\d+(\.\d{1,2})?$/, 'Please enter a valid number'),
  overdue: z.string().min(1, 'Overdue amount is required').regex(/^\d+(\.\d{1,2})?$/, 'Please enter a valid number'),
  loan_amount: z.string().min(1, 'Loan amount is required').regex(/^\d+(\.\d{1,2})?$/, 'Please enter a valid number'),
  Coapplicant: z.string().min(1, 'Co-applicant status is required'),
  Interest_rate: z.string().min(1, 'Interest rate is required').regex(/^\d+(\.\d{1,2})?$/, 'Please enter a valid percentage'),
});

type View = 'form' | 'result' | 'assistant';

/**
 * Extract action items from planner Thai text.
 * The Ai-Credit-Scoring planner uses varying bullet styles: 1), 1., -, *, •,
 * Thai numerals ๑., or section heads like "มาตรการ:". Accept all of them.
 */
/**
 * Strip markdown formatting from a line (bold, italic, headers).
 */
function stripMarkdown(line: string): string {
  return line
    .replace(/^#+\s*/, '') // headers
    .replace(/\*\*(.+?)\*\*/g, '$1') // bold
    .replace(/\*(.+?)\*/g, '$1') // italic
    .replace(/^\*\*+|\*\*+$/g, '') // unclosed bold
    .trim();
}

/**
 * Decide whether a line is an actual action item vs a section header, label,
 * probability readout, or prose. Planner (Gemini) wraps section titles in
 * `**...**` and uses numbered headings like "1. สรุปผล" which are NOT tasks.
 */
function isActionable(raw: string): boolean {
  const line = raw.trim();
  if (line.length < 10) return false; // too short to be a real task
  if (line.length > 300) return false; // too long — probably a paragraph

  // Section headers (Gemini wraps them in bold)
  if (/^\*\*[^*]+\*\*:?$/.test(line)) return false;
  // Numbered section titles like "1. สรุปผลการวิเคราะห์" or "มาตรการที่ 1:"
  if (/^(\*\*)?\d+[.)、]\s*(สรุป|ผลการ|รายการ|ข้อสังเกต|ข้อเสนอแนะ|ข้อจำกัด|หมายเหตุ|เกี่ยวกับ|บทนำ|รายงาน)/.test(line))
    return false;
  if (/^(\*\*)?มาตรการที่\s*\d+/.test(line)) return false;
  // Probability/metadata
  if (/^P\(.*\)\s*=/i.test(line)) return false;
  if (/^(สรุป|ผลการ|หมายเหตุ|รายการเอกสาร|รายงาน|ข้อจำกัด|ข้อสังเกต|ข้อเสนอแนะ)/.test(line))
    return false;
  if (/^(disclaimer|note|summary|report)/i.test(line)) return false;
  // RAG miss sentinel
  if (/ไม่พบข้อมูลในเอกสาร/.test(line)) return false;
  // Pure label line ending with colon, no real content
  if (/^[^:]{1,40}:\s*$/.test(line)) return false;

  return true;
}

/**
 * Extract actionable bullet lines from the planner's Thai text.
 * Accepts numbered/dashed/dotted bullets, ignores markdown headers and prose.
 */
function extractActionsFromText(text?: string): string[] {
  if (!text) return [];

  const bulletPattern =
    /^(?:\d+[.)、]|[๑๒๓๔๕๖๗๘๙๐]+[.)])\s+|^[-*•▪●]\s+|^(?:มาตรการ|ข้อเสนอแนะ|แนวทาง|ขั้นตอน)[:：]\s*/;

  const rawLines = text.split('\n').map((l) => l.trim()).filter(Boolean);

  // Pass 1: bullets that are truly actionable
  const bulletLines: string[] = [];
  for (const raw of rawLines) {
    const stripped = stripMarkdown(raw);
    if (!bulletPattern.test(stripped)) continue;
    const body = stripped.replace(bulletPattern, '').trim();
    // Body itself must be actionable (not a header-like label)
    if (!isActionable(body)) continue;
    // Clip verbose bullets at the first sentence boundary for readability
    const firstSentence = body.split(/(?<=[。.!?])\s+/)[0];
    bulletLines.push(firstSentence.length > 30 ? firstSentence : body);
  }

  if (bulletLines.length > 0) return Array.from(new Set(bulletLines)).slice(0, 6);

  // Pass 2: sentence fallback — split by sentence boundary, keep actionable ones
  const sentences = rawLines
    .flatMap((line) => stripMarkdown(line).split(/(?<=[。.!?])\s+/))
    .map((s) => s.trim())
    .filter((s) => isActionable(s));
  return Array.from(new Set(sentences)).slice(0, 5);
}

function buildChecklist(actions: string[]): ChecklistItem[] {
  return actions.map((task, index) => ({
    id: String(index + 1),
    task,
    completed: false,
  }));
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
    } catch (error) {
      console.error('Error calculating credit score:', error);
      const detail =
        error instanceof Error && error.message
          ? error.message
          : 'Failed to calculate credit score. Please check that the backend is running and try again.';
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
                      Calculate Credit Score
                    </button>
                  </div>
                </form>

                {!isValid && (
                  <p className="mt-3 text-sm text-gray-500">
                    Please complete all required fields before submitting.
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
                <AssistantChat userProfileSummary={userProfileSummary} />
              </div>
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
