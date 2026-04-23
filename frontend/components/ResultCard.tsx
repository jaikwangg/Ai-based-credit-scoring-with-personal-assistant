'use client';

import { CreditInput } from '@/types/credit';

interface ResultCardProps {
  prediction: number;
  confidence: number;
  shapValues: Record<string, number>;
  modelExplanation?: string;
  plannerExplanation?: string;
  plannerError?: string | null;
  userInput: CreditInput;
  distributionWarnings?: string[];
  onTalkToAssistant: () => void;
  onBack?: () => void;
}

const FEATURE_LABELS_TH: Record<string, string> = {
  Sex: 'เพศ',
  Occupation: 'อาชีพ',
  Salary: 'รายได้',
  Marriage_Status: 'สถานภาพสมรส',
  credit_score: 'คะแนนเครดิต',
  credit_grade: 'เกรดเครดิต',
  outstanding: 'ภาระหนี้สินรวม',
  overdue: 'จำนวนวันค้างชำระสูงสุด',
  Coapplicant: 'ผู้กู้ร่วม',
  loan_amount: 'วงเงินขอกู้',
  loan_term: 'ระยะเวลากู้',
  Interest_rate: 'อัตราดอกเบี้ย',
  has_overdue: 'สถานะค้างชำระ',
  dti: 'DTI (หนี้/รายได้)',
  lti: 'LTI (วงเงิน/รายได้)',
};

function labelTh(name: string): string {
  return FEATURE_LABELS_TH[name] || name.replace(/_/g, ' ');
}

type MetricStatus = 'good' | 'warn' | 'bad' | 'neutral';

export default function ResultCard({
  prediction,
  confidence,
  shapValues,
  modelExplanation,
  plannerExplanation,
  plannerError,
  userInput,
  distributionWarnings = [],
  onTalkToAssistant,
  onBack,
}: ResultCardProps) {
  const approved = prediction === 1;
  const pct = Math.round(confidence * 100);

  const entries = Object.entries(shapValues || {});
  const positive = entries
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  const negative = entries
    .filter(([, v]) => v < 0)
    .sort((a, b) => a[1] - b[1])
    .slice(0, 5);
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);

  const salary = parseFloat(userInput.Salary) || 0;
  const outstanding = parseFloat(userInput.outstanding) || 0;
  const loanAmount = parseFloat(userInput.loan_amount) || 0;
  const creditScoreNum = parseInt(userInput.credit_score || '0', 10) || 0;
  const dti = salary > 0 ? outstanding / salary : 0;
  const lti = salary > 0 ? loanAmount / salary : 0;

  const dtiStatus: MetricStatus = dti < 0.4 ? 'good' : dti < 0.6 ? 'warn' : 'bad';
  const ltiStatus: MetricStatus = lti < 5 ? 'good' : lti < 10 ? 'warn' : 'bad';
  const scoreStatus: MetricStatus =
    creditScoreNum >= 700 ? 'good' : creditScoreNum >= 600 ? 'warn' : creditScoreNum > 0 ? 'bad' : 'neutral';

  // Sanity check: model might approve profiles that are obviously unrealistic
  // (e.g. tiny income vs huge loan). Flag this mismatch to the user regardless
  // of what the ML model says — the LGBM model is trained on a research dataset
  // and can give nonsensical outputs outside its training distribution.
  const modelContradicted =
    approved &&
    salary > 0 &&
    (dti >= 1 || lti >= 40 || (creditScoreNum > 0 && creditScoreNum < 500));

  return (
    <div className="space-y-6">
      {onBack && (
        <button
          onClick={onBack}
          className="text-sm text-gray-600 hover:text-gray-900 font-medium"
        >
          กลับไปกรอกข้อมูล
        </button>
      )}

      {distributionWarnings.length > 0 && (
        <div className="bg-yellow-50 border-l-4 border-yellow-500 rounded p-4">
          <p className="font-semibold text-yellow-900">
            ⓘ ข้อมูลบางส่วนอยู่นอกช่วงที่โมเดลเรียนรู้
          </p>
          <p className="text-xs text-yellow-800 mt-1 mb-2">
            ผลการประเมินด้านล่างอาจไม่แม่นยำ — โมเดลถูก train บน dataset ที่มีช่วงค่าจำกัด:
          </p>
          <ul className="text-xs text-yellow-800 space-y-0.5 list-disc list-inside">
            {distributionWarnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {modelContradicted && (
        <div className="bg-amber-50 border-l-4 border-amber-500 rounded p-4">
          <p className="font-semibold text-amber-900">
            คำเตือน: ผลโมเดลอาจไม่สอดคล้องกับโปรไฟล์จริง
          </p>
          <p className="text-sm text-amber-800 mt-1 leading-relaxed">
            แม้โมเดลประเมินว่ามีแนวโน้มอนุมัติ ({pct}%) แต่ตัวชี้วัดทางการเงิน
            {dti >= 1 && ` DTI สูงมาก (${(dti * 100).toFixed(0)}%)`}
            {lti >= 40 && ` LTI สูงมาก (${lti.toFixed(0)} เท่าของรายได้)`}
            {creditScoreNum > 0 && creditScoreNum < 500 && ` คะแนนเครดิตต่ำ (${creditScoreNum})`}
            {' '}อยู่นอกช่วงที่สถาบันการเงินจริงจะพิจารณาอนุมัติได้
            โปรดตรวจสอบข้อมูลที่กรอกหรืออ่านคำแนะนำจาก AI ด้านล่างอย่างละเอียด
          </p>
        </div>
      )}

      {/* ── Hero: Decision + Probability gauge ───────────────────────── */}
      <div
        className={`bg-white rounded-2xl shadow-lg overflow-hidden border-t-8 ${
          approved ? 'border-green-500' : 'border-red-500'
        }`}
      >
        <div className="p-6 md:p-8">
          <div className="flex items-start justify-between gap-4 mb-6">
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
                ผลการประเมินสินเชื่อ
              </p>
              <h1
                className={`text-3xl md:text-4xl font-bold mt-2 ${
                  approved ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {approved ? 'มีแนวโน้มอนุมัติ' : 'มีความเสี่ยงถูกปฏิเสธ'}
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                {approved
                  ? 'โปรไฟล์ของคุณอยู่ในเกณฑ์ที่สถาบันการเงินมีแนวโน้มพิจารณาอนุมัติ'
                  : 'โปรไฟล์ของคุณยังต่ำกว่าเกณฑ์อนุมัติ ต้องปรับปรุงปัจจัยเสี่ยงก่อนยื่นใหม่'}
              </p>
            </div>
            <div
              className={`shrink-0 px-3 py-1.5 rounded-md text-xs font-semibold tracking-wide ${
                approved ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}
            >
              {approved ? 'APPROVED' : 'DECLINED'}
            </div>
          </div>

          {/* Probability Gauge */}
          <div className="mt-6">
            <div className="flex justify-between items-baseline mb-2">
              <span className="text-sm font-medium text-gray-700">
                โอกาสได้รับการอนุมัติ (P-approve)
              </span>
              <span
                className={`text-3xl font-bold ${
                  approved ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {pct}%
              </span>
            </div>
            <div className="relative w-full h-5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className={`absolute left-0 top-0 h-full transition-all duration-1000 ${
                  approved
                    ? 'bg-gradient-to-r from-green-400 to-green-600'
                    : 'bg-gradient-to-r from-red-400 to-red-600'
                }`}
                style={{ width: `${pct}%` }}
              />
              <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-gray-700 opacity-40" />
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1.5">
              <span>0%</span>
              <span className="font-medium">50% เกณฑ์ตัดสิน</span>
              <span>100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Key Metrics Grid ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="DTI"
          sub="หนี้/รายได้"
          value={
            salary > 0
              ? dti >= 10
                ? '>1000%'
                : `${(dti * 100).toFixed(0)}%`
              : '—'
          }
          status={salary > 0 ? dtiStatus : 'neutral'}
          hint={
            salary === 0
              ? 'ไม่มีข้อมูลรายได้'
              : dtiStatus === 'good'
              ? 'อยู่ในเกณฑ์ดี (<40%)'
              : dtiStatus === 'warn'
              ? 'ควรลด (40-60%)'
              : 'สูงเกินเกณฑ์'
          }
        />
        <MetricCard
          label="LTI"
          sub="วงเงิน/รายได้"
          value={
            salary > 0
              ? lti >= 1000
                ? '>1000x'
                : `${lti.toFixed(1)}x`
              : '—'
          }
          status={salary > 0 ? ltiStatus : 'neutral'}
          hint={
            salary === 0
              ? '—'
              : ltiStatus === 'good'
              ? 'พอเหมาะ (<5x)'
              : ltiStatus === 'warn'
              ? 'ค่อนข้างสูง (5-10x)'
              : 'สูงเกินเกณฑ์'
          }
        />
        <MetricCard
          label="คะแนนเครดิต"
          sub="Credit Score"
          value={creditScoreNum > 0 ? creditScoreNum.toString() : '—'}
          status={scoreStatus}
          hint={`เกรด ${userInput.credit_grade || '—'}`}
        />
        <MetricCard
          label="วงเงินขอกู้"
          sub="Loan amount"
          value={
            loanAmount > 0
              ? `฿${loanAmount.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
              : '—'
          }
          status="neutral"
          hint={`ดอกเบี้ย ${userInput.Interest_rate || 0}%`}
        />
      </div>

      {/* ── SHAP Drivers: Positive / Negative ────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-2xl shadow-sm p-6 border border-green-100">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-green-100">
            <h3 className="text-base font-semibold text-green-700">
              ปัจจัยบวก
            </h3>
            <span className="text-xs text-gray-500">ช่วยเพิ่มโอกาสอนุมัติ</span>
          </div>
          {positive.length > 0 ? (
            <div className="space-y-3">
              {positive.map(([name, val]) => (
                <DriverBar
                  key={name}
                  label={labelTh(name)}
                  value={val}
                  maxAbs={maxAbs}
                  positive
                />
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-400 italic">ไม่พบปัจจัยบวกที่มีผลชัดเจน</p>
          )}
        </div>

        <div className="bg-white rounded-2xl shadow-sm p-6 border border-red-100">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-red-100">
            <h3 className="text-base font-semibold text-red-700">
              ปัจจัยลบ
            </h3>
            <span className="text-xs text-gray-500">ลดโอกาสอนุมัติ</span>
          </div>
          {negative.length > 0 ? (
            <div className="space-y-3">
              {negative.map(([name, val]) => (
                <DriverBar
                  key={name}
                  label={labelTh(name)}
                  value={val}
                  maxAbs={maxAbs}
                />
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-400 italic">ไม่พบปัจจัยลบที่มีผลชัดเจน</p>
          )}
        </div>
      </div>

      {/* ── Preview of AI guidance ─────────────────────────────────── */}
      {plannerExplanation && (
        <div className="bg-white rounded-2xl p-6 border border-gray-200">
          <div className="flex items-baseline justify-between mb-3 pb-3 border-b border-gray-200">
            <h3 className="text-base font-semibold text-gray-900">
              สรุปคำแนะนำจากระบบ
            </h3>
            <span className="text-[11px] text-gray-500 uppercase tracking-wide">
              Preview
            </span>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed line-clamp-5">
            {stripMarkdown(plannerExplanation)}
          </p>
          <button
            onClick={onTalkToAssistant}
            className="mt-4 text-sm font-semibold text-gray-900 underline underline-offset-2 hover:text-black"
          >
            อ่านรายงานฉบับเต็มและแหล่งอ้างอิง
          </button>
        </div>
      )}

      {/* ── Technical fallback details (collapsed) ─────────────────── */}
      {modelExplanation && (
        <details className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
          <summary className="cursor-pointer px-4 py-3 font-medium text-gray-700 text-sm hover:bg-gray-100">
            ข้อมูลทางเทคนิค (Raw SHAP breakdown)
          </summary>
          <div className="px-4 py-3 border-t border-gray-200 text-xs text-gray-600 whitespace-pre-wrap font-mono">
            {modelExplanation}
          </div>
        </details>
      )}

      {plannerError && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-xl text-sm text-yellow-900">
          <span className="font-semibold">หมายเหตุระบบ:</span> {plannerError}
        </div>
      )}

      {/* ── CTA ──────────────────────────────────────────────────── */}
      <button
        onClick={onTalkToAssistant}
        className="w-full bg-gray-900 hover:bg-black text-white py-4 px-6 rounded-xl font-semibold text-base shadow-sm hover:shadow-md transition-all"
      >
        ดูรายงานฉบับเต็มและสอบถามข้อมูลเพิ่มเติม
      </button>
    </div>
  );
}

/**
 * Strip markdown (bold, italic, asterisks) for plain-text preview/snippets.
 */
function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/^#+\s*/gm, '')
    .replace(/^\s*\*\s+/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function MetricCard({
  label,
  sub,
  value,
  status,
  hint,
}: {
  label: string;
  sub?: string;
  value: string;
  status: MetricStatus;
  hint?: string;
}) {
  const palette: Record<MetricStatus, { border: string; bg: string; text: string; dot: string }> = {
    good: {
      border: 'border-green-200',
      bg: 'bg-white',
      text: 'text-green-700',
      dot: 'bg-green-500',
    },
    warn: {
      border: 'border-yellow-200',
      bg: 'bg-white',
      text: 'text-yellow-700',
      dot: 'bg-yellow-500',
    },
    bad: {
      border: 'border-red-200',
      bg: 'bg-white',
      text: 'text-red-700',
      dot: 'bg-red-500',
    },
    neutral: {
      border: 'border-gray-200',
      bg: 'bg-white',
      text: 'text-gray-700',
      dot: 'bg-gray-400',
    },
  };
  const c = palette[status];
  return (
    <div className={`rounded-xl p-4 border ${c.border} ${c.bg} shadow-sm`}>
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">{label}</p>
        <span className={`w-2 h-2 rounded-full ${c.dot}`} />
      </div>
      {sub && <p className="text-[10px] text-gray-400 mb-1">{sub}</p>}
      <p className={`text-2xl font-bold mt-1 ${c.text}`}>{value}</p>
      {hint && <p className="text-xs text-gray-500 mt-1">{hint}</p>}
    </div>
  );
}

function DriverBar({
  label,
  value,
  maxAbs,
  positive,
}: {
  label: string;
  value: number;
  maxAbs: number;
  positive?: boolean;
}) {
  const width = Math.max(4, (Math.abs(value) / maxAbs) * 100);
  return (
    <div>
      <div className="flex justify-between items-baseline text-sm mb-1">
        <span className="text-gray-800 font-medium">{label}</span>
        <span
          className={`font-mono text-xs ${positive ? 'text-green-600' : 'text-red-600'}`}
        >
          {value > 0 ? '+' : ''}
          {value.toFixed(3)}
        </span>
      </div>
      <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            positive ? 'bg-green-500' : 'bg-red-500'
          }`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}
