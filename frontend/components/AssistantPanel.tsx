'use client';

import { ChecklistItem } from '@/types/credit';
import PlannerReport from './PlannerReport';

interface RagSource {
  title?: string;
  category?: string;
  institution?: string;
  score?: number;
}

interface AssistantPanelProps {
  approved: boolean;
  confidence: number;
  plannerExplanation?: string;
  plannerMode?: string;
  ragSources: RagSource[];
  plannerError?: string | null;
  checklist: ChecklistItem[];
  onToggleChecklist: (id: string) => void;
  onBack: () => void;
}

export default function AssistantPanel({
  approved,
  confidence,
  plannerExplanation,
  plannerMode,
  ragSources,
  plannerError,
  checklist,
  onToggleChecklist,
  onBack,
}: AssistantPanelProps) {
  const pct = Math.round(confidence * 100);
  const modeLabel =
    plannerMode === 'approved_guidance'
      ? 'แนวทางสำหรับผู้มีโอกาสอนุมัติ'
      : plannerMode === 'improvement_plan'
      ? 'แผนปรับปรุงโปรไฟล์'
      : 'คำแนะนำจาก AI';

  return (
    <div className="space-y-6">
      {/* ── Header bar ─────────────────────────────────────────── */}
      <div className="flex items-center justify-between bg-white rounded-xl shadow-sm p-4 border border-gray-200">
        <button
          onClick={onBack}
          className="text-sm text-gray-600 hover:text-gray-900 font-medium"
        >
          กลับ
        </button>

        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-[10px] text-gray-500 uppercase tracking-wide">ผลการประเมิน</p>
            <p
              className={`text-sm font-semibold ${
                approved ? 'text-green-700' : 'text-red-700'
              }`}
            >
              {approved ? 'มีแนวโน้มอนุมัติ' : 'มีความเสี่ยงถูกปฏิเสธ'} · {pct}%
            </p>
          </div>
          <span className="hidden md:inline-block px-2.5 py-1 rounded-md text-[11px] font-medium bg-gray-100 text-gray-700 border border-gray-200">
            {modeLabel}
          </span>
        </div>
      </div>

      {/* ── Main: AI guidance (formal report) ─────────────────── */}
      {plannerExplanation ? (
        <div className="bg-white rounded-2xl shadow-sm p-6 md:p-10 border border-gray-200">
          <header className="mb-6 pb-4 border-b border-gray-200">
            <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-widest">
              รายงานโดย AI Credit Scoring System
            </p>
            <h2 className="text-xl md:text-2xl font-bold text-gray-900 mt-1">
              รายงานผลการวิเคราะห์และข้อเสนอแนะ
            </h2>
            <p className="text-xs text-gray-500 mt-1">
              สังเคราะห์จากผลโมเดลและเอกสารนโยบายสินเชื่อ (RAG) โดย LLM
            </p>
          </header>

          <PlannerReport text={plannerExplanation} />

          {plannerError && (
            <div className="mt-6 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-900">
              หมายเหตุระบบ: {plannerError}
            </div>
          )}
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-6 text-yellow-900">
          <p className="font-semibold mb-1">ไม่สามารถดึงคำแนะนำจาก AI ได้ในตอนนี้</p>
          {plannerError && <p className="text-sm">{plannerError}</p>}
          <p className="text-sm mt-2">
            สามารถสอบถามข้อมูลเพิ่มเติมได้ที่ช่องแชทด้านล่าง ระบบจะค้นจากฐานความรู้ RAG ให้โดยตรง
          </p>
        </div>
      )}

      {/* ── Action checklist (derived from planner text) ────── */}
      {checklist.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-900">
              รายการดำเนินการ
            </h3>
            <span className="text-xs text-gray-500">
              เสร็จ {checklist.filter((c) => c.completed).length} / {checklist.length} รายการ
            </span>
          </div>
          <p className="text-sm text-gray-500 mb-4">
            ติ๊กเมื่อดำเนินการเสร็จเพื่อติดตามความคืบหน้า
          </p>
          <div className="space-y-2">
            {checklist.map((item, idx) => (
              <label
                key={item.id}
                className={`flex items-start p-3 rounded-lg cursor-pointer transition-colors border ${
                  item.completed
                    ? 'bg-green-50 border-green-200'
                    : 'bg-gray-50 border-transparent hover:bg-gray-100'
                }`}
              >
                <input
                  type="checkbox"
                  checked={item.completed}
                  onChange={() => onToggleChecklist(item.id)}
                  className="w-5 h-5 mt-0.5 text-gray-700 rounded focus:ring-gray-500"
                />
                <span
                  className={`ml-3 flex-1 text-sm leading-relaxed ${
                    item.completed ? 'line-through text-gray-500' : 'text-gray-800'
                  }`}
                >
                  <span className="inline-block w-6 text-gray-400 font-semibold">
                    {idx + 1}.
                  </span>
                  {item.task}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* ── RAG Sources as citations ─────────────────────────── */}
      {ragSources.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200">
          <div className="flex items-baseline justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              แหล่งอ้างอิง
            </h3>
            <span className="text-sm text-gray-500">
              {ragSources.length} เอกสาร
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {ragSources.map((src, idx) => {
              const title = src.title || `เอกสารลำดับที่ ${idx + 1}`;
              const meta = [src.institution, src.category].filter(Boolean).join(' · ');
              const scorePct =
                typeof src.score === 'number' ? Math.round(src.score * 100) : null;
              return (
                <div
                  key={idx}
                  className="p-4 bg-gray-50 rounded-lg border border-gray-200"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-900 text-sm">
                        <span className="text-gray-500 mr-1.5 font-mono">[{idx + 1}]</span>
                        {title}
                      </p>
                      {meta && (
                        <p className="text-xs text-gray-500 mt-1 truncate">{meta}</p>
                      )}
                    </div>
                    {scorePct !== null && (
                      <span
                        title="Similarity score"
                        className="shrink-0 text-[10px] font-mono text-gray-500 bg-white px-1.5 py-0.5 rounded border border-gray-200"
                      >
                        {scorePct}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
