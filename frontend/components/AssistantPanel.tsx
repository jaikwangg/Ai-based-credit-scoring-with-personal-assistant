'use client';

import { ChecklistItem } from '@/types/credit';

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
      ? 'Approval Guidance'
      : plannerMode === 'improvement_plan'
      ? 'Improvement Plan'
      : 'AI Advisor';

  return (
    <div className="space-y-6">
      {/* ── Header bar ─────────────────────────────────────────── */}
      <div className="flex items-center justify-between bg-white rounded-xl shadow-sm p-4 border border-gray-200">
        <button
          onClick={onBack}
          className="text-sm text-gray-600 hover:text-gray-900 font-medium"
        >
          {'<'} Back
        </button>

        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${
              approved ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            }`}
          >
            {approved ? '✓' : '✗'}
          </div>
          <div className="text-right">
            <p className="text-[10px] text-gray-500 uppercase tracking-wide">ผลการประเมิน</p>
            <p className="text-sm font-semibold text-gray-900">
              {approved ? 'มีแนวโน้มอนุมัติ' : 'มีความเสี่ยงถูกปฏิเสธ'} · {pct}%
            </p>
          </div>
          <span className="hidden md:inline-block px-2 py-1 rounded-full text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-100">
            {modeLabel}
          </span>
        </div>
      </div>

      {/* ── Main: AI guidance (hero card) ─────────────────────── */}
      {plannerExplanation ? (
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8 border-l-4 border-blue-500">
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center mr-3 font-bold text-sm">
              AI
            </div>
            <div>
              <h3 className="text-xl md:text-2xl font-bold text-gray-900">
                คำแนะนำจาก AI Assistant
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">
                อิงจากเอกสารนโยบายสินเชื่อจริง (RAG) + ผลวิเคราะห์จากโมเดล
              </p>
            </div>
          </div>
          <div className="text-gray-800 whitespace-pre-wrap leading-relaxed text-[15px]">
            {plannerExplanation}
          </div>
          {plannerError && (
            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-900">
              ⚠ {plannerError}
            </div>
          )}
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-6 text-yellow-900">
          <p className="font-semibold mb-1">ไม่สามารถดึงคำแนะนำจาก AI ได้ในตอนนี้</p>
          {plannerError && <p className="text-sm">{plannerError}</p>}
          <p className="text-sm mt-2">
            ลองถามคำถามที่ช่องแชทด้านล่าง ระบบจะค้นจากฐานความรู้ RAG ให้โดยตรง
          </p>
        </div>
      )}

      {/* ── Action checklist (derived from planner text) ────── */}
      {checklist.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              <span className="mr-2">📋</span> Action Items
            </h3>
            <span className="text-xs text-gray-500">
              {checklist.filter((c) => c.completed).length}/{checklist.length} เสร็จ
            </span>
          </div>
          <p className="text-sm text-gray-500 mb-4">
            ติ๊กเมื่อทำเสร็จเพื่อติดตามความคืบหน้า
          </p>
          <div className="space-y-2">
            {checklist.map((item, idx) => (
              <label
                key={item.id}
                className={`flex items-start p-3 rounded-lg cursor-pointer transition-colors border ${
                  item.completed
                    ? 'bg-green-50 border-green-100'
                    : 'bg-gray-50 border-transparent hover:bg-gray-100'
                }`}
              >
                <input
                  type="checkbox"
                  checked={item.completed}
                  onChange={() => onToggleChecklist(item.id)}
                  className="w-5 h-5 mt-0.5 text-green-600 rounded focus:ring-green-500"
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
        <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
            <span className="mr-2">📚</span> แหล่งอ้างอิง
            <span className="ml-2 text-sm font-normal text-gray-500">
              ({ragSources.length} เอกสาร)
            </span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {ragSources.map((src, idx) => {
              const title = src.title || `Source ${idx + 1}`;
              const meta = [src.institution, src.category].filter(Boolean).join(' · ');
              const scorePct =
                typeof src.score === 'number' ? Math.round(src.score * 100) : null;
              return (
                <div
                  key={idx}
                  className="p-4 bg-gray-50 rounded-lg border border-gray-100 hover:border-blue-200 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-gray-900 text-sm">
                        <span className="text-blue-600 mr-1.5">[{idx + 1}]</span>
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
