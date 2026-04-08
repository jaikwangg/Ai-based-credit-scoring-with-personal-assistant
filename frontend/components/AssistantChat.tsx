'use client';

import { useState, useRef, useEffect } from 'react';
import {
  queryAssistantRag,
  queryAdvisor,
  AdvisorProfile,
  AdvisorResponse,
} from '@/utils/api';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  // Optional structured payload — when present, the message renders as
  // an advisor card instead of plain text.
  advisor?: AdvisorResponse;
}

interface AssistantChatProps {
  userProfileSummary?: string;
  /**
   * Structured user profile for profile-conditioned advisory. When provided,
   * the chat will route eligibility-style questions through the /rag/advisor
   * endpoint (which does pass/fail reasoning) instead of /rag/query (which
   * just paraphrases retrieved chunks).
   */
  advisorProfile?: AdvisorProfile;
}

// Curated example questions verified against the current RAG index.
// Two flavours:
//  - factual lookups → routed through /rag/query (paraphrase mode)
//  - eligibility/advice questions → routed through /rag/advisor (reasoning mode)
//    when an advisorProfile is available.
const FACTUAL_QUESTIONS: string[] = [
  'เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง',
  'รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้',
  'ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้',
];

const ADVISORY_QUESTIONS: string[] = [
  'จากโปรไฟล์ของฉัน มีโอกาสกู้ได้ไหม',
  'ฉันควรปรับปรุงอะไรบ้างเพื่อให้ผ่านเกณฑ์',
  'โปรไฟล์ของฉันผ่านเกณฑ์รายได้ขั้นต่ำหรือไม่',
];

/**
 * Heuristic: should this question be answered by the profile-conditioned
 * advisor (structured eligibility reasoning) instead of plain RAG?
 */
function shouldUseAdvisor(question: string, hasProfile: boolean): boolean {
  if (!hasProfile) return false;
  const triggers = [
    'กู้ได้ไหม',
    'อนุมัติ',
    'มีโอกาส',
    'ผมจะ',
    'ฉันจะ',
    'โปรไฟล์',
    'คุณสมบัติ',
    'ผ่านเกณฑ์',
    'ปรับปรุง',
    'แนะนำ',
    'ควรทำ',
    'ทำยังไง',
    'eligible',
    'qualify',
  ];
  return triggers.some((t) => question.includes(t));
}

export default function AssistantChat({
  userProfileSummary,
  advisorProfile,
}: AssistantChatProps) {
  // Chat welcome is INTENTIONALLY a short greeting, not the planner report.
  // The full report is already displayed above the chat in AssistantPanel.
  const welcomeText =
    'สวัสดีครับ ระบบ AI Assistant พร้อมให้คำปรึกษาเกี่ยวกับการสมัครสินเชื่อบ้าน\n\n' +
    'เลือกคำถามด้านล่างเพื่อเริ่มต้น หรือพิมพ์คำถามของคุณเองก็ได้ ' +
    'ทุกคำตอบจะอ้างอิงจากเอกสารนโยบายสินเชื่อจริงผ่านระบบ RAG';
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: welcomeText,
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const askRag = async (userText: string) => {
    if (!userText.trim() || isTyping) return;
    const text = userText.trim();

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        text,
        isUser: true,
        timestamp: new Date(),
      },
    ]);
    setInputMessage('');
    setIsTyping(true);

    try {
      void userProfileSummary; // legacy prop, kept for backwards compat

      // Decide between profile-conditioned advisor and plain RAG.
      // Advisor: when profile is present AND the question is eligibility-style.
      // Plain RAG: for general factual lookups ("เอกสารใช้อะไรบ้าง").
      const useAdvisor = shouldUseAdvisor(text, !!advisorProfile);

      if (useAdvisor && advisorProfile) {
        const advisor = await queryAdvisor(text, advisorProfile);
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            text: advisor.verdict_summary || 'วิเคราะห์เสร็จแล้ว',
            isUser: false,
            timestamp: new Date(),
            advisor,
          },
        ]);
      } else {
        const rag = await queryAssistantRag(text);
        const rawAnswer = (rag.answer || '').trim();
        const ragFailed = !rawAnswer || /ไม่พบข้อมูลในเอกสาร/.test(rawAnswer);

        const sourceText =
          rag.sources && rag.sources.length > 0
            ? `\n\nแหล่งอ้างอิง:\n${rag.sources
                .slice(0, 3)
                .map(
                  (s, idx) =>
                    `[${idx + 1}] ${s.title || 'ไม่ระบุชื่อเอกสาร'}${
                      s.category ? ` (${s.category})` : ''
                    }`
                )
                .join('\n')}`
            : '';

        const aiText = ragFailed
          ? 'ขออภัย ไม่พบข้อมูลตรงกับคำถามนี้ในเอกสารที่มีอยู่ ลองใช้คำถามที่เจาะจงกว่า หรือเลือกจากคำถามแนะนำด้านล่าง'
          : `${rawAnswer}${sourceText}`;

        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            text: aiText,
            isUser: false,
            timestamp: new Date(),
          },
        ]);
      }
    } catch (error) {
      const detail =
        error instanceof Error && error.message
          ? error.message
          : 'ไม่สามารถเชื่อมต่อ RAG ได้';
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          text: `ขออภัย ระบบไม่สามารถตอบคำถามได้ในขณะนี้: ${detail}`,
          isUser: false,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    await askRag(inputMessage);
  };

  // Hide suggested-question chips once the user has sent at least one message.
  const userHasAsked = messages.some((m) => m.isUser);

  return (
    <div className="flex flex-col h-full max-h-[600px] bg-white rounded-xl shadow-sm border border-gray-200">
      <div className="p-4 border-b border-gray-200 bg-gray-50 rounded-t-xl">
        <h3 className="text-lg font-semibold text-gray-900">AI Personal Assistant</h3>
        <p className="text-sm text-gray-600 mt-1">
          สอบถามข้อมูลเกี่ยวกับผลการประเมินหรือแนวทางการปรับปรุงได้
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-lg p-3 ${
                message.isUser
                  ? 'bg-gray-900 text-white'
                  : message.advisor
                  ? 'bg-white border border-gray-200 shadow-sm w-full'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {message.advisor ? (
                <AdvisorCard data={message.advisor} />
              ) : (
                <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.text}</p>
              )}
            </div>
          </div>
        ))}

        {/* Suggested question chips — only before the user has asked anything */}
        {!userHasAsked && !isTyping && (
          <div className="pt-1 space-y-3">
            {advisorProfile && (
              <div>
                <p className="text-xs text-gray-500 mb-2 font-medium">
                  วิเคราะห์โปรไฟล์ของคุณ (ต้องใช้การคิดวิเคราะห์)
                </p>
                <div className="flex flex-wrap gap-2">
                  {ADVISORY_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => askRag(q)}
                      className="text-xs px-3 py-2 bg-gray-900 text-white rounded-full hover:bg-black transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div>
              <p className="text-xs text-gray-500 mb-2 font-medium">
                คำถามทั่วไปจากเอกสาร
              </p>
              <div className="flex flex-wrap gap-2">
                {FACTUAL_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => askRag(q)}
                    className="text-xs px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-full hover:bg-gray-50 hover:border-gray-400 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg p-3" role="status" aria-live="polite" aria-label="AI is typing">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSend} className="p-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="พิมพ์คำถาม..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isTyping}
            className="px-6 py-2 bg-gray-900 text-white rounded-lg font-semibold hover:bg-black disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            ส่ง
          </button>
        </div>
      </form>
    </div>
  );
}

// === AdvisorCard ============================================================

function AdvisorCard({ data }: { data: AdvisorResponse }) {
  const verdictMeta: Record<
    AdvisorResponse['verdict'],
    { label: string; color: string; bg: string; border: string }
  > = {
    eligible: {
      label: 'มีโอกาสผ่านเกณฑ์',
      color: 'text-green-700',
      bg: 'bg-green-50',
      border: 'border-green-200',
    },
    partially_eligible: {
      label: 'ผ่านบางส่วน',
      color: 'text-amber-700',
      bg: 'bg-amber-50',
      border: 'border-amber-200',
    },
    ineligible: {
      label: 'ยังไม่ผ่านเกณฑ์',
      color: 'text-red-700',
      bg: 'bg-red-50',
      border: 'border-red-200',
    },
    needs_more_info: {
      label: 'ต้องการข้อมูลเพิ่มเติม',
      color: 'text-gray-700',
      bg: 'bg-gray-50',
      border: 'border-gray-200',
    },
  };
  const meta = verdictMeta[data.verdict] || verdictMeta.needs_more_info;

  const statusBadge: Record<
    AdvisorResponse['requirement_checks'][number]['status'],
    { label: string; cls: string }
  > = {
    pass: { label: 'ผ่าน', cls: 'bg-green-100 text-green-800' },
    fail: { label: 'ไม่ผ่าน', cls: 'bg-red-100 text-red-800' },
    unknown: { label: 'ไม่ทราบ', cls: 'bg-gray-100 text-gray-700' },
    not_applicable: { label: 'ไม่เกี่ยว', cls: 'bg-gray-100 text-gray-500' },
  };

  return (
    <div className="space-y-4">
      {/* Verdict header */}
      <div className={`p-3 rounded-md border ${meta.bg} ${meta.border}`}>
        <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide">
          ผลการวิเคราะห์
        </p>
        <p className={`text-base font-bold ${meta.color}`}>{meta.label}</p>
        <p className="text-sm text-gray-700 mt-1 leading-relaxed">
          {data.verdict_summary}
        </p>
      </div>

      {/* Requirement checks */}
      {data.requirement_checks.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-700 mb-2">
            เงื่อนไขที่ตรวจสอบ ({data.requirement_checks.length} ข้อ)
          </p>
          <div className="space-y-2">
            {data.requirement_checks.map((c, i) => {
              const badge = statusBadge[c.status] || statusBadge.unknown;
              return (
                <div
                  key={i}
                  className="text-xs p-2.5 bg-gray-50 rounded border border-gray-200"
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <span className="font-semibold text-gray-900 flex-1">
                      {c.requirement}
                    </span>
                    <span
                      className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-semibold ${badge.cls}`}
                    >
                      {badge.label}
                    </span>
                  </div>
                  <p className="text-gray-600">
                    <span className="text-gray-500">ค่าของคุณ:</span>{' '}
                    <span className="font-medium">{c.user_value}</span>
                  </p>
                  {c.explanation && (
                    <p className="text-gray-500 mt-1 leading-relaxed">{c.explanation}</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recommended actions */}
      {data.recommended_actions.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-700 mb-2">
            แนะนำให้ทำ
          </p>
          <ol className="text-xs text-gray-700 space-y-1 list-decimal list-inside">
            {data.recommended_actions.map((a, i) => (
              <li key={i} className="leading-relaxed">
                {a}
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Sources */}
      {data.sources.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1">
            อ้างอิง
          </p>
          <p className="text-[11px] text-gray-500 leading-relaxed">
            {data.sources
              .slice(0, 4)
              .map((s, i) => `[${i + 1}] ${s.title || 'ไม่ระบุ'}`)
              .join('  ·  ')}
          </p>
        </div>
      )}
    </div>
  );
}
