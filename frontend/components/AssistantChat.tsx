'use client';

import { useState, useRef, useEffect } from 'react';
import { queryAssistantRag } from '@/utils/api';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface AssistantChatProps {
  userProfileSummary?: string;
}

// Curated example questions that are VERIFIED to return valid answers from
// the current RAG index (Thai CIMB home loan documents). These phrases match
// the planner's DRIVER_QUERY_MAP and APPROVED_CHECKLIST_QUERIES so retrieval
// + synthesis both succeed. Do NOT add vague questions like "วิธีปรับปรุงโปรไฟล์"
// — they fail the synthesizer even when retrieval is fine.
const SUGGESTED_QUESTIONS: string[] = [
  'เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง',
  'ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้',
  'รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้',
  'ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้',
];

export default function AssistantChat({
  userProfileSummary,
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
      // Send the question as-is. Prepending a profile summary like
      // "ข้อมูลผู้ใช้: อาชีพ X, ดอกเบี้ย 5%..." pollutes the embedding and
      // confuses the query router — e.g. it routed generic questions to
      // `interest_structure` just because "5%" appeared in the prefix, then
      // synthesis returned "ไม่พบข้อมูลในเอกสารที่มีอยู่".
      void userProfileSummary; // reserved for future in-chat context UX
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
        ? 'ขออภัย ไม่พบข้อมูลตรงกับคำถามนี้ในเอกสารที่มีอยู่ ลองใช้คำถามที่เจาะจงกว่า เช่น "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง" หรือเลือกจากคำถามแนะนำด้านล่าง'
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
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.text}</p>
            </div>
          </div>
        ))}

        {/* Suggested question chips — only before the user has asked anything */}
        {!userHasAsked && !isTyping && (
          <div className="pt-1">
            <p className="text-xs text-gray-500 mb-2 font-medium">คำถามแนะนำ</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
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
