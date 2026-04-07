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

export default function AssistantChat({
  userProfileSummary,
}: AssistantChatProps) {
  // Chat welcome is INTENTIONALLY a short greeting, not the planner report.
  // The full report is already displayed above the chat in AssistantPanel.
  // Showing the same 4000-char text here would be duplication and poor UX.
  const welcomeText =
    'สวัสดีครับ 👋 ผมเป็น AI Assistant ถามได้เลยเกี่ยวกับ:\n\n' +
    '• เอกสารที่ต้องใช้สมัครสินเชื่อ\n' +
    '• คุณสมบัติและเกณฑ์พิจารณา\n' +
    '• วิธีปรับปรุงโปรไฟล์ให้อนุมัติง่ายขึ้น\n' +
    '• รายละเอียดของรายงานวิเคราะห์ด้านบน\n\n' +
    'ทุกคำตอบจะค้นจากฐานความรู้ RAG เอกสารนโยบายสินเชื่อจริง';
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

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    const userText = inputMessage.trim();

    const userMessage: Message = {
      id: Date.now().toString(),
      text: userText,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    try {
      // Give RAG enough context so answers are grounded in the user's
      // actual profile / model decision, not generic loan FAQs.
      const contextParts: string[] = [];
      if (userProfileSummary) contextParts.push(`ข้อมูลผู้ใช้: ${userProfileSummary}`);
      const contextualQuestion =
        contextParts.length > 0
          ? `${contextParts.join(' | ')}\nคำถาม: ${userText}`
          : userText;
      const rag = await queryAssistantRag(contextualQuestion);
      const sourceText =
        rag.sources && rag.sources.length > 0
          ? `\n\nSources:\n${rag.sources
              .slice(0, 3)
              .map((s, idx) => `${idx + 1}. ${s.title || 'Unknown'} (${s.category || 'Uncategorized'})`)
              .join('\n')}`
          : '';

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `${rag.answer || 'No answer returned.'}${sourceText}`,
        isUser: false,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiResponse]);
    } catch (error) {
      const detail =
        error instanceof Error && error.message
          ? error.message
          : 'RAG query failed';
      const aiError: Message = {
        id: (Date.now() + 1).toString(),
        text: `Unable to query assistant right now: ${detail}`,
        isUser: false,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiError]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-h-[600px] bg-white rounded-xl shadow-lg">
      <div className="p-4 border-b border-gray-200 bg-primary-50 rounded-t-xl">
        <h3 className="text-lg font-semibold text-gray-900">AI Personal Assistant</h3>
        <p className="text-sm text-gray-600 mt-1">Ask me anything about your credit score or plan</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.isUser
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{message.text}</p>
            </div>
          </div>
        ))}
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
            placeholder="Ask a question..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isTyping}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
