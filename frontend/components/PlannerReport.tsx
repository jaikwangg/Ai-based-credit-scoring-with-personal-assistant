'use client';

import React from 'react';

/**
 * Minimal Thai-aware renderer for the planner's Gemini output.
 *
 * The planner returns lightly-formatted Markdown using only:
 *   - **bold** inline
 *   - Paragraph blocks separated by blank lines
 *   - Bullet lines starting with `*` or `-`
 *   - Numbered lines like "1." or "มาตรการที่ 1:" (treated as section headings)
 *
 * We intentionally do NOT depend on react-markdown to keep the bundle small
 * and avoid rendering XSS-risky constructs (links, images, HTML). Everything
 * below is plain text transformation.
 */

interface PlannerReportProps {
  text: string;
}

type Block =
  | { kind: 'heading'; level: 1 | 2 | 3; content: string }
  | { kind: 'paragraph'; content: string }
  | { kind: 'bullet-group'; items: string[] }
  | { kind: 'spacer' };

function classifyLine(raw: string): { type: 'heading' | 'bullet' | 'text' | 'empty'; level?: 1 | 2 | 3; content: string } {
  const line = raw.trim();
  if (!line) return { type: 'empty', content: '' };

  // **Heading wrapped in double asterisks** (entire line)
  const boldOnly = line.match(/^\*\*(.+?)\*\*:?$/);
  if (boldOnly) {
    const content = boldOnly[1].trim();
    // Detect heading level by cue words
    if (/^(รายงาน|โดยระบบ)/.test(content)) return { type: 'heading', level: 1, content };
    if (/^(มาตรการที่|หมายเหตุ|สรุป|รายละเอียด|ข้อมูลเพิ่ม)/.test(content))
      return { type: 'heading', level: 2, content };
    return { type: 'heading', level: 3, content };
  }

  // Numbered heading "1. ..."
  if (/^\d+\.\s/.test(line) && line.length < 120) {
    return { type: 'heading', level: 3, content: line.replace(/^\d+\.\s*/, '') };
  }

  // Bullet "* ..." or "- ..."
  if (/^[*\-•]\s+/.test(line)) {
    return { type: 'bullet', content: line.replace(/^[*\-•]\s+/, '') };
  }

  return { type: 'text', content: line };
}

function parseBlocks(text: string): Block[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const blocks: Block[] = [];
  let paragraphBuffer: string[] = [];
  let bulletBuffer: string[] = [];

  const flushParagraph = () => {
    if (paragraphBuffer.length > 0) {
      blocks.push({ kind: 'paragraph', content: paragraphBuffer.join(' ') });
      paragraphBuffer = [];
    }
  };
  const flushBullets = () => {
    if (bulletBuffer.length > 0) {
      blocks.push({ kind: 'bullet-group', items: [...bulletBuffer] });
      bulletBuffer = [];
    }
  };

  for (const raw of lines) {
    const info = classifyLine(raw);
    if (info.type === 'empty') {
      flushParagraph();
      flushBullets();
      continue;
    }
    if (info.type === 'heading') {
      flushParagraph();
      flushBullets();
      blocks.push({ kind: 'heading', level: info.level ?? 3, content: info.content });
      continue;
    }
    if (info.type === 'bullet') {
      flushParagraph();
      bulletBuffer.push(info.content);
      continue;
    }
    // text → accumulate into paragraph
    flushBullets();
    paragraphBuffer.push(info.content);
  }

  flushParagraph();
  flushBullets();
  return blocks;
}

/**
 * Render inline **bold** within a plain string.
 */
function renderInline(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const regex = /\*\*(.+?)\*\*/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <strong key={`b-${key++}`} className="font-semibold text-gray-900">
        {match[1]}
      </strong>
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

export default function PlannerReport({ text }: PlannerReportProps) {
  const blocks = parseBlocks(text);

  return (
    <article className="planner-report text-gray-800 leading-relaxed">
      {blocks.map((block, i) => {
        if (block.kind === 'heading') {
          if (block.level === 1) {
            return (
              <h2
                key={i}
                className="text-xl md:text-2xl font-bold text-gray-900 mt-6 mb-3 pb-2 border-b border-gray-200 first:mt-0"
              >
                {renderInline(block.content)}
              </h2>
            );
          }
          if (block.level === 2) {
            return (
              <h3
                key={i}
                className="text-base md:text-lg font-semibold text-gray-900 mt-5 mb-2"
              >
                {renderInline(block.content)}
              </h3>
            );
          }
          return (
            <h4
              key={i}
              className="text-sm md:text-base font-semibold text-gray-700 mt-4 mb-1.5"
            >
              {renderInline(block.content)}
            </h4>
          );
        }

        if (block.kind === 'paragraph') {
          return (
            <p key={i} className="text-sm md:text-base text-gray-700 my-3 text-justify">
              {renderInline(block.content)}
            </p>
          );
        }

        if (block.kind === 'bullet-group') {
          return (
            <ul key={i} className="my-3 space-y-2 pl-1">
              {block.items.map((item, j) => (
                <li
                  key={j}
                  className="text-sm md:text-base text-gray-700 pl-5 relative leading-relaxed"
                >
                  <span className="absolute left-0 top-[0.6em] w-1.5 h-1.5 rounded-full bg-gray-500" />
                  {renderInline(item)}
                </li>
              ))}
            </ul>
          );
        }

        return null;
      })}
    </article>
  );
}
