'use client';

import { ChecklistItem } from '@/types/credit';

interface ChecklistProps {
  items: ChecklistItem[];
  onToggle: (id: string) => void;
}

export default function Checklist({ items, onToggle }: ChecklistProps) {
  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4">Weekly Checklist</h3>
      <div className="space-y-3">
        {items.map((item) => (
          <div
            key={item.id}
            className="flex items-center p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer"
            onClick={() => onToggle(item.id)}
          >
            <input
              type="checkbox"
              checked={item.completed}
              onChange={() => onToggle(item.id)}
              className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500 cursor-pointer"
              aria-label={item.task}
            />
            <label
              className={`ml-3 flex-1 cursor-pointer ${
                item.completed ? 'line-through text-gray-500' : 'text-gray-900'
              }`}
            >
              {item.task}
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}
