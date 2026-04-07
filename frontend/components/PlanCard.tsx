'use client';

import { Plan } from '@/types/credit';

interface PlanCardProps {
  plan: Plan;
  isSelected: boolean;
  onSelect: () => void;
}

export default function PlanCard({ plan, isSelected, onSelect }: PlanCardProps) {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Conservative':
        return 'bg-green-100 text-green-800';
      case 'Balanced':
        return 'bg-yellow-100 text-yellow-800';
      case 'Aggressive':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div
      className={`border-2 rounded-xl p-6 cursor-pointer transition-all ${
        isSelected
          ? 'border-primary-600 bg-primary-50 shadow-lg'
          : 'border-gray-200 bg-white hover:border-primary-300 hover:shadow-md'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-2xl font-bold text-gray-900">Plan {plan.id}</h3>
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold ${getDifficultyColor(
            plan.difficulty
          )}`}
        >
          {plan.difficulty}
        </span>
      </div>

      <div className="mb-4">
        <p className="text-sm font-medium text-gray-600 mb-1">Objective</p>
        <p className="text-gray-900">{plan.objective}</p>
      </div>

      <div className="mb-4">
        <p className="text-sm font-medium text-gray-600 mb-1">Timeframe</p>
        <p className="text-gray-900 font-semibold">{plan.timeframe}</p>
      </div>

      <div className="mb-4">
        <p className="text-sm font-medium text-gray-600 mb-2">Key Actions</p>
        <ul className="space-y-1">
          {plan.keyActions.map((action, index) => (
            <li key={index} className="text-sm text-gray-700 flex items-start">
              <span className="text-primary-600 mr-2">•</span>
              {action}
            </li>
          ))}
        </ul>
      </div>

      <div className="mb-4 p-3 bg-gray-50 rounded-lg">
        <p className="text-sm font-medium text-gray-600 mb-1">Estimated Score Impact</p>
        <p className="text-lg font-bold text-primary-600">{plan.estimatedScoreImpact}</p>
      </div>

      {plan.warnings && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">{plan.warnings}</p>
        </div>
      )}

      {isSelected && (
        <div className="mt-4 pt-4 border-t border-primary-200">
          <p className="text-sm font-semibold text-primary-600 text-center">
            ✓ Selected Plan
          </p>
        </div>
      )}
    </div>
  );
}
