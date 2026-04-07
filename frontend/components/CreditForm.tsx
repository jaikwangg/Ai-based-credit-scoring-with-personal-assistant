'use client';

import { UseFormRegister, FieldErrors } from 'react-hook-form';
import { CreditInput } from '@/types/credit';

interface CreditFormProps {
  register: UseFormRegister<CreditInput>;
  errors: FieldErrors<CreditInput>;
}

export default function CreditForm({ register, errors }: CreditFormProps) {
  return (
    <div className="space-y-6">
      <div className="mb-6">
        <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">
          Credit Score Assessment
        </h2>
        <p className="text-gray-600 text-sm md:text-base">
          Fill in your financial information to get your personalized credit score
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Sex */}
        <div>
          <label htmlFor="Sex" className="block text-sm font-medium text-gray-700 mb-1">
            Sex <span className="text-red-500">*</span>
          </label>
          <select
            id="Sex"
            {...register('Sex', { required: 'Sex is required' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
          {errors.Sex && (
            <p className="mt-1 text-sm text-red-600">{errors.Sex.message}</p>
          )}
        </div>

        {/* Marital_status */}
        <div>
          <label htmlFor="Marital_status" className="block text-sm font-medium text-gray-700 mb-1">
            Marital Status <span className="text-red-500">*</span>
          </label>
          <select
            id="Marital_status"
            {...register('Marital_status', { required: 'Marital status is required' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select</option>
            <option value="Single">Single</option>
            <option value="Married">Married</option>
            <option value="Divorced">Divorced</option>
            <option value="Widowed">Widowed</option>
          </select>
          {errors.Marital_status && (
            <p className="mt-1 text-sm text-red-600">{errors.Marital_status.message}</p>
          )}
        </div>

        {/* Occupation */}
        <div>
          <label htmlFor="Occupation" className="block text-sm font-medium text-gray-700 mb-1">
            Occupation <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="Occupation"
            {...register('Occupation', { required: 'Occupation is required' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="e.g., Software Engineer, Teacher, Business Owner"
          />
          {errors.Occupation && (
            <p className="mt-1 text-sm text-red-600">{errors.Occupation.message}</p>
          )}
        </div>

        {/* Salary */}
        <div>
          <label htmlFor="Salary" className="block text-sm font-medium text-gray-700 mb-1">
            Salary ($) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="Salary"
            {...register('Salary', {
              required: 'Salary is required',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'Please enter a valid number' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.Salary && (
            <p className="mt-1 text-sm text-red-600">{errors.Salary.message}</p>
          )}
        </div>

        {/* credit_score */}
        <div>
          <label htmlFor="credit_score" className="block text-sm font-medium text-gray-700 mb-1">
            Current Credit Score
          </label>
          <input
            type="text"
            id="credit_score"
            {...register('credit_score', {
              pattern: { value: /^\d+$/, message: 'Please enter a valid score (0-1000)' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0-1000"
          />
          {errors.credit_score && (
            <p className="mt-1 text-sm text-red-600">{errors.credit_score.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">Leave blank if unknown</p>
        </div>

        {/* credit_grade */}
        <div>
          <label htmlFor="credit_grade" className="block text-sm font-medium text-gray-700 mb-1">
            Credit Grade
          </label>
          <select
            id="credit_grade"
            {...register('credit_grade')}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select</option>
            <option value="Excellent">Excellent (A)</option>
            <option value="Good">Good (B)</option>
            <option value="Fair">Fair (C)</option>
            <option value="Poor">Poor (D/F)</option>
          </select>
          {errors.credit_grade && (
            <p className="mt-1 text-sm text-red-600">{errors.credit_grade.message}</p>
          )}
        </div>

        {/* outstanding */}
        <div>
          <label htmlFor="outstanding" className="block text-sm font-medium text-gray-700 mb-1">
            Outstanding Debt ($) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="outstanding"
            {...register('outstanding', {
              required: 'Outstanding debt is required',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'Please enter a valid number' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.outstanding && (
            <p className="mt-1 text-sm text-red-600">{errors.outstanding.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">Total amount you currently owe</p>
        </div>

        {/* overdue */}
        <div>
          <label htmlFor="overdue" className="block text-sm font-medium text-gray-700 mb-1">
            Overdue Amount ($) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="overdue"
            {...register('overdue', {
              required: 'Overdue amount is required',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'Please enter a valid number' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.overdue && (
            <p className="mt-1 text-sm text-red-600">{errors.overdue.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">Amount past due</p>
        </div>

        {/* loan_amount */}
        <div>
          <label htmlFor="loan_amount" className="block text-sm font-medium text-gray-700 mb-1">
            Loan Amount ($) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="loan_amount"
            {...register('loan_amount', {
              required: 'Loan amount is required',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'Please enter a valid number' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.loan_amount && (
            <p className="mt-1 text-sm text-red-600">{errors.loan_amount.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">Amount you&apos;re seeking or have</p>
        </div>

        {/* Interest_rate */}
        <div>
          <label htmlFor="Interest_rate" className="block text-sm font-medium text-gray-700 mb-1">
            Interest Rate (%) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="Interest_rate"
            {...register('Interest_rate', {
              required: 'Interest rate is required',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'Please enter a valid percentage' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.Interest_rate && (
            <p className="mt-1 text-sm text-red-600">{errors.Interest_rate.message}</p>
          )}
        </div>

        {/* Coapplicant */}
        <div>
          <label htmlFor="Coapplicant" className="block text-sm font-medium text-gray-700 mb-1">
            Co-applicant Available? <span className="text-red-500">*</span>
          </label>
          <select
            id="Coapplicant"
            {...register('Coapplicant', { required: 'Co-applicant status is required' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
          {errors.Coapplicant && (
            <p className="mt-1 text-sm text-red-600">{errors.Coapplicant.message}</p>
          )}
        </div>
      </div>

      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-xs text-gray-500 mb-4">
          Your input is sent to the scoring backend to generate model-based results and recommendations.
        </p>
      </div>
    </div>
  );
}
