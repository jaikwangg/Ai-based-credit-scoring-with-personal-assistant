'use client';

import { UseFormRegister, FieldErrors } from 'react-hook-form';
import { CreditInput } from '@/types/credit';

interface CreditFormProps {
  register: UseFormRegister<CreditInput>;
  errors: FieldErrors<CreditInput>;
}

// NOTE: UI labels are in Thai for end-user UX, but option values remain in
// English because the LGBM model was trained on an English dataset
// (e.g. Sex: Male/Female, Occupation: Engineer, Marital_status: Single/Married).
// Do not translate the `value=` attributes below.
export default function CreditForm({ register, errors }: CreditFormProps) {
  return (
    <div className="space-y-6">
      <div className="mb-6">
        <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">
          ประเมินคะแนนเครดิต
        </h2>
        <p className="text-gray-600 text-sm md:text-base">
          กรอกข้อมูลการเงินของคุณเพื่อรับผลการประเมินและคำแนะนำเฉพาะบุคคล
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Sex */}
        <div>
          <label htmlFor="Sex" className="block text-sm font-medium text-gray-700 mb-1">
            เพศ <span className="text-red-500">*</span>
          </label>
          <select
            id="Sex"
            {...register('Sex', { required: 'กรุณาเลือกเพศ' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">เลือก</option>
            <option value="Male">ชาย</option>
            <option value="Female">หญิง</option>
            <option value="Other">อื่น ๆ</option>
          </select>
          {errors.Sex && (
            <p className="mt-1 text-sm text-red-600">{errors.Sex.message}</p>
          )}
        </div>

        {/* Marital_status */}
        <div>
          <label htmlFor="Marital_status" className="block text-sm font-medium text-gray-700 mb-1">
            สถานภาพสมรส <span className="text-red-500">*</span>
          </label>
          <select
            id="Marital_status"
            {...register('Marital_status', { required: 'กรุณาเลือกสถานภาพสมรส' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">เลือก</option>
            <option value="Single">โสด</option>
            <option value="Married">สมรส</option>
            <option value="Divorced">หย่าร้าง</option>
            <option value="Widowed">หม้าย</option>
          </select>
          {errors.Marital_status && (
            <p className="mt-1 text-sm text-red-600">{errors.Marital_status.message}</p>
          )}
        </div>

        {/* Occupation */}
        <div>
          <label htmlFor="Occupation" className="block text-sm font-medium text-gray-700 mb-1">
            อาชีพ <span className="text-red-500">*</span>
          </label>
          <select
            id="Occupation"
            {...register('Occupation', { required: 'กรุณาเลือกอาชีพ' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">เลือก</option>
            <option value="Engineer">วิศวกร</option>
            <option value="Teacher">ครู / อาจารย์</option>
            <option value="Doctor">แพทย์ / พยาบาล</option>
            <option value="Government Officer">ข้าราชการ</option>
            <option value="Business Owner">เจ้าของกิจการ</option>
            <option value="Office Worker">พนักงานบริษัท</option>
            <option value="Freelancer">ฟรีแลนซ์</option>
            <option value="Student">นักเรียน / นักศึกษา</option>
            <option value="Other">อื่น ๆ</option>
          </select>
          {errors.Occupation && (
            <p className="mt-1 text-sm text-red-600">{errors.Occupation.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">
            ค่าที่ส่งให้โมเดลเป็นภาษาอังกฤษเพื่อความเข้ากันได้
          </p>
        </div>

        {/* Salary */}
        <div>
          <label htmlFor="Salary" className="block text-sm font-medium text-gray-700 mb-1">
            รายได้ต่อเดือน (บาท) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="Salary"
            {...register('Salary', {
              required: 'กรุณาระบุรายได้',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'กรุณากรอกตัวเลขให้ถูกต้อง' }
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
            คะแนนเครดิตปัจจุบัน
          </label>
          <input
            type="text"
            id="credit_score"
            {...register('credit_score', {
              pattern: { value: /^\d+$/, message: 'กรุณากรอกคะแนนที่ถูกต้อง (0-1000)' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0-1000"
          />
          {errors.credit_score && (
            <p className="mt-1 text-sm text-red-600">{errors.credit_score.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">เว้นว่างได้หากไม่ทราบ</p>
        </div>

        {/* credit_grade */}
        <div>
          <label htmlFor="credit_grade" className="block text-sm font-medium text-gray-700 mb-1">
            เกรดเครดิต
          </label>
          <select
            id="credit_grade"
            {...register('credit_grade')}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">เลือก</option>
            <option value="Excellent">ดีเยี่ยม (A)</option>
            <option value="Good">ดี (B)</option>
            <option value="Fair">ปานกลาง (C)</option>
            <option value="Poor">ต่ำ (D/F)</option>
          </select>
          {errors.credit_grade && (
            <p className="mt-1 text-sm text-red-600">{errors.credit_grade.message}</p>
          )}
        </div>

        {/* outstanding */}
        <div>
          <label htmlFor="outstanding" className="block text-sm font-medium text-gray-700 mb-1">
            หนี้คงค้าง (บาท) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="outstanding"
            {...register('outstanding', {
              required: 'กรุณาระบุหนี้คงค้าง',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'กรุณากรอกตัวเลขให้ถูกต้อง' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.outstanding && (
            <p className="mt-1 text-sm text-red-600">{errors.outstanding.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">ยอดหนี้รวมที่ยังค้างอยู่ในปัจจุบัน</p>
        </div>

        {/* overdue */}
        <div>
          <label htmlFor="overdue" className="block text-sm font-medium text-gray-700 mb-1">
            ยอดค้างชำระ (บาท) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="overdue"
            {...register('overdue', {
              required: 'กรุณาระบุยอดค้างชำระ',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'กรุณากรอกตัวเลขให้ถูกต้อง' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.overdue && (
            <p className="mt-1 text-sm text-red-600">{errors.overdue.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">ยอดที่เลยกำหนดชำระ (ถ้าไม่มีใส่ 0)</p>
        </div>

        {/* loan_amount */}
        <div>
          <label htmlFor="loan_amount" className="block text-sm font-medium text-gray-700 mb-1">
            วงเงินขอกู้ (บาท) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="loan_amount"
            {...register('loan_amount', {
              required: 'กรุณาระบุวงเงินขอกู้',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'กรุณากรอกตัวเลขให้ถูกต้อง' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="0.00"
          />
          {errors.loan_amount && (
            <p className="mt-1 text-sm text-red-600">{errors.loan_amount.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">จำนวนเงินที่ต้องการขอกู้</p>
        </div>

        {/* Interest_rate */}
        <div>
          <label htmlFor="Interest_rate" className="block text-sm font-medium text-gray-700 mb-1">
            อัตราดอกเบี้ย (%) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="Interest_rate"
            {...register('Interest_rate', {
              required: 'กรุณาระบุอัตราดอกเบี้ย',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'กรุณากรอกเปอร์เซ็นต์ให้ถูกต้อง' }
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
            มีผู้กู้ร่วมหรือไม่ <span className="text-red-500">*</span>
          </label>
          <select
            id="Coapplicant"
            {...register('Coapplicant', { required: 'กรุณาเลือกสถานะผู้กู้ร่วม' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">เลือก</option>
            <option value="Yes">มี</option>
            <option value="No">ไม่มี</option>
          </select>
          {errors.Coapplicant && (
            <p className="mt-1 text-sm text-red-600">{errors.Coapplicant.message}</p>
          )}
        </div>
      </div>

      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-xs text-gray-500 mb-4">
          ข้อมูลของคุณจะถูกส่งไปยังระบบประเมินเพื่อสร้างผลลัพธ์และคำแนะนำจากโมเดล
        </p>
      </div>
    </div>
  );
}
