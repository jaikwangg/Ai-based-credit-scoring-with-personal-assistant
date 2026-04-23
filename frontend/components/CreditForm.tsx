'use client';

import { UseFormRegister, FieldErrors } from 'react-hook-form';
import { CreditInput } from '@/types/credit';

interface CreditFormProps {
  register: UseFormRegister<CreditInput>;
  errors: FieldErrors<CreditInput>;
}

// NOTE: UI labels are in Thai for end-user UX, but the `value=` attributes
// MUST exactly match the categorical labels the LGBM model was trained on
// (auto-detected from data/loan_dataset.csv):
//   Sex            → Male / Female
//   Marriage_Status→ Single / Married / Divorced
//   Occupation     → Salaried_Employee | Government_or_State_Enterprise |
//                    SME_Owner | Professional_Specialist | Freelancer_or_Self_Employed
//   credit_grade   → AA | BB | CC | DD | EE | FF | GG | HH  (8 buckets, not 4)
//   overdue        → days past due, integer in {0, 15, 30, 60, 90, 120}
// Sending labels outside these sets makes the OneHotEncoder output all-zero,
// which silently breaks predictions. Do not localise the value strings.
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
            <option value="Salaried_Employee">พนักงานเงินเดือน (เอกชน)</option>
            <option value="Government_or_State_Enterprise">ข้าราชการ / รัฐวิสาหกิจ</option>
            <option value="SME_Owner">เจ้าของกิจการ SME</option>
            <option value="Professional_Specialist">ผู้ประกอบวิชาชีพเฉพาะ (แพทย์ วิศวกร ทนาย ฯลฯ)</option>
            <option value="Freelancer_or_Self_Employed">ฟรีแลนซ์ / ประกอบอาชีพอิสระ</option>
          </select>
          {errors.Occupation && (
            <p className="mt-1 text-sm text-red-600">{errors.Occupation.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">
            หมวดอาชีพต้องตรงกับที่โมเดลเรียนรู้มาเท่านั้น (5 หมวด)
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
            <option value="AA">AA — ดีเยี่ยม</option>
            <option value="BB">BB — ดีมาก</option>
            <option value="CC">CC — ดี</option>
            <option value="DD">DD — พอใช้</option>
            <option value="EE">EE — ปานกลาง</option>
            <option value="FF">FF — ค่อนข้างต่ำ</option>
            <option value="GG">GG — ต่ำ</option>
            <option value="HH">HH — ต่ำมาก</option>
          </select>
          {errors.credit_grade && (
            <p className="mt-1 text-sm text-red-600">{errors.credit_grade.message}</p>
          )}
        </div>

        {/* outstanding — Total existing debt balance */}
        <div>
          <label htmlFor="outstanding" className="block text-sm font-medium text-gray-700 mb-1">
            ภาระหนี้สินรวมในปัจจุบัน (บาท) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="outstanding"
            {...register('outstanding', {
              required: 'กรุณาระบุภาระหนี้สินรวม',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'กรุณากรอกตัวเลขให้ถูกต้อง' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="เช่น 50000"
          />
          {errors.outstanding && (
            <p className="mt-1 text-sm text-red-600">{errors.outstanding.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">
            ยอดเงินต้นคงเหลือทุกสินเชื่อรวมกัน (บัตรเครดิต รถ สินเชื่อบุคคล ฯลฯ)
            ไม่ต้องรวมสินเชื่อบ้านที่กำลังขอใหม่ ใส่ 0 หากไม่มีหนี้
          </p>
        </div>

        {/* overdue — DAYS past due (model expects integer in {0,15,30,60,90,120}) */}
        <div>
          <label htmlFor="overdue" className="block text-sm font-medium text-gray-700 mb-1">
            จำนวนวันค้างชำระสูงสุดในประวัติ <span className="text-red-500">*</span>
          </label>
          <select
            id="overdue"
            {...register('overdue', { required: 'กรุณาเลือกจำนวนวันค้างชำระ' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">เลือก</option>
            <option value="0">0 วัน — ไม่เคยค้างชำระ</option>
            <option value="15">15 วัน — เคยค้างเล็กน้อย</option>
            <option value="60">60 วัน — เคยค้างปานกลาง</option>
            <option value="120">120 วัน — เคยค้างนาน</option>
          </select>
          {errors.overdue && (
            <p className="mt-1 text-sm text-red-600">{errors.overdue.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">
            จำนวนวันที่เคย <strong>ค้างชำระยาวที่สุด</strong> ในประวัติเครดิตที่ผ่านมา (มาตรฐานเครดิตบูโร)
          </p>
        </div>

        {/* loan_amount — training range 500,000 - 7,400,000 */}
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
            placeholder="เช่น 1500000"
          />
          {errors.loan_amount && (
            <p className="mt-1 text-sm text-red-600">{errors.loan_amount.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">
            500,000 – 7,400,000 บาท
          </p>
        </div>

        {/* loan_term — training range 20-30 years (was missing entirely!) */}
        <div>
          <label htmlFor="loan_term" className="block text-sm font-medium text-gray-700 mb-1">
            ระยะเวลาผ่อน (ปี) <span className="text-red-500">*</span>
          </label>
          <select
            id="loan_term"
            {...register('loan_term', { required: 'กรุณาเลือกระยะเวลาผ่อน' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">เลือก</option>
            {[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30].map((y) => (
              <option key={y} value={String(y)}>
                {y} ปี
              </option>
            ))}
          </select>
          {errors.loan_term && (
            <p className="mt-1 text-sm text-red-600">{errors.loan_term.message}</p>
          )}
        </div>

        {/* Interest_rate — training range 5.69 - 5.89%, narrow */}
        <div>
          <label htmlFor="Interest_rate" className="block text-sm font-medium text-gray-700 mb-1">
            อัตราดอกเบี้ย (%) <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            step="0.01"
            min="5.5"
            max="6.0"
            id="Interest_rate"
            defaultValue="5.75"
            {...register('Interest_rate', {
              required: 'กรุณาระบุอัตราดอกเบี้ย',
              pattern: { value: /^\d+(\.\d{1,2})?$/, message: 'กรุณากรอกเปอร์เซ็นต์ให้ถูกต้อง' }
            })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          {errors.Interest_rate && (
            <p className="mt-1 text-sm text-red-600">{errors.Interest_rate.message}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">
            5.69 – 5.89% (default 5.75%)
          </p>
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
