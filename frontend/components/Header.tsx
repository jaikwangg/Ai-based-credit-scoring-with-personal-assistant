'use client';

export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 py-4 md:py-6">
        <h1 className="text-2xl md:text-3xl font-bold text-gray-900">
          ระบบประเมินสินเชื่อด้วย AI
        </h1>
        <p className="text-sm md:text-base text-gray-600 mt-1">
          รับผลการประเมินเครดิตและคำแนะนำเฉพาะบุคคล พร้อมอ้างอิงเอกสารจริง
        </p>
      </div>
    </header>
  );
}
