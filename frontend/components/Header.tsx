'use client';

export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 py-4 md:py-6">
        <h1 className="text-2xl md:text-3xl font-bold text-gray-900">
          Credit Scoring App
        </h1>
        <p className="text-sm md:text-base text-gray-600 mt-1">
          Get your personalized credit score and recommendations
        </p>
      </div>
    </header>
  );
}
