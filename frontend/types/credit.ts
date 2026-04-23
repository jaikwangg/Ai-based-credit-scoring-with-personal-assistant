// CreditInput type definition - MUST MATCH EXACTLY
export interface CreditInput {
  Sex: string;
  Occupation: string;
  Salary: string;
  Marital_status: string;
  credit_score: string;
  credit_grade: string;
  outstanding: string;
  overdue: string;
  loan_amount: string;
  loan_term: string;
  Coapplicant: string;
  Interest_rate: string;
}

export interface CreditScore {
  score: number;
  grade: string;
  factors: string[];
}

export interface Plan {
  id: string;
  objective: string;
  timeframe: string;
  keyActions: string[];
  difficulty: string;
  estimatedScoreImpact: string;
  warnings?: string;
}

export interface ChecklistItem {
  id: string;
  task: string;
  completed: boolean;
}
