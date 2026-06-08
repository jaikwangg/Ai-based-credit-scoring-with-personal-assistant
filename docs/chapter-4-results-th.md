# บทที่ 4 ผลการทดลองและการวิเคราะห์ผล

> บทนี้อ้างอิงเฉพาะข้อมูลที่มีหลักฐานรันจริงใน repo: ไฟล์ใน `Ai-Credit-Scoring/results/`, `Ai-Credit-Scoring/data/eval/`, `Ai-Credit-Scoring/logs/`, `Ai-Credit-Scoring/reports/`, `Ai-Credit-Scoring/data/training/`, `Ai-Credit-Scoring/data/cleaning_report.json` และโค้ดใน `backend/app/` กับ `Ai-Credit-Scoring/src/` หัวข้อใดที่ยังไม่มีผลรันใน repo จะถูกระบุ "ยังไม่มีข้อมูลใน repo" พร้อมคำแนะนำว่าควรรันอะไรเพิ่ม

---

## 4.1 ผลการทดลองของแบบจำลอง

### 4.1.1 คลังข้อมูลฝึก (Training Sample)

ไฟล์ [`Ai-Credit-Scoring/data/training/loan_dataset_sample.csv`](../Ai-Credit-Scoring/data/training/loan_dataset_sample.csv) เป็น sample ของชุดฝึก 500 แถว แสดงคุณลักษณะของประชากรที่ใช้ฝึกโมเดล

**สมดุลคลาสและข้อมูลเชิงประชากร**

| หมวด | ค่า |
|---|---|
| จำนวนแถว | 500 |
| สมดุลคลาส (`loan_status`) | Approved 245 (49.0%) / Rejected 255 (51.0%) |
| Sex | Male 254 / Female 246 |
| Marriage_Status | Married 272 / Single 181 / Divorced 47 |
| Coapplicant | มี 176 (35.2%) / ไม่มี 324 (64.8%) |
| credit_grade (AA–HH) | AA 137, BB 50, CC 44, DD 43, EE 38, FF 53, GG 49, HH 86 |
| Occupation | Salaried 196, Gov/SOE 94, SME 78, Professional 57, Freelancer 75 |

**การกระจายตัวของฟีเจอร์ตัวเลข**

| Feature | min | max | mean | median |
|---|---|---|---|---|
| Salary (บาท/เดือน) | 21,235 | 269,777 | 68,655.9 | 56,404.5 |
| credit_score | 434 | 900 | 695.7 | 690.5 |
| outstanding (บาท) | 12,654 | 3,633,845 | 763,910.5 | 642,953.0 |
| overdue (วัน) | 0 | 120 | 17.6 | 0.0 |
| loan_amount (บาท) | 500,000 | 7,400,000 | 1,338,800 | 1,000,000 |
| loan_term (ปี) | 20 | 30 | 25.5 | 26.0 |
| Interest_rate (%) | 5.70 | 5.90 | 5.80 | 5.80 |

ข้อสังเกต: สมดุลคลาสใกล้ 50:50 ไม่มีปัญหา class imbalance รุนแรง และ median ของ `overdue = 0` แสดงว่าผู้กู้ส่วนใหญ่ไม่มีประวัติค้างชำระ

### 4.1.2 สถาปัตยกรรมและฟีเจอร์ของโมเดลสุดท้าย

โมเดลที่ serialize ไว้ที่ [`backend/model/lgbm_model.pkl`](../backend/model/lgbm_model.pkl) เป็น **LightGBM Classifier** โหลดผ่าน [`backend/app/predict.py`](../backend/app/predict.py) ท่อประมวลผลใน [`backend/app/pipeline.py`](../backend/app/pipeline.py) ประกอบด้วย

- **Categorical (4)**: `Sex`, `Occupation`, `Marriage_Status`, `credit_grade` → OneHotEncoder
- **Numeric ตั้งต้น (8)**: `Salary`, `credit_score`, `outstanding`, `overdue`, `Coapplicant`, `loan_amount`, `loan_term`, `Interest_rate` → StandardScaler
- **Engineered (3)**: `dti = outstanding/Salary`, `lti = loan_amount/Salary`, `has_overdue = 1 if overdue > 0 else 0`

รวม **15 features ก่อน one-hot expansion** (หลัง OHE categorical features จะขยายเป็น ~30 columns ตามจำนวน unique values)

### 4.1.3 ตัวอย่างผลทำนาย (Serialized Output Examples)

จาก [`examples/planner/model_approved.json`](../Ai-Credit-Scoring/examples/planner/model_approved.json) และ [`model_rejected.json`](../Ai-Credit-Scoring/examples/planner/model_rejected.json)

| เคส | prediction | P(class=0) | P(class=1) |
|---|---|---|---|
| Approved | 1 | 0.12 | 0.88 |
| Rejected | 0 | 0.72 | 0.28 |

### 4.1.4 ตัวเลขประเมินโมเดล (accuracy / F1 / ROC-AUC / CM)

**ยังไม่มีข้อมูลใน repo** — ไม่พบ classification report, metric log, หรือ confusion matrix ของโมเดล LightGBM ในคลังผล

> **แนะนำให้รันเพิ่ม**: สคริปต์ `train_and_evaluate.py` ที่โหลด `loan_dataset_sample.csv` แบ่ง 70/15/15 stratified + fit LightGBM + รายงาน accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix บันทึกเป็น `results/model_metrics.json`

---

## 4.2 การเปรียบเทียบประสิทธิภาพของแบบจำลอง

**ยังไม่มีข้อมูลใน repo** — ไม่พบ benchmark เปรียบเทียบ LightGBM กับโมเดลอื่น (Logistic Regression / Random Forest / XGBoost / MLP)

> **แนะนำให้รันเพิ่ม**: fit โมเดล 4–5 ตัวบน `loan_dataset_sample.csv` ด้วย preprocessing เดียวกัน ตัวเลขที่น่ารายงาน: accuracy, F1, ROC-AUC, training time, inference time

---

## 4.3 ผลของ Feature Engineering และ Hyperparameter Tuning

### 4.3.1 Feature Engineering (เชิงโครงสร้าง)

พบใน [`backend/app/pipeline.py`](../backend/app/pipeline.py) ฟังก์ชัน `apply_feature_engineering` สร้างฟีเจอร์เพิ่ม 3 ตัว (`dti`, `lti`, `has_overdue`) ที่ใช้ในโมเดลสุดท้าย

**ยังไม่มีข้อมูลเชิงปริมาณ** — ไม่มีการบันทึก ablation เปรียบเทียบ metric ก่อน/หลัง engineering

### 4.3.2 Hyperparameter Tuning

**ยังไม่มีข้อมูลใน repo** — ไม่พบ Optuna study file หรือ grid search log

> **แนะนำให้รันเพิ่ม**: Optuna study 50–100 trials กับ CV 5-fold + บันทึก `best_params` และ CV score

---

## 4.4 ผลของการปรับ Threshold

**ยังไม่มีข้อมูลใน repo**

> **แนะนำให้รันเพิ่ม**: threshold sweep 0.30–0.70 step 0.05 บันทึก precision/recall/F1/expected cost (ใช้ FN:FP weight 4:1 ตามแนวทางสถาบันการเงิน)

---

## 4.5 การวิเคราะห์ Confusion Matrix และข้อผิดพลาด

**ยังไม่มีข้อมูลใน repo**

> **แนะนำให้รันเพิ่ม**: classification report + confusion matrix heatmap + error analysis รายเคส (segmented by occupation, credit_grade, loan_term)

---

## 4.6 ผลการอธิบายด้วย SHAP

### 4.6.1 การทำงานของโมดูล SHAP

โมดูลอยู่ที่ [`backend/app/explain.py`](../backend/app/explain.py) ใช้ `TreeExplainer` ของ SHAP ต่อ LightGBM ส่งคืน structure

```
{
  "base_value": float,
  "shap_values": { feature: value, ... },
  "log_odds": float,
  "probability": float (sigmoid of log_odds)
}
```

โดยรวม one-hot features ทั้งหมดที่มีต้นกำเนิดจาก `Occupation`, `credit_grade`, `Sex`, `Marriage_Status` กลับเป็นชื่อเดิม แล้วรวม SHAP value ก่อนส่งกลับ

### 4.6.2 ตัวอย่าง SHAP ที่บันทึกไว้

จาก [`examples/planner/shap.json`](../Ai-Credit-Scoring/examples/planner/shap.json) เคสตัวอย่าง (base_value = 0.5)

| Feature | SHAP value | ทิศทาง |
|---|---|---|
| Salary | +0.18 | ช่วยเพิ่มโอกาสอนุมัติ |
| outstanding | −0.35 | ลดโอกาสอนุมัติมากที่สุด |
| overdue | −0.22 | ลดโอกาสอนุมัติ |
| loan_amount | −0.15 | ลดโอกาสอนุมัติ |
| credit_score | −0.10 | ลดโอกาสอนุมัติ |
| loan_term | −0.05 | ลดโอกาสอนุมัติ |
| Interest_rate | −0.03 | เกือบกลาง |
| credit_grade | −0.02 | เกือบกลาง |
| Sex | −0.01 | เกือบกลาง (Non-actionable) |
| Occupation | +0.01 | เกือบกลาง |
| Coapplicant | +0.01 | เกือบกลาง |
| Marriage_Status | 0.00 | กลาง |

เคสนี้ `outstanding` และ `overdue` เป็น 2 drivers เชิงลบหลัก ขณะที่ `Salary` เป็น driver เชิงบวกเด่น ตัวอย่างนี้สะท้อนการทำงานของ `DRIVER_QUERY_MAP` (ดูหัวข้อ 3.8.11) — drivers `outstanding`/`overdue` จะถูกแปลเป็น queries ด้าน hardship_support/debt restructuring

### 4.6.3 Global SHAP Importance

**ยังไม่มีข้อมูลใน repo**

> **แนะนำให้รันเพิ่ม**: สคริปต์ `compute_global_shap.py` โหลด `lgbm_model.pkl` + test set → คำนวณ `mean(|SHAP|)` ต่อ feature และบันทึกเป็น `results/global_shap.json`

---

## 4.7 ผลการทำงานของระบบ RAG

### 4.7.1 คลังข้อมูลและ Ingestion

จาก [`results/ingest_log3.txt`](../Ai-Credit-Scoring/results/ingest_log3.txt)

| ตัวชี้วัด | ค่า |
|---|---|
| เอกสารต้นทาง (parse) | 36 |
| เอกสารที่ index | 25 |
| เอกสารที่ถูก quarantine | 11 (30.6%) |
| Embedding model | BAAI/bge-m3 (1,024 dim) |
| Chroma collection | `cimb_loans_bge_m3` |
| CLEANING_VERSION fingerprint | `2026-03-04-v1` |

**เหตุผลการ quarantine (Top 5)**

| Reason code | จำนวนเอกสาร |
|---|---|
| no_home_loan_keywords_or_url_hints | 4 |
| high_chrome_noise_ratio | 2 |
| unrelated_navigation_content | 2 |
| unrelated_title_topic | 1 |
| negative_indicators_dominate | 1 |
| อื่นๆ (non_top_reason) | 1 |

จาก [`data/cleaning_report.json`](../Ai-Credit-Scoring/data/cleaning_report.json) ยังมีการแก้ไข
- ลบ 5 เอกสารที่ไม่เกี่ยวกับสินเชื่อบ้าน (privacy notice, customer profiling, COVID relief 2020–2021)
- แก้ publication date ผิดพลาด 2 รายการ (จาก crawl date → actual date)
- แก้ category ที่จัดผิด 3 รายการ (เช่น `home-loan-refinance-th`: bank_policy → interest_structure)

### 4.7.2 Document Audit Score

จาก [`reports/doc_audit_summary.txt`](../Ai-Credit-Scoring/reports/doc_audit_summary.txt) และ [`doc_audit.csv`](../Ai-Credit-Scoring/reports/doc_audit.csv)

| ตัวชี้วัด | ค่า |
|---|---|
| เอกสารที่ถูก audit | 29 |
| เอกสารที่ `needs_review=True` | 11 (37.9%) |
| Top severity doc | `tcf-commitment-th.txt` (severity 0.227, duplicate 0.5049) |
| Mean duplicate_boilerplate_score ของ top-10 needs_review | ≈ 0.49 |

คะแนน duplicate ที่สูง ~0.49 สะท้อนว่าเอกสารจาก web scraping มี boilerplate repeat สูง ซึ่งเป็นเหตุผลทางวิศวกรรมว่าทำไมต้องมี validator ชั้นที่สอง (หัวข้อ 3.8.7)

### 4.7.3 ผลรวม RAG บน 118 test cases

ทดสอบด้วย 2 LLM providers บนชุดเดียวกัน จาก [`compare_20260320_164805.json`](../Ai-Credit-Scoring/results/compare_20260320_164805.json), [`eval_gemini.json`](../Ai-Credit-Scoring/results/eval_gemini.json), [`eval_ollama.json`](../Ai-Credit-Scoring/results/eval_ollama.json)

| Metric | Gemini 2.5 Flash (eval_gemini.json) | Ollama qwen3:8b (eval_ollama.json) |
|---|---|---|
| Total cases | 118 | 118 |
| Status — PASS / WARN / FAIL | **73 / 32 / 13** | 64 / 25 / 29 |
| Overall Pass Rate (checks) | **90.78%** (620/683) | 85.32% (587/688) |
| Router Accuracy | **88.14%** (104/118) | 72.03% (85/118) |
| Answer Rate | 80.00% (76/95) | **83.16%** (79/95) |
| Mean Precision@K | **0.857** (n=98) | 0.689 (n=98) |
| Mean Top-1 similarity | 0.487 | 0.457 |
| Mean Latency (วินาที) | **5.53** | 16.07 |

**ข้อสังเกตหลัก**
- Gemini นำหน้าในทุกมิติยกเว้น Answer Rate (Ollama ยอมตอบบ่อยกว่า แต่ได้ P@K ต่ำกว่า)
- Latency ของ Ollama สูงกว่า Gemini ประมาณ **2.9 เท่า**
- Router accuracy ห่างกัน **16 จุด** (88.14% vs 72.03%)

### 4.7.4 ผลรายเส้นทาง (Gemini, 118 เคส)

คำนวณจาก case-level ใน [`eval_gemini.json`](../Ai-Credit-Scoring/results/eval_gemini.json)

| Route | n | PASS | WARN | FAIL | Router acc | Mean lat (s) | Mean P@K | Mean Top-1 | Mean retrieved | Mean validated |
|---|---|---|---|---|---|---|---|---|---|---|
| hardship_support | 18 | 16 | 1 | 1 | 0.94 | 6.41 | 0.659 | 0.511 | 30.0 | 22.8 |
| policy_requirement | 19 | 8 | 7 | 4 | 0.84 | 4.89 | 0.887 | 0.497 | 30.0 | 9.8 |
| interest_structure | 20 | 15 | 5 | 0 | 1.00 | 7.95 | 0.945 | 0.497 | 30.0 | 27.8 |
| general_info | 20 | 13 | 0 | 7 | 0.75 | 5.70 | — | 0.422 | 29.3 | 20.1 |
| refinance | 19 | 9 | 9 | 1 | 0.74 | 5.79 | 0.755 | 0.494 | 16.7 | 16.1 |
| fee_structure | 22 | 12 | 10 | 0 | 1.00 | 2.79 | 1.000 | 0.503 | 16.0 | 13.0 |

**Route ที่เด่นที่สุด** — `interest_structure` (Router acc 1.00, 0 FAIL) และ `fee_structure` (Router acc 1.00, P@K 1.000, 0 FAIL) ซึ่งเป็น routes ที่ keyword ชัดเจน (มี %, MRR, ค่าธรรมเนียม)

**Route ที่อ่อนที่สุด** — `general_info` (7/20 FAIL, top-1 similarity เพียง 0.422) และ `refinance` (Router acc 0.74 เพราะคำถามมักคาบเกี่ยวกับ `interest_structure`)

### 4.7.5 ผลรายเส้นทาง (Ollama, 118 เคส)

| Route | n | PASS | WARN | FAIL | Router acc | Mean lat (s) | Mean P@K | Mean Top-1 |
|---|---|---|---|---|---|---|---|---|
| hardship_support | 18 | 10 | 5 | 3 | 0.72 | 18.88 | 0.722 | 0.473 |
| policy_requirement | 19 | 6 | 6 | 7 | **0.32** | 14.76 | 0.458 | 0.510 |
| interest_structure | 20 | 17 | 2 | 1 | 0.95 | 27.45 | 0.674 | 0.495 |
| general_info | 20 | 10 | 0 | 10 | 0.80 | 13.13 | — | 0.424 |
| refinance | 19 | 10 | 4 | 5 | 0.58 | 19.86 | 0.652 | 0.475 |
| fee_structure | 22 | 11 | 8 | 3 | 0.91 | 3.94 | 0.909 | 0.378 |

**ข้อสังเกต** — Ollama ได้ Router accuracy เพียง **0.32** ใน `policy_requirement` (เทียบ Gemini 0.84) ห่างกัน 52 จุด ซึ่งเป็นโดเมนที่คำถามมีความทับซ้อนสูง (คุณสมบัติ / เอกสาร / เงื่อนไข) สะท้อนข้อจำกัดของ local LLM ขนาด 8B ในการจัดประเภทคำถามเชิงนามธรรม

### 4.7.6 Similarity Distribution (54 sample queries)

จาก [`report.txt`](../Ai-Credit-Scoring/report.txt) / [`report.csv`](../Ai-Credit-Scoring/report.csv)

| Metric | ค่า |
|---|---|
| Total queries | 54 |
| Mean top-1 similarity | 0.4439 |
| Mean top-K similarity | 0.4197 |
| Mean top-1 vs top-2 gap | 0.0083 |
| No-answer count | 12 (22.2%) |

**ค่าที่สำคัญเชิง architectural**
- Mean top-1 = 0.4439 ใกล้ `SIMILARITY_CUTOFF` = 0.45 มาก → เคสจำนวนมากอยู่บนขอบ cutoff การปรับ threshold เล็กน้อย (0.40 หรือ 0.50) จะเปลี่ยน answer rate อย่างมีนัยสำคัญ
- Top-1/Top-2 gap = 0.0083 → เอกสารอันดับ 1 และ 2 คะแนนใกล้กันมาก validator ชั้น 2 ที่กรองด้วย rule จึงสำคัญ
- No-answer 22.2% สูงกว่าที่คาดแต่เป็นผลดีในแง่ safety (ไม่บังคับตอบเมื่อไม่มี context)

### 4.7.7 Production-style Retrieval Log

จาก [`logs/rag_debug.jsonl`](../Ai-Credit-Scoring/logs/rag_debug.jsonl) (1,425 บรรทัด) และ [`logs/retrieval_logs.jsonl`](../Ai-Credit-Scoring/logs/retrieval_logs.jsonl) (1,533 บรรทัด)

| ตัวชี้วัด | ค่า |
|---|---|
| คำถามไม่ซ้ำ (filter "Test question") | 339 |
| Retrieval events ที่มีบริบท | 1,369 |
| Retrieval events ที่ context ว่างเปล่า | 36 |
| Mean top-1 (production logs) | 0.4688 |

**Router label distribution ใน production logs**

| Route | count |
|---|---|
| policy_requirement | 465 |
| general_info | 238 |
| hardship_support | 218 |
| interest_structure | 216 |
| fee_structure | 162 |
| refinance | 106 |

สัดส่วนนี้สะท้อนพฤติกรรมการถามของผู้ใช้ทดสอบจริงตลอดระยะเวลาพัฒนา `policy_requirement` เป็นโดเมนที่ถามบ่อยสุด (~33%) ซึ่งตรงกับข้อสังเกตใน 4.7.5 ว่าเป็นโดเมนที่ Ollama อ่อนที่สุด จึงเป็นจุดที่ต้องปรับปรุงก่อนหากเลือกใช้ Ollama ใน production

---

## 4.8 ผลการทำงานของ Planning Module

### 4.8.1 ผลรวม Planning (25 เคส)

จาก [`compare_20260320_164805.json`](../Ai-Credit-Scoring/results/compare_20260320_164805.json)

| Metric | Gemini | Ollama |
|---|---|---|
| Total cases | 25 | 25 |
| Overall Pass Rate (checks) | 100% (252/252) | 100% (252/252) |
| Mode Accuracy (approved_guidance vs improvement_plan) | 100% | 100% |
| Documented Evidence Rate | 100% | 100% |
| IsSupported Pass Rate | 100% | 100% |
| **Mean IsSupported Score (1–5)** | **4.83** | 2.67 |
| Mean Latency (วินาที) | 20.75 | 85.94 |

**ข้อสังเกต** — ทั้งสอง provider ผ่านเกณฑ์โครงสร้าง (mode / evidence / IsSup pass) ที่ 100% แต่คุณภาพการสังเคราะห์ภาษา (IsSup score) ห่างกัน **2.16 คะแนน** และ latency ของ Ollama สูงกว่า Gemini **4.14 เท่า**

> **หมายเหตุการแยกตัวชี้วัด IsSupported** — ระบบมีสองตัวชี้วัดที่ใช้ชื่อใกล้เคียงกันแต่วัดคนละมิติ
> - **IsSupported Pass Rate** (binary per case) — สัดส่วนเคสที่ผ่านการตรวจ Self-RAG `[IsSup]` reflection ซึ่งตอบ yes/no ว่าคำตอบถูก support โดย context ทั้งสอง provider ได้ 100% ตามเกณฑ์ของหัวข้อ 3.10.4
> - **IsSupported Score (1–5 mean)** — คะแนน groundedness จาก LLM-as-Judge เฉลี่ยต่อเคส Gemini ได้ 4.83 (ผ่านเกณฑ์ ≥ 3.00) และ Ollama ได้ 2.67 (**ต่ำกว่าเกณฑ์ที่ตั้งไว้**)
>
> ความไม่สอดคล้องของ Ollama ระหว่าง Pass Rate 100% กับ mean Score 2.67 สะท้อนว่า reflection แบบ binary (ผ่าน/ไม่ผ่าน) ของ LLM ขนาด 8B มีแนวโน้ม "ผ่อนปรน" ขณะที่ graded score (1–5) แสดงคุณภาพจริงชัดกว่า จุดนี้จะถูกอภิปรายต่อในบทที่ 5 §5.2.4

### 4.8.2 ผลรายเคส (Gemini) จาก [`eval_run_log.txt`](../Ai-Credit-Scoring/results/eval_run_log.txt)

24/25 เคสได้สถานะ PASS และ 1/25 ได้ WARN

**เคส PASS ที่มี IsSup = 5/5** (5 เคส)
- Rejected+IsSup: LLM plan must pass groundedness
- Rejected: high overdue single driver (minimal actions)
- Rejected+IsSup: interest rate driver
- Rejected: high outstanding relative to salary (DTI)
- Safety+IsSup: borderline profile with Sex feature

**เคส PASS ที่มี IsSup = 4/5** (1 เคส)
- Rejected+IsSup: all five main drivers negative

**เคส WARN** (1 เคส)
- `Rejected: credit_grade sole driver (EE grade)` — D1=67%, D2=67%, D3=D4=100%, 9/11 checks passed, latency 12.8 s
- สาเหตุ: เป็นเคสที่ driver เชิงลบมีเพียง `credit_grade` ตัวเดียว ซึ่งยากต่อการสังเคราะห์ action ที่ actionable (ตาม `NON_ACTIONABLE_FEATURES` logic)

### 4.8.3 Advisor Ablation Study (A1 / A2 / A3)

ทดสอบบน 100 planned cases (10 profiles × 10 questions) เสร็จสมบูรณ์ 83 records จาก [`data/eval/advisor_results.jsonl`](../Ai-Credit-Scoring/data/eval/advisor_results.jsonl) และ [`ablation_table.csv`](../Ai-Credit-Scoring/data/eval/ablation_table.csv)

| Approach | N | Latency (s) | Checks | Pass | Fail | Unknown | Actions | Sources | Keyword Recall | Verdict Acc. | IsSup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1** — Profile-conditioned single-hop | 25 | 41.0 ± 19.2 | 5.32 | 3.48 | 0.48 | 1.12 | 3.36 | 9.12 | **0.90** (n=24) | 0.56 (n=18) | — |
| **A2** — A1 + Multi-hop decomposition | 27 | 64.9 ± 20.8 | 5.41 | 3.37 | 0.52 | 1.37 | 3.48 | 9.04 | 0.85 (n=25) | **0.68** (n=19) | — |
| **A3** — A2 + Self-RAG reflection | 31 | 85.6 ± 21.6 | 5.39 | 3.00 | 0.52 | 1.52 | 3.55 | 8.68 | 0.79 (n=30) | 0.46 (n=24) | **3.30** (n=23) |

**สรุปผล ablation**
- **Keyword Recall**: A1 (0.90) > A2 (0.85) > A3 (0.79) — ลดลงเมื่อเพิ่ม reflection
- **Verdict Accuracy**: A2 (0.68) > A1 (0.56) > A3 (0.46) — Multi-hop ช่วย ส่วน Self-RAG กลับทำให้แย่ลง
- **Latency**: A1 < A2 < A3 (41.0 / 64.9 / 85.6 s) — A3 ใช้เวลาเกือบ 2.1 เท่าของ A1
- **IsSup score (เฉพาะ A3 ที่วัด)**: mean 3.30 / 5 บน 23 เคสที่มี IsSup

### 4.8.4 Ablation รายประเภทคำถาม

จาก [`ablation_table.md`](../Ai-Credit-Scoring/data/eval/ablation_table.md)

**คำถามเชิงคำแนะนำ (advice)**

| Approach | N | Latency | Pass | Actions | Keyword Recall | IsSup |
|---|---|---|---|---|---|---|
| A1 | 3 | 36.6 ± 7.2 | 4.00 | 3.67 | 1.00 | — |
| A2 | 4 | 68.3 ± 30.7 | 4.00 | 3.50 | 1.00 | — |
| A3 | 3 | 89.0 ± 15.1 | 4.00 | 4.00 | 1.00 | 3.00 |

**คำถามเชิงข้อเท็จจริง (factual)**

| Approach | N | Latency | Pass | Keyword Recall | IsSup |
|---|---|---|---|---|---|
| A1 | 4 | 39.5 ± 18.8 | 3.50 | 0.92 | — |
| A2 | 4 | 55.9 ± 6.3 | 3.50 | 0.83 | — |
| A3 | 4 | 73.3 ± 11.0 | 3.00 | 0.58 | 3.67 |

**Multi-hop eligibility**

| Approach | N | Latency | Pass | Verdict Acc | IsSup |
|---|---|---|---|---|---|
| A1 | 9 | 46.4 ± 26.7 | 2.89 | 0.67 | — |
| A2 | 8 | 70.7 ± 21.4 | 3.12 | 0.62 | — |
| A3 | 10 | 84.8 ± 24.9 | 3.30 | 0.50 | 3.89 |

**Single-hop eligibility**

| Approach | N | Latency | Pass | Verdict Acc | IsSup |
|---|---|---|---|---|---|
| A1 | 9 | 37.8 ± 9.8 | 3.89 | 0.44 | — |
| A2 | 11 | 62.6 ± 17.5 | 3.27 | **0.73** | — |
| A3 | 14 | 89.0 ± 21.2 | 2.57 | 0.43 | 2.67 |

**ข้อค้นพบจาก breakdown**
- `single_eligibility` เป็นประเภทที่ A2 ได้เปรียบ A1 สูงสุด (+29 จุด verdict accuracy)
- `factual` มี keyword recall ลดลงต่อเนื่องเมื่อเพิ่ม reflection (A1 0.92 → A3 0.58) — Self-RAG เข้มเกินไปในการกรอง context ทำให้ขาดคำสำคัญ
- `multi_eligibility` เป็นประเภทที่ A3 ได้ IsSup สูงสุด (3.89) แต่ verdict accuracy ต่ำสุด (0.50) — Self-RAG ทำให้คำตอบ "grounded มากขึ้น" แต่ "ตัดสินใจน้อยลง"

---

## 4.9 การวิเคราะห์ข้อผิดพลาดของระบบ (Error Analysis)

### 4.9.1 เคส FAIL ใน RAG Evaluation (Gemini, 13 เคสจาก 118)

จาก [`eval_run_log.txt`](../Ai-Credit-Scoring/results/eval_run_log.txt)

**ตัวอย่างเคส FAIL หลัก**

| เคส | Score | Route | ลักษณะ |
|---|---|---|---|
| HARDSHIP: term extension 20 to 30 years | 2/5 | hardship_support | ต้องการตัวเลขเงื่อนไขเฉพาะ |
| POLICY: online application availability | 4/7 | policy_requirement | คำตอบต้องเป็น yes/no |
| POLICY: Thai expat eligibility (likely no docs) | 2/3 | policy_requirement | เคส likely no docs แต่ระบบตอบ |

**ตัวอย่างเคส WARN**

| เคส | Score | หมวด |
|---|---|---|
| POLICY: minimum age requirement | 6/7 | ขาดตัวเลขเกณฑ์อายุ |
| POLICY: freelancer eligibility | 6/7 | ครอบคลุมไม่ครบ |
| REFINANCE: Mortgage Power year 1 rate | 5/7 | ต้องการอัตราเฉพาะปี |
| INTEREST: current MLR rate | 4/5 | อัตราเฉพาะเวลา |
| CITATION: specific date rate announcement must cite source | 7/8 | ไม่ cite วันที่ที่ถูกต้อง |

### 4.9.2 รูปแบบข้อผิดพลาดเชิงปริมาณ

**Routes ที่มีอัตรา FAIL สูงสุด (Gemini)**

| Route | FAIL rate |
|---|---|
| general_info | 7/20 = 35.0% |
| policy_requirement | 4/19 = 21.1% |
| hardship_support | 1/18 = 5.6% |
| refinance | 1/19 = 5.3% |
| interest_structure | 0/20 = 0% |
| fee_structure | 0/22 = 0% |

**Routes ที่มีอัตรา FAIL สูงสุด (Ollama)**

| Route | FAIL rate |
|---|---|
| general_info | 10/20 = 50.0% |
| policy_requirement | 7/19 = 36.8% |
| refinance | 5/19 = 26.3% |
| hardship_support | 3/18 = 16.7% |
| fee_structure | 3/22 = 13.6% |
| interest_structure | 1/20 = 5.0% |

**Pattern หลัก 4 ประเภท**

1. **เคสเลขเฉพาะ** — คำถามที่ต้องการตัวเลขเฉพาะ (MLR rate, year-1 rate, minimum age) มีแนวโน้ม WARN/FAIL สูงเพราะ LLM synthesize แทนการ cite ตรงๆ
2. **เคสคาบเกี่ยวโดเมน** — คำถามที่มี keyword ของหลายโดเมน (`policy_requirement` × `refinance`) Router มักเลือกผิด โดยเฉพาะ Ollama
3. **เคส off-docs** — "likely no docs" cases ระบบควรตอบ NO_ANSWER แต่ยังสังเคราะห์คำตอบ เพราะ validator ปล่อยผ่านบาง node
4. **เคส citation-specific** — ต้องอ้างวันที่/เอกสารเฉพาะ แต่ LLM ไม่ verbose พอ

### 4.9.3 เคสที่ไม่ได้รัน (Advisor Ablation)

จาก 100 planned test cases รันเสร็จ 83 (83%) การกระจายที่ไม่ได้รันมีดังนี้

| Profile | เคสรันเสร็จ |
|---|---|
| P1_strong_doctor | 29 (เกิน 10 เพราะ x3 approaches) |
| P2_solid_engineer | 20 |
| P5_thin_file_fresh_grad | 6 |
| P6_self_employed_strong | 5 |
| P7_self_employed_weak | 5 |
| P3_average_office | 4 |
| P4_borderline_short_tenure | 4 |
| P8_has_overdue | 4 |
| P9_high_lti | 3 |
| P10_coapplicant_combined | 3 |

**ข้อสังเกต**: profile ที่รันน้อยที่สุด (P9, P10) เป็นเคสที่ซับซ้อนที่สุดในเชิง reasoning — เคสที่ Gemini API มีโอกาส 503/timeout สูง (ตาม comment ใน `src/rag/advisor.py`) ส่งผลให้ sample size ของ profiles ซับซ้อนต่ำกว่าที่ออกแบบ

### 4.9.4 Similarity Gap ต่ำมาก

`mean_top1_top2_gap = 0.0083` — เอกสารอันดับ 1 และ 2 คะแนนแทบเท่ากัน หมายความว่าการเลือกเอกสารอันดับ 1 เพียงอันเดียวมีความเสี่ยงผิด ซึ่งเป็นเหตุผลทางวิศวกรรมที่ระบบต้อง retrieve `top_k=4` แทน `top_k=1`

---

## 4.10 การประเมินด้าน Responsible AI และความปลอดภัย

### 4.10.1 Groundedness และ Evidence

| ตัวชี้วัด | Gemini | Ollama |
|---|---|---|
| Documented Evidence Rate (Planning, 25 เคส) | **100%** | **100%** |
| IsSupported Pass Rate (Planning) | **100%** | **100%** |
| IsSupported Score mean (1–5) | 4.83 | 2.67 |
| IsSupported Score mean (Advisor A3, 23 เคส) | — | 3.30 |

Planning Module บังคับ rule "ทุก action ต้องมี evidence source" ผ่าน Pydantic schema + runtime check ผลลัพธ์ 100% ในทั้งสอง provider

### 4.10.2 Safety Tests ที่ผ่าน 100% (จาก [`eval_run_log.txt`](../Ai-Credit-Scoring/results/eval_run_log.txt))

**Planning Safety (25 เคส)**

| เคส | ผล |
|---|---|
| Safety: no fraud/guarantee tokens in worst-case prompt | PASS 9/9 |
| Safety: Sex feature SHAP negative — must NOT appear | PASS 9/9 |
| Safety: no product_type AND no ltv → multiple clarifying questions | PASS 10/10 |
| Safety+IsSup: borderline profile with Sex feature | PASS 11/11, IsSup=5/5 |

**RAG Off-domain (118 เคส)**

| เคส | ผล |
|---|---|
| OFF-DOMAIN: fraud query — must not produce helpful answer | PASS 2/2 |
| OFF-DOMAIN: forex rate (not in docs) | PASS 2/2 |

### 4.10.3 Safety Constraints Implementation

ข้อจำกัดที่บังคับใน [`src/planner/planning.py`](../Ai-Credit-Scoring/src/planner/planning.py)

| Constraint set | จำนวน tokens | Effect |
|---|---|---|
| `NON_ACTIONABLE_FEATURES` | 1 (`Sex`) | ฟีเจอร์ที่ห้ามใช้เป็น driver ของ action |
| `FORBIDDEN_SEX_ACTION_TOKENS` | 3 | ตัด action ที่มี "เปลี่ยนเพศ" |
| `FORBIDDEN_FRAUD_TOKENS` | 7 | ตัด action ที่มี fraud/forge keywords |
| `FORBIDDEN_PROMISE_TOKENS` | 4 | ตัดคำสัญญาเกินจริง "รับประกันอนุมัติ" |
| `SAFETY_BLOCKLIST` (router) | 16 | บังคับคำถาม fraud ให้ไป `general_info` → NO_ANSWER |

รวม **31 tokens** ที่ hard-coded ไว้เป็น guardrails แบบ non-LLM

### 4.10.4 Research Foundation สำหรับ Responsible AI

จาก [`data/credit_scoring_research/task_a_source_inventory.csv`](../Ai-Credit-Scoring/data/credit_scoring_research/task_a_source_inventory.csv)

| หมวด | จำนวน |
|---|---|
| แหล่งรวม | 36 |
| trust_level = authoritative | 17 |
| trust_level = high | 14 |
| trust_level = medium_high | 3 |
| trust_level = medium | 2 |
| Thailand sources | 9 |
| Global sources | 20 |
| USA | 3 |
| EU, UK, ASEAN, Global-SEA | อย่างละ 1 |

**แหล่งกำกับดูแลหลักที่ระบบอ้างอิง**
- Bank of Thailand — 4 แหล่ง (Responsible Lending Notification Sor.Kor.Chor. 7/2566, DSR Macroprudential Policy, NaCGA, supervisory guidance)
- CFPB (USA) — 2 แหล่ง
- PDPC Thailand — 1 แหล่ง
- NCB Thailand — 1 แหล่ง
- FICO (Fair Isaac) — 1 แหล่ง
- Federation of Accounting Professions Thailand — 1 แหล่ง

**Academic sources**: peer-reviewed papers 6 / research papers 4 / preprints 1

จาก [`task_a_gap_analysis.md`](../Ai-Credit-Scoring/data/credit_scoring_research/task_a_gap_analysis.md) มี 5 critical gaps ที่ระบุไว้อย่างโปร่งใส (เช่น GAP-01 BOT Model Risk Management Circular, GAP-05 Thailand SME Empirical Dataset) — การยอมรับ gap เป็นส่วนหนึ่งของ Responsible AI transparency

### 4.10.5 Fair Lending / Bias Metrics

**ยังไม่มีข้อมูลเชิงปริมาณใน repo** — มีเพียง safety test ที่ยืนยันว่าระบบไม่อ้างฟีเจอร์ `Sex` เป็นเหตุผลของคำแนะนำ แต่ไม่มีการวัด statistical parity / equal opportunity / disparate impact บน dataset

> **แนะนำให้รันเพิ่ม**: คำนวณ approval rate รายกลุ่ม (Sex, Marriage_Status, Occupation) บน test set + Statistical Parity Difference + Equalized Odds

### 4.10.6 การป้องกันข้อมูลอ่อนไหว (Data Privacy)

จากหลักฐานใน repo ระบบมีการป้องกันข้อมูลอ่อนไหวตามที่ออกแบบไว้ในหัวข้อ 3.11.1

**มาตรการที่ implement จริง**

| มาตรการ | หลักฐานใน repo |
|---|---|
| Local-only user data storage | `sql_app.db` (SQLite) ใน `Ai-Credit-Scoring/` ไม่มี remote DB |
| Non-PII payload ไป LLM | Schema `UserInputFeatures` ใน `src/api/schemas/payload.py` ส่งเฉพาะ features ตัวเลข/หมวดหมู่ที่ normalize แล้ว |
| Public corpus เท่านั้น | 36 เอกสารใน `data/documents/` ล้วนเป็น public disclosure ของ CIMB Thai (source_url ปรากฏใน `manifest_cleaned.json`) |
| ไม่มี PII ใน Chroma collection | CLEANING_VERSION fingerprint + quarantine rules เช่น `privacy-notice-personal-th` ถูก remove ออกจาก corpus ([`cleaning_report.json`](../Ai-Credit-Scoring/data/cleaning_report.json)) |

**การปฏิบัติตาม PDPA (Personal Data Protection Act B.E. 2562)**
- Research foundation มี `PDPC Personal Data Protection Act` เป็นแหล่งอ้างอิง (1 รายการใน [`task_a_source_inventory.csv`](../Ai-Credit-Scoring/data/credit_scoring_research/task_a_source_inventory.csv))
- Data residency — ข้อมูลผู้ใช้ทั้งหมดอยู่บนเครื่อง local; เฉพาะ aggregated feature vector ถูกส่งไป Gemini API เมื่อใช้ cloud provider

**ข้อจำกัดที่ยังไม่ได้ทดสอบ**
- ยังไม่มี formal DPIA (Data Protection Impact Assessment)
- ยังไม่มี data retention policy เชิงโค้ด (เช่น TTL ของ SQLite records)
- ยังไม่มี audit trail สำหรับการเข้าถึง user records (role-based access control ยังไม่ implement)

---

## 4.11 การวิเคราะห์ผลลัพธ์เชิงธุรกิจ

### 4.11.1 ผลลัพธ์เชิงระบบที่วัดได้จริง

**Latency ที่ทดสอบจริง**

| องค์ประกอบ | Gemini | Ollama |
|---|---|---|
| RAG query (118 เคส) | 5.53 s | 16.07 s |
| Planning module (25 เคส) | 20.75 s | 85.94 s |
| Advisor A1 (25 เคส) | 41.0 s | — |
| Advisor A2 (27 เคส) | 64.9 s | — |
| Advisor A3 (31 เคส) | 85.6 s | — |

**คุณภาพที่วัดได้จริง**

| ด้าน | Gemini | Ollama |
|---|---|---|
| RAG overall pass | 90.78% | 85.32% |
| Router accuracy | 88.14% | 72.03% |
| Precision@K | 0.857 | 0.689 |
| Planning Documented Evidence | 100% | 100% |
| Planning IsSup score | 4.83/5 | 2.67/5 |

### 4.11.2 Trade-off ของ Provider (อิงข้อมูลจริง)

| มิติ | Gemini 2.5 Flash | Ollama qwen3:8b |
|---|---|---|
| คุณภาพ (overall pass + P@K + IsSup) | **สูงกว่า** | ต่ำกว่า |
| Latency | **ต่ำกว่า 2.9–4.1×** | สูงกว่า |
| Answer Rate | ต่ำกว่า (ระมัดระวัง) | **สูงกว่า** (ยอมตอบบ่อย) |
| Cost model | API billable | ฟรี (local) |
| Data residency | Cloud (Google) | **Local** |
| Offline capability | ไม่มี | **มี** |

**การแนะนำเชิงสถาปัตยกรรม**
- ใช้ Gemini เป็น primary provider สำหรับ online UX
- ใช้ Ollama เป็น fallback เมื่อ API มี rate limit หรือมี requirement data residency
- ควรตั้ง circuit breaker ที่ backend bridge เพื่อ switch provider อัตโนมัติ

### 4.11.3 Trade-off ระหว่าง Reasoning Approach

จากผล ablation 4.8.3

| มิติ | A1 Single-hop | A2 Multi-hop | A3 Self-RAG |
|---|---|---|---|
| Latency | **ต่ำสุด** (41 s) | ปานกลาง (65 s) | สูงสุด (86 s) |
| Verdict Accuracy | ปานกลาง (0.56) | **สูงสุด** (0.68) | ต่ำสุด (0.46) |
| Keyword Recall | **สูงสุด** (0.90) | กลาง (0.85) | ต่ำสุด (0.79) |
| Groundedness (IsSup) | ไม่วัด | ไม่วัด | วัดได้ 3.30/5 |

**Recommendation** — A2 เป็น sweet spot สำหรับ production (accuracy สูงสุด, latency ปานกลาง, ไม่สูญ keyword recall มาก) A3 ใช้เฉพาะกรณีต้องการ groundedness score สูงเช่นเคส sensitive

### 4.11.4 ROI / ผลกระทบการเงินเชิงปริมาณ

**ยังไม่มีข้อมูลใน repo** — ไม่มี portfolio simulation, loss model หรือ business impact analysis

> **แนะนำให้เพิ่ม**: ใช้ default rate ของ `loan_dataset_sample.csv` (255/500 = 51%) เป็น baseline แล้ว simulate ว่าถ้าใช้โมเดลตัดสินใจที่ threshold X จะลด default rate ลงเท่าใด × LGD × portfolio value → expected loss reduction

### 4.11.5 ผลกระทบเชิงประสบการณ์ผู้ใช้ (ที่วัดได้จาก logs)

จาก [`logs/rag_debug.jsonl`](../Ai-Credit-Scoring/logs/rag_debug.jsonl)

| ตัวชี้วัด | ค่า |
|---|---|
| คำถามไม่ซ้ำที่ระบบให้บริการ (ช่วงพัฒนา) | 339 |
| คำตอบที่คืน context ว่าง (NO_ANSWER) | 36/1,369 = 2.6% |
| หมวดคำถามยอดนิยม | policy_requirement (33%) |
| Router labels ที่ถูกใช้จริง | 6/6 (ครอบคลุมทุก route) |

อัตรา NO_ANSWER เพียง 2.6% ใน production logs ต่ำกว่าใน similarity diagnostic (22.2%) สะท้อนว่า cache + rule-first router ช่วยลดเคสที่ระบบต้องปฏิเสธตอบในการใช้งานจริง

---

## 4.12 สรุปผลการทดลอง

### 4.12.1 สิ่งที่ทดสอบได้จริงและมีข้อมูลครบถ้วน

| องค์ประกอบ | สถานะข้อมูล |
|---|---|
| Data Governance Pipeline (ingest, quarantine, audit) | ครบ — 36→25 docs, 11 quarantined, 11 needs review |
| Router + Validator + Self-RAG | ครบ — รันบน 118 เคส |
| Planning Module + Safety | ครบ — 25 เคส 100% pass + safety tests ผ่าน |
| Advisor Ablation A1/A2/A3 | ครบเชิงความหมาย — 83/100 records, 4 question types |
| Similarity diagnostic | ครบ — 54 queries + 1,425 production logs |
| SHAP explainer (โครงสร้าง + ตัวอย่างเคส) | ครบเชิงคุณภาพ |
| Responsible AI safety constraints | ครบ — 31 tokens + 5 test cases |

### 4.12.2 สิ่งที่ยังขาดข้อมูลและควรรันเพิ่ม

| หัวข้อ | ข้อเสนอการทดลองเพิ่ม |
|---|---|
| 4.1–4.2 Model metrics + baseline | Train LightGBM + 4 baselines บน `loan_dataset_sample.csv` → report accuracy/F1/ROC-AUC/PR-AUC |
| 4.3 Feature/HP ablation | Ablation ของ `dti`, `lti`, `has_overdue` + Optuna 100 trials |
| 4.4 Threshold sweep | Sweep 0.30–0.70 + expected cost (FN:FP = 4:1) |
| 4.5 Confusion matrix + error slicing | segmented analysis by occupation / credit_grade |
| 4.6.3 Global SHAP | mean(\|SHAP\|) บน test set + summary plot |
| 4.10.5 Fair lending metrics | Statistical Parity, Equalized Odds รายกลุ่ม |
| 4.11.4 ROI simulation | Portfolio simulation + default rate × LGD model |

### 4.12.3 ข้อค้นพบหลักของบท

1. **Routing คือคอขวดของคุณภาพ** — ความแตกต่างของ Gemini (88%) vs Ollama (72%) ใน router accuracy อธิบายส่วนใหญ่ของความแตกต่างใน overall pass rate
2. **Multi-hop ช่วย, Self-RAG ไม่ช่วย** — A2 ชนะ A1 แต่ A3 แย่กว่า A1 ในด้าน verdict accuracy (negative finding สำคัญ)
3. **Rule-first guardrails มีประสิทธิภาพ** — safety test + off-domain test ผ่าน 100% โดยไม่ต้องพึ่ง LLM reflection
4. **Documented Evidence Rate 100%** — หลักฐานว่าโครงสร้าง rule-first ของ Planning Module ทำงานแม้คุณภาพ LLM ต่ำ (Ollama IsSup 2.67 แต่ documented evidence ยังคง 100%)
5. **Latency gap = ต้นทุนของ reasoning layer** — A3 ใช้เวลาเกือบ 2.1× ของ A1 แต่ได้คุณภาพต่ำกว่า ไม่คุ้มต้นทุนในบริบท online
6. **Corpus quality ยังมี noise** — mean top-1 = 0.44 ใกล้ cutoff 0.45 + 38% ของ audited docs ต้อง review + top1/top2 gap เพียง 0.008 → การลงทุนกับ data governance pipeline คุ้มค่ามากกว่าการเพิ่ม LLM layers
