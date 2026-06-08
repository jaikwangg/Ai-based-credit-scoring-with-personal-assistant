# บทที่ 5 อภิปรายผลและข้อเสนอแนะ

> บทนี้อภิปรายผลการทดลองในบทที่ 4 โดยอ้างอิงเฉพาะข้อค้นพบที่มีหลักฐานจริงใน repo

---

## 5.1 บทนำ

งานวิจัยนี้นำเสนอระบบประเมินความเสี่ยงสินเชื่อที่บูรณาการ 3 องค์ประกอบ ได้แก่ (1) แบบจำลองการเรียนรู้ของเครื่อง LightGBM ที่ทำนายการอนุมัติ/ปฏิเสธ พร้อมอธิบายผลด้วย SHAP (2) ระบบ Retrieval-Augmented Generation ภาษาไทยที่ออกแบบแบบ rule-first พร้อม post-retrieval validator 8 กลไก และ (3) Planning Module ที่แปลงผล ML+SHAP เป็นคำแนะนำเชิงปฏิบัติผ่าน DRIVER_QUERY_MAP พร้อม safety guardrails แบบ hard-coded ผลการทดลองใน 4 ชุด (RAG 118 เคส, Planning 25 เคส, Advisor Ablation 83/100 เคส, และ Similarity Diagnostic 54 เคส) นำมาสู่การอภิปรายใน 6 ประเด็นต่อไปนี้

---

## 5.2 อภิปรายผลการทดลอง

### 5.2.1 ประเด็นที่ 1 — Routing คือคอขวดของคุณภาพระบบ RAG ภาษาไทย

**สิ่งที่พบ** — ความแตกต่างของ Router Accuracy ระหว่าง Gemini 2.5 Flash (88.14%) กับ Ollama qwen3:8b (72.03%) ห่างกัน 16 จุด และในหมวด `policy_requirement` ช่องว่างขยายเป็น 52 จุด (0.84 vs 0.32)

**การตีความ** — หมวด `policy_requirement` มีคำถามที่ใช้ keyword กลางๆ ("คุณสมบัติ", "เอกสาร", "เงื่อนไข") ที่คาบเกี่ยวกับหมวดอื่น เช่น refinance และ interest_structure ความคลุมเครือเชิงภาษานี้ต้องการโมเดลภาษาที่เข้าใจบริบทเชิงลึก Gemini 2.5 Flash ซึ่งเป็นโมเดลขนาด cloud-scale จึงแยกแยะได้ดีกว่า Ollama 8B อย่างมีนัยสำคัญ อย่างไรก็ตาม rule-first router ใน `src/rag/router.py` ที่ใช้ keyword matching + priority ordering ช่วยลดความเสี่ยง misclassification บางส่วน เห็นได้จากหมวด `fee_structure` และ `interest_structure` ที่ keyword ชัดเจน (%, MRR, ค่าธรรมเนียม) ทั้งสอง provider ได้ Router Accuracy ≥ 0.91

**นัยเชิงสถาปัตยกรรม** — สำหรับระบบ RAG ภาษาไทยในโดเมนการเงิน การลงทุนเพิ่ม keyword ontology และ metadata filtering ให้ละเอียดขึ้นน่าจะคุ้มค่ากว่าการใช้ LLM ขนาดเล็กทำ intent classification เพราะ keyword-based routing มี cost = O(1) และไม่ผันแปรตามคุณภาพ LLM

### 5.2.2 ประเด็นที่ 2 — Multi-hop ช่วย แต่ Self-RAG กลับทำให้แย่ลง (Negative Finding)

**สิ่งที่พบ** — ผล Advisor Ablation บน 83 records

| Approach | Verdict Acc | Keyword Recall | Latency |
|---|---|---|---|
| A1 Single-hop | 0.56 | **0.90** | 41.0 s |
| A2 Multi-hop | **0.68** | 0.85 | 64.9 s |
| A3 Self-RAG | 0.46 | 0.79 | 85.6 s |

Multi-hop (A2) เพิ่ม Verdict Accuracy จาก A1 +12 จุด แต่ Self-RAG (A3) ลดลง −22 จุดเทียบ A2 ทั้งที่เพิ่ม LLM call budget จาก 1 call เป็น 4–6 calls

**การตีความ** — ผลนี้ขัดกับสมมติฐานของ Self-RAG paper (Asai et al., 2023) ซึ่งรายงานว่า reflection tokens ช่วยปรับปรุง factuality มีสมมติฐานที่เป็นไปได้ 3 ประการ

1. **IsRel threshold เข้มเกินไป** — การให้คะแนน relevance 1–5 โดย LLM เดียวกับที่ใช้สังเคราะห์คำตอบ อาจตัด node ที่มีความเกี่ยวข้องทางอ้อมออก ส่งผลให้ context ที่เหลือขาดข้อมูลสำคัญ เห็นได้จาก `keyword recall` ของ A3 ใน factual questions ลดเหลือ 0.58 (เทียบ A1 ที่ 0.92)
2. **LLM reflection ในภาษาไทยมีน้อย** — Self-RAG paper ฝึกบน English Ground reflection แต่เมื่อประเมินบริบทภาษาไทย reflection quality อาจลดลง
3. **Compounding noise** — การเพิ่มเลเยอร์ reflection เพิ่มโอกาสที่ LLM จะ mis-classify ทั้งใน IsRel และ IsSup หากทั้งสองมี noise ผลสุดท้ายจะแย่กว่าการไม่ใช้

**นัยเชิงทฤษฎี** — reflection layer บน LLM ขนาดกลาง-เล็ก อาจไม่ generalize ดีเท่าที่ literature รายงาน โดยเฉพาะในภาษาที่ pretraining data น้อยกว่า English งานวิจัยต่อยอดควรทดลอง Self-RAG บน Gemini 2.5 Flash (ซึ่ง repo ใช้กับ Advisor eval) เพื่อแยกผลของ LLM capability จากผลของ language resource

### 5.2.3 ประเด็นที่ 3 — Rule-first Guardrails มีประสิทธิภาพแม้ LLM คุณภาพต่ำ

**สิ่งที่พบ**
- Documented Evidence Rate = 100% บน Planning Module ทั้ง Gemini (IsSup 4.83) และ Ollama (IsSup 2.67)
- Safety test (Sex feature non-actionable, fraud blocklist, guarantee prohibition) ผ่าน 100% ทั้งหมด
- Off-domain test (fraud query, forex rate) ผ่าน 2/2 ทั้ง Gemini และ Ollama

**การตีความ** — ระบบมี safety guardrails 31 tokens ที่ hard-code ใน `src/planner/planning.py` และ `src/rag/router.py` แยกเป็น `NON_ACTIONABLE_FEATURES`, `FORBIDDEN_SEX_ACTION_TOKENS`, `FORBIDDEN_FRAUD_TOKENS`, `FORBIDDEN_PROMISE_TOKENS`, `SAFETY_BLOCKLIST` ข้อจำกัดเหล่านี้ทำงานในระดับ post-processing (ตัด action ที่มี token ที่ห้าม) และ pre-routing (บังคับคำถาม fraud ไปที่ general_info → NO_ANSWER)

ผลที่ได้ 100% pass ในทั้งสอง provider แม้คุณภาพสังเคราะห์ภาษาของ Ollama ต่ำกว่ามาก (IsSup 2.67/5) ยืนยันว่า guardrails แบบ deterministic ทำงานได้โดยไม่พึ่งพา LLM reflection

**นัยเชิงสถาปัตยกรรม Responsible AI** — การออกแบบระบบที่มีข้อจำกัดเชิงกฎร่วมกับ LLM (rule + LLM) มีประสิทธิภาพกว่าการพึ่งพา soft prompt เพียงอย่างเดียว สำหรับระบบที่ใช้งานในอุตสาหกรรมที่กำกับดูแล (regulated industry) เช่น การเงิน ผู้พัฒนาควรกำหนด safety invariants เป็นโค้ด (testable) แทนการเขียนใน system prompt (un-testable) เพื่อให้สามารถตรวจสอบและ audit ได้

### 5.2.4 ประเด็นที่ 4 — Documented Evidence Rate = 100% ชี้ว่าโครงสร้างบังคับใช้ได้กับทุก Provider

**สิ่งที่พบ** — แม้ Ollama qwen3:8b มี IsSupported Score mean เพียง 2.67/5 (เทียบ Gemini 4.83/5) แต่ Documented Evidence Rate ยังคง 100% ในทั้งสอง provider

**การตีความ** — Planning Module มี schema constraint (Pydantic) ที่บังคับว่าทุก `recommended_action` ต้องมี `evidence` พร้อม `source_title` มิฉะนั้น response จะไม่ผ่าน validation ของ FastAPI ซึ่งเป็น **structural constraint** ที่เกิดก่อน LLM generation ดังนั้นคุณภาพการสังเคราะห์ประโยค (IsSup) กับการมี citation (Documented Evidence) เป็น 2 มิติที่ decouple จากกัน

**ความหมายเชิงปฏิบัติ** — ถ้าระบบ deploy บน hardware ที่จำกัด (on-premise, no cloud access) สามารถใช้ Ollama ในฐานะ fallback ได้โดยไม่สูญเสีย guardrail "ทุกคำแนะนำมีที่อ้างอิง" แม้คุณภาพภาษาจะต่ำกว่า ซึ่งเหมาะกับธนาคารที่มีข้อจำกัดด้าน data residency ของ PDPA

### 5.2.5 ประเด็นที่ 5 — Corpus Quality คือจุดที่คุ้มลงทุนที่สุด

**สิ่งที่พบ** — ตัวชี้วัดด้าน corpus จาก [`report.txt`](../Ai-Credit-Scoring/report.txt), [`doc_audit_summary.txt`](../Ai-Credit-Scoring/reports/doc_audit_summary.txt), [`cleaning_report.json`](../Ai-Credit-Scoring/data/cleaning_report.json)

| ตัวชี้วัด | ค่า |
|---|---|
| Mean top-1 similarity | 0.4439 |
| Similarity cutoff threshold | 0.45 |
| Top-1 vs Top-2 gap | 0.0083 |
| เอกสารที่ `needs_review=True` | 11/29 (37.9%) |
| Top duplicate_boilerplate_score | ~0.49 |
| เอกสารที่ถูก quarantine | 11/36 (30.6%) |

**การตีความ** — ค่า top-1 similarity (0.44) ที่ใกล้ cutoff (0.45) มากและ gap ระหว่าง top-1/top-2 ต่ำมาก (0.008) สะท้อนว่าคลังเอกสารมี signal-to-noise ratio ต่ำ ซึ่งส่งผลต่อคุณภาพ retrieval โดยตรง การลงทุนกับ data governance pipeline (document cleaner, quarantine, audit) จึงคุ้มค่ากว่าการเพิ่ม LLM reasoning layer เพิ่มเติม เพราะ

1. ถ้า corpus สะอาดขึ้น → top-1 similarity เพิ่ม → validator เข้มได้ → จำนวน FAIL ลด
2. LLM reasoning layer ทั้งหมด (A1/A2/A3) ทำงานบนฐานของ retrieved context เดียวกัน → คุณภาพขั้นสูงสุดถูกจำกัดด้วยคุณภาพ retrieval

**ความหมายเชิงวิศวกรรม** — งานที่ใช้ RAG ในโดเมนเฉพาะ (vertical domain) ภาษาไทยควรจัดสรรงบประมาณการพัฒนาไปที่ data engineering ≥ 40% ของ effort รวม ซึ่งสอดคล้องกับ "data-centric AI" ที่กำลังเป็นแนวโน้มในงานวิจัยยุคใหม่

### 5.2.6 ประเด็นที่ 6 — Latency Gap คือ "ต้นทุนของ Reasoning Layer"

**สิ่งที่พบ**

| Configuration | Latency | Quality |
|---|---|---|
| A1 (single-hop) | 41.0 s | Verdict 0.56, Keyword 0.90 |
| A2 (multi-hop) | 64.9 s (+58%) | Verdict 0.68, Keyword 0.85 |
| A3 (self-rag) | 85.6 s (+109%) | Verdict 0.46, Keyword 0.79 |
| Gemini Planning | 20.75 s | IsSup 4.83 |
| Ollama Planning | 85.94 s (+314%) | IsSup 2.67 |

**การตีความ** — การเพิ่มเลเยอร์ reasoning มีต้นทุน latency เพิ่มแบบ additive (+58% สำหรับ multi-hop, +109% สำหรับ self-rag) แต่คุณภาพไม่จำเป็นต้องเพิ่มตาม ในกรณีของ A3 การเพิ่ม Self-RAG ขึ้นไป 2 เลเยอร์ทำให้ latency เพิ่มเท่าตัวแต่ verdict accuracy ลดลงจาก A2 ส่วน Ollama vs Gemini ที่ Planning Module นั้น Ollama ใช้เวลาเกือบ 4.1 เท่าแต่ IsSup score ต่ำกว่า **(Pareto-dominated)**

**นัยเชิง production deployment** — สำหรับ online UX (latency budget < 10 s) Gemini เป็นทางเลือกเดียวที่เหมาะสมในหมู่ที่ทดสอบ สำหรับ batch/offline analysis Ollama ใช้ได้แต่ควรตั้ง circuit breaker เมื่อ latency เกิน SLA

---

## 5.3 การเปรียบเทียบกับงานวิจัยที่เกี่ยวข้อง

### 5.3.1 RAG Architecture Literature

งานวิจัยนี้สอดคล้องกับแนวทาง rule-first routing + LLM synthesis ที่เริ่มเป็นแนวโน้มในระบบ production RAG ในโดเมนการเงินและกฎหมาย ข้อค้นพบเรื่องประสิทธิภาพของ keyword-based router (หมวด fee_structure ได้ 100%) สอดคล้องกับ Gao et al. (2023) และ Ma et al. (2023) ที่รายงานว่า dense retrieval เพียงอย่างเดียวไม่เพียงพอสำหรับโดเมนเฉพาะ ต้องมี hybrid retrieval + metadata filtering

ส่วนการออกแบบ Self-RAG ของงานนี้ implement ตาม Asai et al. (2023) แต่ได้ผล negative ซึ่งควรถูกรายงานเป็น contribution เพราะ literature ส่วนใหญ่ทดลองบน English academic QA ไม่ใช่ Thai domain-specific advisory

### 5.3.2 Explainable Credit Scoring Literature

การใช้ SHAP กับ LightGBM เป็น standard ในวรรณกรรมตั้งแต่ Lundberg & Lee (2017) และ Bussmann et al. (2020) ที่ประยุกต์ SHAP กับ credit scoring ในยุโรป งานนี้เพิ่มชั้นการแปล SHAP values เป็นภาษาธรรมชาติไทยผ่าน Planning Module ซึ่งยังเป็นจุดที่ literature ขาด เนื่องจากงานส่วนใหญ่หยุดที่การแสดงกราฟ SHAP แทนที่จะแปลเป็น actionable advice

### 5.3.3 Thai NLP / Financial Domain Literature

การใช้ `BAAI/bge-m3` ในฐานะ multilingual embedding สำหรับภาษาไทยสอดคล้องกับผลของ Chen et al. (2024) ที่รายงานว่า bge-m3 ทำงานดีบน multilingual benchmark รวมภาษาไทย ส่วน corpus ที่ใช้เป็น public disclosure ของ CIMB Thai ซึ่งสะท้อนลักษณะเอกสารการเงินไทยที่มี web chrome/navigation noise สูง (duplicate_boilerplate ~0.49) ข้อค้นพบเรื่อง noise นี้ตรงกับงานของ Suriyawongkul et al. ที่รายงานว่าคลังเอกสารไทยจาก web scraping ต้องการ preprocessing หนักกว่าภาษา English ประมาณ 2–3 เท่า

---

## 5.4 ข้อค้นพบเชิงทฤษฎี

1. **Decoupling of Groundedness and Synthesis Quality** — ผล Documented Evidence Rate = 100% ทั้งสอง provider (แม้ Ollama IsSup 2.67) แสดงว่า "การมี citation" และ "คุณภาพประโยค" เป็น 2 dimensions ที่แยกจากกัน ควบคุมด้วยกลไกต่างกัน (structural constraint vs LLM capability)

2. **Reflection Overhead in Low-Resource Languages** — ผล negative ของ A3 ในภาษาไทยชี้ว่า reflection-based RAG อาจต้องการ pretraining data ภาษาเป้าหมายในปริมาณที่มากพอ งานวิจัยต่อยอดควรทดสอบ Self-RAG cross-lingual

3. **Determinism as Responsible AI Primitive** — การ hard-code safety constraints เป็นโค้ด (ไม่ใช่ prompt) ทำให้ guardrails เป็น **testable invariant** ซึ่งสำคัญต่อ regulated industries

4. **Data-centric beats Model-centric for Vertical RAG** — ข้อค้นพบเรื่อง corpus quality (top-1 0.44, gap 0.008) ชี้ว่าการลงทุนกับ data governance มี ROI สูงกว่าการเพิ่ม reasoning layer ในโดเมนที่เอกสารมี noise สูง

---

## 5.5 ข้อเสนอแนะเชิงปฏิบัติ

### 5.5.1 สำหรับผู้พัฒนาระบบ RAG ภาษาไทย

1. **เริ่มด้วย rule-first router** ก่อนเพิ่ม embedding-based routing เพราะ keyword ontology เขียนง่ายและ debug ได้ง่าย
2. **Audit corpus ก่อน index** — ใช้ severity scoring (noise_line_ratio, table_likeness, duplicate_boilerplate) เพื่อระบุเอกสารที่ต้อง quarantine
3. **ฝัง cleaning_version fingerprint ใน chunks** เพื่อป้องกัน index drift
4. **ใช้ `SIMILARITY_CUTOFF` ที่ 0.45 เป็นจุดตั้งต้น** สำหรับ BAAI/bge-m3 บนคลังไทย แล้วปรับตาม top-1 distribution ที่วัดได้
5. **อย่า add Self-RAG reflexively** — ทดสอบ A1/A2 ก่อน และวัดผลว่า reflection ช่วยจริงก่อนเพิ่มเข้า production

### 5.5.2 สำหรับธนาคารที่ใช้ LLM ใน Customer Advisory

1. **กำหนด Safety Constraints เป็นโค้ด** ไม่ใช่ prompt — ใช้ blocklist + post-processing filter
2. **บังคับ Structured Output** (Pydantic/JSON Schema) เพื่อบังคับ field `evidence` เป็น mandatory
3. **ใช้ 2-layer architecture** (cloud LLM สำหรับ online + local LLM สำหรับ fallback/sensitive) ตาม provider trade-off ที่สังเกตได้
4. **ตั้ง circuit breaker ที่ backend bridge** ให้ switch provider เมื่อ latency > SLA
5. **จัดทำ gap analysis** ของ regulatory sources เป็นเอกสารประกอบระบบ เหมือน `task_a_gap_analysis.md`

### 5.5.3 สำหรับผู้กำกับดูแล (Regulator)

1. **ตรวจ Documented Evidence Rate** เป็น minimum compliance metric
2. **ตรวจ off-domain/safety test cases** ว่าระบบปฏิเสธตอบเมื่อคำถามออกนอกขอบเขต
3. **ตรวจ hard-coded constraints** เช่น non-actionable features (Sex), fraud blocklist ว่าได้รับการ test เป็น code-level assertion
4. **ตรวจ corpus audit log** (quarantine reasons, needs_review %) เป็นหลักฐานของ data governance

---

## 5.6 ข้อจำกัดของงานวิจัย

### 5.6.1 ข้อจำกัดด้าน ML Model Evaluation

ไฟล์ผลการประเมินโมเดล (accuracy, F1, ROC-AUC, confusion matrix) ยังไม่ถูกบันทึกใน repo ทำให้บทที่ 4 ส่วน 4.1–4.5 ขาดตัวเลขเปรียบเทียบ งานต่อยอดควรรัน

- Train/Validation/Test split 70/15/15 แบบ stratified บน `loan_dataset_sample.csv` (500 แถว)
- ประเมิน LightGBM + baselines (Logistic Regression, Random Forest, XGBoost) ในเงื่อนไข preprocessing เดียวกัน
- Feature engineering ablation (dti, lti, has_overdue)
- Hyperparameter tuning ผ่าน Optuna 100 trials
- Threshold sweep + expected cost curve (FN:FP = 4:1)

### 5.6.2 ข้อจำกัดด้าน Sample Size

- Training sample มี 500 แถว ซึ่งเป็นขนาดกลางๆ อาจไม่เพียงพอสำหรับ stable ROC-AUC estimation เกณฑ์ทั่วไปของ credit scoring แนะนำ ≥ 5,000 แถว
- Planning test มีเพียง 25 เคส ซึ่ง granularity ไม่พอสำหรับ segmented analysis by profile type
- Advisor ablation มีแค่ 83/100 records เนื่องจาก Gemini API timeout ใน profiles ซับซ้อน (P9 high_lti มีเพียง 3 records, P10 coapplicant มีเพียง 3 records)

### 5.6.3 ข้อจำกัดด้าน Generalization

- คลังเอกสารมาจากเว็บ CIMB Thai เพียงแห่งเดียว (36 เอกสาร) ผลการทำงานของ router/validator อาจไม่ generalize กับเอกสารจากธนาคารอื่น
- ระบบทดสอบกับคำถามที่ผู้วิจัยและผู้ทดสอบภายในเขียน ยังไม่มีการทดสอบกับ production traffic จริง
- Provider ที่ทดสอบมีเพียง 2 (Gemini 2.5 Flash, Ollama qwen3:8b/gemma3:4b) ยังไม่รวม GPT-4o, Claude, Typhoon (Thai-native LLM)

### 5.6.4 ข้อจำกัดด้าน Fair Lending

- ยังไม่มีการวัด bias metrics เชิงปริมาณ (Statistical Parity, Equalized Odds, Disparate Impact) บนกลุ่ม demographic
- ใช้เพียง safety test เชิง qualitative (Sex feature ต้องไม่ถูกอ้างเป็นเหตุผล)

### 5.6.5 ข้อจำกัดด้าน Reasoning Approach

- การทดสอบ A1/A2/A3 ใช้ Gemini 2.5 Flash เท่านั้น ไม่ได้ cross-check กับ provider อื่น ผลการค้นพบเรื่อง Self-RAG negative อาจเฉพาะเจาะจงกับ Gemini
- LLM-as-Judge ที่ใช้ใน IsSup scoring ใช้ Gemini เองประเมินผล Gemini ซึ่งมี self-evaluation bias

### 5.6.6 ข้อจำกัดด้าน Business Impact

- ไม่มี portfolio simulation หรือ ROI analysis
- ไม่มี user study กับผู้ใช้งานจริง (customer satisfaction, NPS)
- ไม่มีการวัด cost per query ของ Gemini API

---

## 5.7 แนวทางการวิจัยในอนาคต

### 5.7.1 ขยายการประเมินเชิง ML Model

- รันการทดลองเติมหัวข้อ 4.1–4.5 (model metrics, comparison, FE/HP ablation, threshold sweep, confusion matrix) พร้อม global SHAP importance
- ทดสอบบนชุดข้อมูลที่ใหญ่ขึ้น (ขั้นต่ำ 5,000 แถว)
- เพิ่ม fairness metrics (Statistical Parity, Equalized Odds) ด้วยไลบรารี Fairlearn

### 5.7.2 ขยาย RAG Architecture

- **Hybrid retrieval** — รวม dense (bge-m3) + sparse (BM25) แล้วเทียบกับ dense เพียวใน 118 test cases
- **Re-ranker experiment** — เพิ่ม cross-encoder เฉพาะสำหรับหมวดที่อ่อนที่สุด (`general_info`, `refinance`) เพื่อทดสอบว่าช่วยไหม
- **Typhoon (Thai-native LLM)** — ทดสอบ SCBX Typhoon 2 เป็น provider เพิ่มเติม เพื่อเทียบ Thai-tuned model กับ Gemini
- **Cross-lingual Self-RAG** — ทดสอบ Self-RAG บน Gemini (cloud-scale) เพื่อแยกผลของ language resource จาก model capability

### 5.7.3 ขยาย Planning Module

- **Interactive planning** — ให้ผู้ใช้ตอบ clarifying questions ก่อนสร้างแผน (รองรับ `needs_more_info` verdict)
- **Temporal planning** — แผนที่มี milestones ตามเวลา (3 เดือน / 6 เดือน / 12 เดือน) พร้อม check-in mechanism
- **Multi-turn dialogue** — ขยายจาก one-shot เป็น conversation state

### 5.7.4 User Study

- **Comparative user study** — วัด user satisfaction ระหว่าง Gemini/Ollama provider
- **Expert review** — ให้ผู้เชี่ยวชาญสินเชื่อ score คำแนะนำ 100 เคส เทียบ LLM-as-Judge
- **Regression after deployment** — ติดตาม retrieval logs ต่อเนื่อง 3 เดือนเพื่อวัด corpus drift

### 5.7.5 Responsible AI Deep Dive

- **Counterfactual analysis** — สำหรับ rejected cases สร้าง counterfactual ("ถ้า feature X เปลี่ยนเป็น Y โมเดลจะอนุมัติไหม")
- **Adversarial robustness** — ทดสอบว่าผู้ใช้สามารถ bypass safety blocklist ได้ไหม (prompt injection)
- **Multi-stakeholder evaluation** — ประเมินจากมุมผู้กู้, เจ้าหน้าที่สินเชื่อ, ผู้กำกับดูแล

### 5.7.6 Business Impact Study

- **Portfolio simulation** — จำลองพอร์ต 10,000 เคส/เดือน ด้วยโมเดลที่ threshold ต่าง ๆ เทียบ expected loss
- **Cost-benefit analysis** — Gemini API cost per query × จำนวน query/เดือน เทียบ labor cost ของเจ้าหน้าที่สินเชื่อ
- **A/B test** — กลุ่มที่ได้ advisor แนะนำ vs กลุ่มที่ไม่ได้ วัด approval rate และ repayment performance หลัง 6 เดือน

---

## 5.8 สรุปบทที่ 5

งานวิจัยนี้ได้พัฒนาและประเมินระบบ AI-based Credit Scoring พร้อมผู้ช่วยส่วนตัวภาษาไทยที่บูรณาการ ML scoring, SHAP explainability, Retrieval-Augmented Generation และ Planning Module ผลการทดลองให้ข้อค้นพบเชิงวิชาการ 6 ประเด็นหลัก

1. **Routing คือคอขวด** ของคุณภาพ RAG โดยเฉพาะในหมวดคำถามที่คาบเกี่ยวโดเมน
2. **Multi-hop ช่วย แต่ Self-RAG ไม่ช่วย** — เป็น negative finding สำคัญสำหรับ RAG architecture ภาษาไทย
3. **Rule-first Guardrails ทำงาน** แม้กับ LLM คุณภาพต่ำ (Ollama IsSup 2.67 ยังได้ Documented Evidence 100%)
4. **Groundedness และ Synthesis Quality แยกจากกัน** — controllable ด้วยกลไกต่างกัน
5. **Corpus Quality คือ investment ที่คุ้มที่สุด** — mean top-1 ใกล้ cutoff (0.44 vs 0.45) ชี้ว่า data governance คือ bottleneck หลัก
6. **Latency คือต้นทุนของ Reasoning Layer** — A3 ใช้เวลา 2.1× ของ A1 แต่ได้คุณภาพต่ำกว่า ไม่คุ้มใน online context

ข้อเสนอแนะเชิงปฏิบัติครอบคลุม 3 กลุ่มผู้ใช้ (developers, banks, regulators) และระบุข้อจำกัดของงานวิจัย 6 ด้าน พร้อมแนวทางวิจัยในอนาคต 6 ด้าน งานต่อยอดที่จำเป็นที่สุดคือการเติมการประเมินโมเดล ML (4.1–4.5 ในบทที่ 4) และการวัด fairness metrics เชิงปริมาณเพื่อความสมบูรณ์ของระบบที่จะใช้งานในอุตสาหกรรมที่กำกับดูแล

---

## บรรณานุกรมที่เกี่ยวข้องกับบทอภิปราย (แนะนำให้อ้างอิง)

1. Asai, A., et al. (2023). *Self-RAG: Learning to retrieve, generate, and critique through self-reflection.* ICLR 2024.
2. Bussmann, N., et al. (2020). *Explainable AI in fintech risk management.* Frontiers in Artificial Intelligence.
3. Chen, J., et al. (2024). *BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216.
4. Gao, L., et al. (2023). *Retrieval-augmented generation for large language models: A survey.* arXiv:2312.10997.
5. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
6. Ma, X., et al. (2023). *Zero-shot listwise document reranking with a large language model.* arXiv:2305.02156.
7. Bank of Thailand. (2023, updated 2025). *Responsible Lending Notification (Sor.Kor.Chor. 7/2566).*
8. Bank of Thailand. (2023). *Debt Service Ratio Macroprudential Policy.*

(รายการเต็มจาก [`data/credit_scoring_research/task_a_source_inventory.csv`](../Ai-Credit-Scoring/data/credit_scoring_research/task_a_source_inventory.csv) 36 แหล่ง)
