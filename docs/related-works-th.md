# งานวิจัยที่เกี่ยวข้อง (Related Works) — สำหรับบทที่ 2

> เอกสารนี้รวบรวมงานวิจัยที่เกี่ยวข้องกับวิทยานิพนธ์ แบ่งเป็น 2 กลุ่ม
> - **กลุ่ม A**: 36 แหล่งที่ผู้วิจัยคัดสรรไว้แล้วใน [`Ai-Credit-Scoring/data/credit_scoring_research/task_a_sources.json`](../Ai-Credit-Scoring/data/credit_scoring_research/task_a_sources.json) — ครอบคลุมด้านกำกับดูแล, scorecard, ML for credit
> - **กลุ่ม B**: งานวิจัยเพิ่มเติมที่ค้นมาใหม่ เพื่ออุดช่องว่างด้าน **RAG, Self-RAG, Thai NLP, LLM Evaluation, Safety** ที่บทที่ 3–5 ใช้อ้างอิงแต่ยังไม่มีใน inventory

---

## กลุ่ม A — แหล่งใน repo (สรุป)

จาก [task_a_source_inventory.csv](../Ai-Credit-Scoring/data/credit_scoring_research/task_a_source_inventory.csv)

| หมวด | จำนวน | แหล่งเด่น |
|---|---|---|
| Regulatory Compliance | 7 | BOT Responsible Lending, DSR, RBP Sandbox, PDPA, EU AI Act, TFRS 9 |
| ML Models | 4 | XGBoost/LightGBM comparison, DL ensemble, SMOTE |
| Explainability/XAI | 4 | Lundberg & Lee SHAP, LIME, Credit Scoring XAI |
| Credit Scoring Fundamentals | 4 | NCB Thailand, FICO, Basel III IRB, PD Model Guide |
| Scorecard Methodology | 3 | WoE/IV, LR Scorecard |
| SME Lending | 3 | NaCGA, ADB SME Thailand, ASEAN Digital Lending |
| Alternative Data | 3 | World Bank, AFI, SSRN |
| Fairness/Bias | 3 | CFPB AI/ML, CFPB Fair Lending 2023, Fair ML Survey |
| Financial Inclusion | 2 | World Bank Findex 2021, IFC SME |
| RAG | **1** ⚠️ | Lewis et al. 2020 (Meta AI) — **ช่องว่างหลัก** |
| Other | 4 | Model Governance SR 11-7, PSI Monitoring, etc. |

**ข้อสังเกต** — หมวด RAG / LLM / Thai NLP ยังขาดมาก กลุ่ม B จะเสริมส่วนนี้

---

## กลุ่ม B — งานวิจัยเพิ่มเติม (จับคู่กับบทในวิทยานิพนธ์)

### B.1 RAG Architecture Foundations (ใช้ในบท 2, 3.8, 5.3.1)

**B1-01. Lewis et al. (2020) — RAG สำหรับ Knowledge-Intensive NLP**
- ผู้เขียน: Lewis, Perez, Piktus, Petroni, Karpukhin, Goyal, Küttler, Lewis, Yih, Rocktäschel, Riedel, Kiela
- Venue: NeurIPS 2020
- arXiv: 2005.11401
- **ความสำคัญต่อวิทยานิพนธ์**: paper ต้นฉบับของ RAG ที่รวม pre-trained parametric (BART) + non-parametric (dense retriever) เสนอสองรูปแบบคือ RAG-Sequence และ RAG-Token สถาปัตยกรรมของวิทยานิพนธ์นี้เป็นการต่อยอดแนวคิดดังกล่าวด้วยการเพิ่ม rule-first router, validator และ Self-RAG reflection
- URL: https://arxiv.org/abs/2005.11401

**B1-02. Gao et al. (2023) — RAG for LLMs: A Survey**
- Venue: arXiv 2312.10997 (v5 submitted Mar 2024)
- **ความสำคัญ**: Survey แบ่ง RAG เป็น 3 ยุค (Naive → Advanced → Modular) วิทยานิพนธ์นี้จัดเป็น **Modular RAG** เพราะแยก router, validator, synthesizer ออกจากกันอย่างชัดเจน
- อ้างใน: บท 2 RAG Taxonomy, บท 5.3.1
- URL: https://arxiv.org/abs/2312.10997

### B.2 Self-RAG & Multi-hop (ใช้ในบท 3.8.8, 4.8.3, 5.2.2)

**B2-01. Asai et al. (2023/2024) — Self-RAG** ⭐ Core reference
- ผู้เขียน: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- Venue: **ICLR 2024 Oral** (top 1%)
- arXiv: 2310.11511
- **ความสำคัญ**: paper ต้นฉบับของ Self-RAG ที่เสนอ reflection tokens `[Retrieve]`, `[IsRel]`, `[IsSup]` วิทยานิพนธ์ implement 3 token นี้ใน `src/rag/self_rag.py` และได้ผล **negative finding** ว่า Self-RAG ทำให้ verdict accuracy ลดลง ซึ่งขัดกับสิ่งที่ paper ต้นฉบับรายงาน — เป็นจุดอภิปรายเด่นในบท 5
- GitHub: https://github.com/AkariAsai/self-rag
- URL: https://arxiv.org/abs/2310.11511 และ https://selfrag.github.io/

**B2-02. Question Decomposition / Multi-hop RAG**
- บทความสำรวจล่าสุด (2024–2025) ระบุว่า retrieval-reasoning coupling เป็นกลุ่มงานที่ใหญ่ที่สุด (55% ของ 252 papers)
- ตระกูลเด่น: ReAct (Yao et al., 2022), Self-Ask (Press et al., 2022), Decomposed Prompting (Khot et al., 2022), Plan-and-Solve (Wang et al., 2023), Least-to-Most (Zhou et al., 2022)
- **ความสำคัญ**: A2 ของวิทยานิพนธ์ implement query decomposition เป็น 2–4 sub-questions คล้าย Decomposed Prompting และ Plan-and-Solve
- Survey: https://arxiv.org/html/2601.00536v1 (Retrieval-Reasoning Processes for Multi-hop QA)

### B.3 Embedding Models สำหรับภาษาไทย (ใช้ในบท 3.8.4, 5.3.3)

**B3-01. Chen et al. (2024) — BGE-M3 (M3-Embedding)** ⭐ Core reference
- ผู้เขียน: Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu
- Venue: **Findings of ACL 2024**
- arXiv: 2402.03216
- **ความสำคัญ**: paper ต้นฉบับของ embedding model ที่วิทยานิพนธ์ใช้ (`BAAI/bge-m3`) รองรับ 100+ ภาษา (รวมไทย), ทำได้ทั้ง dense, sparse, multi-vector retrieval, รองรับ input 8,192 tokens ใช้ self-knowledge distillation ในการฝึก
- HuggingFace: https://huggingface.co/BAAI/bge-m3
- URL: https://arxiv.org/abs/2402.03216

**B3-02. SCB 10X (2023) — Typhoon: Thai Large Language Model**
- ผู้พัฒนา: SCB 10X (subsidiary of SCBX Group)
- arXiv: 2312.13951 (Dec 2023)
- **ความสำคัญ**: Thai-tuned LLM ที่ performance เทียบ GPT-3.5 (Typhoon 7B) และ Typhoon 1.5X Instruct (70B) ชนะ GPT-4 Turbo และ Claude 3 Sonnet ใน ThaiExam leaderboard
- **อ้างในบท 5.7.2** เป็น future work ของการเปรียบเทียบ provider (ทดลองกับ Thai-native LLM แทน Gemini/Ollama)
- URL: https://arxiv.org/pdf/2312.13951 และ https://www.scbx.com/en/news/scb-10x-unveils-large-language-model-typhoon/

### B.4 Evaluation Framework (ใช้ในบท 3.10, 5.2.4)

**B4-01. Es et al. (2023) — RAGAS: Automated RAG Evaluation**
- ผู้เขียน: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- Venue: **EACL 2024 Demo**
- arXiv: 2309.15217
- Metrics หลัก 4 ตัว: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **ความสำคัญ**: Framework ที่ทำ reference-free evaluation — วิทยานิพนธ์นี้ **ไม่ได้ใช้ RAGAS** แต่ออกแบบ custom metrics ที่เหมาะกับ advisory system (Router Accuracy, IsSup Score, Documented Evidence Rate, Verdict Accuracy) ต้องอธิบายใน Ch 3.10 เหตุผลว่าทำไมไม่ใช้ RAGAS
- URL: https://arxiv.org/abs/2309.15217

**B4-02. LLM-as-a-Judge Surveys (2024–2025)**
- Survey: Gu et al. (2024) — A Survey on LLM-as-a-Judge, arXiv 2411.15594
- Bias study: Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge, arXiv 2410.02736 (NeurIPS 2024)
- **ความสำคัญ**: ระบุ biases หลัก (position, verbosity, self-enhancement, primacy/recency) ที่วิทยานิพนธ์นี้ใช้ Gemini เป็น Judge ของผลลัพธ์จาก Gemini เอง จึงมี **self-enhancement bias** ต้องยอมรับเป็น limitation ในบท 5.6.5
- Multilingual reliability: "How Reliable is Multilingual LLM-as-a-Judge?" ACL Findings 2025 — ชี้ว่าการประเมินภาษาที่ไม่ใช่ English ยังไม่น่าเชื่อถือเท่า
- URLs: https://arxiv.org/abs/2411.15594 และ https://arxiv.org/abs/2410.02736

### B.5 ML Foundations (ใช้ในบท 3.x, 4.1)

**B5-01. Ke et al. (2017) — LightGBM** ⭐ Core reference
- ผู้เขียน: Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu
- Venue: **NeurIPS 2017**
- **ความสำคัญ**: paper ต้นฉบับของ LightGBM ที่วิทยานิพนธ์ใช้เป็นโมเดลหลัก ([`backend/model/lgbm_model.pkl`](../backend/model/lgbm_model.pkl)) เสนอ 2 เทคนิคหลัก: Gradient-based One-Side Sampling (GOSS) และ Exclusive Feature Bundling (EFB) เพิ่มความเร็วการฝึก 20 เท่าเทียบ GBDT ปกติ
- URL: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

**B5-02. Lundberg & Lee (2017) — SHAP** ⭐ Core reference (มีใน 36 แหล่งแล้ว)
- Venue: NeurIPS 2017 Oral
- arXiv: 1705.07874
- **ความสำคัญ**: ฐานของโมดูล `backend/app/explain.py` ที่ใช้ `TreeExplainer` ต่อ LightGBM
- URL: https://arxiv.org/abs/1705.07874

### B.6 XAI in Credit Scoring (ใช้ในบท 2, 4.6, 5.3.2)

**B6-01. Bussmann et al. (2020) — Explainable AI in Fintech Risk Management** ⭐
- Venue: **Frontiers in Artificial Intelligence**, Vol. 3, Article 26
- Dataset: 15,000 SMEs P2P lending
- **ความสำคัญ**: paper อ้างอิงหลักสำหรับการใช้ SHAP ในบริบทสินเชื่อ SME สอดคล้องกับบริบทไทย (NaCGA, CIMB Thai home loan) วิทยานิพนธ์ต่อยอดด้วยการ "แปล SHAP → คำแนะนำภาษาไทย" ซึ่งเป็นจุดที่ Bussmann ไม่ได้ทำ
- URL: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00026/full

**B6-02. PLOS ONE (2024) — Shapley Values for Credit Scorecards**
- "A novel framework for enhancing transparency in credit scoring: Leveraging Shapley values for interpretable credit scorecards"
- **ความสำคัญ**: งานล่าสุดที่ใช้ SHAP กับ XGBoost/Random Forest/LightGBM/CatBoost และแปลผลเป็น credit scorecard แบบ interpretable
- URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308718

**B6-03. Nature Sci. Reports (2025) — Hybrid Boosted Attention LightGBM for Credit Risk**
- "Hybrid boosted attention-based LightGBM framework for enhanced credit risk assessment in digital finance"
- **ความสำคัญ**: งานล่าสุดที่ enhance LightGBM ด้วย attention mechanism สามารถอ้างเป็น future direction
- URL: https://www.nature.com/articles/s41599-025-05230-y

### B.7 Credit Scoring ใน Thai Context (ใช้ในบท 2, 5.3.3)

**B7-01. Puey Ungphakorn Institute (2021) — Credit Risk Database: Credit Scoring Models for Thai SMEs**
- สถาบัน: Puey Ungphakorn Institute for Economic Research (pier.or.th) — ร่วมกับ BOT
- **ความสำคัญ**: ผลการทดลอง ML บนข้อมูล Thai SMEs พบว่า random forest ดีที่สุด แต่ logistic regression user-friendly กว่า — แสดงให้เห็น interpretability-accuracy trade-off ในบริบทไทย
- URL: https://ideas.repec.org/p/pui/dpaper/168.html

**B7-02. Credit Risk & LLM for P2P Lending (2024)**
- "Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending"
- arXiv: 2401.16458
- **ความสำคัญ**: ใช้ BERT สร้าง risk score จาก loan description text ต่อยอดเป็น signal ของโมเดล structured
- URL: https://arxiv.org/html/2401.16458v1

**B7-03. NLP for Credit Risk (2024)**
- "Application of Natural Language Processing in Financial Risk Detection" — arXiv 2406.09765
- **ความสำคัญ**: Survey NLP techniques สำหรับ credit/financial risk สามารถอ้างในบท 2 เพื่อวาง context
- URL: https://arxiv.org/abs/2406.09765

### B.8 Vector Stores & Infrastructure (ใช้ในบท 3.8.5)

**B8-01. ChromaDB vs FAISS Comparison (Industry Analysis 2024–2025)**
- **ความสำคัญ**: งานเทียบที่มักสรุปว่า ChromaDB เหมาะกับ 10k–200k vectors (ขนาดของวิทยานิพนธ์นี้อยู่ในช่วงนี้) และมี metadata filtering ที่ดีกว่า FAISS — ตรงกับเหตุผลที่วิทยานิพนธ์เลือก ChromaDB (3.8.5)
- Reference: LiquidMetal AI Vector DB Comparison 2025, Firecrawl Best Vector DB 2026
- URL: https://liquidmetal.ai/casesAndBlogs/vector-comparison/

### B.9 LLM Safety & Structured Output (ใช้ในบท 3.9.4, 3.11, 4.10)

**B9-01. Guardrails AI (Open-source framework)**
- GitHub: guardrails-ai/guardrails
- **ความสำคัญ**: Framework สำหรับ structural + type + quality validation ของ LLM output ใช้ Pydantic-style schemas วิทยานิพนธ์นี้ implement แนวคิดเดียวกัน (Pydantic AssistantResponse + hard-coded blocklists) โดยไม่ใช้ library ภายนอก
- URL: https://github.com/guardrails-ai/guardrails

**B9-02. Safeguarding LLMs Survey (PMC 2025)**
- "Safeguarding large language models: a survey"
- **ความสำคัญ**: Survey 4 ด้านของ guardrails: input checks, runtime constraints, output filtering, policy compliance — mapping กับ safety guardrails ในวิทยานิพนธ์ (5 ด้านจากบท 3.11)
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12532640/

**B9-03. Hybrid Retrieval / BM25+Dense (2024)**
- "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG" — arXiv 2503.23013
- "Integrate sparse and dense vectors" — AWS OpenSearch blog 2024
- **ความสำคัญ**: แนวทาง hybrid retrieval ที่วิทยานิพนธ์เสนอเป็น future work ใน 5.7.2 — RRF (Reciprocal Rank Fusion) เป็นเทคนิคที่อ้างได้
- URL: https://arxiv.org/pdf/2503.23013

---

## แนวทางจัดเรียงงานวิจัยที่เกี่ยวข้องใน **บทที่ 2**

เสนอโครงสร้างบทที่ 2 ดังนี้

### 2.1 การประเมินความเสี่ยงสินเชื่อ
- Basel III IRB (GL-BASEL-001, 36-sources)
- FICO 5-factor model (GL-FICO-001)
- NCB Thailand (TH-NCB-001)
- **B7-01** Thai SME Credit Scoring (Puey Institute)

### 2.2 การเรียนรู้ของเครื่องสำหรับ Credit Scoring
- **B5-01** LightGBM (Ke et al., 2017)
- XGBoost/LightGBM comparison (GL-ML-001)
- **B6-02** PLOS ONE 2024 — Shapley Scorecards
- **B6-03** Nature 2025 — Hybrid Attention LightGBM

### 2.3 Explainable AI และ SHAP
- **B5-02** SHAP (Lundberg & Lee, 2017)
- **B6-01** Bussmann Fintech Risk (2020) ⭐
- LIME (GL-XAI-002)
- XAI Credit Scoring CFA (GL-XAI-003)

### 2.4 Retrieval-Augmented Generation
- **B1-01** Lewis et al. (2020) — RAG foundations
- **B1-02** Gao et al. (2023) — RAG Survey
- **B2-01** Asai et al. (2023) — Self-RAG ⭐
- **B2-02** Multi-hop QA decomposition (Wang 2023, Khot 2022)

### 2.5 Multilingual Embeddings และ Thai NLP
- **B3-01** BGE-M3 (Chen et al., 2024) ⭐
- **B3-02** Typhoon LLM (SCB 10X, 2023)
- **B7-02** Credit Risk + LLM P2P (2024)

### 2.6 การประเมินระบบ RAG
- **B4-01** RAGAS (Es et al., 2023)
- **B4-02** LLM-as-Judge Survey (2024)
- **B4-02** LLM-as-Judge Biases (2024)

### 2.7 Responsible AI และกำกับดูแล
- BOT Responsible Lending (TH-BOT-001)
- PDPA (TH-PDPA-001)
- CFPB AI/ML (GL-FAIR-001, GL-FAIR-002)
- Fair ML Survey (GL-FAIRML-001)
- EU AI Act Annex III (GL-FAIR-003)
- **B9-02** Safeguarding LLMs Survey (2025)

### 2.8 Vector Stores และ Infrastructure (สั้น ๆ)
- **B8-01** ChromaDB vs FAISS — industry analysis
- **B9-01** Guardrails AI — framework comparison

---

## สรุป

- แหล่งเดิมใน repo 36 รายการ + งานใหม่ 15+ รายการ = **50+ citations พร้อมใช้**
- ⭐ Core references ที่ **ต้องมีใน bibliography**: Ke 2017 (LightGBM), Lundberg & Lee 2017 (SHAP), Lewis 2020 (RAG), Asai 2024 (Self-RAG), Chen 2024 (BGE-M3), Bussmann 2020 (XAI Fintech), Es 2024 (RAGAS)
- ช่องว่างที่ยังควรค้นเพิ่ม (ถ้ามีเวลา):
  1. Thai-specific BERT/embedding evaluation (WangchanBERTa, HoogBERTa)
  2. Credit scoring ที่ใช้ RAG หรือ LLM โดยตรง (2024–2026) — ยังใหม่มาก
  3. Responsible AI monitoring frameworks (EU AI Act compliance toolkits)

---

## แหล่งอ้างอิงทั้งหมด (URLs)

**Papers & Surveys**
- Lewis et al. 2020 RAG: https://arxiv.org/abs/2005.11401
- Gao et al. 2023 RAG Survey: https://arxiv.org/abs/2312.10997
- Asai et al. 2024 Self-RAG: https://arxiv.org/abs/2310.11511
- Chen et al. 2024 BGE-M3: https://arxiv.org/abs/2402.03216
- Es et al. 2024 RAGAS: https://arxiv.org/abs/2309.15217
- Ke et al. 2017 LightGBM: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- Lundberg & Lee 2017 SHAP: https://arxiv.org/abs/1705.07874
- Bussmann et al. 2020 XAI Fintech: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00026/full
- SCB 10X 2023 Typhoon: https://arxiv.org/pdf/2312.13951
- LLM-as-Judge Survey 2024: https://arxiv.org/abs/2411.15594
- LLM-as-Judge Biases 2024: https://arxiv.org/abs/2410.02736
- Thai SME Credit (Puey Institute): https://ideas.repec.org/p/pui/dpaper/168.html
- P2P Lending + LLM 2024: https://arxiv.org/html/2401.16458v1
- Shapley Credit Scorecards 2024 PLOS: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308718
- Hybrid LightGBM Attention 2025 Nature: https://www.nature.com/articles/s41599-025-05230-y
- NLP Financial Risk Survey 2024: https://arxiv.org/abs/2406.09765
- Safeguarding LLMs Survey 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12532640/
- Multi-hop QA Survey: https://arxiv.org/html/2601.00536v1
- DAT Hybrid Retrieval 2024: https://arxiv.org/pdf/2503.23013

**Tools & Implementations**
- Self-RAG GitHub: https://github.com/AkariAsai/self-rag
- BGE-M3 HuggingFace: https://huggingface.co/BAAI/bge-m3
- RAGAS GitHub: https://github.com/explodinggradients/ragas
- Guardrails AI: https://github.com/guardrails-ai/guardrails
- ChromaDB: https://www.trychroma.com/

**Regulatory**
- BOT Responsible Lending: https://www.bot.or.th/en/financial-institutions/financial-consumer-protection/responsible-lending.html
- BOT DSR: https://www.bot.or.th/en/financial-system/macro-prudential-policy/debt-service-ratio.html
- CFPB AI/ML Guidance: (ใน 36-sources)

**Vector DB Analyses**
- LiquidMetal AI 2025: https://liquidmetal.ai/casesAndBlogs/vector-comparison/
- Firecrawl Best Vector DB 2026: https://www.firecrawl.dev/blog/best-vector-databases
