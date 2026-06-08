# บทที่ 3 วิธีดำเนินการวิจัย (ฉบับแก้ไข — Section 3.8–3.14)

> เอกสารนี้แก้สิ่งที่ไม่ตรงกับ repo และเพิ่ม section ที่ควรมีอ้างอิงหลักฐานจริงในโค้ด การอ้างอิง path ใช้รูปแบบ `path/to/file` เพื่อให้ตรวจสอบย้อนกลับได้

---

## 3.8 การพัฒนาสถาปัตยกรรมระบบ Retrieval-Augmented Generation สำหรับการอธิบายผลการประเมินเครดิต

ภายหลังจากการพัฒนาแบบจำลองการเรียนรู้ของเครื่องสำหรับการประเมินความเสี่ยงทางเครดิตในขั้นตอนก่อนหน้า ผู้วิจัยพบว่าแม้แบบจำลองสามารถให้ผลลัพธ์เชิงการทำนายได้อย่างมีประสิทธิภาพ แต่ผลลัพธ์ดังกล่าวยังอยู่ในรูปแบบคะแนนความน่าจะเป็นหรือการจำแนกประเภท ซึ่งยากต่อการตีความสำหรับผู้ใช้งานทั่วไป อีกทั้งยังไม่สามารถให้เหตุผลเชิงบริบทหรือข้อเสนอแนะที่อ้างอิงจากองค์ความรู้ภายนอกได้โดยตรง

เพื่อแก้ไขข้อจำกัดดังกล่าว งานวิจัยนี้จึงพัฒนาสถาปัตยกรรม Retrieval-Augmented Generation (RAG) ที่ทำหน้าที่เป็นชั้นการประมวลผลเชิงความรู้ (Knowledge Layer) เชื่อมต่อระหว่างผลลัพธ์ของแบบจำลองเครดิตสกอริ่งกับฐานความรู้ด้านการเงินภาษาไทย

### 3.8.1 สถาปัตยกรรม End-to-End แบบ 3 บริการ

ระบบในงานวิจัยนี้ถูกใช้งานแบบ microservices 3 บริการ ตามที่ระบุใน `INTEGRATION.md`

| บริการ | พอร์ต | หน้าที่หลัก |
|---|---|---|
| Frontend (Next.js) | 3000 | รับข้อมูลผู้ขอสินเชื่อและแสดงผล พร้อม API proxy |
| Backend Bridge (FastAPI) | 8000 | ประมวลผลโมเดล LightGBM + SHAP และ orchestrate การเรียก planner |
| Planner / RAG (FastAPI) | 8001 | RAG retrieval + validator + Self-RAG + Planning Module |
| Ollama (optional) | 11434 | ให้บริการ LLM ภายในเครื่องเมื่อ `USE_OLLAMA=true` |

**ระบบประกอบด้วย 5 ชั้นเชิงตรรกะ**

1. **Presentation** — Next.js UI รับข้อมูลผู้ขอสินเชื่อและแสดงผลคะแนน/คำแนะนำ
2. **API Proxy** — Next.js API routes ทำหน้าที่ proxy ไป backend bridge (`/api/predict`, `/api/rag/query`, `/api/rag/advisor`)
3. **Decisioning** — FastAPI backend เรียกโมเดล LightGBM และคำนวณ SHAP พร้อมส่ง request ไปยัง planner
4. **Knowledge** — RAG engine (router → metadata filter → retrieval → validator → synthesis → source attachment)
5. **Data** — SQLite สำหรับ operational records, ChromaDB สำหรับเวกเตอร์เอกสารนโยบาย

> รูปที่ 3.X End-to-end สถาปัตยกรรมระบบ RAG สำหรับการอธิบายผลการประเมินเครดิต (ไฟล์ต้นฉบับ: `docs/architecture-end-to-end.mmd`)

### 3.8.2 การเตรียมเอกสารและ Data Governance Pipeline

ฐานความรู้ที่มีคุณภาพเป็นปัจจัยสำคัญต่อประสิทธิภาพของระบบ RAG ผู้วิจัยรวบรวมเอกสารสินเชื่อบ้านจากเว็บไซต์สาธารณะของ CIMB Thai (public disclosure) และแหล่งกำกับดูแลภาษาไทย จากนั้นผ่าน data governance pipeline 4 ชั้น ก่อนเข้า vector store

**ชั้นที่ 1 — Manifest & Provenance Tracking**

ทุกเอกสารถูกบันทึกใน `Ai-Credit-Scoring/data/manifest_cleaned.json` พร้อม metadata ได้แก่ `id`, `title`, `source_url`, `institution`, `publication_date`, `category`, `file_path`, และ flag `needs_rescrape` สำหรับเอกสารที่ต้องดึงใหม่

**ชั้นที่ 2 — Document Cleaning with Versioned Fingerprint**

โมดูล `StructuredDocumentParser` ที่ `Ai-Credit-Scoring/src/document_parser.py` (1,374 บรรทัด) ทำความสะอาดเอกสารเว็บและฝัง fingerprint `CLEANING_VERSION="2026-03-04-v1"` ในทุก chunk เพื่อป้องกัน index drift (การที่เอกสารถูกแก้แต่ index ยังเป็นฉบับเก่า) ระบบมี runtime check ว่า fingerprint ของ collection ตรงกับเวอร์ชันที่ใช้ขณะ ingest หรือไม่ บันทึกใน `results/ingest_log3.txt` ว่า `Fingerprint sanity check passed: CLEANING_VERSION=2026-03-04-v1`

การทำความสะอาดใช้ rule-based patterns ดังนี้
- `NOISE_TERMS` (18 คำ) — กรองแถบนำทางของเว็บ เช่น "ติดต่อเรา", "โปรโมชั่น", "cookie"
- `LOAN_CONTEXT_TERMS` (6 คำ) — ยืนยันว่าเอกสารเกี่ยวข้องกับสินเชื่อบ้าน ("สินเชื่อบ้าน", "รีไฟแนนซ์", "mortgage"…)
- `TABLE_SIGNAL_TERMS` — ตรวจจับโครงสร้างตาราง ("MRR", "MLR", "MOR", "LTV", "%", "ปีที่")
- `CHROME_TOKENS` — ตรวจจับ web chrome ("search", "quicklinks", "sitemap")

**ชั้นที่ 3 — Relevance Quarantine**

เอกสารที่ไม่ผ่านเกณฑ์จะถูก quarantine ออกจากชุดที่นำไป index โดยมี reason code 5 ประเภท

| Reason code | ความหมาย |
|---|---|
| `no_home_loan_keywords_or_url_hints` | ไม่พบคำหลักสินเชื่อบ้านหรือ URL hint |
| `high_chrome_noise_ratio` | อัตราส่วน noise จาก web chrome สูงเกินเกณฑ์ |
| `unrelated_title_topic` | หัวข้อเอกสารไม่เกี่ยวกับสินเชื่อบ้าน |
| `unrelated_navigation_content` | เป็นเนื้อหา navigation มากกว่าสาระ |
| `negative_indicators_dominate` | สัญญาณลบ (ไม่เกี่ยว) มากกว่าสัญญาณบวก |

ผลจากการรัน ingest (จาก `Ai-Credit-Scoring/results/ingest_log3.txt`)

| ตัวชี้วัด | จำนวน |
|---|---|
| เอกสารต้นทาง (scraped) | 36 |
| เอกสารที่เข้า index | 25 |
| เอกสารที่ถูก quarantine | 11 |
| Collection name | `cimb_loans_bge_m3` |
| CLEANING_VERSION | `2026-03-04-v1` |

นอกจากนี้ `data/cleaning_report.json` บันทึกการแก้ไขเอกสารอย่างเป็นระบบ ได้แก่ ลบเอกสารที่ไม่เกี่ยวข้องจาก corpus 5 รายการ (เช่น privacy-notice, customer-profiling, COVID relief ปี 2020-2021) แก้วันที่ผิดพลาดจาก crawl date 2 รายการ และปรับ category ที่จัดกลุ่มผิด เช่น `home-loan-refinance-th` จาก `bank_policy` → `interest_structure`

**ชั้นที่ 4 — Document Audit with Severity Scoring**

ก่อน index ทุกเอกสารถูก audit ด้วยตัวชี้วัดเชิงปริมาณ บันทึกใน `Ai-Credit-Scoring/reports/doc_audit.csv` และสรุปใน `doc_audit_summary.txt`

| ตัวชี้วัด | ความหมาย |
|---|---|
| `noise_line_ratio` | สัดส่วนบรรทัดที่เข้าข่าย noise |
| `table_likeness_score` | ความน่าจะเป็นว่าเป็นตาราง |
| `row_conversion_score` | คะแนนการแปลงตารางเป็นข้อความเรียง |
| `duplicate_boilerplate_score` | คะแนนข้อความซ้ำที่เป็น boilerplate |
| `severity` | คะแนนรวม (ถ่วงน้ำหนัก) ใช้ตัดสิน `needs_review` |

จากการรันครั้งล่าสุด มีเอกสารทั้งหมด 29 ไฟล์ที่ถูก audit และ 11 ไฟล์ (≈37.9%) ถูก flag ว่า `needs_review=True` เพื่อรอตรวจทานรอบต่อไป

### 3.8.3 การแบ่ง Chunk และการแทนข้อความเชิงความหมาย

ผู้วิจัยใช้ `SentenceSplitter` ของ LlamaIndex ที่ `src/data_loader.py` และ `src/ingest.py` โดยตั้งค่าจากไฟล์กลาง `Ai-Credit-Scoring/config/settings.py`

| พารามิเตอร์ | ค่า | ที่มา |
|---|---|---|
| `CHUNK_SIZE` | 512 tokens | env/default |
| `CHUNK_OVERLAP` | 50 tokens | env/default |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | env/default |
| Embedding dimension | 1,024 | จาก model card |
| Tokenizer | BGE-M3 native tokenizer (HF AutoTokenizer) | ใช้ตัวเดียวกับ embedder |

การใช้ tokenizer ของ `BAAI/bge-m3` โดยตรงเป็นสิ่งสำคัญสำหรับภาษาไทย เพราะไทยไม่มี whitespace แบ่งคำ การนับ token ด้วย tokenizer เดียวกับ embedder ทำให้ขอบเขตของ chunk สอดคล้องกับการ embed จริง

> หมายเหตุ: ค่า `CHUNK_SIZE=512` และ `CHUNK_OVERLAP=50` เป็น default ที่ระบบใช้จริง การเปรียบเทียบค่า overlap ที่ 0/30/50/100 ยังไม่มีการบันทึกใน repo หากต้องการอ้างเป็นผลการทดลองในเล่ม ควรรันเพิ่มและบันทึกตัวเลข

### 3.8.4 การเลือก Embedding Model

ผู้วิจัยเลือก `BAAI/bge-m3` ด้วยเหตุผล 3 ประการ
1. **Multilingual native support** — รองรับภาษาไทยและอังกฤษใน embedding space เดียวกัน เหมาะกับคลังเอกสารที่มีทั้งสองภาษาปน
2. **Dense + sparse + multi-vector representation** — เป็น M3 (multi-functionality, multi-lingual, multi-granularity)
3. **Open-source ไม่มีข้อจำกัดการใช้งานเชิงพาณิชย์**

> หมายเหตุ: benchmark Recall@5 เปรียบเทียบกับ `multilingual-e5-large` และ `paraphrase-multilingual-mpnet` ยังไม่มีหลักฐานการรันใน repo ถ้าจะใส่ตารางเปรียบเทียบในบท 3 ต้องรันเพิ่ม

### 3.8.5 Vector Store

งานวิจัยนี้ใช้ **ChromaDB** (ไม่ใช่ FAISS) ด้วยเหตุผลด้าน metadata filtering ที่จำเป็นต่อ rule-first router

| การตั้งค่า | ค่า |
|---|---|
| Vector store | ChromaDB (persisted) |
| Persist dir | `./storage/chroma` |
| Collection name | `cimb_loans_bge_m3` |
| Similarity metric | Cosine (normalized) |
| `SIMILARITY_TOP_K` | 4 |
| `SIMILARITY_CUTOFF` | 0.45 |

ChromaDB รองรับ native metadata filtering ซึ่งใช้กำกับ retrieval ตาม `category`, `topic_tags`, `doc_kind` หลังจาก router จัดประเภทคำถาม (ดู 3.8.6)

### 3.8.6 Rule-first Router + Metadata Gating

แทนที่จะใช้ cross-encoder rerank ผู้วิจัยออกแบบกลไก **rule-first routing** ที่ `Ai-Credit-Scoring/src/rag/router.py` เพื่อควบคุม domain drift และ hallucination

**6 route labels** (`ROUTE_LABELS`)
`policy_requirement`, `interest_structure`, `fee_structure`, `refinance`, `hardship_support`, `general_info`

**Route keywords (เฉพาะโดเมนสินเชื่อบ้านไทย)**

| Route | ตัวอย่าง keywords |
|---|---|
| interest_structure | "ดอกเบี้ย", "mrr", "fixed", "floating", "%", "ปีแรก" |
| fee_structure | "ค่าธรรมเนียม", "ค่าปรับ", "ปิดบัญชี", "จดจำนอง", "ประกันอัคคีภัย" |
| refinance | "รีไฟแนนซ์", "บ้านแลกเงิน", "mortgage power" |
| hardship_support | "ผ่อนไม่ไหว", "ปรับโครงสร้างหนี้", "พักชำระ", "โควิด", "น้ำท่วม", "ขยายระยะเวลา" |
| policy_requirement | "คุณสมบัติ", "เอกสาร", "รายได้ขั้นต่ำ", "อาชีพ", "สัญชาติ", "เงื่อนไข" |

**Route priority** — เมื่อคำถามมี keyword หลายโดเมน ระบบใช้ลำดับความสำคัญ `fee_structure > interest_structure > hardship_support > refinance > policy_requirement`

**Safety Blocklist** — คำถามที่มี tokens ต่อไปนี้จะถูกบังคับให้เป็น `general_info` เพื่อให้ validator คืน `NO_ANSWER` แทนที่จะดึงเอกสารนโยบายออกมา

```
ปลอมแปลง, ปลอม, ทุจริต, ฉ้อโกง, แอบอ้าง, โกง,
หลบเลี่ยง, เลี่ยง, ซ่อน, ซ่อนเงิน, ฟอกเงิน,
forgery, fraud, fake, falsify, launder
```

เมื่อ router กำหนด route แล้ว จะสร้าง `MetadataFilters` ของ LlamaIndex เพื่อให้ Chroma ดึงเฉพาะเอกสารที่ `category` หรือ `topic_tags` สอดคล้องกับ route

### 3.8.7 Post-retrieval Validator

ภายหลัง retrieval ผู้วิจัยใช้ validator ชั้นที่สองที่ `Ai-Credit-Scoring/src/rag/validator.py` เพื่อกรองเอกสารที่ไม่เกี่ยวข้องออกก่อนส่งเข้า LLM

**กลไกของ validator**

1. **Similarity cutoff** — ตัด nodes ที่ `score < SIMILARITY_CUTOFF` (0.45)
2. **Doc kind allow-list** — คงเฉพาะ `{"policy", "rate_sheet", "form"}`
3. **Home domain keywords** — ทุก node ต้องมีอย่างน้อยหนึ่งคำจาก `HOME_DOMAIN_KEYWORDS` ("สินเชื่อบ้าน", "รีไฟแนนซ์", "จำนอง", "mortgage", "home loan"…)
4. **Global blocklist** — ตัด nodes ที่มีคำจาก `GLOBAL_BLOCKLIST` ("เงินฝาก", "บัตรเครดิต", "NDID", "พร้อมเพย์", "ภาษี", "fx"…) ยกเว้นคำถามมี `EXPLICIT_ALLOW_TERMS` ด้วย
5. **Route must-have** — nodes ต้องมี token อย่างน้อยหนึ่งคำจาก `ROUTE_MUST_HAVE[route]`
6. **Route-specific blocklist** — ตัด token เฉพาะ route เช่น `refinance` ห้ามมี "NDID/พร้อมเพย์/กรมสรรพากร/ภาษี"
7. **Prepayment hints** — ตรวจ `PREPAYMENT_HINTS` ("ก่อน 5 ปี", "1% ของวงเงินกู้", "ค่าปรับ") สำหรับคำถามเกี่ยวกับการปิดสินเชื่อก่อนกำหนด
8. **Clarification message** — หากคำถามกำกวม ("ปิดบัญชี") ส่งคำถามกลับผู้ใช้ให้ยืนยันว่า "ปิดสินเชื่อก่อนกำหนด" หรือ "ปิดบัญชีเงินฝาก"

หาก validator ไม่เหลือ node ที่ผ่านเกณฑ์ ระบบจะคืน `NO_ANSWER_MESSAGE = "ไม่พบข้อมูลในเอกสารที่มีอยู่"` แทนที่จะบังคับให้ LLM สังเคราะห์คำตอบ

### 3.8.8 Self-RAG Orchestrator

ผู้วิจัยพัฒนาชั้น reflection แบบ Self-RAG ที่ `Ai-Credit-Scoring/src/rag/self_rag.py` ใช้ 3 reflection tokens ต่อยอดจากงานของ Asai et al. (2023)

| Token | หน้าที่ |
|---|---|
| `[Retrieve]` | ตัดสินว่าคำถามจำเป็นต้องค้นหาเอกสารหรือไม่ (yes/no binary) |
| `[IsRel]` | ให้คะแนน relevance ของแต่ละ retrieved node (1–5) และตัด node ที่ต่ำกว่า threshold |
| `[IsSup]` | ประเมินว่า generated answer ได้รับการ support โดย context หรือไม่ หากไม่ผ่านจะ retry 1 ครั้ง |

**LLM call budget ต่อ query**

| เงื่อนไข | จำนวน LLM calls |
|---|---|
| Happy path (IsSup ผ่าน) | 4 (Retrieve + IsRel + synthesis + IsSup) |
| IsSup retry | 6 (เพิ่ม retry_synthesis + retry_IsSup) |
| Retrieve=No (off-domain/greeting) | 1 (คืน NO_ANSWER ทันที) |

Budget นี้ถูกใช้คำนวณ latency ของ A3 approach ในบทที่ 4

### 3.8.9 LRU Cache + TTL

เพื่อลด latency และลดการเรียก LLM ซ้ำ ผู้วิจัยสร้าง cache ที่ `Ai-Credit-Scoring/src/rag/cache.py` โดยใช้เฉพาะ stdlib (ไม่พึ่ง Redis) เพื่อลด deployment footprint

| การตั้งค่า | ค่า default |
|---|---|
| Max entries | 256 (LRU) |
| TTL | 3,600 วินาที |
| Thread-safe | ใช้ `threading.Lock` |
| Stats tracked | hits, misses, evictions, hit_rate |

Cache key สร้างจาก `(question, top_k)` เพื่อให้คำถามเดียวกันที่ขอ top-k ต่างกันไม่ชนกัน

### 3.8.10 กระบวนการสร้างคำตอบ (Generation)

ระบบรวม nodes ที่ผ่าน validator + Self-RAG เข้ากับคำถามของผู้ใช้ แล้วส่งให้ LLM provider ที่กำหนด

**ตัวเลือก LLM provider ที่ทดสอบ** (ไม่ใช่ GPT-4o)

| Provider | Model | เหตุผลในการทดสอบ |
|---|---|---|
| Google Gemini | `gemini-2.5-flash` | Cloud API ที่ latency ต่ำและคุณภาพสูง |
| Ollama (local) | `qwen3:8b` (default) / `gemma3:4b` (ingest) | ใช้ offline/fallback ลด data egress |

การเลือก provider ควบคุมผ่าน env (`USE_OLLAMA`, `USE_GEMINI`) ใน `Ai-Credit-Scoring/config/settings.py`

**Context injection format** — ทุก chunk ถูก prepend ด้วย metadata ("title", "category", "cleaning_version") เพื่อให้ LLM อ้างอิงแหล่งที่มาได้ และมี `source_titles` ส่งกลับมาใน response schema

**Response schema** — `RAGQueryResponse` ที่ `src/api/schemas/payload.py`

```python
class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    router_label: str           # route ที่ detect
    sources: List[RAGSource]    # เอกสารต้นทางพร้อม score
    retrieved_count: int        # nodes ก่อน validator
    validated_count: int        # nodes หลัง validator
    ...
```

### 3.8.11 การเชื่อมโยงระบบ RAG กับผลลัพธ์ของแบบจำลองเครดิต

จุดเด่นของงานวิจัยนี้คือการเชื่อมผล ML + SHAP เข้ากับ RAG ผ่านกลไกที่เรียกว่า **DRIVER_QUERY_MAP** ที่ `Ai-Credit-Scoring/src/planner/planning.py`

แทนที่จะส่ง SHAP values ดิบเป็นข้อความให้ LLM สังเคราะห์ (ซึ่งเสี่ยง hallucinate) ผู้วิจัย hard-code mapping จาก feature → curated Thai queries ที่ถูกทดสอบแล้วว่า retrieve เอกสารที่ถูกต้อง

| SHAP Driver | Query ที่ส่งเข้า RAG (Thai) |
|---|---|
| `overdue` | "ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้", "ขอขยายระยะเวลาผ่อนได้ไหม", "มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง" |
| `outstanding` | (ใช้ queries เดียวกับ overdue — เน้นมาตรการลดหนี้) |
| `loan_amount`, `loan_term`, `Interest_rate` | "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้", "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่", "มี fixed rate หรือ floating rate บ้าง" |
| `Occupation`, `Salary` | "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้", "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง" |
| `credit_score`, `credit_grade` | "เครดิตบูโรสำคัญอย่างไร" |

กลไกนี้มีคุณสมบัติ 3 ประการ
1. **Deterministic** — feature เดียวกันไปสู่ query ชุดเดียวกันเสมอ ช่วย reproducibility
2. **Expert-curated** — query ถูกตรวจแล้วว่าแม่นกับ corpus CIMB Thai ลด misrouting
3. **Explainable bridge** — ผู้ใช้สามารถตรวจได้ว่า driver ใดถูกแปลเป็น query ใด

**Production speed controls** (จาก `INTEGRATION.md`) กำหนดจำนวน query ต่อเคส

| ENV | ค่า default | ความหมาย |
|---|---|---|
| `PLANNER_MAX_ACTION_DRIVERS` | 3 | จำกัดจำนวน drivers ที่ถูกแปลเป็นคำแนะนำ |
| `PLANNER_MAX_RAG_QUERIES_PER_DRIVER` | 1 | จำกัด RAG queries ต่อ driver |
| `PLANNER_APPROVED_MAX_RAG_QUERIES` | 2 | เพดานรวมสำหรับเคสที่อนุมัติ |
| `PLANNER_ENABLE_LLM_SYNTHESIS` | false | ปิด LLM synthesis เพื่อใช้ rule-first plan |

---

## 3.9 การพัฒนา Planning Module สำหรับการสร้างคำแนะนำเชิงปฏิบัติ

แม้ระบบ RAG จะอธิบายผลลัพธ์และให้ข้อมูลได้ แต่ผู้ใช้ต้องการทราบว่า "ควรทำอย่างไรต่อไป" มากกว่าเพียงเหตุผล Planning Module จึงทำหน้าที่เป็น Decision Support Layer แปลงผลวิเคราะห์เป็นแผนปฏิบัติ

### 3.9.1 สถาปัตยกรรมของ Planning Module

Planning Module อยู่ที่ `Ai-Credit-Scoring/src/planner/` ประกอบด้วย 4 ไฟล์หลัก

| ไฟล์ | หน้าที่ | ขนาด |
|---|---|---|
| `planning.py` | สร้าง plan (rule-first + RAG evidence) | 961 บรรทัด |
| `scoring.py` | คำนวณ risk_prob + SHAP-like values (fallback เมื่อไม่มีโมเดล) | 79 บรรทัด |
| `rag_bridge.py` | เชื่อม planner กับ RAG manager และ lookup helpers | 155 บรรทัด |
| `demo.py` | ตัวอย่างการเรียกใช้งาน | 76 บรรทัด |

### 3.9.2 การออกแบบคำแนะนำจาก SHAP (Driver Prioritization)

Planning Module รับค่า SHAP จากโมเดลหลัก (LightGBM ใน `backend/app/explain.py`) แล้วเรียงตาม magnitude ของ contribution เชิงลบ เพื่อจัดลำดับความสำคัญของ action

- Driver ที่มี SHAP เชิงลบสูงสุด → ถูกแปลเป็น action ข้อแรก
- Driver ที่เป็น `NON_ACTIONABLE_FEATURES` (เช่น `Sex`) → ถูกข้ามอัตโนมัติ
- จำกัดไว้สูงสุด `PLANNER_MAX_ACTION_DRIVERS=3` drivers ต่อเคส

### 3.9.3 การผสานข้อมูลจาก RAG

สำหรับ driver แต่ละตัว ระบบเรียก RAG ด้วย queries จาก `DRIVER_QUERY_MAP` (3.8.11) แล้วรวบรวม source ที่ได้มาเป็น evidence ผูกกับ action แต่ละข้อ ทำให้ข้อเสนอแนะมี citation กลับไปยังเอกสารต้นฉบับเสมอ

### 3.9.4 Safety Constraints Hard-coded

ผู้วิจัยกำหนดข้อจำกัดด้าน Responsible AI เป็น hard constraints ในโค้ด (ไม่พึ่ง soft prompt เพียงอย่างเดียว) ที่ `src/planner/planning.py`

| Constraint | รายการ | ผลบังคับ |
|---|---|---|
| `NON_ACTIONABLE_FEATURES` | `{"Sex"}` | ฟีเจอร์นี้ถูกข้ามอัตโนมัติในขั้นเลือก driver |
| `FORBIDDEN_SEX_ACTION_TOKENS` | "เปลี่ยนเพศ", "change sex", "เปลี่ยน gender" | ทุก action ถูก scan ก่อนส่งกลับผู้ใช้ หากพบ token จะถูก drop |
| `FORBIDDEN_FRAUD_TOKENS` | "ปลอม", "ปลอมแปลง", "แก้เอกสาร", "แก้ไขเอกสาร", "fake", "forg", "fraud" (7 tokens) | เช่นเดียวกับข้างต้น |
| `FORBIDDEN_PROMISE_TOKENS` | "รับประกันอนุมัติ", "อนุมัติแน่นอน", "guarantee approval", "guaranteed approval" (4 tokens) | ป้องกันคำสัญญาเกินจริง |

ข้อจำกัดเหล่านี้ถูกทดสอบด้วย safety test cases 3 เคสในชุดทดสอบ Planning (ดูบทที่ 4)

### 3.9.5 โครงสร้างคำตอบ (Structured Output)

งานวิจัยนี้ไม่ใช้ Structured Output ของ GPT-4o แต่ใช้ Pydantic schemas เป็น contract ระหว่าง Python services Planning Module มี output 2 รูปแบบ

**รูปแบบที่ 1 — `AssistantResponse`** (ที่ `src/schema.py`)

```python
AssistantResponse:
    decision: "approve" | "decline" | "need_more_info" | "review"
    summary: str
    reasons: List[Reason]       # type: rule/model/policy + evidence
    missing_info: List[str]
    next_actions: List[str]
    customer_message_draft: Optional[str]
    risk_note: Optional[str]
```

**รูปแบบที่ 2 — `AdvisorResponse`** (ที่ `src/api/schemas/payload.py`) สำหรับ endpoint `/rag/advisor`

```python
AdvisorResponse:
    question: str
    verdict: "eligible" | "partially_eligible" | "ineligible" | "needs_more_info"
    verdict_summary: str
    requirement_checks: List[AdvisorRequirementCheck]
        requirement, user_value, status (pass/fail/unknown/not_applicable), explanation
    recommended_actions: List[str]
    sources: List[RAGSource]
    reasoning_trace: AdvisorReasoningTrace
        used_multihop, sub_questions, sources_per_hop,
        used_self_rag, issup_score (1-5), issup_passed, elapsed_seconds
```

`requirement_checks` แตกต่างจากการตอบแบบ free-text เพราะเป็น checklist ที่ frontend render เป็น row pass/fail ได้ตรง (ดู `frontend/components/ResultCard.tsx`)

---

## 3.10 การออกแบบกรอบการประเมินผลระบบ

งานวิจัยนี้ใช้ชุดตัวชี้วัด custom ที่ออกแบบให้เหมาะกับ retrieval-based advisory system (ไม่ใช่ RAGAS) เนื่องจากตัวชี้วัดของ RAGAS เช่น Context Precision/Recall ต้องการ ground-truth context ต่อคำถาม ซึ่งไม่เหมาะกับลักษณะงานที่ต้องการประเมินด้าน routing, structured verdict และ groundedness ของ plan

### 3.10.1 ชุดทดสอบ RAG (118 เคส)

ชุดทดสอบอยู่ที่ `Ai-Credit-Scoring/data/eval/test_cases.jsonl` และขยายในสคริปต์ evaluator ครอบคลุม 118 เคส กระจายใน 6 route labels

| Route | จำนวน cases |
|---|---|
| hardship_support | 18 |
| policy_requirement | 19 |
| interest_structure | 20 |
| general_info | 20 |
| refinance | 19 |
| fee_structure | 22 |
| **รวม** | **118** |

รูปแบบ case แต่ละอันใน JSONL

```json
{"question": "...", "expected_route": "interest_structure",
 "expected_keywords": ["%", "ดอกเบี้ย"], "should_answer": true}
```

นอกจากเคส in-domain ยังมีเคส off-domain (fraud/forex) เพื่อทดสอบว่า validator คืน NO_ANSWER ถูกต้อง

### 3.10.2 ชุดทดสอบ Advisor (100 เคส, Crossed Design)

ชุดทดสอบ Advisor อยู่ที่ `Ai-Credit-Scoring/data/eval/advisor_test_set.jsonl` ออกแบบเป็น matrix 10 profiles × 10 questions

**Profiles 10 แบบ**

| Label | คำอธิบาย |
|---|---|
| P1_strong_doctor | High-income professional, clean credit |
| P2_solid_engineer | ฐานรายได้มั่นคง, credit ดี |
| P3_average_office | พนักงานออฟฟิศ ระดับเฉลี่ย |
| P4_borderline_short_tenure | อายุงานสั้น, ใกล้เกณฑ์ |
| P5_thin_file_fresh_grad | ประวัติสินเชื่อบาง, จบใหม่ |
| P6_self_employed_strong | อาชีพอิสระ รายได้ดี |
| P7_self_employed_weak | อาชีพอิสระ รายได้ไม่มั่นคง |
| P8_has_overdue | มีประวัติค้างชำระ |
| P9_high_lti | LTI สูง |
| P10_coapplicant_combined | มีผู้กู้ร่วม |

**Question types 4 ประเภท** (รวม 100 เคส)

| Type | จำนวน |
|---|---|
| single_eligibility | 40 |
| factual | 30 |
| advice | 20 |
| multi_eligibility | 10 |

Ground truth ประกอบด้วย `expected_keywords`, `ground_truth_requirements`, และ `expected_verdict` (เฉลย 50 เคส ไม่เฉลย 50 เคสสำหรับเคสที่ขึ้นกับการตีความ)

### 3.10.3 ชุดทดสอบ Planning Module (25 เคส)

ชุดทดสอบ Planning อยู่ใน evaluator `scripts/evaluate_planning.py` ครอบคลุม

| ประเภท | ตัวอย่าง |
|---|---|
| Approved-path | "high income+good credit", "government employee perfect profile", "married couple high combined income" |
| Rejected-path | "overdue+low credit", "salary-driven rejection", "self-employed all drivers negative", "high outstanding (DTI)" |
| Boundary | "prediction=0 but p_approve=0.49 (near threshold)" |
| Missing-info | "missing product_type → clarifying question", "missing coapplicant_income" |
| Safety | "no fraud/guarantee tokens", "Sex feature SHAP negative must NOT appear", "no product_type AND no ltv" |
| IsSup-specific | "LLM plan must pass groundedness check", "all five main drivers negative" |

### 3.10.4 ตัวชี้วัด (Custom RAG + Planning Metrics)

| ตัวชี้วัด | ใช้กับ | วิธีวัด | เกณฑ์ผ่าน |
|---|---|---|---|
| Router Accuracy | RAG | เทียบ `router_label` กับ `expected_route` | ≥ 0.75 (`ROUTE_ACCURACY_MIN` ใน `tests/test_rag_eval.py`) |
| Answer Rate | RAG | % คำถามที่ระบบตอบ (ไม่ใช่ NO_ANSWER) เมื่อ `should_answer=True` | — |
| No-Answer Accuracy | RAG | % คำถาม off-domain ที่ระบบปฏิเสธตอบ | ≥ 0.80 (`NO_ANSWER_ACCURACY_MIN`) |
| Keyword Hit Rate | RAG | % คำตอบที่มีอย่างน้อย 1 `expected_keywords` | ≥ 0.60 (`KEYWORD_HIT_MIN`) |
| Precision@K | RAG | สัดส่วน retrieved sources ที่เป็น relevant (LLM-as-Judge) | — |
| Top-1 / Top-K Similarity | RAG | mean similarity ของ top-1 และ top-K | — |
| Mean Latency | RAG, Plan | วินาทีต่อ query | — |
| Mode Accuracy | Planning | `approved_guidance` vs `improvement_plan` ตรงกับเคสหรือไม่ | 1.00 |
| Documented Evidence Rate | Planning | % คำแนะนำที่แนบ source | 1.00 |
| IsSupported Pass Rate | Planning | % เคสที่ Self-RAG IsSup ผ่าน | 1.00 |
| IsSupported Score (1–5) | Planning | LLM-as-Judge groundedness | ≥ 3.00 |
| Verdict Accuracy | Advisor | `verdict` ตรงกับ `expected_verdict` | — |

### 3.10.5 การออกแบบ Ablation Study (A1 / A2 / A3)

เพื่อพิสูจน์คุณค่าของกลไก reasoning แต่ละชั้น ผู้วิจัยออกแบบ 3 approach ที่ increment กัน (ไม่ใช่ Baseline vs No-rerank vs Rerank) รันบนชุด 100 เคสใน 3.10.2

| Approach | รายละเอียด | Flag |
|---|---|---|
| **A1** Profile-conditioned single-hop | RAG retrieval ครั้งเดียว ใช้ profile เป็น context เสริมใน prompt | `use_multihop=false, use_self_rag=false` |
| **A2** A1 + Multi-hop decomposition | LLM แตกคำถามเป็น 2–4 sub-questions, retrieve แยก, dedup, synthesize | `use_multihop=true, use_self_rag=false` |
| **A3** A2 + Self-RAG reflection | เพิ่ม `[IsRel]` (ให้คะแนน node 1–5) และ `[IsSup]` (ตรวจ groundedness + retry) | `use_multihop=true, use_self_rag=true` |

ผลจะถูกนำเสนอในบทที่ 4 พร้อม breakdown by 4 question types (`advice`, `factual`, `multi_eligibility`, `single_eligibility`)

### 3.10.6 Similarity Distribution Report

นอกจากการประเมินรายเคส ผู้วิจัยเก็บ retrieval logs ที่ `logs/rag_debug.jsonl` และ `logs/retrieval_logs.jsonl` และมีสคริปต์ `src/rag/report.py` สรุป similarity distribution เช่น mean top-1, mean top-K, top-1/top-2 gap, และ no-answer rate เพื่อติดตาม corpus quality ต่อเนื่อง

---

## 3.11 การกำกับดูแล Responsible AI และความปลอดภัยของระบบ

ระบบนี้เกี่ยวข้องกับการตัดสินใจเชิงการเงิน ผู้วิจัยออกแบบกลไก Responsible AI 6 ด้าน

### 3.11.1 การป้องกันข้อมูลอ่อนไหว

- ข้อมูลผู้ใช้ถูกเก็บใน SQLite local เท่านั้น (`sql_app.db`) ไม่มีการส่งออก
- เวลาเรียก LLM provider (Gemini/Ollama) จะส่งเฉพาะ features ตัวเลข/หมวดหมู่ที่ถูก normalize แล้ว ไม่ส่งข้อมูล PII
- Corpus ใน ChromaDB เป็น public disclosure ของ CIMB Thai ไม่มีข้อมูลลูกค้า

### 3.11.2 Groundedness (การควบคุมความถูกต้องของคำตอบ)

- `SIMILARITY_CUTOFF=0.45` ตัด nodes ที่ similarity ต่ำ
- Post-retrieval validator (3.8.7) กรองด้วย 8 กลไก
- หากไม่มี node ผ่านเกณฑ์ ระบบคืน `NO_ANSWER_MESSAGE = "ไม่พบข้อมูลในเอกสารที่มีอยู่"`
- Self-RAG `[IsSup]` ตรวจ groundedness และ retry หนึ่งครั้งถ้าไม่ผ่าน
- Planning Module บังคับทุก action ต้องมี evidence source (`Documented Evidence Rate = 100%`)

### 3.11.3 การป้องกันคำถามนอกขอบเขต (Out-of-domain)

ใช้ `SAFETY_BLOCKLIST` ใน router (3.8.6) — คำถาม fraud/forgery ถูกบังคับไปที่ `general_info` ซึ่ง validator จะไม่พบเอกสารตรงและคืน NO_ANSWER ทดสอบด้วยเคส

- `OFF-DOMAIN: fraud query — must not produce helpful answer`
- `OFF-DOMAIN: forex rate (not in docs)`

### 3.11.4 การตรวจสอบความสอดคล้องของผลลัพธ์ (Consistency Check)

Planning Module บังคับว่า action ต้องสอดคล้องกับ verdict — ผู้ที่มี `overdue` สูงจะไม่ได้รับคำแนะนำให้ก่อหนี้เพิ่ม Safety constraint นี้ทำผ่านการเลือก `DRIVER_QUERY_MAP` ที่เน้น hardship support สำหรับ driver `overdue` และ `outstanding`

### 3.11.5 Research Foundation สำหรับ Responsible AI

ระบบตั้งอยู่บนฐานกำกับดูแลที่รวบรวมใน `data/credit_scoring_research/task_a_sources.json` จำนวน 36 แหล่ง กรอง `trust_level` authoritative 17, high 14, medium_high 3, medium 2 ครอบคลุม

- **Thai regulatory (primary)**: Bank of Thailand Responsible Lending Notification Sor.Kor.Chor. 7/2566, BOT DSR Macroprudential Policy, PDPC Personal Data Protection Act, NCB Thailand
- **Global regulatory**: CFPB (USA), Basel III IRB Framework, FICO 5-Factor Model
- **Academic**: peer-reviewed papers 6 รายการ, research papers 4 รายการ, preprints 1 รายการ

ไฟล์ `task_a_gap_analysis.md` ยังระบุ 5 critical gaps (เช่น GAP-01 "BOT Model Risk Management Circular", GAP-05 "Thailand SME Credit Scoring Empirical Dataset") เพื่อความโปร่งใสว่าฐานความรู้ยังขาดมิติใด

### 3.11.6 การพิจารณา Fair Lending และ Bias Mitigation

ในบริบทการปล่อยสินเชื่อ fair lending เป็นข้อกังวลสำคัญทั้งเชิงจริยธรรมและเชิงกฎหมาย ผู้วิจัยออกแบบมาตรการ 3 ชั้นเพื่อบรรเทา bias

**ชั้นที่ 1 — Non-actionable Feature Exclusion**

`NON_ACTIONABLE_FEATURES = {"Sex"}` ใน `src/planner/planning.py` ห้าม Planning Module อ้างฟีเจอร์ `Sex` เป็นเหตุผลของคำแนะนำ แม้ SHAP value จะชี้ว่า `Sex` มี contribution เพราะการกระทำ "เปลี่ยนเพศ" ไม่ถือเป็น actionable recommendation ตามหลัก fair lending

**ชั้นที่ 2 — Safety Token Screening**

`FORBIDDEN_SEX_ACTION_TOKENS` สแกน recommended_actions ก่อนส่งกลับผู้ใช้ หากพบ token เช่น "เปลี่ยนเพศ" / "change sex" จะตัดออก เพื่อป้องกัน LLM synthesize action ที่ละเมิดหลักฐาน

**ชั้นที่ 3 — Research-grounded Compliance Framework**

ระบบอ้างอิง
- Bank of Thailand Responsible Lending Notification Sor.Kor.Chor. 7/2566 ที่กำหนด ability-to-repay และ Risk-Based Pricing
- Basel III IRB framework ที่แยก PD/LGD/EAD ตามความเสี่ยงจริง ไม่ใช่ demographic
- CFPB fair lending guidance เป็นแนวปฏิบัติระดับสากล

**ข้อจำกัดที่ระบุใน `task_a_gap_analysis.md`**
- GAP-03: การขาด NCB Thailand factor weights อาจทำให้การเทียบเกณฑ์ fair lending ท้องถิ่นไม่ชัดเจน

> หมายเหตุ: การวัดเชิงปริมาณ (Statistical Parity, Equalized Odds, Disparate Impact) ยังไม่มีผลรันใน repo ผู้วิจัยวางไว้เป็นหัวข้อวิจัยต่อยอด (ดูบทที่ 5 §5.6.4 และ §5.7.5)

---

## 3.12 สภาพแวดล้อมการทดลองและการควบคุมการทำซ้ำ

### 3.12.1 เครื่องมือและเวอร์ชันที่ใช้จริง (จาก `requirements.txt`)

**Backend / ML stack**

| ส่วนประกอบ | เวอร์ชัน | หน้าที่ |
|---|---|---|
| Python | 3.12 | ภาษาหลัก |
| FastAPI | 0.135.1 | API framework |
| Uvicorn | 0.41.0 | ASGI server |
| Pydantic | 2.12.5 | Data validation/schema |
| Pandas | 3.0.1 | Data processing |
| NumPy | 2.4.2 | Numerical |
| Scikit-learn | 1.6.1 | Preprocessing (ColumnTransformer) |
| LightGBM | 4.6.0 | โมเดลหลัก |
| SHAP | 0.48.0 | Explainability |

**RAG stack**

| ส่วนประกอบ | เวอร์ชัน | หน้าที่ |
|---|---|---|
| llama-index-core | ≥ 0.14.0 | RAG orchestration |
| llama-index-llms-google-genai | latest | Gemini 2.5 Flash integration |
| llama-index-llms-ollama | latest | Ollama local LLM |
| llama-index-embeddings-huggingface | latest | BGE-M3 embeddings |
| llama-index-vector-stores-chroma | latest | Chroma integration |
| ChromaDB | latest | Vector store |
| Ollama | 0.6.1 | Local LLM runtime |
| BAAI/bge-m3 | — | Embedding model (1,024 dim) |

**Testing**

| ส่วนประกอบ | เวอร์ชัน | หน้าที่ |
|---|---|---|
| pytest | 7.4.3 | Unit + integration tests |
| pytest-asyncio | 0.21.1 | async test support |
| hypothesis | 6.92.1 | Property-based testing |

**Frontend**

- Next.js (React) พอร์ต 3000
- Bun / npm เป็น package manager (มี `bun.lock` และ `package.json`)

### 3.12.2 การตั้งค่า Deployment Orchestration

ใช้ PowerShell scripts เพื่อ reproducibility (ที่ `scripts/`)

| สคริปต์ | หน้าที่ |
|---|---|
| `dev_up.ps1` | เริ่ม 3 services พร้อม health check (planner→backend→frontend) |
| `dev_down.ps1` | stop ทุก service |
| `prod_smoke.ps1` | smoke test API endpoints |

`dev_up.ps1 -Clean` เคลียร์ process บน port 3000/8000/8001 ก่อนเริ่มใหม่ และรอ health endpoint ตอบ 200 ก่อนประกาศ ready — เพื่อให้ผลรัน benchmark ไม่ถูกรบกวนจาก warmup

### 3.12.3 การควบคุมการทำซ้ำ (Reproducibility)

- **Code versioning**: Git (branch `main`)
- **Environment**: `requirements.txt` + `uv.lock` (ใช้ `uv` เป็น package manager)
- **Data fingerprint**: `CLEANING_VERSION="2026-03-04-v1"` ฝังใน chunks และตรวจทุกครั้งที่โหลด index
- **Test logging**: ผลการรัน evaluation บันทึกใน `Ai-Credit-Scoring/results/` พร้อม timestamp ในชื่อไฟล์ (`compare_20260320_164805.json`) และ meta field `timestamp`, `llm_provider`, `embed_model`, `total_cases`
- **Retrieval audit**: ทุก query ถูก log ที่ `logs/rag_debug.jsonl` (1,425 records) และ `logs/retrieval_logs.jsonl` (1,533 records) เพื่อวิเคราะห์ post-hoc
- **Configuration**: `.env` template ใน `INTEGRATION.md` ระบุค่าที่ reproducible
- **Test suite**: 14 test files ~2,300 lines ที่ `tests/` รวมถึง property-based tests (`test_preservation_properties.py`)

---

## 3.13 การทดสอบระบบ (Test Infrastructure)

ผู้วิจัยพัฒนา test suite เพื่อรับประกันความถูกต้องของระบบและใช้เป็น executable specification

| Test file | บรรทัด | ครอบคลุม |
|---|---|---|
| `test_document_parser_cleaning.py` | 41 | การทำความสะอาดเอกสารและ fingerprint |
| `test_document_parser_metadata.py` | 52 | การฝัง metadata ลง chunks |
| `test_document_parser_quarantine.py` | 101 | logic ของ quarantine 5 reason codes |
| `test_indexer_chroma_reset.py` | 72 | การ rebuild collection และ sanity check |
| `test_metadata_formatting.py` | 209 | การ format metadata สำหรับ context injection |
| `test_planner_quality.py` | 377 | คุณภาพ plan (documented evidence, safety) |
| `test_planning_engine.py` | 232 | Planning engine end-to-end |
| `test_preservation_properties.py` | 337 | Property-based tests (hypothesis) |
| `test_query.py` | 268 | Query engine behaviour |
| `test_query_ollama.py` | 67 | Ollama integration |
| `test_rag_eval.py` | 140 | RAG metrics thresholds |
| `test_rag_router_validator.py` | 64 | Router + validator correctness |
| `test_scoring_api.py` | 94 | API contract ของ scoring endpoint |
| `test_bug_condition_exploration.py` | 207 | Regression tests จาก bug reports |
| **รวม** | **~2,261** | **14 files** |

Test suite นี้ทำงานกับ Chroma collection จริง (ไม่ mock) เพื่อให้ข้อผิดพลาดจาก integration layer ไม่ถูกซ่อน

---

## 3.14 สรุปวิธีดำเนินการ

ระบบที่พัฒนาในงานวิจัยนี้เป็นการบูรณาการ 3 องค์ประกอบ

1. **ML Scoring + SHAP** — LightGBM บน 15 features (11 numeric + 4 categorical, engineered 3 ตัว: `dti`, `lti`, `has_overdue`) พร้อม TreeExplainer
2. **RAG Knowledge Layer** — ChromaDB + BGE-M3 + Rule-first Router (6 routes + priority + safety blocklist) + Post-retrieval Validator (8 กลไก) + Self-RAG (3 reflection tokens + LLM call budget 4–6 calls) + LRU Cache (256 entries, TTL 1 ชม.)
3. **Planning Module** — `AssistantResponse` / `AdvisorResponse` schemas พร้อม `DRIVER_QUERY_MAP` (model→RAG bridge), safety constraints 4 ชุด, และ checklist-style verdict

การประเมินใช้ชุดตัวชี้วัด custom (Router Accuracy, Answer Rate, No-Answer Accuracy, P@K, Keyword Hit, Mean Latency, IsSup Score, Documented Evidence Rate, Verdict Accuracy) รันบน 118 RAG cases, 25 Planning cases, และ 100 Advisor cases (10×10 crossed design) พร้อม ablation 3 approaches (A1/A2/A3) และทดสอบกับ 2 LLM providers (Gemini 2.5 Flash, Ollama) ผลการทดลองนำเสนอในบทที่ 4

---

## ภาคผนวก (เสนอเพิ่ม)

### ก. Prompt templates ที่ใช้จริง

- RAG synthesis prompt (เน้นอ้างอิงเอกสาร)
- Self-RAG `[Retrieve]` / `[IsRel]` / `[IsSup]` templates ที่ `src/rag/self_rag.py`
- Multi-hop decomposition prompt ที่ `src/rag/multihop.py`

### ข. Route keyword dictionary และ Safety blocklist เต็ม

อ้างจาก `src/rag/router.py` และ `src/rag/validator.py`

### ค. Research source inventory 36 รายการ

สำเนาจาก `data/credit_scoring_research/task_a_source_inventory.csv`

### ง. Gap analysis

สำเนาจาก `data/credit_scoring_research/task_a_gap_analysis.md`
