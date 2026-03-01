# E-Commerce RAG Chatbot – ShopEasy Support Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that answers customer queries strictly from a
set of curated E-Commerce policy and product documents.  Built with **LangChain + OpenAI + FAISS**.

---

## Table of Contents
1. [Project Overview](#-project-overview)
2. [Knowledge Base Documents](#-knowledge-base-documents)
3. [Architecture](#-architecture)
4. [Folder Structure](#-folder-structure)
5. [Setup & Installation](#-setup--installation)
6. [Environment Variables](#-environment-variables)
7. [How to Run](#-how-to-run)
8. [Sample Conversation](#-sample-conversation)
9. [Functional Requirements Checklist](#-functional-requirements-checklist)
10. [Key Design Decisions](#-key-design-decisions)

---

## 📦 Project Overview

| Item | Detail |
|---|---|
| **Domain** | E-Commerce customer support |
| **Brand** | ShopEasy |
| **LLM** | OpenAI `gpt-4o-mini` |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Store** | FAISS (local, CPU) |
| **Framework** | LangChain v0.2 |
| **Memory** | In-process conversation history (last 10 turns) |

---

## 📄 Knowledge Base Documents

The chatbot is backed by ShopEasy's internal policy and product documents, stored as PDFs under `documents/`:

| File | What's inside |
|---|---|
| `return_policy.pdf` | Return window by category, eligibility criteria, refund timelines, damaged/defective item process |
| `shipping_policy.pdf` | Domestic and international shipping options, tracking, failed delivery handling, address changes |
| `warranty_policy.pdf` | Warranty periods by product category, coverage details, claim steps, ShopEasy Protect extended plan |
| `smarthub_x1_manual.pdf` | Full user manual for the SmartHub X1 – setup, specs, LED indicators, device pairing, troubleshooting |

To use your own documents, just drop PDFs into the `documents/` folder and re-run `ingest.py`.

---

## 🏗 Architecture

```
Customer Query
      │
      ▼
 ┌──────────┐     FAISS Similarity    ┌──────────────────────┐
 │  chatbot │ ──────────────────────► │  FAISS Vector Store  │
 │   .py    │ ◄──── Top-5 Chunks ──── │ (ecommerce_faiss_    │
 └──────────┘                         │  index/)             │
      │                               └──────────────────────┘
      │  Build prompt with:
      │   • System instructions
      │   • Conversation history (last 10 turns)
      │   • Retrieved context (5 chunks)
      │   • Current question
      ▼
 ┌──────────────┐
 │  GPT-4o-mini │  → Grounded answer or fallback message
 └──────────────┘
```

**Ingestion pipeline** (`ingest.py`):
```
PDF files  →  PyPDFLoader  →  RecursiveCharacterTextSplitter (700 tokens, 120 overlap)
           →  OpenAIEmbeddings  →  FAISS.from_documents  →  save_local()
```

---

## 📁 Folder Structure

```
E-Commerce_RAG_Chatbot/
│
├── documents/                    # Company policy and product PDFs
│   ├── return_policy.pdf
│   ├── shipping_policy.pdf
│   ├── warranty_policy.pdf
│   ├── smarthub_x1_manual.pdf
│
├── ecommerce_faiss_index/        # FAISS index (created after running ingest.py)
│   ├── index.faiss
│   └── index.pkl
│
├── ingest.py                     # Document ingestion & FAISS indexing
├── chatbot.py                    # RAG chatbot with conversation memory
├── ecommerce_rag_notebook.ipynb  # Jupyter Notebook with full walkthrough
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── README.md                     # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- OpenAI API key

### 1. Clone / navigate to the project folder
```bash
cd "Week 15/E-Commerce_RAG_Chatbot"
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file
```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```
Edit `.env` and set your `OPENAI_API_KEY`.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `OPENAI_API_BASE` | Optional | Custom API endpoint (Azure OpenAI or proxy) |

---

## 🚀 How to Run

### Step 1 – Add your documents
Place your company PDF files inside the `documents/` folder. The chatbot will only answer based on what's in these files.

### Step 2 – Ingest and index documents
```bash
python ingest.py
```
This loads the PDFs, splits them into chunks, embeds them, and saves the FAISS index to `ecommerce_faiss_index/`.

### Step 3 – Start the chatbot
```bash
python chatbot.py
```

### Special commands during chat
| Command | Action |
|---|---|
| `clear` | Resets conversation history |
| `exit` | Exits the chatbot |

---

## 💬 Sample Conversation

```
==================================================================
  ShopEasy Customer Support Chatbot
  Powered by RAG + GPT-4o-mini
  Type 'exit' to quit | Type 'clear' to reset conversation
==================================================================

You: What is the return policy for electronics?

ShopEasy Bot: For electronics and gadgets, ShopEasy offers a **15-day return window**
from the date of delivery. To be eligible, the item must:
  • Be in original, unused condition
  • Include all original packaging, accessories, and documentation
  • Not be on the non-returnable list (e.g., opened software titles)
  • Have a valid proof of purchase (order ID or receipt)

A restocking fee of 10% applies if the item is returned without the original
factory seal intact.

(Source: Return & Refund Policy, Section 2 & 7)

-----------------------------------------------------------------

You: What if I received a defective one?

ShopEasy Bot: If you received a defective electronic item, please contact ShopEasy
**within 48 hours of delivery**. Here's what to do:

  1. Attach clear photographs of the defect along with your order ID.
  2. Raise a complaint via My Orders > Report Issue, or contact:
     - Email: returns@shopeasy.com
     - Phone: 1-800-SHOP-EASY (Mon–Sat, 9 AM – 6 PM IST)
     - Live Chat: Available 24/7 on the ShopEasy app

ShopEasy will arrange a **free pick-up** and either replace the item or issue a
**full refund** — your choice.

Note: This falls under the damaged/defective items clause and is not subject to
the 15-day electronics return window restriction.

-----------------------------------------------------------------

You: How long does the refund take?

ShopEasy Bot: Refund processing times after the returned item is received and
quality-checked:

  • Credit/Debit Card: 5–7 business days
  • Net Banking: 3–5 business days
  • UPI / Wallets: 1–3 business days
  • ShopEasy Credit (store credit): Instant upon return approval
  • Cash on Delivery orders: 7 business days (via ShopEasy Credit or bank transfer)

(Source: Return & Refund Policy, Section 5 & Customer Service FAQ)

-----------------------------------------------------------------

You: Tell me how to setup SmartHub X1

ShopEasy Bot: Here's the initial setup process for the ShopEasy SmartHub X1:

  1. Connect the hub to your router using the provided Ethernet cable.
  2. Plug in the 12V/2A power adapter. The LED ring glows **red** during boot
     (takes 30–60 seconds).
  3. Download the **'ShopEasy Home'** app (App Store or Google Play).
  4. Open the app → tap '+' → select 'SmartHub X1'.
  5. Follow the in-app pairing wizard.
  6. The LED ring turns **blue** when successfully connected.
  7. Create your home profile and add rooms to start pairing smart devices.

(Source: SmartHub X1 User Manual, Section 4 – Initial Setup)

-----------------------------------------------------------------

You: What colors does it support for LED?

ShopEasy Bot: The SmartHub X1 does not display multiple LED colours for decoration —
instead, the LED ring uses specific colours to indicate **device status**:

  • Solid Red     – Booting up or no internet connection
  • Blinking Red  – Firmware update in progress (do not unplug)
  • Solid Blue    – Connected and operating normally
  • Blinking Blue – Pairing mode active
  • Solid Yellow  – Local network connected but no internet
  • Fast Blinking Yellow – Device overheating warning
  • Off           – No power

(Source: SmartHub X1 User Manual, Section 5 – LED Status Indicators)

-----------------------------------------------------------------

You: What about its warranty?

ShopEasy Bot: The SmartHub X1 falls under the **ShopEasy Private Label Electronics**
category and is covered by a **1-year ShopEasy warranty**.

Coverage includes:
  ✅ Manufacturing defects present at time of purchase
  ✅ Hardware failures under normal use
  ✅ Display/LED defects (above threshold)

NOT covered:
  ❌ Physical damage (drops, liquid damage)
  ❌ Damage from unauthorised repair or modification
  ❌ Normal wear and tear
  ❌ Tampered serial numbers

To claim warranty: My Orders > Warranty Claim, or visit shopeasy.com/warranty-claim.

You can also purchase **ShopEasy Protect** for an extended 1 or 2-year plan, which
also covers 1 accidental damage incident.

(Source: Warranty Policy, Sections 2, 3, 4, 6)

-----------------------------------------------------------------

You: exit

Thank you for using ShopEasy Support. Goodbye! 👋
```

---

## ✅ Functional Requirements Checklist

| Requirement | Status | Implementation |
|---|---|---|
| Load documents from local folder | ✅ | `PyPDFLoader` in `ingest.py` |
| Split into semantic chunks | ✅ | `RecursiveCharacterTextSplitter` (700 chars, 120 overlap) |
| OpenAI embeddings | ✅ | `text-embedding-3-small` |
| FAISS vector store | ✅ | `FAISS.from_documents` + `save_local` |
| Top-k retrieval | ✅ | `k=5` similarity search |
| Context injection into prompt | ✅ | `ChatPromptTemplate` with retrieved chunks |
| Strict document grounding | ✅ | System prompt instructions |
| Conversation history / memory | ✅ | `MessagesPlaceholder` + rolling history (10 turns) |
| Follow-up question support | ✅ | History passed to every LLM call |
| Fallback for missing info | ✅ | `"I don't have enough information in the provided documents."` |
| README with setup & run steps | ✅ | This file |
| Jupyter Notebook walkthrough | ✅ | `ecommerce_rag_notebook.ipynb` |

---

## 🔧 Key Design Decisions

| Decision | Rationale |
|---|---|
| `RecursiveCharacterTextSplitter` with `chunk_size=700` | Policy documents tend to have short numbered clauses, so smaller chunks prevent unrelated content mixing into the same chunk |
| `top_k=5` instead of 4 | Customer queries often touch multiple documents at once (e.g., defective item – return policy + warranty), so a wider retrieval net helps |
| Rolling 10-turn history | Keeps the context window manageable while still supporting natural follow-up questions |
| `temperature=0` | Policy answers need to be consistent and deterministic, not creative |
| Source + page metadata in context | Helps the model cite the right document section and makes responses more trustworthy |
