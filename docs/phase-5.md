# Phase 5: Evidence-Grounded Financial RAG (Retrieval-Augmented Generation)

## 1. Executive Summary

Phase 5 equips Aegis with an **evidence-grounded Financial RAG engine**. Every material fundamental claim, valuation thesis, and news catalyst made by the multi-agent network is grounded in authoritative corporate filings from the **US (SEC EDGAR)** and **India (NSE/BSE)**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          AUTHORITATIVE SOURCES                         │
├───────────────────────────────────┬────────────────────────────────────┤
│           US MARKETS              │           INDIA MARKETS            │
│  • SEC 10-K (Annual Reports)      │  • NSE/BSE Regulatory Filings      │
│  • SEC 10-Q (Quarterly Reports)   │  • Audited Annual Reports          │
│  • SEC 8-K (Material Events)      │  • Quarterly Financial Results     │
│  • Earnings Transcripts           │  • Investor Presentations          │
└─────────────────┬─────────────────┴─────────────────┬──────────────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   FinancialDocumentParser   │
                     │  (Extracts SEC/NSE Sections)│
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │      FinancialChunker       │
                     │ (Section & Page Breadcrumbs)│
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │    LocalFinancialEmbedder   │
                     │  (Apple Silicon M5 + Cache) │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │      SQLiteVectorStore      │
                     │   (Dense + Keyword Hybrid)  │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │    FinancialRAGRetriever    │
                     │  (Metadata Filtering Top-K) │
                     └──────────────┬──────────────┘
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
┌──────────────────────────────┐              ┌──────────────────────────────────────┐
│       FundamentalAgent       │              │              NewsAgent               │
│  (Grounded Balance Sheet &   │              │     (Grounded 8-K Disclosures &      │
│     Valuation Multiples)     │              │          Corporate Catalysts)        │
└──────────────┬───────────────┘              └──────────────────┬───────────────────┘
               │                                                 │
               └────────────────────────────┬────────────────────┘
                                            ▼
                             ┌─────────────────────────────┐
                             │      CitationVerifier       │
                             │   (Strict Anti-Hallucination│
                             │    "Insufficient evidence.")│
                             └─────────────────────────────┘
```

---

## 2. Core Guardrails & Evidence Grounding Rules

1. **Mandatory Evidence Grounding**: `FundamentalAgent` and `NewsAgent` must retrieve authoritative filing chunks before generating material claims.
2. **Strict Anti-Hallucination**: Citations are never fabricated or extrapolated.
3. **Graceful Insufficient Evidence Handling**: If no authoritative filings exist for a queried asset or relevance falls below threshold:
   - Output explicitly sets `evidence: ["Insufficient evidence."]`.
   - Returns neutral `HOLD` with reduced confidence ($0.30$).
4. **Local M5 Acceleration**: Document parsing, section extraction, chunking, and dense vector embeddings run locally on Apple Silicon / M5 Mac with persistent SQLite caching, resulting in **zero unnecessary API costs** and sub-millisecond query execution.
5. **Separation of Concerns**: RAG retrieval informs the reasoning and conviction of the domain agents. Trade sizing, margin arithmetic, buying power, and portfolio NAV calculations remain strictly deterministic Python code.

---

## 3. Metadata Schema

Every indexed filing chunk retains the canonical `DocumentMetadata` schema:

| Metadata Field | Type | Description |
| :--- | :--- | :--- |
| `company` | `str` | Full corporate name (e.g., `Apple Inc.`, `Reliance Industries Limited`) |
| `ticker` | `str` | Standardized ticker symbol (e.g., `AAPL`, `RELIANCE.NS`) |
| `market` | `us` \| `india` | Geographic market identifier |
| `document_type` | `DocumentType` | `10-K`, `10-Q`, `8-K`, `earnings_transcript`, `quarterly_results`, etc. |
| `published_date` | `str` (YYYY-MM-DD) | Date filed with SEC EDGAR or NSE/BSE |
| `source` | `str` | Authoritative source (`SEC EDGAR`, `NSE India`, `BSE`) |
| `section` | `str` | Detected section (e.g., `Item 1A. Risk Factors`, `Item 7. MD&A`, `Segment Reporting`) |
| `page_number` | `int` | Document page index breadcrumb |

---

## 4. Citation Format & Examples

Verifiable evidence citations are formatted into structured strings and bracketed inline references:

### Inline Bracket Tag:
`[Apple Inc. (AAPL) 10-K, Item 7. MD&A, p.1, 2024-10-31 (SEC EDGAR)]`

### Evidence Output Sample:
```json
{
  "agent": "FundamentalAgent",
  "signal": "BUY",
  "confidence": 0.80,
  "reasons": [
    "10B Neural Model: High ROE (145.0%) and disciplined capital structure (D/E: 1.20) [Apple Inc. (AAPL) 10-K, Item 7. MD&A, p.1, 2024-10-31 (SEC EDGAR)]",
    "Attractive valuation multiple of 28.4x P/E relative to growth profile [Apple Inc. (AAPL) 10-K, Item 7. MD&A, p.1, 2024-10-31 (SEC EDGAR)]"
  ],
  "risks": [
    "Filing disclosure risk per SEC EDGAR (2024-10-31)",
    "Execution variance relative to management guidance in latest disclosures"
  ],
  "evidence": [
    "Source: SEC EDGAR | 10-K (2024-10-31), Section: Item 7. MD&A, p.1 - \"Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations Total net sales were $383...\" (Relevance: 0.27)"
  ]
}
```

---

## 5. Verification & Test Suite

The test suite in [`tests/test_phase5_rag.py`](file:///Users/varun/Coding/AI%20Financial%20Tool/tests/test_phase5_rag.py) comprehensively verifies:
- US SEC (10-K, 10-Q, 8-K) and Indian NSE/BSE section extraction
- Chunking with section breadcrumbs and page preservation
- Local M5 embedding generator with persistent SQLite cache
- Hybrid dense cosine + keyword vector store filtering
- Citation integrity and anti-hallucination validation
- Non-existent asset "Insufficient evidence." safety handling
- Full integration with `FundamentalAgent` and `NewsAgent`

**Total Test Suite**: **102 / 102 tests passing** (`pytest tests/ -v`).
