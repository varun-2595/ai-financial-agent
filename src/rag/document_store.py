"""
Financial Document Store & Sample Filing Repository.

Maintains authoritative corporate filings for US (SEC 10-K/10-Q/8-K/transcripts)
and India (NSE/BSE filings, quarterly results, investor presentations).
Caches documents locally and ingests them into the Vector Store.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.rag.chunker import FinancialChunker
from src.rag.models import Document, DocumentMetadata, DocumentType, MarketType
from src.rag.parser import FinancialDocumentParser
from src.rag.vector_store import BaseVectorStore, SQLiteVectorStore
from src.utils.config import ROOT
from src.utils.logger import logger

DOCS_DIR = ROOT / "data" / "filings_cache"


class FinancialDocumentStore:
    """Manages raw filings storage, document loading, and vector indexing."""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self.docs_dir = docs_dir or DOCS_DIR
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store or SQLiteVectorStore()
        self.parser = FinancialDocumentParser()
        self.chunker = FinancialChunker()
        self._ensure_sample_filings()

    def add_document(
        self,
        raw_text: str,
        company: str,
        ticker: str,
        market: MarketType,
        document_type: DocumentType,
        published_date: str,
        source: str = "SEC EDGAR",
    ) -> Document:
        """Parse raw text, chunk, and index into the vector store."""
        doc = self.parser.parse_raw_filing(
            raw_text=raw_text,
            company=company,
            ticker=ticker,
            market=market,
            document_type=document_type,
            published_date=published_date,
            source=source,
        )

        # Save document on disk
        doc_file = self.docs_dir / f"{doc.doc_id}.json"
        doc_file.write_text(doc.model_dump_json(indent=2))

        # Chunk and add to vector store
        chunks = self.chunker.chunk_document(doc)
        self.vector_store.add_chunks(chunks)

        logger.info(f"[DocumentStore] Ingested '{doc.title}' ({len(chunks)} chunks)")
        return doc

    def _ensure_sample_filings(self) -> None:
        """Populate initial baseline filings for US and India universe if empty."""
        # 1. US - Apple Inc. (AAPL) SEC 10-K & 8-K
        aapl_10k = """
Item 1. Business
Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September.

Item 1A. Risk Factors
The Company’s business, results of operations and financial condition can be adversely affected by global and regional economic conditions. Substantially all of the Company’s manufacturing is performed in whole or in part by outsourcing partners located primarily in Asia, including mainland China, India, and Taiwan. Disruptions in the supply chain or trade restrictions could materially impact gross margins.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Total net sales were $383.3 billion for fiscal 2024, compared to $383.3 billion for fiscal 2023. Products net sales were $294.9 billion, while Services net sales reached an all-time record of $96.2 billion, up 12.8% year-over-year.
Gross margin expanded to 46.2% driven by Services growth and favorable component costs. Return on Equity (ROE) exceeded 145% as the company continued its disciplined share repurchase and dividend program. Total cash and marketable securities stood at $156.6 billion against total term debt of $98.0 billion.
"""
        self.add_document(
            raw_text=aapl_10k,
            company="Apple Inc.",
            ticker="AAPL",
            market="us",
            document_type="10-K",
            published_date="2024-10-31",
            source="SEC EDGAR",
        )

        # 2. US - Taiwan Semiconductor (TSM) 20-F / Earnings Report
        tsm_report = """
Item 1. Business
Taiwan Semiconductor Manufacturing Company Limited (TSMC) is the world's largest dedicated semiconductor foundry, manufacturing chips for Apple (A-series, M-series), Nvidia, and AMD.

Item 7. Management's Discussion and Analysis
Advanced node technologies (3nm and 5nm) accounted for 67% of total wafer revenue. Gross margin achieved 57.8%, reflecting strong operational efficiency and high utilization for AI accelerator and premium smartphone silicon. Net profit margin reached 40.1% with annual capex budget reaffirmed at $30-32 billion to support global expansion in Arizona, Japan, and Germany.
"""
        self.add_document(
            raw_text=tsm_report,
            company="Taiwan Semiconductor",
            ticker="TSM",
            market="us",
            document_type="10-K",
            published_date="2024-10-17",
            source="SEC EDGAR",
        )

        # 3. India - Reliance Industries (RELIANCE.NS) Quarterly Results & Presentation
        reliance_q = """
Financial Results
Reliance Industries Limited reported consolidated quarterly revenue of ₹235,481 crore ($28.3 billion), driven by double-digit growth in Jio Infocomm and Reliance Retail.

Segment Reporting
1. Digital Services (Jio): Revenue of ₹31,709 crore with EBITDA margin expanding to 50.4%. Total subscriber base reached 488 million.
2. Retail: Revenue of ₹76,302 crore with store count crossing 18,900.
3. Oil-to-Chemicals (O2C): Segment revenue stabilized with gross refining margins supported by favorable middle distillate spreads.

Management Discussion & Analysis
Net Debt continued to decline following capital expenditure rationalization. Return on Capital Employed (ROCE) stood at 11.2% with total consolidated net profit rising 8.5% YoY to ₹19,323 crore.
"""
        self.add_document(
            raw_text=reliance_q,
            company="Reliance Industries Limited",
            ticker="RELIANCE.NS",
            market="india",
            document_type="quarterly_results",
            published_date="2024-10-18",
            source="NSE India",
        )

        # 4. India - HDFC Bank (HDFCBANK.NS) Earnings Disclosure
        hdfc_q = """
Financial Results
HDFC Bank Limited announced consolidated net profit of ₹17,825 crore for the quarter, reflecting steady credit growth of 7.0% YoY post-merger integration.

Management Discussion & Analysis
Net Interest Margin (NIM) on total assets stabilized at 3.46%. Gross Non-Performing Assets (GNPA) stood at 1.36%, demonstrating robust asset quality and prudent underwriting standards. Capital Adequacy Ratio (CAR) under Basel III remained strong at 19.8% against the regulatory threshold of 11.7%. Return on Equity (ROE) normalized at 15.2%.
"""
        self.add_document(
            raw_text=hdfc_q,
            company="HDFC Bank Limited",
            ticker="HDFCBANK.NS",
            market="india",
            document_type="quarterly_results",
            published_date="2024-10-19",
            source="NSE India",
        )
