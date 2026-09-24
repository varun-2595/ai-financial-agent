"""
RAG Data Models for Evidence-Grounded Financial Reasoning.

Defines schemas for corporate filings, chunks, metadata, citations, and retrieval results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


DocumentType = Literal[
    "10-K",
    "10-Q",
    "8-K",
    "earnings_transcript",
    "nse_filing",
    "bse_filing",
    "annual_report",
    "quarterly_results",
    "investor_presentation",
    "press_release",
]

MarketType = Literal["india", "us"]


class DocumentMetadata(BaseModel):
    """Canonical metadata schema for all financial documents."""
    company: str
    ticker: str
    market: MarketType
    document_type: DocumentType
    published_date: str  # YYYY-MM-DD
    source: str          # e.g., "SEC EDGAR", "NSE India", "BSE", "Company Investor Relations"
    page_number: Optional[int] = None
    section: Optional[str] = None  # e.g., "Item 1A. Risk Factors", "MD&A", "Segment Revenue"
    extra_attributes: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Full corporate document before chunking."""
    doc_id: str
    title: str
    metadata: DocumentMetadata
    raw_text: str
    created_at: datetime = Field(default_factory=_utcnow)


class DocumentChunk(BaseModel):
    """Atomic text chunk with embedding and specific section metadata."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: DocumentMetadata
    token_count: int
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=_utcnow)


class EvidenceCitation(BaseModel):
    """Structured verifiable citation linking a claim to its source document."""
    company: str
    ticker: str
    market: MarketType
    document_type: DocumentType
    published_date: str
    source: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    exact_quote: str
    relevance_score: float

    def format_inline(self) -> str:
        """Format citation as an inline bracket tag."""
        sec = f", {self.section}" if self.section else ""
        pg = f", p.{self.page_number}" if self.page_number else ""
        return f"[{self.company} ({self.ticker}) {self.document_type}{sec}{pg}, {self.published_date} ({self.source})]"


class RetrievalQuery(BaseModel):
    """Query parameters for evidence retrieval."""
    ticker: str
    market: Optional[MarketType] = None
    query_text: str
    document_types: Optional[list[DocumentType]] = None
    top_k: int = 4
    min_similarity: float = 0.25


class RetrievalResult(BaseModel):
    """Result of RAG retrieval containing matching chunks and citations."""
    query: RetrievalQuery
    chunks: list[DocumentChunk] = Field(default_factory=list)
    citations: list[EvidenceCitation] = Field(default_factory=list)
    has_sufficient_evidence: bool = False
    context_text: str = ""
    retrieval_latency_ms: float = 0.0
