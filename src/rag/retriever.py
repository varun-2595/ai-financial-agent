"""
Evidence Retriever for Financial RAG.

Fetches authoritative document chunks from the vector store,
extracts verbatim quotes, and structures ground-truth evidence citations.
"""
from __future__ import annotations

import time
from typing import Optional

from src.rag.models import (
    DocumentChunk,
    EvidenceCitation,
    MarketType,
    RetrievalQuery,
    RetrievalResult,
)
from src.rag.vector_store import BaseVectorStore, SQLiteVectorStore
from src.utils.logger import logger


class FinancialRAGRetriever:
    """Retrieves relevant corporate filing chunks and formats ground-truth evidence."""

    def __init__(self, vector_store: Optional[BaseVectorStore] = None):
        self.vector_store = vector_store or SQLiteVectorStore()

    def retrieve_evidence(
        self,
        ticker: str,
        query_text: str,
        market: Optional[MarketType] = None,
        top_k: int = 4,
        min_similarity: float = 0.20,
    ) -> RetrievalResult:
        """
        Execute hybrid search and extract structured citations.
        """
        t0 = time.perf_counter()
        query = RetrievalQuery(
            ticker=ticker.upper(),
            market=market.lower() if market else None,
            query_text=query_text,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        scored_chunks = self.vector_store.search(query)
        latency = (time.perf_counter() - t0) * 1000.0

        if not scored_chunks:
            return RetrievalResult(
                query=query,
                chunks=[],
                citations=[],
                has_sufficient_evidence=False,
                context_text="[Insufficient evidence: No corporate filings or disclosures found matching this query.]",
                retrieval_latency_ms=latency,
            )

        chunks: list[DocumentChunk] = []
        citations: list[EvidenceCitation] = []
        context_parts: list[str] = []

        for chunk, score in scored_chunks:
            chunks.append(chunk)
            meta = chunk.metadata

            # Extract first 200 characters as quote snippet
            quote_snippet = chunk.text.strip().replace("\n", " ")[:200] + "..."

            citation = EvidenceCitation(
                company=meta.company,
                ticker=meta.ticker,
                market=meta.market,
                document_type=meta.document_type,
                published_date=meta.published_date,
                source=meta.source,
                section=meta.section,
                page_number=meta.page_number,
                exact_quote=quote_snippet,
                relevance_score=score,
            )
            citations.append(citation)

            # Build grounded context block for LLM prompt
            context_parts.append(
                f"--- EVIDENCE CHUNK ({citation.format_inline()}) ---\n"
                f"Section: {meta.section} | Page: {meta.page_number or 'N/A'}\n"
                f"{chunk.text}\n"
            )

        context_text = "\n".join(context_parts)
        has_sufficient = len(chunks) > 0 and scored_chunks[0][1] >= min_similarity

        logger.debug(
            f"[Retriever] Retrieved {len(chunks)} chunks for {ticker} "
            f"(Top score: {scored_chunks[0][1]:.3f}, Latency: {latency:.1f}ms)"
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            citations=citations,
            has_sufficient_evidence=has_sufficient,
            context_text=context_text,
            retrieval_latency_ms=latency,
        )
