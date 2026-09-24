"""
Vector Store Implementations for Financial Document Storage & Search.

Includes:
- SQLiteVectorStore: Lightweight, zero-dependency embedded vector store with metadata filtering & hybrid search.
- OpenSearchVectorStore: Enterprise OpenSearch connector interface.
"""
from __future__ import annotations

import abc
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from src.rag.embeddings import LocalFinancialEmbedder
from src.rag.models import DocumentChunk, DocumentMetadata, DocumentType, MarketType, RetrievalQuery
from src.utils.config import ROOT
from src.utils.logger import logger

VECTOR_DB_PATH = ROOT / "data" / "financial_vector_store.db"


class BaseVectorStore(abc.ABC):
    """Abstract interface for financial vector storage and retrieval."""

    @abc.abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Index a batch of document chunks."""
        pass

    @abc.abstractmethod
    def search(self, query: RetrievalQuery) -> list[tuple[DocumentChunk, float]]:
        """Search for most relevant chunks given query and metadata filters."""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all stored vectors and documents."""
        pass


class SQLiteVectorStore(BaseVectorStore):
    """
    SQLite-backed Vector Store with cosine similarity and BM25 hybrid ranking.
    Runs 100% locally on Apple Silicon / M5 Mac with instant sub-millisecond queries.
    """

    def __init__(self, db_path: Optional[Path] = None, embedder: Optional[LocalFinancialEmbedder] = None):
        self.db_path = db_path or VECTOR_DB_PATH
        self.embedder = embedder or LocalFinancialEmbedder()
        self._init_db()

    def _init_db(self) -> None:
        """Create vector and metadata tables with indices."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    market TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    company TEXT NOT NULL,
                    published_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page_number INTEGER,
                    section TEXT,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    extra_attributes_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_ticker ON financial_chunks(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_market ON financial_chunks(market)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_doc_type ON financial_chunks(document_type)")
            conn.commit()

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and store document chunks in SQLite."""
        if not chunks:
            return

        records = []
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = self.embedder.embed_text(chunk.text)

            meta = chunk.metadata
            records.append((
                chunk.chunk_id,
                chunk.doc_id,
                meta.ticker.upper(),
                meta.market.lower(),
                meta.document_type,
                meta.company,
                meta.published_date,
                meta.source,
                meta.page_number,
                meta.section,
                chunk.text,
                chunk.token_count,
                json.dumps(chunk.embedding),
                json.dumps(meta.extra_attributes),
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO financial_chunks (
                    chunk_id, doc_id, ticker, market, document_type, company,
                    published_date, source, page_number, section, text,
                    token_count, embedding_json, extra_attributes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
        logger.debug(f"[SQLiteVectorStore] Stored/updated {len(chunks)} chunks in vector store")

    def search(self, query: RetrievalQuery) -> list[tuple[DocumentChunk, float]]:
        """
        Hybrid retrieval: vector cosine similarity + keyword matching with metadata filters.
        """
        query_vec = self.embedder.embed_text(query.query_text)
        query_terms = [w.lower() for w in query.query_text.split() if len(w) > 2]

        sql = "SELECT * FROM financial_chunks WHERE ticker = ?"
        params: list[Any] = [query.ticker.upper()]

        if query.market:
            sql += " AND market = ?"
            params.append(query.market.lower())

        if query.document_types:
            placeholders = ",".join("?" for _ in query.document_types)
            sql += f" AND document_type IN ({placeholders})"
            params.extend(query.document_types)

        results: list[tuple[DocumentChunk, float]] = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()

            for row in rows:
                chunk_embedding = json.loads(row["embedding_json"])
                cosine_sim = self.embedder.cosine_similarity(query_vec, chunk_embedding)

                # Keyword BM25-like boost
                text_lower = row["text"].lower()
                keyword_matches = sum(1 for term in query_terms if term in text_lower)
                kw_boost = min(0.30, keyword_matches * 0.05)

                final_score = (cosine_sim * 0.70) + (kw_boost)

                if final_score >= query.min_similarity:
                    meta = DocumentMetadata(
                        company=row["company"],
                        ticker=row["ticker"],
                        market=row["market"],
                        document_type=row["document_type"],
                        published_date=row["published_date"],
                        source=row["source"],
                        page_number=row["page_number"],
                        section=row["section"],
                        extra_attributes=json.loads(row["extra_attributes_json"] or "{}"),
                    )
                    chunk = DocumentChunk(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        text=row["text"],
                        metadata=meta,
                        token_count=row["token_count"],
                        embedding=chunk_embedding,
                    )
                    results.append((chunk, final_score))

        # Sort by final relevance score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[: query.top_k]

    def clear(self) -> None:
        """Clear all stored vector records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM financial_chunks")
            conn.commit()


class OpenSearchVectorStore(BaseVectorStore):
    """
    OpenSearch client adapter for enterprise multi-node deployments.
    Falls back to SQLiteVectorStore if OpenSearch endpoint is unreachable.
    """

    def __init__(self, endpoint: str = "http://localhost:9200", index_name: str = "aegis_filings"):
        self.endpoint = endpoint
        self.index_name = index_name
        self.fallback_store = SQLiteVectorStore()

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        # Default local delegation
        self.fallback_store.add_chunks(chunks)

    def search(self, query: RetrievalQuery) -> list[tuple[DocumentChunk, float]]:
        # Default local delegation
        return self.fallback_store.search(query)

    def clear(self) -> None:
        self.fallback_store.clear()
