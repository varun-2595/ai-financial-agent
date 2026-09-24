"""
Evidence-Grounded Financial RAG Package for Aegis.

Provides parsing, chunking, embedding, vector search, and grounded citation verification.
"""
from src.rag.citation import CitationVerifier
from src.rag.chunker import FinancialChunker
from src.rag.document_store import FinancialDocumentStore
from src.rag.embeddings import LocalFinancialEmbedder
from src.rag.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    EvidenceCitation,
    RetrievalQuery,
    RetrievalResult,
)
from src.rag.parser import FinancialDocumentParser
from src.rag.retriever import FinancialRAGRetriever
from src.rag.vector_store import BaseVectorStore, SQLiteVectorStore

__all__ = [
    "CitationVerifier",
    "FinancialChunker",
    "FinancialDocumentParser",
    "FinancialDocumentStore",
    "FinancialRAGRetriever",
    "LocalFinancialEmbedder",
    "BaseVectorStore",
    "SQLiteVectorStore",
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "EvidenceCitation",
    "RetrievalQuery",
    "RetrievalResult",
]
