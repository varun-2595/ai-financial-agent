"""
Test Suite for Phase 5: Evidence-Grounded Financial RAG.

Covers:
- Parser & Section Extraction for US (SEC) and India (NSE/BSE)
- Financial Chunker with breadcrumbs and page preservation
- Local M5 Embedder & Persistent SQLite Cache
- Vector Store Search with Ticker/Market/DocType Filtering
- Retrieval Quality and Relevance Scoring
- Citation Integrity, Grounding, and Anti-Hallucination Checks
- 'Insufficient evidence.' handling for ungrounded assets
- FundamentalAgent & NewsAgent RAG Integration
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.agents.fundamental_agent import FundamentalAgent
from src.agents.news_agent import NewsAgent
from src.data.models import Fundamentals, NewsItem, StockSnapshot
from src.rag.citation import CitationVerifier
from src.rag.chunker import FinancialChunker
from src.rag.document_store import FinancialDocumentStore
from src.rag.embeddings import LocalFinancialEmbedder
from src.rag.models import Document, DocumentMetadata, EvidenceCitation, RetrievalQuery
from src.rag.parser import FinancialDocumentParser
from src.rag.retriever import FinancialRAGRetriever
from src.rag.vector_store import SQLiteVectorStore


@pytest.fixture
def temp_rag_env():
    """Create isolated temporary directory for test RAG vector store and cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test_vec.db"
        cache_path = tmp_path / "test_embed_cache.db"
        docs_dir = tmp_path / "filings"

        embedder = LocalFinancialEmbedder(cache_db_path=cache_path)
        vec_store = SQLiteVectorStore(db_path=db_path, embedder=embedder)
        doc_store = FinancialDocumentStore(docs_dir=docs_dir, vector_store=vec_store)
        retriever = FinancialRAGRetriever(vector_store=vec_store)

        yield {
            "tmp_path": tmp_path,
            "embedder": embedder,
            "vec_store": vec_store,
            "doc_store": doc_store,
            "retriever": retriever,
        }


def test_parser_sec_and_india_sections():
    """Test parsing and section boundary detection for US and India corporate filings."""
    parser = FinancialDocumentParser()

    # US 10-K text
    us_text = """
Item 1. Business
Apple Inc. designs, manufactures and markets smartphones and computers.

Item 1A. Risk Factors
Supply chain disruptions in Asia could affect gross margins.

Item 7. Management's Discussion and Analysis
Total net sales were $383.3 billion for fiscal 2024. Services grew 12.8%.
"""
    sections_us = parser.extract_sections(us_text, market="us")
    assert len(sections_us) == 3
    section_names_us = [s[0] for s in sections_us]
    assert "Item 1. Business" in section_names_us
    assert "Item 1A. Risk Factors" in section_names_us
    assert "Item 7. MD&A" in section_names_us

    # India Quarterly Results
    india_text = """
Financial Results
Reliance reported quarterly revenue of ₹235,481 crore with 8.5% YoY net profit growth.

Segment Reporting
Retail revenue crossed ₹76,302 crore with store count crossing 18,900.

Management Discussion & Analysis
Net debt continued to decline following capex rationalization.
"""
    sections_in = parser.extract_sections(india_text, market="india")
    assert len(sections_in) == 3
    section_names_in = [s[0] for s in sections_in]
    assert "Financial Results" in section_names_in
    assert "Segment Reporting" in section_names_in


def test_chunker_metadata_breadcrumbs():
    """Test that chunker creates chunks with full metadata breadcrumbs and page numbers."""
    parser = FinancialDocumentParser()
    chunker = FinancialChunker(chunk_size_chars=400, chunk_overlap_chars=50)

    raw = """
Item 1. Business
First paragraph discussing core business operations.

Second paragraph discussing manufacturing facilities.

Item 1A. Risk Factors
Critical risk regarding component supply chains and currency fluctuations.
"""
    doc = parser.parse_raw_filing(
        raw_text=raw,
        company="Test Corp",
        ticker="TEST",
        market="us",
        document_type="10-K",
        published_date="2024-11-01",
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata.ticker == "TEST"
        assert chunk.metadata.company == "Test Corp"
        assert chunk.metadata.document_type == "10-K"
        assert chunk.metadata.published_date == "2024-11-01"
        assert chunk.metadata.section in ["Item 1. Business", "Item 1A. Risk Factors", "Overview & Operations"]
        assert chunk.metadata.page_number >= 1


def test_local_embedder_caching_and_similarity(temp_rag_env):
    """Test local embedding generation, deterministic caching, and cosine similarity."""
    embedder = temp_rag_env["embedder"]

    text1 = "Apple services revenue reached record high gross margin expanded to 46.2%"
    text2 = "Apple services segment revenue and gross margin expansion in quarterly 10-K"
    text3 = "Unrelated geopolitical agricultural commodities harvest report"

    vec1 = embedder.embed_text(text1)
    vec2 = embedder.embed_text(text2)
    vec3 = embedder.embed_text(text3)

    assert len(vec1) == embedder.dim
    # Cache hit check
    vec1_cached = embedder.embed_text(text1)
    assert vec1 == vec1_cached

    sim_1_2 = embedder.cosine_similarity(vec1, vec2)
    sim_1_3 = embedder.cosine_similarity(vec1, vec3)

    assert sim_1_2 > sim_1_3
    assert sim_1_2 > 0.40


def test_vector_store_filtering_and_retrieval(temp_rag_env):
    """Test SQLiteVectorStore filtering by ticker, market, and document type."""
    vec_store = temp_rag_env["vec_store"]
    doc_store = temp_rag_env["doc_store"]

    # Ingest custom documents
    doc_store.add_document(
        raw_text="Item 7. MD&A\nNvidia datacenter revenue surged 150% driven by Hopper GPU demand.",
        company="NVIDIA Corporation",
        ticker="NVDA",
        market="us",
        document_type="10-K",
        published_date="2024-03-15",
    )

    doc_store.add_document(
        raw_text="Item 8.01 Other Events\nNvidia announced next-gen Blackwell architecture launch.",
        company="NVIDIA Corporation",
        ticker="NVDA",
        market="us",
        document_type="8-K",
        published_date="2024-06-20",
    )

    # 1. Search 10-K only
    q_10k = RetrievalQuery(
        ticker="NVDA",
        market="us",
        query_text="datacenter revenue GPU demand",
        document_types=["10-K"],
    )
    results_10k = vec_store.search(q_10k)
    assert len(results_10k) == 1
    assert results_10k[0][0].metadata.document_type == "10-K"

    # 2. Search 8-K only
    q_8k = RetrievalQuery(
        ticker="NVDA",
        market="us",
        query_text="Blackwell architecture launch",
        document_types=["8-K"],
    )
    results_8k = vec_store.search(q_8k)
    assert len(results_8k) == 1
    assert results_8k[0][0].metadata.document_type == "8-K"


def test_retriever_evidence_and_citation_formatting(temp_rag_env):
    """Test FinancialRAGRetriever creates verified inline citations with accurate fields."""
    retriever = temp_rag_env["retriever"]

    result = retriever.retrieve_evidence(
        ticker="AAPL",
        query_text="services revenue gross margin 10-K",
        market="us",
    )

    assert result.has_sufficient_evidence is True
    assert len(result.citations) >= 1

    top_citation = result.citations[0]
    assert top_citation.ticker == "AAPL"
    assert top_citation.company == "Apple Inc."
    assert top_citation.document_type == "10-K"
    assert top_citation.published_date == "2024-10-31"
    assert top_citation.source == "SEC EDGAR"

    formatted = top_citation.format_inline()
    assert "[Apple Inc. (AAPL) 10-K" in formatted
    assert "2024-10-31 (SEC EDGAR)]" in formatted


def test_insufficient_evidence_when_no_filings_exist(temp_rag_env):
    """Verify that querying an unknown asset returns 'Insufficient evidence.' without hallucinations."""
    retriever = temp_rag_env["retriever"]

    result = retriever.retrieve_evidence(
        ticker="NONEXISTENT_TICKER",
        query_text="quarterly revenue growth",
        market="us",
    )

    assert result.has_sufficient_evidence is False
    assert len(result.chunks) == 0
    assert len(result.citations) == 0
    assert "Insufficient evidence" in result.context_text

    # Verify CitationVerifier output
    reasons, risks, evidence = CitationVerifier.verify_and_ground_claims(
        reasons=["High growth projection"],
        retrieval=result,
        agent_name="FundamentalAgent",
    )
    assert evidence == ["Insufficient evidence."]
    assert "Insufficient evidence" in reasons[0]


def test_fundamental_agent_rag_integration(temp_rag_env):
    """Test FundamentalAgent utilizes retrieved filing evidence for grounded analysis."""
    retriever = temp_rag_env["retriever"]
    agent = FundamentalAgent(retriever=retriever)

    stock = StockSnapshot(
        ticker="AAPL",
        name="Apple Inc.",
        market="us",
        currency="USD",
        current_price=225.50,
        fundamentals=Fundamentals(pe_ratio=28.4, return_on_equity=1.45, debt_to_equity=1.2),
    )

    output = agent.analyze(stock)
    assert output.signal in ["BUY", "HOLD"]
    assert output.confidence > 0.50
    assert len(output.evidence) >= 1
    # Evidence must contain verifiable SEC citation
    assert any("SEC EDGAR" in e or "10-K" in e for e in output.evidence)
    # Reasons must include grounded bracket citation tag
    assert any("[Apple Inc. (AAPL) 10-K" in r for r in output.reasons)


def test_news_agent_rag_integration(temp_rag_env):
    """Test NewsAgent grounds catalyst assessments with news headlines and filing context."""
    retriever = temp_rag_env["retriever"]
    agent = NewsAgent(retriever=retriever)

    news = [
        NewsItem(title="Apple reports record quarterly services revenue expansion", publisher="Reuters")
    ]
    stock = StockSnapshot(
        ticker="AAPL",
        name="Apple Inc.",
        market="us",
        currency="USD",
        current_price=225.50,
        recent_news=news,
    )

    output = agent.analyze(stock)
    assert output.signal in ["BUY", "HOLD"]
    assert output.confidence > 0.50
    assert any("Headline:" in e for e in output.evidence)


def test_indian_filing_retrieval_and_grounding(temp_rag_env):
    """Test Indian corporate filings (NSE/BSE) retrieval and grounding."""
    retriever = temp_rag_env["retriever"]
    agent = FundamentalAgent(retriever=retriever)

    stock = StockSnapshot(
        ticker="RELIANCE.NS",
        name="Reliance Industries",
        market="india",
        currency="INR",
        current_price=2950.0,
        fundamentals=Fundamentals(pe_ratio=24.5, return_on_equity=0.14, debt_to_equity=0.65),
    )

    output = agent.analyze(stock)
    assert output.signal in ["BUY", "HOLD"]
    # Evidence must contain NSE India citation
    assert any("NSE India" in e or "quarterly_results" in e for e in output.evidence)
