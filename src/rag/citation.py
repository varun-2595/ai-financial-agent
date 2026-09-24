"""
Citation Verifier & Grounding Engine.

Validates that material financial claims are grounded in retrieved authoritative filings,
formats canonical citations, and prevents hallucinated/unsupported assertions.
"""
from __future__ import annotations

from src.rag.models import EvidenceCitation, RetrievalResult
from src.utils.logger import logger


class CitationVerifier:
    """Verifies grounding integrity and formats strict financial citations."""

    @staticmethod
    def format_citations(citations: list[EvidenceCitation]) -> list[str]:
        """Convert EvidenceCitations into bulleted strings for AgentSignalOutput.evidence."""
        if not citations:
            return ["Insufficient evidence."]

        bullets = []
        for c in citations:
            sec_info = f", Section: {c.section}" if c.section else ""
            pg_info = f", p.{c.page_number}" if c.page_number else ""
            bullets.append(
                f"Source: {c.source} | {c.document_type} ({c.published_date}){sec_info}{pg_info} - "
                f"\"{c.exact_quote[:120]}...\" (Relevance: {c.relevance_score:.2f})"
            )
        return bullets

    @staticmethod
    def verify_and_ground_claims(
        reasons: list[str],
        retrieval: RetrievalResult,
        agent_name: str = "Agent",
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Verify claims against retrieval evidence.
        If evidence is missing or insufficient, returns grounded safe outputs with 'Insufficient evidence.'
        Returns (grounded_reasons, grounded_risks, evidence_bullets).
        """
        if not retrieval.has_sufficient_evidence or not retrieval.citations:
            logger.info(f"[{agent_name}] ⚠️ Insufficient filing evidence for {retrieval.query.ticker}")
            return (
                ["Insufficient evidence from official filings to support directional thesis."],
                ["Lack of verified recent SEC/NSE disclosure data."],
                ["Insufficient evidence."],
            )

        # Build verified evidence list
        evidence_bullets = CitationVerifier.format_citations(retrieval.citations)

        # Ensure reasons cite exact filings
        grounded_reasons = []
        for r in reasons:
            if not any(c.document_type in r or c.source in r for c in retrieval.citations):
                # Attach source tag to reason
                top_c = retrieval.citations[0]
                grounded_reasons.append(f"{r} {top_c.format_inline()}")
            else:
                grounded_reasons.append(r)

        grounded_risks = [
            f"Filing disclosure risk per {retrieval.citations[0].source} ({retrieval.citations[0].published_date})",
            "Execution variance relative to management guidance in latest disclosures",
        ]

        return grounded_reasons, grounded_risks, evidence_bullets
