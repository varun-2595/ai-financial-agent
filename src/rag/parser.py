"""
Financial Document Parser for US and India Filings.

Handles:
- US: SEC 10-K, 10-Q, 8-K, earnings call transcripts
- India: NSE/BSE corporate filings, annual reports, quarterly financial results, investor presentations
"""
from __future__ import annotations

import re
from typing import Optional

from src.rag.models import Document, DocumentMetadata, DocumentType, MarketType


class FinancialDocumentParser:
    """Parses corporate filings into structured documents with detected section boundaries."""

    # US SEC Item regex patterns
    SEC_SECTION_PATTERNS = [
        (r"(Item\s+1\.\s+Business)", "Item 1. Business"),
        (r"(Item\s+1A\.\s+Risk\s+Factors)", "Item 1A. Risk Factors"),
        (r"(Item\s+1B\.\s+Unresolved\s+Staff\s+Comments)", "Item 1B. Unresolved Staff Comments"),
        (r"(Item\s+2\.\s+Properties)", "Item 2. Properties"),
        (r"(Item\s+3\.\s+Legal\s+Proceedings)", "Item 3. Legal Proceedings"),
        (r"(Item\s+7\.\s+Management['’]s\s+Discussion\s+and\s+Analysis)", "Item 7. MD&A"),
        (r"(Item\s+7A\.\s+Quantitative\s+and\s+Qualitative\s+Disclosures)", "Item 7A. Market Risk"),
        (r"(Item\s+8\.\s+Financial\s+Statements\s+and\s+Supplementary\s+Data)", "Item 8. Financial Statements"),
        (r"(Item\s+9\.\s+Controls\s+and\s+Procedures)", "Item 9. Controls"),
        (r"(Item\s+2\.02\s+Results\s+of\s+Operations)", "Item 2.02 Results of Operations"),
        (r"(Item\s+7\.01\s+Regulation\s+FD\s+Disclosure)", "Item 7.01 Reg FD Disclosure"),
        (r"(Item\s+8\.01\s+Other\s+Events)", "Item 8.01 Other Events"),
    ]

    # India NSE/BSE filing section patterns
    INDIA_SECTION_PATTERNS = [
        (r"(Financial\s+Results|Quarterly\s+Results|Unaudited\s+Financial\s+Results)", "Financial Results"),
        (r"(Management\s+Discussion\s+(?:and|&)\s+Analysis|MD&A)", "Management Discussion & Analysis"),
        (r"(Segment\s+Reporting|Segment\s+Revenue|Segment\s+Results)", "Segment Reporting"),
        (r"(Corporate\s+Governance\s+Report)", "Corporate Governance"),
        (r"(Board['’]s\s+Report|Directors['’]\s+Report)", "Board's Report"),
        (r"(Auditor['’]s\s+Report|Independent\s+Auditor['’]s\s+Report)", "Auditor's Report"),
        (r"(Investor\s+Presentation|Conference\s+Call\s+Transcript|Key\s+Highlights)", "Investor Presentation Highlights"),
        (r"(Capital\s+Expenditure|Capex\s+Plans|Capacity\s+Expansion)", "Capex & Expansion"),
        (r"(Risk\s+Management\s+and\s+Internal\s+Controls)", "Risk Management"),
    ]

    def parse_raw_filing(
        self,
        raw_text: str,
        company: str,
        ticker: str,
        market: MarketType,
        document_type: DocumentType,
        published_date: str,
        source: str = "SEC EDGAR",
        doc_id: Optional[str] = None,
    ) -> Document:
        """Parse raw text filing into a canonical Document model with metadata."""
        cleaned_text = self._clean_text(raw_text)
        generated_id = doc_id or f"{market.lower()}_{ticker.upper()}_{document_type.replace(' ', '_')}_{published_date}"

        metadata = DocumentMetadata(
            company=company,
            ticker=ticker.upper(),
            market=market,
            document_type=document_type,
            published_date=published_date,
            source=source,
            section="Full Document",
        )

        title = f"{company} ({ticker.upper()}) {document_type} - {published_date}"

        return Document(
            doc_id=generated_id,
            title=title,
            metadata=metadata,
            raw_text=cleaned_text,
        )

    def extract_sections(self, text: str, market: MarketType) -> list[tuple[str, str, int]]:
        """
        Split document into recognized sections.
        Returns list of (section_name, section_text, estimated_page_number).
        """
        patterns = self.SEC_SECTION_PATTERNS if market == "us" else self.INDIA_SECTION_PATTERNS
        combined_pattern = "|".join(f"(?P<sec_{i}>{p[0]})" for i, p in enumerate(patterns))

        sections: list[tuple[str, str, int]] = []
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))

        if not matches:
            # If no formal headers found, treat as unified main section
            return [("Overview & Operations", text, 1)]

        for i, match in enumerate(matches):
            # Determine which pattern matched
            matched_name = "General Disclosure"
            for idx, (_, canonical_name) in enumerate(patterns):
                if match.group(f"sec_{idx}"):
                    matched_name = canonical_name
                    break

            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_content = text[start_idx:end_idx].strip()

            # Estimate page number roughly based on character count (~2500 chars/page)
            est_page = max(1, start_idx // 2500 + 1)
            sections.append((matched_name, section_content, est_page))

        return sections

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace, remove HTML artifacts, and standardize financial punctuation."""
        # Replace HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        text = text.replace("&#160;", " ").replace("&nbsp;", " ")
        # Strip excessive blank lines and trailing whitespaces
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
