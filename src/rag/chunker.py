"""
Financial Document Chunker.

Creates semantically coherent chunks with section breadcrumbs and page numbering,
avoiding truncation of financial metric tables and accounting disclosures.
"""
from __future__ import annotations

import re
from typing import Optional

from src.rag.models import Document, DocumentChunk, DocumentMetadata
from src.rag.parser import FinancialDocumentParser


class FinancialChunker:
    """Splits financial documents into atomic chunks with complete metadata breadcrumbs."""

    def __init__(
        self,
        chunk_size_chars: int = 1200,
        chunk_overlap_chars: int = 200,
        min_chunk_chars: int = 50,
    ):
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self.parser = FinancialDocumentParser()

    def chunk_document(self, doc: Document) -> list[DocumentChunk]:
        """Chunk a Document into DocumentChunks with section and page metadata."""
        sections = self.parser.extract_sections(doc.raw_text, doc.metadata.market)
        chunks: list[DocumentChunk] = []
        chunk_counter = 1

        for section_name, section_text, start_page in sections:
            # Split section text into paragraphs / financial blocks
            paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]

            curr_block: list[str] = []
            curr_len = 0
            page_offset = 0

            for para in paragraphs:
                para_len = len(para)

                # If adding this paragraph exceeds chunk size, finalize current chunk
                if curr_len + para_len > self.chunk_size_chars and curr_block:
                    chunk_text = "\n\n".join(curr_block)
                    if len(chunk_text) >= self.min_chunk_chars:
                        chunk = self._create_chunk(
                            doc=doc,
                            text=chunk_text,
                            section=section_name,
                            page_number=start_page + page_offset,
                            chunk_index=chunk_counter,
                        )
                        chunks.append(chunk)
                        chunk_counter += 1

                    # Handle overlap: keep last paragraph if it fits
                    if self.chunk_overlap_chars > 0 and curr_block:
                        last_p = curr_block[-1]
                        if len(last_p) <= self.chunk_overlap_chars:
                            curr_block = [last_p, para]
                            curr_len = len(last_p) + para_len
                        else:
                            curr_block = [para]
                            curr_len = para_len
                    else:
                        curr_block = [para]
                        curr_len = para_len

                    page_offset = curr_len // 2500
                else:
                    curr_block.append(para)
                    curr_len += para_len

            # Finalize remaining block in this section
            if curr_block:
                chunk_text = "\n\n".join(curr_block)
                if len(chunk_text) >= self.min_chunk_chars:
                    chunk = self._create_chunk(
                        doc=doc,
                        text=chunk_text,
                        section=section_name,
                        page_number=start_page + page_offset,
                        chunk_index=chunk_counter,
                    )
                    chunks.append(chunk)
                    chunk_counter += 1

        # Fallback if text was short
        if not chunks and doc.raw_text.strip():
            chunk = self._create_chunk(
                doc=doc,
                text=doc.raw_text.strip(),
                section="Overview",
                page_number=1,
                chunk_index=1,
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        doc: Document,
        text: str,
        section: str,
        page_number: int,
        chunk_index: int,
    ) -> DocumentChunk:
        """Helper to construct a typed DocumentChunk."""
        token_count = max(1, len(text) // 4)
        chunk_id = f"{doc.doc_id}_chunk_{chunk_index:03d}"

        metadata = DocumentMetadata(
            company=doc.metadata.company,
            ticker=doc.metadata.ticker,
            market=doc.metadata.market,
            document_type=doc.metadata.document_type,
            published_date=doc.metadata.published_date,
            source=doc.metadata.source,
            page_number=page_number,
            section=section,
            extra_attributes=doc.metadata.extra_attributes,
        )

        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            text=text,
            metadata=metadata,
            token_count=token_count,
        )
