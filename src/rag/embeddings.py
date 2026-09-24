"""
Embedding Generator & Persistent Cache for Local M5 Mac Execution.

Generates dense semantic vector embeddings optimized for financial text,
with persistent SQLite caching to eliminate redundant calculations.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Optional

from src.utils.config import ROOT
from src.utils.logger import logger

DEFAULT_EMBEDDING_DIM = 128
DB_PATH = ROOT / "data" / "embeddings_cache.db"


class LocalFinancialEmbedder:
    """
    High-performance local financial text embedder running on Apple Silicon / Mac.
    Combines subword hash projection and TF-IDF financial keyword weighting.
    """

    # Domain-specific financial keywords that receive amplified semantic weight
    FINANCIAL_TERMS = {
        "revenue": 3.0, "net income": 3.0, "ebitda": 2.5, "operating margin": 2.5,
        "gross margin": 2.5, "cash flow": 2.5, "free cash flow": 3.0, "debt": 2.0,
        "liabilities": 2.0, "diluted eps": 3.0, "dividend": 2.0, "capex": 2.5,
        "guidance": 3.0, "headwind": 2.5, "tailwind": 2.5, "sec": 2.0, "10-k": 2.5,
        "10-q": 2.5, "8-k": 2.5, "nse": 2.0, "bse": 2.0, "quarterly": 2.0,
        "growth": 2.0, "profit": 2.0, "loss": 2.0, "risk": 2.5, "litigation": 2.5,
        "restructuring": 2.5, "acquisition": 2.5, "tariffs": 2.5, "semiconductor": 2.0,
        "supply chain": 2.0, "roe": 2.5, "roce": 2.5, "pe ratio": 2.0,
    }

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM, cache_db_path: Optional[Path] = None):
        self.dim = dim
        self.db_path = cache_db_path or DB_PATH
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        """Ensure SQLite embeddings cache table exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    text_hash TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> list[float]:
        """Generate normalized vector embedding for input text with caching."""
        if not text or not text.strip():
            return [0.0] * self.dim

        text_hash = self._get_hash(text)

        # 1. Check persistent SQLite cache
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding_json FROM embeddings_cache WHERE text_hash = ? AND dim = ?",
                    (text_hash, self.dim),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as exc:
            logger.debug(f"[Embedder] Cache lookup failed: {exc}")

        # 2. Compute local dense embedding
        vec = self._compute_dense_vector(text)

        # 3. Store in persistent cache
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings_cache (text_hash, dim, embedding_json) VALUES (?, ?, ?)",
                    (text_hash, self.dim, json.dumps(vec)),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"[Embedder] Cache store failed: {exc}")

        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed_text(t) for t in texts]

    def _compute_dense_vector(self, text: str) -> list[float]:
        """Compute normalized dense projection from text tokens and financial n-grams."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\.\%]+\b", text.lower())
        vec = [0.0] * self.dim

        if not words:
            return vec

        # Token frequencies & domain boosts
        for i, word in enumerate(words):
            # Check single word boost
            weight = self.FINANCIAL_TERMS.get(word, 1.0)

            # Check 2-word bigrams
            if i + 1 < len(words):
                bigram = f"{word} {words[i+1]}"
                if bigram in self.FINANCIAL_TERMS:
                    weight *= self.FINANCIAL_TERMS[bigram]

            # Deterministic hash feature mapping
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign * weight

            # Subword 3-grams mapping
            if len(word) >= 4:
                for j in range(len(word) - 2):
                    sub = word[j:j+3]
                    sub_h = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16)
                    sub_idx = sub_h % self.dim
                    sub_sign = 1.0 if ((sub_h >> 8) % 2 == 0) else -1.0
                    vec[sub_idx] += sub_sign * 0.3 * weight

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]

        return vec

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two normalized vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))
