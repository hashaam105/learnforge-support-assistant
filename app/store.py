"""SQLite persistence layer.

Everything the system knows lives in one file: documents, chunks, vectors, the
lexical index, conversation state and the escalation queue. For a 40-document
corpus that is the right call — see the trade-offs section of the README for
why, and docs/DATA_SCHEMA.md for the pgvector migration path.

The class is deliberately a thin, explicit data-access layer rather than an ORM
so that the SQL in schema.sql *is* the schema, with nothing hidden behind
model metaclasses.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from app.config import ROOT, Settings, settings
from app.text import content_terms

SCHEMA_FILE = ROOT / "schema.sql"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pack_vector(values: Sequence[float]) -> bytes:
    """float32 little-endian blob. 768 dims = 3 KB/chunk; ~150 chunks = 450 KB."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


class Store:
    def __init__(self, cfg: Settings | None = None, db_path: str | Path | None = None) -> None:
        self.cfg = cfg or settings
        self.path = Path(db_path) if db_path else self.cfg.db_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._vector_cache: tuple[str, list[str], np.ndarray] | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------
    # ingest writes
    # ------------------------------------------------------------------
    def get_document_hash(self, doc_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row["content_hash"] if row else None

    def upsert_document(self, doc: dict[str, Any]) -> None:
        """Insert or replace a document, bumping `version` when content changed."""
        previous = self.conn.execute(
            "SELECT version, content_hash FROM documents WHERE doc_id = ?", (doc["doc_id"],)
        ).fetchone()
        version = 1
        if previous:
            version = previous["version"] + (1 if previous["content_hash"] != doc["content_hash"] else 0)

        self.conn.execute(
            """
            INSERT INTO documents (doc_id, source_type, title, source_path, source_uri,
                                   authority_tier, effective_date, date_label,
                                   has_explicit_date, ticket_status, topics, raw_text,
                                   content_hash, version, ingested_at, is_active)
            VALUES (:doc_id, :source_type, :title, :source_path, :source_uri,
                    :authority_tier, :effective_date, :date_label,
                    :has_explicit_date, :ticket_status, :topics, :raw_text,
                    :content_hash, :version, :ingested_at, 1)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_type=excluded.source_type, title=excluded.title,
                source_path=excluded.source_path, source_uri=excluded.source_uri,
                authority_tier=excluded.authority_tier,
                effective_date=excluded.effective_date, date_label=excluded.date_label,
                has_explicit_date=excluded.has_explicit_date,
                ticket_status=excluded.ticket_status, topics=excluded.topics,
                raw_text=excluded.raw_text, content_hash=excluded.content_hash,
                version=excluded.version, ingested_at=excluded.ingested_at, is_active=1
            """,
            {**doc, "version": version, "ingested_at": utcnow()},
        )

    def replace_chunks(self, doc_id: str, chunks: list[dict[str, Any]]) -> None:
        """Delete-then-insert. Cascades clear stale embeddings, and we clear the
        FTS rows explicitly because FTS5 virtual tables have no foreign keys."""
        old = [r["chunk_id"] for r in self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
        )]
        if old:
            self.conn.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?", [(c,) for c in old]
            )
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM deprecated_claims WHERE doc_id = ?", (doc_id,))

        now = utcnow()
        self.conn.executemany(
            """
            INSERT INTO chunks (chunk_id, doc_id, ordinal, text, heading, token_estimate,
                                source_type, authority_tier, effective_date, topics,
                                has_deprecation_notice, content_hash, ingested_at)
            VALUES (:chunk_id, :doc_id, :ordinal, :text, :heading, :token_estimate,
                    :source_type, :authority_tier, :effective_date, :topics,
                    :has_deprecation_notice, :content_hash, :ingested_at)
            """,
            [{**c, "ingested_at": now} for c in chunks],
        )
        self.conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, text, heading, topics) VALUES (?, ?, ?, ?)",
            [(c["chunk_id"], c["text"], c["heading"] or "", c["topics"]) for c in chunks],
        )
        self._vector_cache = None

    def add_deprecated_claims(self, claims: list[dict[str, Any]]) -> None:
        if not claims:
            return
        now = utcnow()
        self.conn.executemany(
            """
            INSERT INTO deprecated_claims (doc_id, chunk_id, claim_text, marker,
                                           superseded_by, detected_at)
            VALUES (:doc_id, :chunk_id, :claim_text, :marker, :superseded_by, :detected_at)
            """,
            [{**c, "detected_at": now} for c in claims],
        )

    def write_embeddings(self, model: str, dim: int, rows: Iterable[tuple[str, Sequence[float]]]) -> int:
        now = utcnow()
        payload = [(cid, model, dim, pack_vector(vec), now) for cid, vec in rows]
        if not payload:
            return 0
        self.conn.executemany(
            """
            INSERT INTO embeddings (chunk_id, model, dim, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id, model) DO UPDATE SET
                vector=excluded.vector, dim=excluded.dim, created_at=excluded.created_at
            """,
            payload,
        )
        self._vector_cache = None
        return len(payload)

    def chunks_missing_embeddings(self, model: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT c.chunk_id, c.text, c.heading
                FROM chunks c
                LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
                WHERE e.chunk_id IS NULL
                ORDER BY c.chunk_id
                """,
                (model,),
            )
        )

    # ------------------------------------------------------------------
    # retrieval reads
    # ------------------------------------------------------------------
    def lexical_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """BM25 over FTS5. Returns (chunk_id, score) with higher = better.

        SQLite's bm25() returns a *negative* relevance (more negative = better),
        so we flip the sign to keep every scorer in this codebase ascending.
        """
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT f.chunk_id AS chunk_id, bm25(chunks_fts, 1.0, 0.6, 0.4) AS score
                FROM chunks_fts f
                JOIN chunks c ON c.chunk_id = f.chunk_id
                JOIN documents d ON d.doc_id = c.doc_id AND d.is_active = 1
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS expression (rare after sanitising) must not 500.
            return []
        return [(r["chunk_id"], -float(r["score"])) for r in rows]

    def vector_matrix(self, model: str) -> tuple[list[str], np.ndarray]:
        """All vectors for a model as one matrix, cached in memory.

        Brute force over ~150 rows is microseconds and exact. The cache is
        invalidated on any write; see README trade-offs for when this stops
        being the right answer (roughly 10^5 chunks).
        """
        if self._vector_cache and self._vector_cache[0] == model:
            return self._vector_cache[1], self._vector_cache[2]

        rows = self.conn.execute(
            """
            SELECT e.chunk_id, e.vector
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id AND d.is_active = 1
            WHERE e.model = ?
            ORDER BY e.chunk_id
            """,
            (model,),
        ).fetchall()
        if not rows:
            empty = (model, [], np.zeros((0, 0), dtype="float32"))
            self._vector_cache = empty
            return [], empty[2]

        ids = [r["chunk_id"] for r in rows]
        matrix = np.vstack([unpack_vector(r["vector"]) for r in rows]).astype("float32")
        self._vector_cache = (model, ids, matrix)
        return ids, matrix

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"""
            SELECT c.*, d.title, d.date_label, d.ticket_status, d.source_uri,
                   d.has_explicit_date, d.version
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id IN ({marks})
            """,
            tuple(chunk_ids),
        ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def deprecated_claims_for(self, doc_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        marks = ",".join("?" * len(doc_ids))
        rows = self.conn.execute(
            f"SELECT * FROM deprecated_claims WHERE doc_id IN ({marks})", tuple(doc_ids)
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        def one(sql: str) -> int:
            return int(self.conn.execute(sql).fetchone()[0])

        by_type = {
            r["source_type"]: r["n"]
            for r in self.conn.execute(
                "SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type"
            )
        }
        last_run = self.conn.execute(
            "SELECT * FROM ingest_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return {
            "documents": one("SELECT COUNT(*) FROM documents"),
            "documents_by_type": by_type,
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "embeddings": one("SELECT COUNT(*) FROM embeddings"),
            "deprecated_claims": one("SELECT COUNT(*) FROM deprecated_claims"),
            "escalations": one("SELECT COUNT(*) FROM escalations"),
            "last_ingest": dict(last_run) if last_run else None,
        }

    # ------------------------------------------------------------------
    # conversation state
    # ------------------------------------------------------------------
    def ensure_conversation(self, session_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            return dict(row)
        now = utcnow()
        self.conn.execute(
            "INSERT INTO conversations (session_id, created_at, last_active_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        self.conn.commit()
        return {
            "session_id": session_id,
            "created_at": now,
            "last_active_at": now,
            "summary": "",
            "slots": "{}",
            "turn_count": 0,
            "escalated": 0,
        }

    def update_conversation(
        self, session_id: str, *, slots: dict[str, Any], summary: str, escalated: bool
    ) -> None:
        self.conn.execute(
            """
            UPDATE conversations
               SET slots = ?, summary = ?, last_active_at = ?,
                   turn_count = turn_count + 1,
                   escalated = MAX(escalated, ?)
             WHERE session_id = ?
            """,
            (json.dumps(slots), summary, utcnow(), int(escalated), session_id),
        )
        self.conn.commit()

    def history(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT role, content, turn_index FROM messages
            WHERE session_id = ? ORDER BY turn_index DESC, message_id DESC LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def next_turn_index(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS n FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["n"])

    def add_message(self, **kwargs: Any) -> int:
        cols = [
            "session_id", "turn_index", "role", "content", "standalone_query",
            "retrieved_chunk_ids", "citations", "retrieval_score", "score_margin",
            "groundedness", "confidence", "action", "reason_codes", "latency_ms", "model",
        ]
        values = {c: kwargs.get(c) for c in cols}
        values["created_at"] = utcnow()
        placeholders = ", ".join(f":{c}" for c in cols + ["created_at"])
        cur = self.conn.execute(
            f"INSERT INTO messages ({', '.join(cols + ['created_at'])}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # escalations
    # ------------------------------------------------------------------
    def add_escalation(self, payload: dict[str, Any]) -> str:
        escalation_id = f"ESC-{uuid.uuid4().hex[:10].upper()}"
        self.conn.execute(
            """
            INSERT INTO escalations (escalation_id, session_id, message_id, created_at,
                                     queue, priority, reason_codes, user_intent, summary,
                                     attempted_answer, context_snapshot, resolved)
            VALUES (:escalation_id, :session_id, :message_id, :created_at, :queue,
                    :priority, :reason_codes, :user_intent, :summary, :attempted_answer,
                    :context_snapshot, 0)
            """,
            {**payload, "escalation_id": escalation_id, "created_at": utcnow()},
        )
        self.conn.commit()
        return escalation_id

    def list_escalations(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM escalations ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # ingest runs
    # ------------------------------------------------------------------
    def start_ingest_run(self, kb_path: str, embedding_model: str | None) -> str:
        run_id = f"RUN-{uuid.uuid4().hex[:10].upper()}"
        self.conn.execute(
            """INSERT INTO ingest_runs (run_id, started_at, kb_path, embedding_model, status)
               VALUES (?, ?, ?, ?, 'running')""",
            (run_id, utcnow(), kb_path, embedding_model),
        )
        self.conn.commit()
        return run_id

    def finish_ingest_run(self, run_id: str, **counters: Any) -> None:
        self.conn.execute(
            """
            UPDATE ingest_runs
               SET finished_at = ?, docs_seen = ?, docs_changed = ?,
                   chunks_written = ?, vectors_written = ?, status = ?
             WHERE run_id = ?
            """,
            (
                utcnow(),
                counters.get("docs_seen", 0),
                counters.get("docs_changed", 0),
                counters.get("chunks_written", 0),
                counters.get("vectors_written", 0),
                counters.get("status", "ok"),
                run_id,
            ),
        )
        self.conn.commit()


def _to_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR-query over content terms.

    Two jobs. First, safety: user text goes straight into a MATCH expression
    where `"`, `*`, `^` and `-` are operators, so tokenising to word
    characters makes FTS-syntax injection impossible. Second, precision:
    stopwords are dropped, because OR-ing "the" against a 66-chunk corpus
    matches everything and turns an out-of-scope question into five
    confident-looking hits.
    """
    tokens = content_terms(query)
    if not tokens:
        return ""
    return " OR ".join(tokens[:32])
