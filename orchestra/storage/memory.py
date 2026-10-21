from __future__ import annotations
"""Hierarchical Semantic memory (Project Atlas) with dependency tracking,
importance-weighted decay, TTL, and ownership."""

import json
import sqlite3
import os
import time
import logging
import threading
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from orchestra.engine.context_compression import validate_embedding_dimension
from orchestra.storage.sqlite import transaction
from orchestra.providers.ollama import OllamaProvider

# Try to import hnswlib
try:
    import hnswlib
    _HNSW_AVAILABLE = True
except ImportError:
    _HNSW_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "hnswlib not installed. SemanticMemory will use O(N) numpy fallback. "
        "Install with: pip install hnswlib>=0.8.0"
    )

_ollama = OllamaProvider()


def _identity(row: dict) -> dict:
    """What the event log keeps of a deleted row: which one it was, not what it said."""
    return {"source": row.get("source"), "owner": row.get("owner"),
            "sensitivity_label": row.get("sensitivity_label")}
logger = logging.getLogger(__name__)


class SemanticMemory:
    """Project Atlas: Vector storage + Dependency Graph using SQLite.

    Supports importance-weighted decay:
    - TTL-based expiration (default 30 days)
    - Explicit user pinning
    - Citation-count boosting
    - Sensitivity labels
    - Owner tracking
    """

    DEFAULT_TTL_HOURS = 720  # 30 days
    PINNED_BOOST = 0.05      # Similarity bonus for pinned items
    CITATION_BOOST = 0.02    # Per-citation similarity bonus (capped)

    def __init__(
        self,
        db_path: str = ".orchestra/memory.db",
        *,
        embeddings_table: str = "embeddings",
        index_path: Optional[str] = None,
        now_fn=None,
    ):
        self.db_path = db_path
        self._table = embeddings_table
        self._now_fn = now_fn or datetime.utcnow
        self._init_db()
        self._index: Any = None
        self._index_meta: Dict[str, Any] = {}
        self._index_dirty = True
        self._last_add_time = 0.0
        self._index_path = index_path if index_path is not None else self._derive_index_path()
        self._dim: Optional[int] = None
        self._lock = threading.RLock()
        self._pending_vectors: list[tuple[int, np.ndarray]] = []
        self._deleted_ids: set[int] = set()

    def _derive_index_path(self) -> str:
        base = os.path.splitext(self.db_path)[0]
        return f"{base}_hnsw.idx"

    def _init_db(self):
        try:
            with transaction(self.db_path) as conn:
                # Main embeddings table with decay support
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        vector BLOB NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        ttl_hours INTEGER DEFAULT 720,
                        sensitivity_label TEXT DEFAULT 'normal',
                        source TEXT,
                        owner TEXT DEFAULT 'orchestra',
                        pinned INTEGER DEFAULT 0,
                        citation_count INTEGER DEFAULT 0
                    )
                """)

                # Robustly add missing columns (schema migration)
                existing = {row[1] for row in conn.execute(f"PRAGMA table_info({self._table})").fetchall()}
                missing_columns = {
                    "summary": "TEXT",
                    "file_path": "TEXT",
                    "expires_at": "TIMESTAMP",
                    "ttl_hours": "INTEGER DEFAULT 720",
                    "sensitivity_label": "TEXT DEFAULT 'normal'",
                    "source": "TEXT",
                    "owner": "TEXT DEFAULT 'orchestra'",
                    "pinned": "INTEGER DEFAULT 0",
                    "citation_count": "INTEGER DEFAULT 0",
                }
                for col, dtype in missing_columns.items():
                    if col not in existing:
                        conn.execute(f"ALTER TABLE {self._table} ADD COLUMN {col} {dtype}")

                # Backfill expires_at for rows that don't have it
                conn.execute(f"""
                    UPDATE {self._table}
                    SET expires_at = datetime(created_at, '+' || CAST(IFNULL(ttl_hours, 720) AS TEXT) || ' hours')
                    WHERE expires_at IS NULL AND created_at IS NOT NULL
                """)

                # Dependency tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS relations (
                        source_path TEXT,
                        target_path TEXT,
                        relation_type TEXT,
                        PRIMARY KEY (source_path, target_path, relation_type)
                    )
                """)

                # Append-only audit trail for compliance and investigations
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        item_id INTEGER,
                        actor TEXT DEFAULT 'orchestra',
                        details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(
                "Memory DB schema migration failed (read-only filesystem?). "
                "Continuing with degraded memory: %s",
                e,
            )

    def _log_memory_event(
        self,
        event_type: str,
        item_id: int | None = None,
        actor: str = "orchestra",
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Append an immutable audit event for memory operations."""
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory_events (event_type, item_id, actor, details) VALUES (?, ?, ?, ?)",
                (event_type, item_id, actor, json.dumps(details) if details else None),
            )
            conn.commit()

    def _is_index_fresh(self) -> bool:
        if not os.path.exists(self._index_path):
            return False
        idx_mtime = os.path.getmtime(self._index_path)
        if idx_mtime <= self._last_add_time:
            return False  # this store wrote after the index was built
        # Table-specific freshness: the index is stale iff THIS table's row count no
        # longer matches what the index was built from. Using the shared DB file's
        # mtime would wrongly invalidate this index whenever a *sibling* typed-memory
        # table (episodic/procedural, same memory.db) is written — a self-inflicted
        # O(N) rebuild on every cross-store write. Row count is per-table.
        expected = self._index_meta.get("table_count", self._index_meta.get("count"))
        if expected is None:  # fresh instance: read the count from the persisted meta file
            try:
                with open(self._index_path + ".meta", "r", encoding="utf-8") as f:
                    persisted = json.load(f)
                    expected = persisted.get("table_count", persisted.get("count"))
            except Exception:
                return False
        if expected is None:
            return False
        try:
            with transaction(self.db_path) as conn:
                count = conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]
        except Exception:
            return False
        return count == expected

    def _load_hnsw_index(self) -> bool:
        """Load a pre-built HNSW index from disk if it is fresh."""
        if not _HNSW_AVAILABLE:
            return False
        if not self._is_index_fresh():
            return False

        meta_path = self._index_path + ".meta"
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            dim = meta.get("dim", 0)
            if dim == 0:
                return False

            index = hnswlib.Index(space="cosine", dim=dim)
            index.load_index(self._index_path)

            self._index = index
            self._index_meta = meta
            self._dim = dim
            return True
        except Exception:
            return False

    def _build_hnsw_index(self):
        """Build or incrementally update HNSW index from SQLite and persist to disk."""
        if not _HNSW_AVAILABLE:
            return

        # If we have an existing index, pending additions, and no deletions,
        # try an incremental append instead of a full rebuild.
        if (
            self._index is not None
            and self._pending_vectors
            and not self._deleted_ids
            and self._index_meta.get("dim") == self._dim
        ):
            current_count = self._index.get_current_count()
            max_elements = self._index.get_max_elements()
            pending_count = len(self._pending_vectors)

            if current_count + pending_count > max_elements:
                new_max = max(current_count + pending_count + 1000, int(max_elements * 2))
                self._index.resize_index(new_max)
                self._index_meta["max_elements"] = new_max

            new_ids = []
            new_data = []
            for db_id, vec in self._pending_vectors:
                new_ids.append(db_id)
                new_data.append(vec)

            if new_data:
                data = np.stack(new_data).astype(np.float32)
                labels = np.arange(current_count, current_count + len(new_data), dtype=np.int32)
                self._index.add_items(data, labels)
                self._index_meta["ids"].extend(new_ids)
                self._index_meta["count"] = current_count + len(new_data)
                if "table_count" in self._index_meta:
                    self._index_meta["table_count"] += len(new_data)

            self._pending_vectors = []
            self._index_dirty = False

            os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
            self._index.save_index(self._index_path)
            meta_path = self._index_path + ".meta"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._index_meta, f)
            return

        # Full rebuild path
        with transaction(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, vector FROM {self._table} ORDER BY id"
            ).fetchall()

        if not rows:
            self._index = None
            self._index_meta = {"ids": [], "dim": 0, "count": 0, "max_elements": 0}
            self._index_dirty = False
            self._pending_vectors = []
            self._deleted_ids = set()
            # Clean up stale index files
            for p in (self._index_path, self._index_path + ".meta"):
                if os.path.exists(p):
                    os.remove(p)
            return

        ids = [r[0] for r in rows]
        vectors = [np.frombuffer(r[1], dtype=np.float32) for r in rows]

        # The newest row says which embedding model the store is on. Vectors of another size
        # come from an earlier model: they cannot be compared with it, and stacking them
        # with the rest used to raise and take search down for good.
        dim = len(vectors[-1])
        comparable = [(row_id, vector) for row_id, vector in zip(ids, vectors) if len(vector) == dim]
        if len(comparable) < len(rows):
            logger.warning("%d memory rows have another embedding size than the current %d and are not searched",
                           len(rows) - len(comparable), dim)
        ids = [row_id for row_id, _ in comparable]
        vectors = [vector for _, vector in comparable]
        self._dim = dim
        count = len(vectors)
        max_elements = max(count + 1000, 1000)

        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        index.set_ef(50)

        data = np.stack(vectors).astype(np.float32)
        index.add_items(data, np.arange(count, dtype=np.int32))

        self._index = index
        self._index_meta = {
            "ids": ids,
            "dim": dim,
            "count": count,
            "table_count": len(rows),      # what freshness is judged by: rows, indexed or not
            "max_elements": max_elements,
        }
        self._index_dirty = False
        self._pending_vectors = []
        self._deleted_ids = set()

        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        index.save_index(self._index_path)
        meta_path = self._index_path + ".meta"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._index_meta, f)

    def add(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        summary: str = None,
        *,
        ttl_hours: int = None,
        sensitivity_label: str = "normal",
        source: str = None,
        owner: str = "orchestra",
        pinned: bool = False,
        memory_type: Optional[str] = None,
    ) -> bool:
        """Add content and its summary to Atlas.

        ``memory_type`` is accepted for interface parity with TypedMemory and
        ignored by the plain single-table store.

        Args:
            content: The raw content to store.
            metadata: Free-form JSON metadata.
            summary: Optional summary for embedding (falls back to content[:1000]).
            ttl_hours: Time-to-live in hours (default: 720 = 30 days).
            sensitivity_label: Data sensitivity tag (e.g., "normal", "pii", "classified").
            source: Where this came from (e.g., run_id, user_id).
            owner: Owner of this memory entry.
            pinned: If True, item never expires and gets search boost.
        """
        vector = _ollama.embed(summary or content[:1000])
        if not vector:
            return False

        vector_arr = np.array(vector, dtype=np.float32)
        if self._dim != len(vector):
            # First row, or the embedding model changed. The new size is the store's size from
            # here on; the index is rebuilt from the rows that have it (see _build_hnsw_index).
            if self._dim is not None:
                logger.warning("Embedding size changed from %s to %s: older rows are no longer searched",
                               self._dim, len(vector))
            self._dim = len(vector)

        vector_blob = vector_arr.tobytes()
        file_path = metadata.get("path") if metadata else None
        meta_json = json.dumps(metadata) if metadata else None
        ttl = ttl_hours if ttl_hours is not None else self.DEFAULT_TTL_HOURS
        pinned_int = 1 if pinned else 0

        with self._lock:
            with transaction(self.db_path) as conn:
                cursor = conn.execute(
                    f"""
                    INSERT INTO {self._table}
                    (content, summary, vector, metadata, file_path,
                     ttl_hours, sensitivity_label, source, owner, pinned, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            datetime('now', '+' || CAST(? AS TEXT) || ' hours'))
                    """,
                    (content, summary, vector_blob, meta_json, file_path,
                     ttl, sensitivity_label, source, owner, pinned_int, ttl)
                )
                conn.commit()
                item_id = cursor.lastrowid

            self._log_memory_event(
                "add",
                item_id=item_id,
                actor=owner,
                details={
                    "source": source,
                    "sensitivity_label": sensitivity_label,
                    "ttl_hours": ttl,
                    "pinned": bool(pinned),
                },
            )
            self._pending_vectors.append((item_id, vector_arr))
            self._index_dirty = True
            self._last_add_time = time.time()
        return True

    def delete(self, item_id: int, *, actor: str = "orchestra") -> bool:
        """Delete an item by id and invalidate the index."""
        with self._lock:
            with transaction(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    f"""
                    SELECT id, content, summary, metadata, sensitivity_label, source, owner
                    FROM {self._table} WHERE id = ?
                    """,
                    (item_id,),
                ).fetchone()
                if row is None:
                    return False
                tombstone = dict(row)
                cursor = conn.execute(f"DELETE FROM {self._table} WHERE id = ?", (item_id,))
                conn.commit()
                deleted = cursor.rowcount > 0

            if deleted:
                self._log_memory_event(
                    "tombstone",
                    item_id=item_id,
                    actor=actor,
                    details={"reason": "delete", **_identity(tombstone)},
                )
                self._deleted_ids.add(item_id)
                self._index_dirty = True
                self._last_add_time = time.time()
            return deleted

    def pin(self, item_id: int, *, actor: str = "orchestra") -> bool:
        """Pin an item so it never expires and gets search boost."""
        with self._lock:
            with transaction(self.db_path) as conn:
                cursor = conn.execute(
                    f"UPDATE {self._table} SET pinned = 1, expires_at = NULL WHERE id = ?",
                    (item_id,)
                )
                conn.commit()
                ok = cursor.rowcount > 0
            if ok:
                self._log_memory_event("pin", item_id=item_id, actor=actor)
            return ok

    def unpin(self, item_id: int, *, actor: str = "orchestra") -> bool:
        """Unpin an item and restore its TTL-based expiration."""
        with self._lock:
            with transaction(self.db_path) as conn:
                cursor = conn.execute(
                    f"""
                    UPDATE {self._table}
                    SET pinned = 0,
                        expires_at = datetime('now', '+' || CAST(ttl_hours AS TEXT) || ' hours')
                    WHERE id = ?
                    """,
                    (item_id,)
                )
                conn.commit()
                ok = cursor.rowcount > 0
            if ok:
                self._log_memory_event("unpin", item_id=item_id, actor=actor)
            return ok

    def increment_citation(self, item_id: int, *, actor: str = "orchestra") -> bool:
        """Increment citation count for an item (used for importance-weighted search)."""
        with self._lock:
            with transaction(self.db_path) as conn:
                cursor = conn.execute(
                    f"UPDATE {self._table} SET citation_count = citation_count + 1 WHERE id = ?",
                    (item_id,)
                )
                conn.commit()
                ok = cursor.rowcount > 0
            if ok:
                self._log_memory_event("cite", item_id=item_id, actor=actor)
            return ok

    def cleanup_expired(self, *, actor: str = "orchestra") -> int:
        """Remove expired, non-pinned items. Returns count of deleted rows."""
        with self._lock:
            with transaction(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"""
                    SELECT id, content, summary, metadata, sensitivity_label, source, owner
                    FROM {self._table}
                    WHERE pinned = 0 AND expires_at < datetime('now')
                    """
                ).fetchall()
                candidates = [dict(r) for r in rows]
                cursor = conn.execute(
                    f"DELETE FROM {self._table} WHERE pinned = 0 AND expires_at < datetime('now')"
                )
                conn.commit()
                deleted = cursor.rowcount

            if deleted:
                for row in candidates:
                    self._log_memory_event(
                        "tombstone",
                        item_id=row["id"],
                        actor=actor,
                        details={"reason": "cleanup_expired", **_identity(row)},
                    )
                    self._deleted_ids.add(row["id"])
                self._index_dirty = True
                self._last_add_time = time.time()
            return deleted

    def delete_by_source(self, source: str, *, actor: str = "orchestra") -> int:
        """Delete every row that came from *source* (a run id). Returns how many."""
        with transaction(self.db_path) as conn:
            ids = [row[0] for row in conn.execute(f"SELECT id FROM {self._table} WHERE source = ?", (source,))]
        return sum(1 for item_id in ids if self.delete(int(item_id), actor=actor))

    def sweep(self, *, event_days: int = 30, actor: str = "orchestra") -> int:
        """Housekeeping: remove expired rows and log entries older than *event_days*.

        Returns the number of rows removed. Without this the TTL was only a filter at search
        time and the event log grew with every search.
        """
        removed = self.cleanup_expired(actor=actor)
        with self._lock:
            with transaction(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM memory_events WHERE created_at < datetime('now', '-' || CAST(? AS TEXT) || ' days')",
                    (int(event_days),),
                )
                conn.commit()
        return removed

    def search(self, query: str, limit: int = 5, *, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar content and include neighbor info.

        Filters out expired items (unless pinned). Boosts pinned and
        highly-cited items in ranking. ``memory_type`` is accepted for interface
        parity with TypedMemory and ignored by the plain single-table store.
        """
        query_vector = _ollama.embed(query)
        if not query_vector:
            return []

        query_vec = np.array(query_vector, dtype=np.float32)
        with self._lock:
            return self._search_locked(query_vec, query, limit)

    def _search_locked(
        self, query_vec: np.ndarray, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Locked search implementation — caller must hold ``self._lock``."""
        # Build SQL filter for expired items
        # Pinned items bypass expiration; non-pinned must have expires_at > now or be NULL
        expired_filter = (
            "(pinned = 1 OR expires_at IS NULL OR expires_at > datetime('now'))"
        )

        # Try HNSW path
        if _HNSW_AVAILABLE:
            if self._index_dirty:
                self._build_hnsw_index()
            elif self._index is None:
                loaded = self._load_hnsw_index()
                if not loaded:
                    self._build_hnsw_index()
            elif not self._is_index_fresh():
                self._build_hnsw_index()

            count = self._index_meta.get("count", 0)
            if self._index is not None and count > 0 and self._index_meta.get("dim") == len(query_vec):
                ids_map = self._index_meta["ids"]
                q = query_vec.reshape(1, -1).astype(np.float32)
                k = min(limit * 4, len(ids_map))  # Over-fetch to account for filtering
                labels, _ = self._index.knn_query(q, k=k)
                candidate_indices = labels[0]

                placeholders = ",".join("?" * len(candidate_indices))
                db_ids = [ids_map[i] for i in candidate_indices]

                results: List[Dict[str, Any]] = []
                with transaction(self.db_path) as conn:
                    cursor = conn.execute(
                        f"""
                        SELECT id, content, summary, vector, metadata, file_path,
                               pinned, citation_count, ttl_hours, sensitivity_label,
                               source, owner, created_at, expires_at
                        FROM {self._table}
                        WHERE id IN ({placeholders})
                          AND {expired_filter}
                        """,
                        tuple(db_ids)
                    )
                    rows = {row[0]: row[1:] for row in cursor.fetchall()}

                for idx, db_id in zip(candidate_indices, db_ids):
                    if db_id not in rows:
                        continue
                    (content, summary, vector_blob, meta_json, file_path,
                     pinned, citation_count, ttl_hours, sensitivity_label,
                     source, owner, created_at, expires_at) = rows[db_id]
                    vec = np.frombuffer(vector_blob, dtype=np.float32)
                    if len(vec) != len(query_vec):
                        continue
                    similarity = float(
                        np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
                    )
                    # Apply importance boosts
                    boost = 0.0
                    if pinned:
                        boost += self.PINNED_BOOST
                    if citation_count:
                        boost += min(citation_count * self.CITATION_BOOST, 0.10)
                    boost += self._recency_boost(created_at)
                    results.append({
                        "id": db_id,
                        "content": content,
                        "summary": summary,
                        "similarity": similarity + boost,
                        "raw_similarity": similarity,
                        "file_path": file_path,
                        "metadata": json.loads(meta_json) if meta_json else {},
                        "pinned": bool(pinned),
                        "citation_count": citation_count,
                        "ttl_hours": ttl_hours,
                        "sensitivity_label": sensitivity_label,
                        "source": source,
                        "owner": owner,
                        "created_at": created_at,
                        "expires_at": expires_at,
                    })

                results.sort(key=lambda x: x["similarity"], reverse=True)
                top_results = results[:limit]

                for res in top_results:
                    if res["file_path"]:
                        res["dependencies"] = self.get_neighbors(res["file_path"])
                self._audit_pii_access(top_results, query)
                self._log_memory_event(
                    "search",
                    actor="orchestra",
                    details={"query_chars": len(query), "limit": limit, "results": len(top_results)},
                )
                return top_results

        # Fallback numpy path
        results: List[Dict[str, Any]] = []
        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT id, content, summary, vector, metadata, file_path,
                       pinned, citation_count, ttl_hours, sensitivity_label,
                       source, owner, created_at, expires_at
                FROM {self._table}
                WHERE {expired_filter}
                """
            )
            for (item_id, content, summary, vector_blob, meta_json, file_path,
                 pinned, citation_count, ttl_hours, sensitivity_label,
                 source, owner, created_at, expires_at) in cursor:
                vec = np.frombuffer(vector_blob, dtype=np.float32)
                if len(vec) != len(query_vec):
                    continue
                similarity = float(
                    np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
                )
                boost = 0.0
                if pinned:
                    boost += self.PINNED_BOOST
                if citation_count:
                    boost += min(citation_count * self.CITATION_BOOST, 0.10)
                boost += self._recency_boost(created_at)
                results.append({
                    "id": item_id,
                    "content": content,
                    "summary": summary,
                    "similarity": similarity + boost,
                    "raw_similarity": similarity,
                    "file_path": file_path,
                    "metadata": json.loads(meta_json) if meta_json else {},
                    "pinned": bool(pinned),
                    "citation_count": citation_count,
                    "ttl_hours": ttl_hours,
                    "sensitivity_label": sensitivity_label,
                    "source": source,
                    "owner": owner,
                    "created_at": created_at,
                    "expires_at": expires_at,
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:limit]

        for res in top_results:
            if res["file_path"]:
                res["dependencies"] = self.get_neighbors(res["file_path"])

        self._audit_pii_access(top_results, query)
        self._log_memory_event(
            "search",
            actor="orchestra",
            details={"query_chars": len(query), "limit": limit, "results": len(top_results)},
        )
        return top_results

    def _recency_boost(self, created_at) -> float:
        """Exponential half-life recency boost; 0.0 when disabled or unparseable."""
        from orchestra import config
        if not config.memory_recency_enabled() or not created_at:
            return 0.0
        try:
            ts = datetime.strptime(str(created_at)[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return 0.0
        half_life = config.memory_recency_half_life_hours()
        if half_life <= 0:
            return 0.0
        age_hours = max(0.0, (self._now_fn() - ts).total_seconds() / 3600.0)
        # True half-life: the boost halves every ``half_life`` hours.
        return float(config.memory_recency_weight() * (0.5 ** (age_hours / half_life)))

    def _audit_pii_access(self, results: List[Dict[str, Any]], query: str) -> None:
        """Log access to PII-labelled memory entries for compliance."""
        for res in results:
            if res.get("sensitivity_label") == "pii":
                logger.info(
                    "PII_ACCESS: query=%r item_id=%d owner=%s source=%s",
                    query, res["id"], res.get("owner"), res.get("source"),
                )

    def get_memory_ledger(self) -> List[Dict[str, Any]]:
        """Return a complete audit ledger of all memory entries.

        Useful for compliance, debugging, and ownership verification.
        """
        with transaction(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"""
                SELECT id, content, metadata, created_at, expires_at,
                       ttl_hours, sensitivity_label, source, owner, pinned, citation_count
                FROM {self._table}
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_memory_audit_log(
        self,
        item_id: int | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return the append-only audit trail for memory operations.

        Args:
            item_id: Filter events for a specific memory item.
            event_type: Filter by event type (add, search, pin, unpin, cite, tombstone).
            limit: Maximum number of events to return.

        Returns:
            List of audit events ordered by newest first.
        """
        with transaction(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conditions = []
            params: List[Any] = []
            if item_id is not None:
                conditions.append("item_id = ?")
                params.append(item_id)
            if event_type is not None:
                conditions.append("event_type = ?")
                params.append(event_type)
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            query = f"""
                SELECT id, event_type, item_id, actor, details, created_at
                FROM memory_events
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(limit)
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                rec = dict(row)
                if rec.get("details"):
                    rec["details"] = json.loads(rec["details"])
                results.append(rec)
            return results

    def add_relation(self, source: str, target: str, rel_type: str = "depends_on"):
        """Record a dependency between two project files."""
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO relations (source_path, target_path, relation_type) VALUES (?, ?, ?)",
                (source, target, rel_type)
            )
            conn.commit()

    def get_neighbors(self, file_path: str) -> List[str]:
        """Find related files in the project atlas."""
        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT target_path FROM relations WHERE source_path = ?", (file_path,)
            )
            return [row[0] for row in cursor.fetchall()]


def is_task_record(item: Dict[str, Any]) -> bool:
    """True for a row written automatically when a run finished: that run's task text.

    Such rows exist so the router can find similar earlier tasks. They are not knowledge and
    must not be shown to an agent as context: an agent answers them.
    """
    return bool((item.get("metadata") or {}).get("run_id"))


def search_notes(memory: Any, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for what someone chose to remember, leaving out the task text of earlier runs."""
    rows = memory.search(query, limit=max(limit * 10, 50)) or []
    return [row for row in rows if not is_task_record(row)][:limit]


def get_memory():
    from orchestra import config
    db_path = config.memory_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if config.memory_typed_enabled():
        from orchestra.storage.typed_memory import TypedMemory
        return TypedMemory(str(db_path))
    return SemanticMemory(str(db_path))
