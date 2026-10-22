"""The HNSW index of the memory store: where it lives, whether it matches the rows, loading
and rebuilding it. A mixin of `SemanticMemory`, moved out of memory.py to keep it under 800
lines (docs/plans/refactor-oversized-modules.md)."""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

import numpy as np

from orchestra.storage.sqlite import transaction

logger = logging.getLogger("orchestra.storage.memory")


def _hnsw_available() -> bool:
    from orchestra.storage import memory   # the flag lives there; tests switch it off

    return bool(memory._HNSW_AVAILABLE)


class HnswIndexMixin:
    """Uses the attributes `SemanticMemory.__init__` sets: db_path, the index fields, _lock."""

    # Set by SemanticMemory.__init__; declared here for the type checker.
    db_path: str
    _table: str
    _index: Any
    _index_meta: Dict[str, Any]
    _index_dirty: bool
    _last_add_time: float
    _index_path: str
    _dim: Optional[int]
    _lock: threading.RLock
    _pending_vectors: list[tuple[int, np.ndarray]]
    _deleted_ids: set[int]

    def _derive_index_path(self) -> str:
        base = os.path.splitext(self.db_path)[0]
        return f"{base}_hnsw.idx"

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
        return bool(count == expected)

    def _load_hnsw_index(self) -> bool:
        """Load a pre-built HNSW index from disk if it is fresh."""
        if not _hnsw_available():
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

            import hnswlib

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
        if not _hnsw_available():
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

        import hnswlib

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
