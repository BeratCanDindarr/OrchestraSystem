from __future__ import annotations
"""Typed memory facade: routes add/search across three physically separate
SemanticMemory stores (episodic / semantic / procedural), each with its own
table and HNSW index over the same memory.db. Opt-in via config.memory_typed_enabled().
"""
import os
from typing import Any, Dict, List, Optional

from orchestra.storage.memory import SemanticMemory

MEMORY_TYPES = ("semantic", "episodic", "procedural")
_DEFAULT_TYPE = "semantic"
_TABLES = {
    "semantic": "embeddings",              # reuse existing table — no migration
    "episodic": "embeddings_episodic",
    "procedural": "embeddings_procedural",
}


class TypedMemory:
    """Facade over three SemanticMemory stores keyed by memory_type."""

    def __init__(
        self,
        db_path: str = ".orchestra/memory.db",
        *,
        stores: Optional[Dict[str, SemanticMemory]] = None,
        now_fn=None,
    ):
        if stores is not None:
            self._stores = stores
        else:
            base = os.path.splitext(db_path)[0]
            self._stores = {
                mem_type: SemanticMemory(
                    db_path,
                    embeddings_table=table,
                    index_path=(f"{base}_hnsw.idx" if mem_type == "semantic"
                                else f"{base}_hnsw_{mem_type}.idx"),
                    now_fn=now_fn,
                )
                for mem_type, table in _TABLES.items()
            }

    def _store(self, memory_type: Optional[str]) -> SemanticMemory:
        return self._stores.get(memory_type or _DEFAULT_TYPE, self._stores[_DEFAULT_TYPE])

    def add(self, content, metadata=None, summary=None, *, memory_type=None, **kwargs) -> bool:
        return self._store(memory_type).add(content, metadata, summary, **kwargs)

    def search(self, query, limit: int = 5, *, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if memory_type is not None:
            return self._store(memory_type).search(query, limit=limit)
        merged: List[Dict[str, Any]] = []
        for store in self._stores.values():
            merged.extend(store.search(query, limit=limit) or [])
        merged.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return merged[:limit]

    # ── id-based ops route by type (default semantic) ────────────────────────
    def pin(self, item_id, *, memory_type=_DEFAULT_TYPE, **kwargs) -> bool:
        return self._store(memory_type).pin(item_id, **kwargs)

    def unpin(self, item_id, *, memory_type=_DEFAULT_TYPE, **kwargs) -> bool:
        return self._store(memory_type).unpin(item_id, **kwargs)

    def delete(self, item_id, *, memory_type=_DEFAULT_TYPE, **kwargs) -> bool:
        return self._store(memory_type).delete(item_id, **kwargs)

    def increment_citation(self, item_id, *, memory_type=_DEFAULT_TYPE, **kwargs) -> bool:
        return self._store(memory_type).increment_citation(item_id, **kwargs)

    # ── aggregate / shared-table ops ─────────────────────────────────────────
    def cleanup_expired(self, **kwargs) -> int:
        return sum(store.cleanup_expired(**kwargs) for store in self._stores.values())

    def delete_by_source(self, source: str, **kwargs) -> int:
        """Delete a source's rows in every store: a row's id says nothing about its type."""
        return sum(store.delete_by_source(source, **kwargs) for store in self._stores.values())

    def sweep(self, **kwargs) -> int:
        return sum(store.sweep(**kwargs) for store in self._stores.values())

    def get_memory_ledger(self) -> List[Dict[str, Any]]:
        ledger: List[Dict[str, Any]] = []
        for store in self._stores.values():
            ledger.extend(store.get_memory_ledger())
        return ledger

    def get_neighbors(self, file_path: str) -> List[str]:
        return self._stores[_DEFAULT_TYPE].get_neighbors(file_path)

    def add_relation(self, source: str, target: str, rel_type: str = "depends_on"):
        return self._stores[_DEFAULT_TYPE].add_relation(source, target, rel_type)
