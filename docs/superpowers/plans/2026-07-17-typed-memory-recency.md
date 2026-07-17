# Typed Memory + Recency Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `memory.db`'s single vector store into three physically separate typed stores (episodic/semantic/procedural), each with its own table + HNSW index, and add exponential half-life recency decay as a first-class retrieval signal — all opt-in, config-gated default-off, zero behavior change when disabled.

**Architecture:** Parameterize `SemanticMemory` by table name + index path + injectable clock (behavior-preserving refactor, defaults = today's values). A thin new `TypedMemory` facade holds three `SemanticMemory` instances over the same `memory.db`, routing `add`/`search` by a `memory_type` argument. `get_memory()` returns the facade when `memory.typed_enabled`, else today's plain instance. Recency boost lives inside the shared `search` scoring.

**Tech Stack:** Python 3.11+, SQLite (WAL), hnswlib (+ numpy O(N) fallback), pytest.

## Global Constraints

- **Zero regression when gates off.** `memory.typed_enabled` (default False) and `memory.recency_enabled` (default False) must both leave behavior byte-identical to today. Every existing test in `tests/test_memory.py`, `tests/test_rag_unit.py`, `tests/test_semantic_indexing.py` must stay green.
- **No code duplication** (coding-style): the three typed stores share one `SemanticMemory` code path — no copy-paste of add/search/rebuild.
- **No migration:** existing `embeddings` table = the `semantic` store as-is. Never re-embed or move existing rows.
- **DI-testable:** no real Ollama/DB in tests — mirror the `patch("orchestra.storage.memory._ollama.embed")` pattern and inject `now_fn` for recency.
- **Config accessors only** via `orchestra/config.py` `get(section, key, default)` — never parse TOML directly.
- **Conventional Commits:** `refactor:` for Task 1, `feat:` for the rest. No attribution footer.
- Run tests from a scratch dir: `.venv/bin/python -m pytest <file> -p no:cacheprovider --no-cov`.

---

### Task 1: Parameterize `SemanticMemory` (table + index path + clock)

Behavior-preserving refactor. Add three constructor params with defaults equal to today's hardcoded values, and route every `embeddings`-table SQL statement through `self._table`. `relations` and `memory_events` stay literal (shared).

**Files:**
- Modify: `orchestra/storage/memory.py`
- Test: `tests/test_typed_memory.py` (create)

**Interfaces:**
- Produces: `SemanticMemory(db_path=".orchestra/memory.db", *, embeddings_table="embeddings", index_path=None, now_fn=None)` — `index_path=None` derives today's `{base}_hnsw.idx`; `now_fn=None` defaults to `datetime.utcnow`.

- [ ] **Step 1: Write the failing test**

Add to a new file `tests/test_typed_memory.py`:

```python
"""Tests for parameterized SemanticMemory, TypedMemory facade, and recency."""
import sqlite3
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pytest

from orchestra import config
from orchestra.storage.memory import SemanticMemory


def _mock_embed(text: str, dim: int = 384) -> list:
    h = hash(text) % (2**32)
    np.random.seed(h)
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def test_custom_table_isolates_rows(tmp_path):
    """A SemanticMemory bound to a custom table writes only to that table."""
    db = str(tmp_path / "memory.db")
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        mem = SemanticMemory(
            db, embeddings_table="embeddings_episodic",
            index_path=str(tmp_path / "epi_hnsw.idx"),
        )
        assert mem.add("an episode") is True
        results = mem.search("episode", limit=1)
        assert len(results) == 1 and results[0]["content"] == "an episode"
    with sqlite3.connect(db) as conn:
        epi = conn.execute("SELECT COUNT(*) FROM embeddings_episodic").fetchone()[0]
        # Default 'embeddings' table exists (schema created) but holds nothing here
        base = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert epi == 1
    assert base == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py::test_custom_table_isolates_rows -p no:cacheprovider --no-cov -v`
Expected: FAIL — `SemanticMemory.__init__() got an unexpected keyword argument 'embeddings_table'`.

- [ ] **Step 3: Add imports and parameterize the constructor**

In `orchestra/storage/memory.py`, add to the top imports (after `import numpy as np`):

```python
import math
from datetime import datetime
```

Replace `__init__` and `_derive_index_path` (lines 46-61):

```python
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
```

- [ ] **Step 4: Route every `embeddings` SQL reference through `self._table`**

In `orchestra/storage/memory.py`, replace the literal table name `embeddings` with `{self._table}` in these statements (leave `relations` and `memory_events` untouched). Each becomes an f-string:

`_init_db` — the CREATE/PRAGMA/ALTER/UPDATE block:
```python
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

                conn.execute(f"""
                    UPDATE {self._table}
                    SET expires_at = datetime(created_at, '+' || CAST(IFNULL(ttl_hours, 720) AS TEXT) || ' hours')
                    WHERE expires_at IS NULL AND created_at IS NOT NULL
                """)
```

Then replace the table name in each remaining statement (add `f` prefix to the string):
- `add`: `INSERT INTO {self._table} (...)`
- `delete`: `SELECT ... FROM {self._table} WHERE id = ?` and `DELETE FROM {self._table} WHERE id = ?`
- `pin`: `UPDATE {self._table} SET pinned = 1, ...`
- `unpin`: `UPDATE {self._table} SET pinned = 0, ...`
- `increment_citation`: `UPDATE {self._table} SET citation_count = ...`
- `cleanup_expired`: `SELECT ... FROM {self._table} WHERE pinned = 0 ...` and `DELETE FROM {self._table} WHERE pinned = 0 ...`
- `_build_hnsw_index`: `SELECT id, vector FROM {self._table} ORDER BY id`
- `_search_locked`: both `SELECT ... FROM {self._table} WHERE ...` statements (HNSW path and numpy fallback)
- `get_memory_ledger`: `SELECT ... FROM {self._table} ORDER BY created_at DESC`

> Note: `self._table` is an internal, trusted constant (never user input) — f-string interpolation carries no injection risk.

- [ ] **Step 5: Run the new test + full memory suite to verify pass and zero regression**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py tests/test_memory.py -p no:cacheprovider --no-cov -v`
Expected: PASS — new test green, all 20 existing `test_memory.py` tests green (they use the default `embeddings` table).

- [ ] **Step 6: Commit**

```bash
git add orchestra/storage/memory.py tests/test_typed_memory.py
git commit -m "refactor(memory): parameterize SemanticMemory by table/index/clock"
```

---

### Task 2: Recency boost + config accessors

Add four config accessors and an exponential half-life recency boost wired into both `search` scoring paths. Gated by `memory.recency_enabled` (default off → boost is exactly 0.0).

**Files:**
- Modify: `orchestra/config.py` (after `rag_max_chars`, ~line 765)
- Modify: `orchestra/storage/memory.py` (`_recency_boost` helper + both boost sites)
- Test: `tests/test_typed_memory.py`

**Interfaces:**
- Consumes: `SemanticMemory(..., now_fn=...)` from Task 1.
- Produces: `config.memory_typed_enabled() -> bool`, `config.memory_recency_enabled() -> bool`, `config.memory_recency_half_life_hours() -> float`, `config.memory_recency_weight() -> float`; `SemanticMemory._recency_boost(created_at) -> float`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_typed_memory.py`:

```python
def test_recency_boost_decays_with_age(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_recency_enabled", lambda: True)
    monkeypatch.setattr(config, "memory_recency_half_life_hours", lambda: 10.0)
    monkeypatch.setattr(config, "memory_recency_weight", lambda: 0.2)
    fixed_now = datetime(2026, 7, 17, 12, 0, 0)
    mem = SemanticMemory(str(tmp_path / "m.db"), now_fn=lambda: fixed_now)
    fresh = mem._recency_boost("2026-07-17 12:00:00")   # age 0h
    one_half_life = mem._recency_boost("2026-07-17 02:00:00")  # age 10h == half-life
    assert fresh == pytest.approx(0.2)
    assert one_half_life == pytest.approx(0.1, abs=1e-6)


def test_recency_boost_zero_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_recency_enabled", lambda: False)
    mem = SemanticMemory(str(tmp_path / "m.db"))
    assert mem._recency_boost("2026-07-17 12:00:00") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py::test_recency_boost_decays_with_age -p no:cacheprovider --no-cov -v`
Expected: FAIL — `AttributeError: module 'orchestra.config' has no attribute 'memory_recency_enabled'`.

- [ ] **Step 3: Add config accessors**

In `orchestra/config.py`, after `rag_max_chars()` (before `context_memory_config()`):

```python
# ── Typed Memory & Recency Configuration ─────────────────────────────────────

def memory_typed_enabled() -> bool:
    """Split semantic memory into episodic/semantic/procedural stores (default off)."""
    return bool(get("memory", "typed_enabled", False))


def memory_recency_enabled() -> bool:
    """Add an exponential recency decay boost to memory retrieval scoring (default off)."""
    return bool(get("memory", "recency_enabled", False))


def memory_recency_half_life_hours() -> float:
    """Age (hours) at which the recency boost halves."""
    return float(get("memory", "recency_half_life_hours", 168.0))


def memory_recency_weight() -> float:
    """Maximum recency boost (added to similarity for a brand-new entry)."""
    return float(get("memory", "recency_weight", 0.1))
```

- [ ] **Step 4: Add the `_recency_boost` helper**

In `orchestra/storage/memory.py`, add this method to `SemanticMemory` (e.g. just before `_audit_pii_access`):

```python
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
        return config.memory_recency_weight() * math.exp(-age_hours / half_life)
```

- [ ] **Step 5: Wire the boost into both scoring sites**

In `_search_locked`, the HNSW path (after the `citation_count` boost, ~line 546) and the numpy fallback (~line 604), add one line to each `boost` computation:

```python
                    if citation_count:
                        boost += min(citation_count * self.CITATION_BOOST, 0.10)
                    boost += self._recency_boost(created_at)
```

(Both branches already unpack `created_at` from the row.)

- [ ] **Step 6: Run tests to verify pass and zero regression**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py tests/test_memory.py -p no:cacheprovider --no-cov -v`
Expected: PASS — recency tests green; existing tests unaffected (recency disabled by default → boost 0.0).

- [ ] **Step 7: Commit**

```bash
git add orchestra/config.py orchestra/storage/memory.py tests/test_typed_memory.py
git commit -m "feat(memory): add half-life recency boost to retrieval scoring (gated)"
```

---

### Task 3: `memory_type` shim on plain `SemanticMemory`

Add a `memory_type` keyword to `add` and `search` that the plain class accepts and ignores, so all callers pass it uniformly regardless of whether typed mode is on.

**Files:**
- Modify: `orchestra/storage/memory.py` (`add`, `search` signatures)
- Test: `tests/test_typed_memory.py`

**Interfaces:**
- Produces: `SemanticMemory.add(..., memory_type=None)` and `SemanticMemory.search(query, limit=5, *, memory_type=None)` — `memory_type` ignored by the plain class.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_typed_memory.py`:

```python
def test_plain_memory_ignores_memory_type(tmp_path):
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        mem = SemanticMemory(str(tmp_path / "m.db"))
        assert mem.add("x content", memory_type="episodic") is True
        results = mem.search("x content", limit=1, memory_type="episodic")
        assert len(results) == 1 and results[0]["content"] == "x content"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py::test_plain_memory_ignores_memory_type -p no:cacheprovider --no-cov -v`
Expected: FAIL — `add() got an unexpected keyword argument 'memory_type'`.

- [ ] **Step 3: Add the ignored kwarg to `add` and `search`**

In `add` (line 285), add `memory_type: Optional[str] = None,` to the keyword-only args (e.g. after `pinned: bool = False,`). Do not use it in the body.

In `search` (line 468), change the signature to:

```python
    def search(self, query: str, limit: int = 5, *, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
```

Leave the body unchanged (`memory_type` intentionally unused in the plain class).

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py tests/test_memory.py -p no:cacheprovider --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orchestra/storage/memory.py tests/test_typed_memory.py
git commit -m "feat(memory): accept-and-ignore memory_type on plain SemanticMemory"
```

---

### Task 4: `TypedMemory` facade

New file holding three `SemanticMemory` instances over one `memory.db`, routing by `memory_type` and merging cross-type searches by final score.

**Files:**
- Create: `orchestra/storage/typed_memory.py`
- Test: `tests/test_typed_memory.py`

**Interfaces:**
- Consumes: `SemanticMemory(db_path, *, embeddings_table, index_path, now_fn)` (Task 1); `memory_type` shim (Task 3).
- Produces: `TypedMemory(db_path=".orchestra/memory.db", *, stores=None, now_fn=None)` with `add(content, metadata=None, summary=None, *, memory_type=None, **kwargs) -> bool`, `search(query, limit=5, *, memory_type=None) -> List[Dict]`, plus delegated `pin/unpin/delete/increment_citation(item_id, *, memory_type="semantic", **kwargs)`, `cleanup_expired(**kwargs)` (all stores), `get_neighbors/add_relation` (semantic store), `get_memory_ledger()` (merged).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_typed_memory.py`:

```python
def test_add_routes_to_typed_table(tmp_path):
    from orchestra.storage.typed_memory import TypedMemory
    db = str(tmp_path / "memory.db")
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        tm = TypedMemory(db)
        assert tm.add("a lesson", memory_type="episodic") is True
        assert tm.add("a fact", memory_type="semantic") is True
    with sqlite3.connect(db) as conn:
        sem = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        epi = conn.execute("SELECT COUNT(*) FROM embeddings_episodic").fetchone()[0]
    assert sem == 1 and epi == 1


def test_search_merges_across_types(tmp_path):
    from orchestra.storage.typed_memory import TypedMemory
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        tm = TypedMemory(str(tmp_path / "memory.db"))
        tm.add("alpha semantic", memory_type="semantic")
        tm.add("alpha episodic", memory_type="episodic")
        contents = {r["content"] for r in tm.search("alpha", limit=5)}
    assert "alpha semantic" in contents and "alpha episodic" in contents


def test_search_filters_by_type(tmp_path):
    from orchestra.storage.typed_memory import TypedMemory
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        tm = TypedMemory(str(tmp_path / "memory.db"))
        tm.add("only semantic", memory_type="semantic")
        tm.add("only episodic", memory_type="episodic")
        contents = {r["content"] for r in tm.search("only", limit=5, memory_type="episodic")}
    assert contents == {"only episodic"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py::test_add_routes_to_typed_table -p no:cacheprovider --no-cov -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestra.storage.typed_memory'`.

- [ ] **Step 3: Create the facade**

Create `orchestra/storage/typed_memory.py`:

```python
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

    def get_memory_ledger(self) -> List[Dict[str, Any]]:
        ledger: List[Dict[str, Any]] = []
        for store in self._stores.values():
            ledger.extend(store.get_memory_ledger())
        return ledger

    def get_neighbors(self, file_path: str) -> List[str]:
        return self._stores[_DEFAULT_TYPE].get_neighbors(file_path)

    def add_relation(self, source: str, target: str, rel_type: str = "depends_on"):
        return self._stores[_DEFAULT_TYPE].add_relation(source, target, rel_type)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py -p no:cacheprovider --no-cov -v`
Expected: PASS — routing, merge, and filter tests green.

- [ ] **Step 5: Commit**

```bash
git add orchestra/storage/typed_memory.py tests/test_typed_memory.py
git commit -m "feat(memory): add TypedMemory facade (episodic/semantic/procedural)"
```

---

### Task 5: `get_memory()` gate

Return `TypedMemory` when `memory.typed_enabled`, else the plain instance (today's behavior).

**Files:**
- Modify: `orchestra/storage/memory.py` (`get_memory`, lines 726-729)
- Test: `tests/test_typed_memory.py`

**Interfaces:**
- Consumes: `config.memory_typed_enabled()` (Task 2), `TypedMemory` (Task 4).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_typed_memory.py`:

```python
def test_get_memory_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from orchestra.storage.memory import get_memory
    from orchestra.storage.typed_memory import TypedMemory

    monkeypatch.setattr(config, "memory_typed_enabled", lambda: True)
    assert isinstance(get_memory(), TypedMemory)

    monkeypatch.setattr(config, "memory_typed_enabled", lambda: False)
    assert isinstance(get_memory(), SemanticMemory)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py::test_get_memory_gate -p no:cacheprovider --no-cov -v`
Expected: FAIL — `get_memory()` always returns `SemanticMemory`, so the first assert fails.

- [ ] **Step 3: Add the gate**

In `orchestra/storage/memory.py`, replace `get_memory` (lines 726-729):

```python
def get_memory():
    import os
    os.makedirs(".orchestra", exist_ok=True)
    from orchestra import config
    if config.memory_typed_enabled():
        from orchestra.storage.typed_memory import TypedMemory
        return TypedMemory()
    return SemanticMemory()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py -p no:cacheprovider --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orchestra/storage/memory.py tests/test_typed_memory.py
git commit -m "feat(memory): gate get_memory on memory.typed_enabled"
```

---

### Task 6: Writer → type wiring

Tag each writer with its memory type: reflection lessons and run-finalize summaries → `episodic`; MCP add and project indexing → `semantic`. RAG needs no change (its `search(prompt, limit)` call leaves `memory_type=None` → merge across all types when typed is on).

**Files:**
- Modify: `orchestra/engine/reflection.py` (`_record_lesson`, ~line 85)
- Modify: `orchestra/engine/runners/common.py` (`_finalize` indexing, ~line 735)
- Modify: `orchestra/service.py` (`memory_add_entry` line 704, `memory_index_project_entries` lines 736/739)
- Test: `tests/test_typed_memory.py`, `tests/test_semantic_indexing.py`

**Interfaces:**
- Consumes: `memory_type` kwarg on `add` (Tasks 3/4).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_typed_memory.py`:

```python
def test_record_lesson_tags_episodic(monkeypatch):
    from orchestra.engine import reflection
    monkeypatch.setattr(reflection.config, "reflection_record_lessons", lambda: True)
    captured = {}

    class FakeMem:
        def add(self, content, **kwargs):
            captured["content"] = content
            captured.update(kwargs)
            return True

    reflection._record_lesson("some task", ["was wrong"], "do better", FakeMem())
    assert captured["memory_type"] == "episodic"


def test_memory_add_entry_tags_semantic(monkeypatch):
    captured = {}

    class FakeMem:
        def add(self, content, metadata=None, summary=None, **kwargs):
            captured.update(kwargs)
            return True

    monkeypatch.setattr("orchestra.storage.memory.get_memory", lambda: FakeMem())
    from orchestra.service import memory_add_entry
    memory_add_entry("hello world")
    assert captured["memory_type"] == "semantic"
```

Add to `tests/test_semantic_indexing.py` (new test, reusing existing fixtures/patches):

```python
@patch('orchestra.engine.runner.artifacts')
@patch('orchestra.engine.runner.append_event')
@patch('orchestra.storage.memory.get_memory')
def test_finalize_tags_episodic(mock_get_memory, mock_append_event, mock_artifacts, sample_completed_run):
    """Completed-run summaries are indexed as episodic memory."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory
    _finalize(sample_completed_run)
    assert mock_memory.add.called
    assert mock_memory.add.call_args[1]["memory_type"] == "episodic"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py::test_record_lesson_tags_episodic tests/test_semantic_indexing.py::test_finalize_tags_episodic -p no:cacheprovider --no-cov -v`
Expected: FAIL — `KeyError: 'memory_type'` (writers don't pass it yet).

- [ ] **Step 3: Tag reflection lessons as episodic**

In `orchestra/engine/reflection.py`, the `memory.add(...)` call in `_record_lesson` (line 85), add `memory_type="episodic"`:

```python
        memory.add(
            task,
            metadata={"reflection_lesson": lesson},
            source="reflection_lesson",
            owner="orchestra",
            sensitivity_label="normal",
            memory_type="episodic",
        )
```

- [ ] **Step 4: Tag run-finalize summaries as episodic**

In `orchestra/engine/runners/common.py`, the `memory.add(...)` call in `_finalize` (line 735), add `memory_type="episodic"`:

```python
            memory.add(
                run.task,
                metadata=metadata,
                summary=summary,
                source=run.run_id,
                owner="orchestra",
                sensitivity_label="normal",
                memory_type="episodic",
            )
```

- [ ] **Step 5: Tag MCP add and project indexing as semantic**

In `orchestra/service.py`:

`memory_add_entry` (line 704):
```python
    ok = mem.add(content, metadata or {}, summary=summary, memory_type="semantic")
```

`memory_index_project_entries` — both `mem.add(...)` calls (lines 736 and 739):
```python
                    if mem.add(chunk, {"path": str(file_path), "chunk": chunk_index}, memory_type="semantic"):
```
```python
                if mem.add(content, {"path": str(file_path)}, memory_type="semantic"):
```

- [ ] **Step 6: Run tests to verify pass and zero regression**

Run: `.venv/bin/python -m pytest tests/test_typed_memory.py tests/test_semantic_indexing.py tests/test_rag_unit.py -p no:cacheprovider --no-cov -v`
Expected: PASS — new wiring tests green; existing finalize/rag tests still green.

- [ ] **Step 7: Commit**

```bash
git add orchestra/engine/reflection.py orchestra/engine/runners/common.py orchestra/service.py tests/test_typed_memory.py tests/test_semantic_indexing.py
git commit -m "feat(memory): tag writers by memory type (reflection/run->episodic, mcp/index->semantic)"
```

---

## Final Verification

Run the fast suite + type check to confirm zero regression across the engine:

```bash
.venv/bin/python -m pytest tests/test_typed_memory.py tests/test_memory.py tests/test_semantic_indexing.py tests/test_rag_unit.py -p no:cacheprovider --no-cov -v
.venv/bin/pytest -m "not slow" -q          # full fast suite
.venv/bin/mypy orchestra/storage/memory.py orchestra/storage/typed_memory.py orchestra/config.py
```

Manual smoke (from a scratch dir — CWD pollution warning):

```bash
# typed + recency on
ORCHESTRA_MEMORY__TYPED_ENABLED=true ORCHESTRA_MEMORY__RECENCY_ENABLED=true \
  orchestra ask cdx-deep "test task"   # writes run summary to embeddings_episodic
```

Update the progress log in `docs/superpowers/plans/2026-07-17-typed-memory-recency.md` and record the branch state in the `orchestra-agentic-core-branch` memory once merged.

## Progress Log

| Date | Tasks | Notes |
|------|-------|-------|
| 2026-07-17 | 1-6 | Branch `feat/typed-memory-recency`. All 6 tasks done TDD, one commit each. `test_typed_memory.py` 11 tests + `test_semantic_indexing.py::test_finalize_tags_episodic`. 41 memory/rag/finalize tests green; mypy adds **zero** new errors (recency boost `float()`-wrapped). Recency formula uses **true half-life** `0.5 ** (age/half_life)` (halves at `half_life_hours`) rather than the spec's original `exp(-age/half_life)` — the config name demands halving semantics. `procedural` store created but has no writer yet (per Non-Goals). |

## Self-Review

- **Spec coverage:** three physical stores (Task 4) ✓; recency first-class (Task 2) ✓; writer→type mapping A (Task 6) ✓; RAG merge across types (Task 4 search + no-change note) ✓; config gates default-off (Tasks 2/5) ✓; no migration (Task 4 reuses `embeddings`) ✓; DI-testable (all tasks) ✓.
- **Placeholders:** none — every step carries concrete code/commands.
- **Type consistency:** `memory_type` kwarg name uniform across `SemanticMemory`, `TypedMemory`, and all writers; `_recency_boost`/`now_fn`/`embeddings_table`/`index_path` names consistent between Tasks 1, 2, and 4.
- **Note (procedural):** the `embeddings_procedural` store is created but has no writer yet — intentional per spec Non-Goals (reserved for a future phase).
