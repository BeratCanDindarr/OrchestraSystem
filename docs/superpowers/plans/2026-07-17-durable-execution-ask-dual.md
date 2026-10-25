# ask/dual Durable Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ask` ve `dual` modlarına, `critical` modda mevcut olan crash-recovery resume + signal-based suspend yeteneğini kazandır.

**Architecture:** Yeni saf modül `orchestra/engine/durable_execution.py` iki yeteneği taşır: (A) `resume_ask`/`resume_dual` — `_resume_critical` semantiğini yansıtan, dependency-injected checkpoint-boundary devam fonksiyonları; (B) `suspend_on_signal` — config açıkken SIGINT/SIGTERM'de run state'i `paused_runs`'a yazan context manager. `parallel.py`'ye dokunulmaz. Feature B ile suspend edilen run'lar mevcut `resume_runner` DB dalıyla, sert crash ise yeni fallback routing ile devam eder.

**Tech Stack:** Python 3.11+, stdlib `signal`/`contextlib`, SQLite (`state_suspension`), pytest.

## Global Constraints

- Config-gate `durable_execution.enabled` **varsayılan `False`**; gate kapalıyken davranış **hiç değişmez**.
- Immutability tercihi: yeni obje üret, mevcut olanı gereksiz mutasyon etme (signal handler'ın run mutasyonu crash-recovery için bilinçli istisna, event log'da belgelenir).
- `parallel.py` **değiştirilmez** (CLAUDE.md: lock altında mutasyon riski).
- Testler scratch dizininden çalıştırılır: `.venv/bin/python -m pytest <dosya> -p no:cacheprovider --no-cov` (repo'da path-isolation fixture yok — testler CWD'ye artifact yazar).
- Conventional Commits; `feat:` / `test:` / `docs:` prefiksleri.
- Dürüstlük sınırı: agent çağrısı **sırasında** kill → o subprocess çıktısı kaybolur; garanti **son tamamlanan checkpoint'ten** devamdır (critical ile aynı).

---

### Task 1: `resume_ask` / `resume_dual` — checkpoint-boundary devam (Feature A)

**Files:**
- Create: `orchestra/engine/durable_execution.py`
- Test: `tests/test_durable_execution.py`

**Interfaces:**
- Consumes: `orchestra.engine.artifacts.append_event`, `artifacts.write_manifest`; `orchestra.models.{OrchestraRun,RunStatus,AgentRun,AgentStatus}`; `orchestra.state.ApprovalState`.
- Produces:
  - `resume_ask(run, checkpoint, *, finalize_fn, load_paused_fn, delete_paused_fn, get_db_fn, force_continue=False, **kwargs) -> OrchestraRun`
  - `resume_dual(run, checkpoint, *, synthesize_fn, finalize_fn, load_paused_fn, delete_paused_fn, get_db_fn, force_continue=False, **kwargs) -> OrchestraRun`
  - `_prepare_resume(run, checkpoint, force_continue) -> str | None` (values: `"wait"`, `"continue"`, `None`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_durable_execution.py
import sqlite3

import pytest

from orchestra.engine.durable_execution import resume_ask, resume_dual
from orchestra.models import AgentRun, OrchestraRun, RunStatus
from orchestra.state import ApprovalState


def _mem_db():
    return sqlite3.connect(":memory:")


def _run(mode):
    r = OrchestraRun(mode=mode, task="t")
    r.status = RunStatus.RUNNING
    return r


def test_resume_dual_continues_from_running_checkpoint():
    run = _run("dual")
    calls = {"synth": 0, "final": 0}
    resume_dual(
        run,
        {"label": "verification_complete", "status": "running"},
        synthesize_fn=lambda r, task, **kw: calls.__setitem__("synth", calls["synth"] + 1),
        finalize_fn=lambda r: calls.__setitem__("final", calls["final"] + 1),
        load_paused_fn=lambda c, rid: None,
        delete_paused_fn=lambda c, rid: None,
        get_db_fn=_mem_db,
    )
    assert calls == {"synth": 1, "final": 1}
    assert run.approval_state == ApprovalState.RESUMED


def test_resume_ask_finalizes_but_never_synthesizes():
    run = _run("ask")
    called = []
    resume_ask(
        run,
        {"label": "ask_completion", "status": "running"},
        finalize_fn=lambda r: called.append("final"),
        load_paused_fn=lambda c, rid: None,
        delete_paused_fn=lambda c, rid: None,
        get_db_fn=_mem_db,
    )
    assert called == ["final"]


def test_resume_waits_at_pending_approval_gate():
    run = _run("ask")
    run.approval_state = ApprovalState.PENDING
    called = []
    out = resume_ask(
        run,
        {"label": "approval_gate", "status": "waiting_approval"},
        finalize_fn=lambda r: called.append("final"),
        load_paused_fn=lambda c, rid: None,
        delete_paused_fn=lambda c, rid: None,
        get_db_fn=_mem_db,
    )
    assert out.status == RunStatus.WAITING_APPROVAL
    assert called == []


def test_resume_dual_restores_paused_state():
    run = _run("dual")
    paused = _run("dual")
    agent = AgentRun(alias="x", provider="p", model="m")
    agent.stdout_log = "hi"
    paused.agents = [agent]
    paused.reviews = [{"r": 1}]
    paused.summary = "SUMMARY"
    deleted = []
    out = resume_dual(
        run,
        {"label": "round1_complete", "status": "running"},
        synthesize_fn=lambda r, task, **kw: None,
        finalize_fn=lambda r: None,
        load_paused_fn=lambda c, rid: paused,
        delete_paused_fn=lambda c, rid: deleted.append(rid),
        get_db_fn=_mem_db,
    )
    assert out.summary == "SUMMARY"
    assert len(out.agents) == 1 and out.agents[0].alias == "x"
    assert out.reviews == [{"r": 1}]
    assert deleted == [run.run_id]
```

- [ ] **Step 2: Run test to verify it fails**

Run (scratch dir): `.venv/bin/python -m pytest tests/test_durable_execution.py -p no:cacheprovider --no-cov -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestra.engine.durable_execution'`

- [ ] **Step 3: Write minimal implementation**

```python
# orchestra/engine/durable_execution.py
"""Durable execution for ask/dual: crash-recovery resume + signal-based suspend.

Mirrors critical_runner._resume_critical for the ask/dual modes.

Honesty boundary: if the process dies WHILE an agent subprocess is running, that
agent's partial output is lost. Resume continues from the last COMPLETED
checkpoint — identical to critical mode's guarantee.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from orchestra.engine import artifacts
from orchestra.engine.artifacts import append_event
from orchestra.models import OrchestraRun, RunStatus
from orchestra.state import ApprovalState

logger = logging.getLogger("orchestra.engine.runner")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _restore_paused_state(run, load_paused_fn, delete_paused_fn, get_db_fn) -> None:
    """Sync agents/reviews/summary from DB-persisted paused state, if present."""
    try:
        connection = get_db_fn()
        paused_run = load_paused_fn(connection, run.run_id)
        if paused_run:
            run.agents = paused_run.agents
            run.reviews = paused_run.reviews
            run.summary = paused_run.summary
            delete_paused_fn(connection, run.run_id)
            append_event(
                run.run_id,
                {"event": "paused_state_restored", "agent_count": len(run.agents)},
            )
        connection.close()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to restore paused state: %s", e, exc_info=True)


def _prepare_resume(run, checkpoint, force_continue: bool) -> Optional[str]:
    """Shared gate/transition. Returns 'wait', 'continue', or None (unchanged)."""
    checkpoint_label = (checkpoint or {}).get("label", "")
    checkpoint_status = (checkpoint or {}).get("status", "")

    if (
        checkpoint_label == "approval_gate"
        and run.approval_state == ApprovalState.PENDING
        and not force_continue
    ):
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(
            run.run_id,
            {"event": "run_resume_checkpointed", "mode": run.mode, "label": checkpoint_label},
        )
        return "wait"

    if (
        force_continue
        or run.approval_state in {ApprovalState.APPROVED, ApprovalState.RESUMED}
        or checkpoint_status == RunStatus.RUNNING.value
    ):
        run.status = RunStatus.RUNNING
        run.approval_state = ApprovalState.RESUMED
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(
            run.run_id,
            {"event": "run_resumed", "mode": run.mode, "label": checkpoint_label or "manifest"},
        )
        return "continue"

    if checkpoint_status == RunStatus.WAITING_APPROVAL.value:
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        return "wait"

    return None


def resume_ask(
    run,
    checkpoint,
    *,
    finalize_fn,
    load_paused_fn,
    delete_paused_fn,
    get_db_fn,
    force_continue: bool = False,
    **kwargs,
) -> OrchestraRun:
    """Resume an ask-mode run from its last checkpoint (single agent, no synthesis)."""
    if _prepare_resume(run, checkpoint, force_continue) != "continue":
        return run
    _restore_paused_state(run, load_paused_fn, delete_paused_fn, get_db_fn)
    finalize_fn(run)
    return run


def resume_dual(
    run,
    checkpoint,
    *,
    synthesize_fn,
    finalize_fn,
    load_paused_fn,
    delete_paused_fn,
    get_db_fn,
    force_continue: bool = False,
    **kwargs,
) -> OrchestraRun:
    """Resume a dual-mode run from its last checkpoint (synthesize then finalize)."""
    if _prepare_resume(run, checkpoint, force_continue) != "continue":
        return run
    _restore_paused_state(run, load_paused_fn, delete_paused_fn, get_db_fn)
    synthesize_fn(run, run.task, **kwargs)
    finalize_fn(run)
    return run
```

- [ ] **Step 4: Run test to verify it passes**

Run (scratch dir): `.venv/bin/python -m pytest tests/test_durable_execution.py -p no:cacheprovider --no-cov -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestra/engine/durable_execution.py tests/test_durable_execution.py
git commit -m "feat(engine): add ask/dual checkpoint-boundary resume (Phase 2)"
```

---

### Task 2: config gate + `suspend_on_signal` (Feature B)

**Files:**
- Modify: `orchestra/config.py` (add accessor near other feature gates, ~line 393)
- Modify: `orchestra/engine/durable_execution.py` (add `suspend_on_signal`)
- Test: `tests/test_durable_execution.py` (append)

**Interfaces:**
- Consumes: `orchestra.config.get`; stdlib `signal`, `contextlib.contextmanager`.
- Produces:
  - `config.durable_execution_enabled() -> bool`
  - `suspend_on_signal(run, *, suspend_fn, get_db_fn, enabled_fn=None)` — context manager. `suspend_fn(connection, run, paused_by="signal")` matches `state_suspension.suspend_run`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_durable_execution.py
import signal

from orchestra.engine.durable_execution import suspend_on_signal


def test_suspend_on_signal_noop_when_disabled():
    run = OrchestraRun(mode="dual", task="t")
    before = signal.getsignal(signal.SIGINT)
    with suspend_on_signal(
        run,
        suspend_fn=lambda c, r, **k: None,
        get_db_fn=_mem_db,
        enabled_fn=lambda: False,
    ):
        assert signal.getsignal(signal.SIGINT) is before
    assert signal.getsignal(signal.SIGINT) is before


def test_suspend_on_signal_persists_and_restores_when_enabled():
    run = OrchestraRun(mode="dual", task="t")
    run.status = RunStatus.RUNNING
    seen = {}

    def suspend_fn(conn, r, paused_by="system"):
        seen["by"] = paused_by
        seen["status"] = r.status

    before = signal.getsignal(signal.SIGINT)
    with suspend_on_signal(
        run, suspend_fn=suspend_fn, get_db_fn=_mem_db, enabled_fn=lambda: True
    ):
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not before
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    assert seen["by"] == "signal"
    assert seen["status"] == RunStatus.WAITING_APPROVAL
    assert run.approval_state == ApprovalState.RESUMED
    assert signal.getsignal(signal.SIGINT) is before
```

- [ ] **Step 2: Run test to verify it fails**

Run (scratch dir): `.venv/bin/python -m pytest tests/test_durable_execution.py -p no:cacheprovider --no-cov -q`
Expected: FAIL — `ImportError: cannot import name 'suspend_on_signal'`

- [ ] **Step 3: Write minimal implementation**

Add accessor to `orchestra/config.py` (after `reflection_enabled`, ~line 396):

```python
def durable_execution_enabled() -> bool:
    """Persist run state on SIGINT/SIGTERM and enable ask/dual crash-recovery
    resume from the last checkpoint (default off)."""
    return bool(get("durable_execution", "enabled", False))
```

Add to `orchestra/engine/durable_execution.py` (imports + function):

```python
# add to the top-level imports
import signal
from contextlib import contextmanager

from orchestra import config


@contextmanager
def suspend_on_signal(run, *, suspend_fn, get_db_fn, enabled_fn=None):
    """Install SIGINT/SIGTERM handlers that persist the run before exiting.

    No-op (behavior unchanged) unless durable execution is enabled. On signal:
    mark the run WAITING_APPROVAL, persist it to paused_runs, restore the prior
    handlers, and re-raise as KeyboardInterrupt for a clean exit. signal.signal
    only works in the main thread; if unavailable, the guard degrades to no-op.
    """
    if enabled_fn is None:
        enabled_fn = config.durable_execution_enabled
    if not enabled_fn():
        yield
        return

    triggered = {"done": False}
    previous: dict = {}

    def _handler(signum, frame):
        del frame
        if triggered["done"]:
            return
        triggered["done"] = True
        try:
            connection = get_db_fn()
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_state = ApprovalState.RESUMED
            run.updated_at = _now()
            suspend_fn(connection, run, paused_by="signal")
            connection.close()
            append_event(run.run_id, {"event": "durable_suspend", "signal": int(signum)})
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to suspend run on signal: %s", e, exc_info=True)
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not in the main thread — cannot install; leave as no-op.
            previous.pop(sig, None)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run (scratch dir): `.venv/bin/python -m pytest tests/test_durable_execution.py -p no:cacheprovider --no-cov -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestra/config.py orchestra/engine/durable_execution.py tests/test_durable_execution.py
git commit -m "feat(engine): add signal-based suspend for durable execution (Phase 2)"
```

---

### Task 3: `resume_runner` fallback routing for ask/dual (Feature A wiring)

**Files:**
- Modify: `orchestra/engine/runners/resume_runner.py:108-112`
- Test: `tests/test_durable_execution.py` (append)

**Interfaces:**
- Consumes: `resume_ask`, `resume_dual` (Task 1); `resume_runner._finalize`, `_synthesize_if_possible`; `state_suspension.resume_run`, `delete_paused_run`; `storage.db.get_db`.
- Produces: extended fallback routing — a dual/ask manifest with a RUNNING checkpoint and no `paused_runs` row now continues instead of returning unchanged.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_durable_execution.py
from orchestra.engine import artifacts
from orchestra.engine.runners import resume_runner
from orchestra.models import AgentStatus


def test_resume_runner_routes_dual_hard_crash(monkeypatch):
    run = OrchestraRun(mode="dual", task="hard crash task")
    run.status = RunStatus.RUNNING
    agent = AgentRun(alias="a", provider="p", model="m")
    agent.status = AgentStatus.COMPLETED
    agent.stdout_log = "partial"
    run.agents = [agent]
    artifacts.write_manifest(run)
    artifacts.write_checkpoint(run, "verification_complete")

    calls = []
    monkeypatch.setattr(
        resume_runner, "_synthesize_if_possible", lambda r, task, **kw: calls.append("synth")
    )
    monkeypatch.setattr(resume_runner, "_finalize", lambda r: calls.append("final"))

    out = resume_runner.resume_run(run.run_id)
    assert calls == ["synth", "final"]
    assert out.approval_state == ApprovalState.RESUMED
```

- [ ] **Step 2: Run test to verify it fails**

Run (scratch dir): `.venv/bin/python -m pytest tests/test_durable_execution.py::test_resume_runner_routes_dual_hard_crash -p no:cacheprovider --no-cov -q`
Expected: FAIL — `assert [] == ['synth', 'final']` (fallback returns run without continuing for dual)

- [ ] **Step 3: Write minimal implementation**

In `orchestra/engine/runners/resume_runner.py`, replace the fallback tail (currently):

```python
    run = OrchestraRun.from_manifest(manifest)
    if run.mode == "critical":
        return _runner._resume_critical(run, checkpoint, **kwargs)

    return run
```

with:

```python
    run = OrchestraRun.from_manifest(manifest)
    if run.mode == "critical":
        return _runner._resume_critical(run, checkpoint, **kwargs)

    if run.mode in ("dual", "ask"):
        from orchestra.engine.durable_execution import resume_ask, resume_dual
        from orchestra.engine.state_suspension import (
            delete_paused_run,
            resume_run as load_paused_state,
        )
        from orchestra.storage.db import get_db

        if run.mode == "dual":
            return resume_dual(
                run,
                checkpoint,
                synthesize_fn=_synthesize_if_possible,
                finalize_fn=_finalize,
                load_paused_fn=load_paused_state,
                delete_paused_fn=delete_paused_run,
                get_db_fn=get_db,
                **kwargs,
            )
        return resume_ask(
            run,
            checkpoint,
            finalize_fn=_finalize,
            load_paused_fn=load_paused_state,
            delete_paused_fn=delete_paused_run,
            get_db_fn=get_db,
            **kwargs,
        )

    return run
```

- [ ] **Step 4: Run test to verify it passes**

Run (scratch dir): `.venv/bin/python -m pytest tests/test_durable_execution.py -p no:cacheprovider --no-cov -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestra/engine/runners/resume_runner.py tests/test_durable_execution.py
git commit -m "feat(engine): route ask/dual crash recovery through durable_execution (Phase 2)"
```

---

### Task 4: Wire `suspend_on_signal` into `dual_runner` and `_run_ask_impl` + config template

**Files:**
- Modify: `orchestra/engine/runners/dual_runner.py` (around the main `try` at line 51)
- Modify: `orchestra/engine/runners/common.py` (`_run_ask_impl`, around the main `try` after span setup)
- Modify: `.orchestra/config.toml` (add documented, default-off `[durable_execution]` block)
- Test: `tests/test_durable_execution.py` (append gate-off regression)

**Interfaces:**
- Consumes: `suspend_on_signal` (Task 2); `state_suspension.suspend_run`; `storage.db.get_db`.
- Produces: with the gate off (default), the wrapping is a no-op — signal handlers untouched, existing behavior preserved.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_durable_execution.py
import orchestra.engine.runners.common as _common


def test_run_ask_gate_off_leaves_signals_untouched(monkeypatch):
    before = signal.getsignal(signal.SIGINT)

    def fake_parallel(run, agents, **kwargs):
        alias, _prompt = agents[0]
        a = AgentRun(alias=alias, provider="p", model="m")
        a.status = AgentStatus.COMPLETED
        a.stdout_log = "ok"
        a.estimated_cost_usd = 0.0
        run.agents.append(a)
        return [a]

    monkeypatch.setattr(_common, "run_parallel", fake_parallel)
    monkeypatch.setattr("orchestra.config.durable_execution_enabled", lambda: False)

    run = _common._run_ask_impl("cdx-deep", "hello world")
    assert signal.getsignal(signal.SIGINT) is before
    assert run.status in (RunStatus.COMPLETED, RunStatus.WAITING_APPROVAL)
```

- [ ] **Step 2: Run test to verify it fails**

Run (scratch dir): `.venv/bin/python -m pytest tests/test_durable_execution.py::test_run_ask_gate_off_leaves_signals_untouched -p no:cacheprovider --no-cov -q`
Expected: FAIL — `AttributeError`/`NameError` for the not-yet-imported `suspend_on_signal` wiring (or PASS-by-accident before wiring; if it errors on missing wiring, that is the RED state). If it passes trivially, still add the wiring in Step 3 for the gate-on path.

- [ ] **Step 3: Write minimal implementation**

In `orchestra/engine/runners/dual_runner.py`, after `t0 = time.time()` (line ~49) and before `try:` (line ~51), install the guard manually (no body re-indentation), and close it in the existing `finally`:

```python
    span = _get_tracer().start_agent_span(run.run_id, agent_alias, f"dual")
    t0 = time.time()

    from orchestra.engine.durable_execution import suspend_on_signal
    from orchestra.engine.state_suspension import suspend_run as _suspend_fn
    from orchestra.storage.db import get_db as _get_db
    _suspend_ctx = suspend_on_signal(run, suspend_fn=_suspend_fn, get_db_fn=_get_db)
    _suspend_ctx.__enter__()

    try:
        # ... existing body unchanged ...
    except Exception as e:
        if span:
            _get_tracer().set_error(span, str(e))
        raise
    finally:
        _suspend_ctx.__exit__(None, None, None)
        if span:
            _get_tracer().set_timing(span, "total_latency_ms", (time.time() - t0) * 1000)
            span.end()
```

Apply the identical pattern in `orchestra/engine/runners/common.py` `_run_ask_impl`, after its `span`/`t0` setup and before the main `try:`, closing `_suspend_ctx.__exit__(None, None, None)` at the start of that function's `finally` block.

Add to `.orchestra/config.toml`:

```toml
# Durable execution for ask/dual (default off). When enabled, SIGINT/SIGTERM
# persists the run to paused_runs so `orchestra resume <id>` continues from the
# last completed checkpoint. Note: an agent killed mid-call loses its partial
# output — resume continues from the last checkpoint, not the interrupted call.
[durable_execution]
enabled = false
```

- [ ] **Step 4: Run tests to verify they pass**

Run (scratch dir):
```
.venv/bin/python -m pytest tests/test_durable_execution.py -p no:cacheprovider --no-cov -q
.venv/bin/python -m pytest tests/ -m "not slow" -k "dual or ask or resume" -p no:cacheprovider --no-cov -q
```
Expected: `test_durable_execution.py` all PASS (8 passed); dual/ask/resume regression suite shows no NEW failures vs. baseline (pre-existing `test_approval_*` failures that need a real agent/CLI are unrelated — see the branch memory note).

- [ ] **Step 5: Commit**

```bash
git add orchestra/engine/runners/dual_runner.py orchestra/engine/runners/common.py .orchestra/config.toml tests/test_durable_execution.py
git commit -m "feat(engine): enable signal-based suspend in ask/dual runners (Phase 2)"
```

---

## Post-Implementation

- [ ] Run `.venv/bin/mypy orchestra/engine/durable_execution.py` — expect clean.
- [ ] Update the branch memory note (`orchestra-agentic-core-branch.md`) to mark Faz 2 ask/dual durable execution complete.

## Self-Review Notes

- **Spec coverage:** Feature A (`resume_ask`/`resume_dual`) → Task 1 + routing Task 3; Feature B (`suspend_on_signal`) → Task 2 + wiring Task 4; config gate default-off → Task 2 + Task 4 template; honesty boundary → documented in module docstring + config.toml comment + event `durable_suspend`. Signal-suspended resume reuses existing `resume_runner` DB branch (no change needed — verified).
- **Type consistency:** `resume_ask`/`resume_dual`/`_prepare_resume`/`suspend_on_signal`/`durable_execution_enabled` names identical across tasks and call sites.
- **Placeholder scan:** no TBD/TODO; all code shown in full.
