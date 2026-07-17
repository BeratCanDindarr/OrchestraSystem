# Faz 4 kalan — Observability + Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans / subagent-driven-development. Steps use `- [ ]`.

**Goal:** OTel span'a token/cost emit + üç-katman eval (unit judge persistence + regression + production sampling) + çekimserlik metriği — hepsi config-gate default-off, sıfır regresyon.

**Architecture:** Bkz `docs/superpowers/specs/2026-07-17-faz4-observability-eval-design.md`. 5 bağımsız blok (A-E).

## Global Constraints

- Gate'ler default-off; kapalıyken davranış birebir. Mevcut test suite yeşil kalır.
- DI-testli, gerçek model/OTel collector yok. OTel: tracer **çağrı doğrulama** (span-attribute değil).
- Robust-column deseni (`migrations.COLUMNS`) — yeni migration modülü yok.
- Conventional commits (`feat:`/`test:`). Scratch dizininden test.

---

### Task A: OTel token/cost span emission

**Files:** Modify `orchestra/engine/runners/common.py`, `dual_runner.py`, `critical_runner.py`. Test `tests/test_span_usage.py` (create).

**Interfaces:** Produces `common._emit_span_usage(span, run, alias) -> None`.

- [ ] **A1** Write failing test `tests/test_span_usage.py`: patch `common._get_tracer` → MagicMock tracer; MagicMock span; build a run with one agent (alias="x", `estimated_completion_tokens=42`, `estimated_cost_usd=0.5`); call `_emit_span_usage(span, run, "x")`; assert `tracer.set_tokens.assert_called_once_with(span, 0, 42)` and `tracer.set_cost.assert_called_once_with(span, 0.5)`. Second test: `span=None` → no calls; missing alias → no calls.
- [ ] **A2** Run → FAIL (`_emit_span_usage` undefined).
- [ ] **A3** Add `_emit_span_usage` to common.py (see spec code).
- [ ] **A4** Wire into all three finally blocks before `span.end()`: `_emit_span_usage(span, run, alias)` (ask: `alias`; dual/critical: `agent_alias`). dual/critical `from orchestra.engine.runners.common import _emit_span_usage`.
- [ ] **A5** Run test + `tests/test_tracer.py` → PASS.
- [ ] **A6** Commit `feat(observability): emit token/cost to OTel agent spans`.

---

### Task B: config accessors + calibration wiring

**Files:** Modify `orchestra/config.py`, `orchestra/engine/runners/common.py`. Test `tests/test_eval_calibration_wiring.py` (create).

**Interfaces:** Produces `config.eval_calibration_enabled() -> bool`, `config.eval_sample_rate() -> float`.

- [ ] **B1** Write failing test: monkeypatch `config.eval_calibration_enabled → True`; patch `orchestra.engine.calibration.assess_calibration` → recording stub returning a default dict; call the finalize calibration path (test a small extracted helper `common._maybe_assess_calibration(run) -> dict | None`) with a COMPLETED run + summary; assert stub called with `(run.task, run.summary)`. Gate off → returns None, stub not called.
- [ ] **B2** Run → FAIL.
- [ ] **B3** Add accessors to config.py (after `calibration_judge_alias`):
```python
def eval_calibration_enabled() -> bool:
    return bool(get("eval", "calibration_enabled", False))

def eval_sample_rate() -> float:
    return float(get("eval", "sample_rate", 1.0))
```
- [ ] **B4** Add `common._maybe_assess_calibration(run)` helper: `if not config.eval_calibration_enabled() or not run.summary: return None`; `from orchestra.engine import calibration`; `return calibration.assess_calibration(run.task, run.summary)` (wrapped in try/except → None).
- [ ] **B5** Run test → PASS.
- [ ] **B6** Commit `feat(eval): add calibration gate + sample_rate config, calibration helper`.

---

### Task C: judge score persistence

**Files:** Modify `orchestra/storage/migrations.py`, `orchestra/storage/eval_tracker.py`. Test `tests/test_eval_tracker_scores.py` (create).

**Interfaces:** Produces `EvalTracker.log_score(run_id, task_hash, scores: dict) -> None`.

- [ ] **C1** Write failing test: in-memory sqlite, `ensure_schema(conn)`; patch `eval_tracker.get_db` → returns this conn (context-manager safe wrapper); `EvalTracker().log_score("r1","h1", {"answer_quality":0.8,"factual_grounding":0.7,"source_attribution":1.0,"conciseness":0.9,"calibration":0.6,"composite_score":0.8})`; query `eval_results` → row has task_hash="h1", composite_score=0.8, calibration=0.6.
- [ ] **C2** Run → FAIL.
- [ ] **C3** Extend `migrations.COLUMNS["eval_results"]` with `task_hash TEXT, answer_quality REAL, factual_grounding REAL, source_attribution REAL, conciseness REAL, calibration REAL, composite_score REAL`.
- [ ] **C4** Add `EvalTracker.log_score(run_id, task_hash, scores)` — INSERT into eval_results (run_id, task_hash, outcome='SCORED', 5 axes, calibration, composite_score, created_at). Use `.get(k, 0.0)` per field.
- [ ] **C5** Run test + `tests/test_migrations.py`-adjacent → PASS.
- [ ] **C6** Commit `feat(eval): persist judge + calibration scores to eval_results`.

---

### Task D: regression tier

**Files:** Modify `orchestra/storage/eval_tracker.py`. Test extends `tests/test_eval_tracker_scores.py`.

**Interfaces:** Produces `EvalTracker.regression_delta(task_hash) -> dict` = `{"task_hash","latest","historical_mean","delta","n"}`.

- [ ] **D1** Write failing test: log_score three rows for "h1" composite 0.6, 0.8, 0.4 (created_at ascending); `regression_delta("h1")` → n=3, latest=0.4, historical_mean=mean(0.6,0.8)=0.7, delta=0.4-0.7=-0.3. n<2 → delta 0.0.
- [ ] **D2** Run → FAIL.
- [ ] **D3** Add `regression_delta` — SELECT composite_score FROM eval_results WHERE task_hash=? ORDER BY created_at/id; latest = last, historical = mean of the rest.
- [ ] **D4** Run test → PASS.
- [ ] **D5** Commit `feat(eval): add regression_delta historical comparison`.

---

### Task E: production trace sampling

**Files:** Create `orchestra/engine/eval_sampler.py`. Test `tests/test_eval_sampler.py` (create).

**Interfaces:** Produces `eval_sampler.should_sample(run_id: str, rate: float) -> bool`.

- [ ] **E1** Write failing test: `should_sample("r1", 1.0) is True`; `should_sample("r1", 0.0) is False`; deterministic: same run_id+rate → same result across calls; over many run_ids at rate=0.5, roughly half sampled (assert 0.3<frac<0.7 for 200 ids).
- [ ] **E2** Run → FAIL (module missing).
- [ ] **E3** Create `eval_sampler.py` (see spec: sha256 hash → [0,1) < rate; edges 1.0/0.0).
- [ ] **E4** Run test → PASS.
- [ ] **E5** Commit `feat(eval): add deterministic production trace sampler`.

---

### Task F: wire B/C/E into _finalize judge block

**Files:** Modify `orchestra/engine/runners/common.py` (judge block ~682-706). Test extends `tests/test_eval_calibration_wiring.py`.

- [ ] **F1** Write failing test: monkeypatch `config.eval_harness_enabled→True`, `eval_sample_rate→0.0`; patch `eval_harness.run_judge`; run `_finalize`-judge-path; assert judge NOT called (sampled out). Then `sample_rate→1.0` + patch run_judge + `EvalTracker.log_score`; assert `log_score` called with composite from judge result. (Use existing `test_semantic_indexing` fixtures for a COMPLETED run + patched artifacts/append_event/get_memory.)
- [ ] **F2** Run → FAIL.
- [ ] **F3** In common.py judge block: guard with `and eval_sampler.should_sample(run.run_id, config.eval_sample_rate())`; after `result = run_judge(...)`, call `_maybe_assess_calibration(run)`, build scores dict (4 axes + calibration + composite), `EvalTracker().log_score(run.run_id, eval_harness.task_hash(run.task), scores)`; include calibration in the `eval_judge_completed` event. All wrapped in the existing try/except.
- [ ] **F4** Run test + `tests/test_semantic_indexing.py` + `tests/test_eval_harness_unit.py` → PASS.
- [ ] **F5** Commit `feat(eval): wire sampling+calibration+persistence into finalize judge`.

---

## Final Verification

```bash
.venv/bin/python -m pytest tests/test_span_usage.py tests/test_eval_tracker_scores.py \
  tests/test_eval_sampler.py tests/test_eval_calibration_wiring.py tests/test_tracer.py \
  tests/test_eval_harness_unit.py tests/test_calibration_unit.py tests/test_semantic_indexing.py \
  -p no:cacheprovider --no-cov -q
.venv/bin/mypy orchestra/engine/eval_sampler.py orchestra/storage/eval_tracker.py orchestra/config.py
```
Confirm 77 pre-existing full-suite failures unchanged (baseline diff vs main).

## Progress Log

| Date | Tasks | Notes |
|------|-------|-------|

## Self-Review

- Spec coverage: A(token/cost)✓ B(calibration gate+helper)✓ C(persistence)✓ D(regression)✓ E(sampling)✓ F(finalize wiring)✓.
- Placeholders: none — code in spec + steps.
- Type consistency: `_emit_span_usage`, `log_score`, `regression_delta`, `should_sample`, `eval_calibration_enabled`, `eval_sample_rate` uniform across tasks.
