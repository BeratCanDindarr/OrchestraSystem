# Design: Faz 4 kalan — OTel token/cost emit + üç-katman eval + çekimserlik metriği

**Date:** 2026-07-17
**Status:** Draft
**Phase:** Faz 4 (Gözlemlenebilirlik & Eval) — kalan iş
**Roadmap:** `~/.claude/plans/benim-i-in-ara-t-rma-yapman-tender-panda.md`, Bölüm 4 · Faz 4

## Problem

Faz 4'ün dashboard + `calibration.py` (anti-sycophancy ölçümü) kısmı bitti. İki eksik kaldı:

1. **OTel span'a token/cost emit edilmiyor.** `tracer.py`'de `set_tokens`/`set_cost` metotları **var** ama üç runner (ask/dual/critical) sadece `set_timing`+`set_error` çağırıyor — token/cost span'a hiç yazılmıyor. Veri (`AgentRun.estimated_completion_tokens`, `estimated_cost_usd`) finalize anında dolu ama kayboluyor.

2. **Eval tek katman.** `EvalTracker.log_run` sadece PASS/FAIL (`eval_runs`) yazıyor. LLM-judge (`eval_harness.run_judge`, 4 eksen + composite) çalışsa bile skorlar sadece event'e düşüyor, **kalıcı tabloya yazılmıyor**. Çekimserlik/anti-sycophancy ölçümü (`calibration.assess_calibration` — mevcut, DI'lı) **`_finalize`'a bağlı değil**. Regresyon karşılaştırması ve production örnekleme yok.

## Goals

- **Block A:** Üç runner'ın span'ına token/cost emit et (DRY helper).
- **Block B:** Çekimserlik/anti-sycophancy metriğini (`assess_calibration`) eval akışına bağla (gated).
- **Block C:** LLM-judge skorlarını (4 eksen + calibration + composite) kalıcı `eval_results` tablosuna yaz.
- **Block D:** Regresyon katmanı — aynı task için geçmiş vs güncel composite skoru karşılaştır.
- **Block E:** Production trace sampling temeli — run'ların bir kesrini deterministik örnekle (maliyet kontrolü).
- Tümü **opt-in, config-gate default-off**, DI-testli, kapalıyken davranış birebir aynı.

## Non-Goals

- Periyodik background regresyon **job scheduler**'ı yok (Faz 6 dashboard/SPA'ya ait; burada sadece sorgulanabilir `regression_delta` temeli).
- `parallel.py` provider-span'ına token/cost (legacy tracer; kapsam dışı).
- input token (`tokens_in`) modelde yok → span'a `tokens_in=0` yazılır (dürüst; sadece output+cost anlamlı).
- Dashboard'ı SPA'ya çevirme (Faz 6).

## Proposed Solution — 5 blok (depth-first, her biri bağımsız deploy)

### Block A — OTel token/cost span emission

`common.py`'ye DRY helper:
```python
def _emit_span_usage(span, run, alias):
    if span is None: return
    agent = next((a for a in run.agents if a.alias == alias), None)
    if agent is None: return
    tracer = _get_tracer()
    tracer.set_tokens(span, 0, getattr(agent, "estimated_completion_tokens", 0))
    tracer.set_cost(span, getattr(agent, "estimated_cost_usd", 0.0))
```
Üç runner'ın `finally` bloğunda `span.end()`'den ÖNCE çağrılır (ask: common.py; dual; critical — `_get_tracer`'ı zaten common'dan alıyorlar, helper'ı da oradan import ederler). Test: mock tracer + mock span, `set_tokens`/`set_cost` doğru argümanlarla çağrıldı mı (span-attribute testi kırılgan; çağrı doğrulama stabil).

### Block B — Çekimserlik/anti-sycophancy metriği

`_finalize`'daki judge bloğunda, judge sonrası (gated `eval_calibration_enabled`):
```python
if config.eval_calibration_enabled():
    assessment = calibration.assess_calibration(run.task, run.summary)
    # assessment["calibration_score"], ["appropriately_abstained"], ["sycophantic"], ["overconfident"]
```
Skor `eval_judge_completed` event'ine ve (Block C ile) persistence'a eklenir. `assess_calibration` zaten DI'lı + tolerant-fail (hata → conservative default). Config: `eval_calibration_enabled()` default False.

### Block C — Judge skor persistence

`eval_results` tablosunu robust-column deseniyle genişlet (yeni migration modülü YOK — `migrations.COLUMNS["eval_results"]`'e ekle):
```
task_hash TEXT, answer_quality REAL, factual_grounding REAL,
source_attribution REAL, conciseness REAL, calibration REAL, composite_score REAL
```
`EvalTracker.log_score(run_id, task_hash, scores: dict)` bir satır INSERT eder. `_finalize`'da judge (+calibration) sonrası çağrılır (gate: `eval_harness_enabled`, judge zaten o gate ile çalışıyor). Test: in-memory db, log_score → satır doğrula.

### Block D — Regresyon katmanı

`EvalTracker.regression_delta(task_hash) -> dict`:
```python
{"task_hash": ..., "latest": float, "historical_mean": float, "delta": float, "n": int}
```
`eval_results`'tan `task_hash`'e ait composite skorları çeker; en yeni vs önceki-ortalama farkını döner (n<2 → delta 0). Pure sorgu, event opsiyonel. Test: in-memory db, birden çok skor → beklenen delta.

### Block E — Production trace sampling temeli

Yeni `engine/eval_sampler.py`:
```python
def should_sample(run_id: str, rate: float) -> bool:
    if rate >= 1.0: return True
    if rate <= 0.0: return False
    # deterministik: run_id hash'i [0,1)'e map → rate ile eşikle (stabil, Math.random YOK)
    h = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < rate
```
`_finalize` judge gate'i: `eval_harness_enabled() AND should_sample(run.run_id, config.eval_sample_rate())`. Config: `eval_sample_rate()` default **1.0** (eval açıksa hepsini judge et; <1.0 production'da maliyet için örnekler). Test: rate=0→hiç, rate=1→hep, sabit run_id deterministik.

## Config (yeni accessor'lar — `config.py`)

| Accessor | Anahtar | Default |
|----------|---------|---------|
| `eval_calibration_enabled()` | `eval.calibration_enabled` | `False` |
| `eval_sample_rate()` | `eval.sample_rate` | `1.0` |

(`eval_harness_enabled`, `eval_judge_alias`, `calibration_judge_alias` zaten var.)

## Değişecek Dosyalar

- `orchestra/engine/runners/common.py` — `_emit_span_usage` helper (A); 3 finally'den biri (ask) + judge bloğu calibration/persistence/sampling wiring (B/C/E).
- `orchestra/engine/runners/dual_runner.py`, `critical_runner.py` — `_emit_span_usage` çağrısı (A).
- `orchestra/storage/eval_tracker.py` — `log_score` (C), `regression_delta` (D).
- `orchestra/storage/migrations.py` — `eval_results` COLUMNS genişlet (C).
- `orchestra/engine/eval_sampler.py` — **YENİ** (E).
- `orchestra/config.py` — 2 accessor (B/E).
- Testler: `tests/test_span_usage.py` (A, YENİ), `tests/test_eval_tracker_scores.py` (C/D, YENİ), `tests/test_eval_sampler.py` (E, YENİ), `tests/test_eval_calibration_wiring.py` (B, YENİ).

## Testing (DI ile, gerçek model/collector olmadan)

- A: mock `_get_tracer()` + MagicMock span → `set_tokens(span,0,N)`, `set_cost(span,C)` çağrı doğrula; agent yok/span None → no-op.
- B: `calibration.assess_calibration`'ı fake `run_ask_fn` ile; gate off → çağrılmaz.
- C: in-memory db → `log_score` satırı + kolonlar.
- D: çok skorlu db → `regression_delta` beklenen değerler.
- E: `should_sample` deterministik (sabit run_id), rate uçları.
- Testler scratch dizininden: `.venv/bin/python -m pytest <dosya> -p no:cacheprovider --no-cov`.

## Risks

| Risk | Mitigation |
|------|------------|
| OTel span attribute testi kırılgan | Span-attribute yerine tracer **çağrı doğrulama** (mock) — collector'a bağımlı değil |
| `eval_results` şema değişimi mevcut satırları bozar | Robust-column ALTER ADD (idempotent, nullable/default); mevcut `outcome`/`pass_at_1` korunur |
| Judge maliyeti production'da artar | `eval_sample_rate` ile örnekleme (Block E); tümü default eval-gate arkasında |
| calibration ekstra judge run = ekstra maliyet | Ayrı gate `eval_calibration_enabled` default off |

## Alternatives Considered

| Yaklaşım | Neden reddedildi |
|----------|------------------|
| Calibration'ı judge rubriğine 5. eksen gömmek | `calibration.py` zaten ayrı, test-edilmiş, DI'lı modül — yeniden kullanmak DRY |
| Yeni `eval_scores` tablosu + migration modülü | Mevcut `eval_results` + robust-column deseni yeterli, daha az invaziv |
| Tam background regresyon job | Faz 6 dashboard'a bağımlı; şimdilik sorgulanabilir `regression_delta` temeli (YAGNI) |

## Open Questions

Yok — kapsam (5 blok) ve gate'ler netleşti.
