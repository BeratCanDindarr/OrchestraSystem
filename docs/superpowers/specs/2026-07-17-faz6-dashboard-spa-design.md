# Design: Faz 6 — Dashboard'ı Canlı Etkileşimli Araca Çevir (HTMX + SSE + aiohttp)

**Date:** 2026-07-17
**Status:** Implemented (2026-07-17, dal `feat/faz6-dashboard-spa`; end-to-end smoke-verified)
**Phase:** Faz 6 (R&D) — tüm özellikler
**Roadmap:** Bölüm 4 · Faz 6 (dashboard'ı dinamik görsel araca çevir)

## Problem

Mevcut `monitoring/web_dashboard.py` **senkron `http.server`** üstünde statik inline HTML + tek-seferlik client fetch. Canlı güncelleme, run drill-down, etkileşimli HITL, gözlemlenebilirlik panelleri yok. Roadmap bunu **gerçek zamanlı, etkileşimli** bir araca dönüştürmek istiyor.

## Goals (tüm Faz 6 özellikleri)

- **Canlı akış:** polling yerine **SSE** ile run/agent durumu anlık güncelleme.
- **Modern frontend:** **HTMX** + sunucu-render HTML partial'lar + inline SVG sparkline grafikleri (maliyet/latency/failure trend, provider dağılımı).
- **Run drill-down:** bir run'a tıklayınca agent'lar, event timeline (events.db `EventLog.replay`), OTel span hiyerarşisi, synthesis özeti.
- **Etkileşimli HITL:** bekleyen onaylar için canlı approve/reject kartları.
- **Gözlemlenebilirlik panelleri:** incident drill-down, token/cost attribution, canlı budget göstergesi, eval skorları (composite + calibration).
- Backend **tamamen pytest-testli** (aiohttp TestClient + saf render fn'ler). Mevcut `compute_*` fn'ler ve testleri korunur.

## Non-Goals

- npm/build toolchain (HTMX vendored/CDN, build yok).
- Auth (dashboard trusted-network; `dashboard_approvals_enabled` gate zaten var).
- Eski `http.server` dashboard'u silme (bırakılır; CLI yeni app'e yönlendirilir).
- Gerçek OTel collector zorunluluğu (spans yoksa trace paneli boş).

## Proposed Solution — mimari

**Katmanlar (many small files):**
1. `monitoring/dashboard_render.py` — **saf** `render_*(data) -> str` HTML partial üreticileri (metrics kartları, runs tablosu, run detail, event timeline, HITL kartları, budget, sparkline SVG). Unit-testable (string içeriği).
2. `monitoring/dashboard_app.py` — **aiohttp** app: route'lar (JSON değil, HTML partial döner — HTMX swap), SSE endpoint, static asset servisi. Mevcut `web_dashboard.compute_*` + `service` + `EventLog` + `tracer`'dan okur.
3. `monitoring/dashboard/` — static: `index.html` (shell: htmx + sse ext yükler, panel `hx-get`'leri + `hx-ext="sse" sse-connect="/api/stream"`), `style.css`, opsiyonel vendored `htmx.min.js`.
4. CLI `orchestra dashboard` → yeni aiohttp app'i çalıştırır.

**HTMX deseni:** panel elemanları `hx-get="/api/metrics"` ile yüklenir (partial swap); canlı akış için `sse-swap="run_update"` elemanları SSE'den gelen HTML fragment'i alır. Approve/reject `hx-post="/api/approve"` → güncellenmiş kart partial'i döner.

### SSE canlı akış (`/api/stream`)

Event bus (`events/bus.py`) handler'ları **thread-pool'da** çalışır → asyncio SSE loop'una köprü:
```python
async def _stream(request):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    def handler(run_id, event):   # runs in bus thread
        loop.call_soon_threadsafe(queue.put_nowait, (run_id, event))
    bus.subscribe_all(handler)
    try:
        while True:
            run_id, event = await queue.get()
            frag = render_run_update(run_id, event)   # HTML fragment
            await resp.write(sse_pack("run_update", frag))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        bus.unsubscribe_all(handler)
    return resp
```
`sse_pack(event, data)` → `f"event: {event}\ndata: {data}\n\n".encode()` (çok satırlı data tek satıra düzleştirilir). Test: fake bus emit → queue → fragment; `sse_pack` format.

### Run drill-down

- `GET /api/runs` → runs tablosu partial (satır tıklanabilir `hx-get="/api/runs/{id}"`).
- `GET /api/runs/{id}` → detail partial: agents tablosu + summary + timeline'a `hx-get`.
- `GET /api/runs/{id}/events` → `EventLog.replay(run_id)` seq-sıralı timeline partial.
- `GET /api/runs/{id}/trace` → OTel in-memory spans (varsa) hiyerarşi partial; yoksa "trace yok".

### HITL kartları

- `GET /api/paused` → `list_paused_runs()` her biri approve/reject butonlu kart (`hx-post`).
- `POST /api/approve` (gate `dashboard_approvals_enabled`) → `apply_approval` + güncellenmiş kart/kuyruk partial. Gate kapalıyken 403 partial.

### Gözlemlenebilirlik panelleri

- `GET /api/metrics` → mevcut `compute_metrics` → kartlar + provider/mode dağılımı sparkline.
- `GET /api/incidents` → `count_incidents` → incident tipleri + drill-down.
- `GET /api/eval` → `compute_eval_summary` → pass-rate + composite/calibration.
- `GET /api/budget` → **YENİ** `compute_budget(conn)` → toplam cost, run-başı attribution, config budget'a göre kalan; canlı gösterge.

## Config (yeni accessor — `config.py`)

| Accessor | Anahtar | Default |
|----------|---------|---------|
| `dashboard_budget_usd()` | `dashboard.budget_usd` | `0.0` (0 = limitsiz) |

(`dashboard_approvals_enabled` zaten var.)

## Değişecek/Yeni Dosyalar

- `orchestra/monitoring/dashboard_render.py` — **YENİ** (saf render fn'ler + `sse_pack`).
- `orchestra/monitoring/dashboard_app.py` — **YENİ** (aiohttp app, route'lar, SSE, static).
- `orchestra/monitoring/dashboard/index.html`, `style.css` — **YENİ** (HTMX shell).
- `orchestra/monitoring/web_dashboard.py` — `compute_budget` eklenir (compute fn'ler burada toplu).
- `orchestra/config.py` — `dashboard_budget_usd()`.
- `orchestra/cli.py` — `dashboard` komutu yeni app'i çalıştırır (eski fallback).
- Testler: `tests/test_dashboard_render.py`, `tests/test_dashboard_app.py`, `tests/test_dashboard_sse.py` — YENİ.

## Testing (DI ile, gerçek tarayıcı/collector yok)

- **Render fn'ler (saf):** her `render_*` beklenen HTML içerir (run_id, `hx-get`, sse attr, sparkline `<svg>`).
- **aiohttp app (TestClient):** her endpoint 200 + doğru partial; `/` shell htmx + sse-connect içerir; static asset servis edilir; approve gate kapalı→403.
- **SSE:** fake bus handler → asyncio queue → fragment; `sse_pack` format; abonelik başlar/biter (subscribe/unsubscribe çağrıldı).
- **compute_budget:** in-memory conn → toplam/kalan doğru.
- Mevcut `compute_*` testleri değişmeden yeşil kalır.
- Scratch dizininden: `.venv/bin/python -m pytest <dosya> -p no:cacheprovider --no-cov`.

## Risks

| Risk | Mitigation |
|------|------------|
| Bus handler thread → asyncio queue race | `loop.call_soon_threadsafe` + bounded queue; test fake-emit |
| SSE bağlantısı sızıntısı | `finally: unsubscribe_all`; StreamResponse ConnectionReset yakalar |
| HTMX CDN air-gapped ağda çalışmaz | `/static/htmx.min.js` route; vendored kopya varsa onu servis eder, yoksa CDN fallback (operatör vendorlar) |
| Eski http.server testleri kırılır | web_dashboard.py compute fn'lere dokunulmaz; yeni app ayrı modül |
| OTel spans production'da yok | trace paneli "yok" gösterir; gate'siz best-effort |

## Alternatives Considered

| Yaklaşım | Neden reddedildi |
|----------|------------------|
| Vanilla JS tek dosya | Kullanıcı HTMX seçti; HTMX daha az client JS, server-render partial testli |
| React/Svelte bundle | npm/build yok, pytest'te test edilemez, repoya artifact |
| Eski http.server'a SSE ekle | Senkron; uzun-ömürlü SSE + async bus için aiohttp doğru (gateway deseniyle hizalı) |
| JSON API + client render | HTMX server-render partial → daha az client kod, backend'de test edilir |

## Open Questions

Yok — teknoloji (HTMX+SSE+aiohttp) ve kapsam (5 özellik ekseni) netleşti.
