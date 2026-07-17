# Design: Federation Background Sync Daemon + Token Refresh (Faz 5 tamamlama)

**Date:** 2026-07-17
**Status:** Implemented (2026-07-17, dal `feat/faz5-federation-daemon`)
**Phase:** Faz 5 son parça — federation'ı uçtan uca aktive et
**Roadmap:** Bölüm 4 · Faz 5 (Federation'ı aktive et)

## Problem

Federation altyapısı (CRDT, HTTP server/client, summarizer, retry, TTL cleanup) tamam ve config-gate'li. Ama iki eksik uçtan uca aktivasyonu engelliyor:

1. **Background sync daemon yok:** `sync_all_peers()` sadece manuel çağrılıyor; `sync_interval_seconds` config'i tanımlı ama **kullanılmıyor**. Periyodik senkron için bir loop gerek.
2. **Token refresh sinyali yok:** `is_token_expired()` var ama süresi dolan token'lar sessizce atlanıyor (operatöre uyarı yok, pre-expiry sinyali yok). Shared-secret federasyonda "refresh" = operatör config'te token'ı döndürür; sistem bunu **fark edip loglamalı** ve bir sonraki tick'te otomatik almalı (`_peer_from_config` her tick config'i yeniden okur).

## Goals

- **Block A:** `sync_tick` (bir tick: sync + TTL cleanup) + `run_sync_loop` (periyodik, DI'lı) daemon. Config `federation.sync_interval_seconds` (default 300) + `federation_sync_enabled()` gate.
- **Block B:** `FederationPeer.needs_refresh(now, margin)` predicate + `peers_needing_refresh` helper; daemon pre-expiry uyarısı loglar. Config `federation.token_refresh_margin_seconds` (default 3600).
- **Block C:** `attach_sync_daemon(app)` → federation HTTP `create_app` `on_startup`/`on_cleanup`'a bağla (gate'liyken otomatik başlar).
- DI-testli (sleep_fn / should_continue / now_fn) → gerçek zamanlayıcı/asyncio-forever yok. Federation default-off → gate kapalıyken hiçbir şey başlamaz.

## Non-Goals

- OAuth-tarzı otomatik token rotation endpoint'i (shared-secret için over-engineering; config-driven rotation yeterli — operatör token'ı config'te değiştirir, `_peer_from_config` alır).
- Embedding-based cross-instance search (tasarım gereği keyword).

## Proposed Solution

### Block A — Daemon (`engine/federation_daemon.py`, YENİ)

```python
async def sync_tick(*, now_fn=time.time) -> dict:
    from orchestra.federation_server.federation_client import sync_all_peers
    from orchestra.engine.federation import FederationBus
    result = await sync_all_peers()
    cleaned = FederationBus().cleanup_expired(now_fn())
    return {"sync": result, "cleaned": cleaned}

async def run_sync_loop(*, interval=None, sleep_fn=None, should_continue=None, now_fn=time.time) -> None:
    import asyncio
    if interval is None:
        interval = int(config.federation_config().get("sync_interval_seconds", 300))
    sleep_fn = sleep_fn or asyncio.sleep
    should_continue = should_continue or (lambda: True)
    while should_continue():
        try:
            await sync_tick(now_fn=now_fn)
            _warn_peers_needing_refresh(now_fn())
        except Exception:
            logger.warning("federation sync tick failed", exc_info=True)
        await sleep_fn(interval)
```

Config (`config.py`):
```python
def federation_sync_enabled() -> bool:
    cfg = config.federation_config()
    return bool(cfg.get("enabled", False)) and int(cfg.get("sync_interval_seconds", 300)) > 0
```

### Block B — Token refresh sinyali

`FederationPeer` (federation.py):
```python
def needs_refresh(self, now: float, margin_seconds: float) -> bool:
    """True when a TTL'd token is within ``margin`` of expiry (operator should rotate)."""
    if self.token_expires_at <= 0.0:
        return False
    return now > self.token_expires_at - margin_seconds
```
Daemon helper `_warn_peers_needing_refresh(now)` → config peer'larını `_peer_from_config` ile kurar, `needs_refresh(now, config.federation_token_refresh_margin_seconds())` olanları WARN loglar. Config accessor `federation_token_refresh_margin_seconds()` default 3600.

### Block C — HTTP app'e bağla

`federation_http.py`:
```python
def attach_sync_daemon(app):
    async def _on_startup(app):
        if config.federation_sync_enabled():
            app["_sync_task"] = asyncio.create_task(run_sync_loop())
    async def _on_cleanup(app):
        task = app.get("_sync_task")
        if task: task.cancel()
    app.on_startup.append(_on_startup); app.on_cleanup.append(_on_cleanup)
    return app
# create_app() sonunda: attach_sync_daemon(app)
```

## Config (yeni accessor'lar — `config.py`)

| Accessor | Anahtar | Default |
|----------|---------|---------|
| `federation_sync_enabled()` | türetilmiş (`federation.enabled` && `sync_interval_seconds`>0) | `False` |
| `federation_token_refresh_margin_seconds()` | `federation.token_refresh_margin_seconds` | `3600.0` |

## Değişecek Dosyalar

- `orchestra/engine/federation_daemon.py` — **YENİ** (sync_tick, run_sync_loop, _warn_peers_needing_refresh).
- `orchestra/engine/federation.py` — `FederationPeer.needs_refresh`.
- `orchestra/config.py` — `federation_sync_enabled`, `federation_token_refresh_margin_seconds`.
- `orchestra/federation_server/federation_http.py` — `attach_sync_daemon` + `create_app` wiring.
- Testler: `tests/test_federation_daemon.py` — YENİ.

## Testing (DI ile — gerçek sleep/asyncio-forever yok)

- `sync_tick`: patch `sync_all_peers` (async) + `FederationBus.cleanup_expired` → dönen `{"sync","cleaned"}` doğru; ikisi de çağrıldı.
- `run_sync_loop`: `should_continue` = 3 tick sonra False sayaç; `sleep_fn` async recorder; `sync_tick` patch → 3 tick çalışır, 3 sleep(interval). Tick exception → loop devam eder (yutulur).
- `needs_refresh`: `token_expires_at=0` → False; margin içinde → True; margin dışında → False (sabit now).
- `federation_sync_enabled`: enabled+interval>0 → True; enabled=false → False; interval=0 → False (monkeypatch federation_config).
- `attach_sync_daemon`: `on_startup`/`on_cleanup` handler'ları kaydeder; `_on_startup` gate kapalıyken task açmaz (patch config + fake app dict).
- Scratch dizininden: `.venv/bin/python -m pytest <dosya> -p no:cacheprovider --no-cov`.

## Risks

| Risk | Mitigation |
|------|------------|
| Daemon sonsuz loop testi asılır | `should_continue` DI + `sleep_fn` stub → deterministik, gerçek uyku yok |
| Tick hatası loop'u öldürür | try/except tick içinde → loop devam eder, loglar |
| Gate kapalıyken daemon başlar | `federation_sync_enabled` (enabled && interval>0), `_on_startup` gate kontrol eder |
| Task cleanup'ta iptal edilmez → sızıntı | `on_cleanup` task.cancel() |

## Alternatives Considered

| Yaklaşım | Neden reddedildi |
|----------|------------------|
| OAuth token rotation endpoint | Shared-secret federasyon için over-engineering; config-driven rotation yeterli |
| `threading.Thread` daemon | aiohttp app zaten asyncio; `create_task` + app lifecycle doğal, test-edilebilir |
| CLI `orchestra federation-serve` komutu | Ayrı iş; wiring `create_app`'e (mevcut `run_app` yolu) yeterli |

## Open Questions

Yok.
