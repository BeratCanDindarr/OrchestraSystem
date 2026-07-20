# Design: Model Fleet as Characters + Task Dispatch

**Date:** 2026-07-20
**Status:** Implemented (branch `fix/review-blockers`, on top of Faz 6 dashboard)
**Origin:** Kullanıcı isteği — dashboard'da hangi modelin çalıştığını/boşta/mevcut olduğunu insan-formunda karakterler olarak gör + oradan görev ata.

## Problem

Faz 6 dashboard run'ları/metrikleri gösteriyor ama **model filosunu** göstermiyor. Kullanıcı modelleri **birer karakter/personel** olarak görmek (müsait/çalışıyor/boşta/çevrimdışı) ve **oradan görev atamak** istiyor.

## Solution — 4 saf modül (Faz 6 HTMX+SSE deseni üstüne)

- **`monitoring/characters.py`** — `ALIAS_TABLE` → `Character` kadrosu. Persona otomatik türetilir (isim havuzu, sıralı-pozisyonla benzersiz; rol = provider archetype + tier), `[characters.<alias>]` config ile override edilebilir. Deterministik geometrik **avatar SVG** (bağımlılıksız). `dispatch_task(character_id, task, dispatch_fn=None)` → arka plan thread'de `run_ask`, DI'lı.
- **`monitoring/presence.py`** — `PresenceService` event bus'ı dinler (`agent_started/completed/failed`, hepsi `alias` taşır), in-flight sayacı + son-bitiş zamanı tutar → `state_for(alias, available, now)` = **busy / available / idle / offline**. `now` parametreli (deterministik test), thread-safe.
- **`dashboard_render.py`** — `render_character_card(char, state, dispatch_enabled)` + `render_fleet(items, dispatch_enabled)`. Persona metni escape'li; avatar bizim güvenli SVG'imiz. Dispatch formu yalnız gate açıkken.
- **`dashboard_app.py`** — `GET /api/fleet` (kadro + presence snapshot → render), `POST /api/dispatch` (gate'li: `dashboard_dispatch_enabled` default off → 403; açıkken `dispatch_task` + yenilenmiş filo). Shell'de `#fleet` paneli `hx-trigger="load, every 3s"` ile canlı.

## Presence durum makinesi

```
not is_available()              → OFFLINE (gri, soluk)
agent_started (in-flight > 0)   → BUSY    (mavi, nabız animasyonu)
in-flight 0 + son iş yakın      → AVAILABLE (yeşil)
in-flight 0 + iş yok/eski       → IDLE    (sarı)
```

## Güvenlik

- `POST /api/dispatch` **default-off gate** (`dashboard.dispatch_enabled`) — görev atama kod çalıştırır (RCE), bilinçli opt-in.
- Tüm persona metni `html.escape` (XSS); avatar üretilen güvenli SVG.

## Config

| Accessor | Anahtar | Default |
|----------|---------|---------|
| `dashboard_dispatch_enabled()` | `dashboard.dispatch_enabled` | `False` |
| `[characters.<alias>]` | name / role override | (auto) |

## Testing

`test_characters.py` (roster/persona/avatar/dispatch), `test_presence.py` (durum makinesi), `test_dashboard_render.py` (kart/filo/XSS), `test_dashboard_app.py` (fleet/dispatch endpoint + gate). Uçtan uca smoke: 26 kart, `agent_started` → busy, dispatch gate off→403 / on→200 + yenilenen filo.

## Eklenen ileri özellikler (2026-07-20)

- **Auto-cast** — `characters.auto_cast(task, roster)`: önce learned routing (`router.semantic.
  suggest_best_alias`), yoksa tier heuristiği (`classify_to_tier`); yalnız müsait/idle karakterlerden,
  insan-okunur gerekçeyle. UI'da "Auto-cast ✨" kutusu → `POST /api/dispatch` (character_id yok).
- **Karakter sohbeti** — `characters.chat_with(id, message, chat_fn=)` (senkron, DI); `GET /api/character/{id}`
  (persona + sohbet paneli) + gated `POST /api/character/{id}/chat` → balon (you/them). Kart "Chat" butonu.
- **Uzak `alias@node` karakterler** — `characters.remote_characters(fetch_card_fn=)`: her remote node'un
  A2A Agent Card'ını (`agent_card.fetch_agent_card`) çekip skill=alias'ları `alias@node` karakter yapar;
  best-effort (erişilemeyen node atlanır). Gate `fleet.include_remote_nodes` (default off). **Not:** bu,
  review'da dead-code işaretlenen `fetch_agent_card`'a gerçek çağıran kazandırdı.

## Deferred

- Sürükle-bırak atama (frontend-only), filoyu HTMX SSE ile push (şu an 3s polling), uzak karakterlerin
  canlı presence'ı (remote node event stream), sohbet streaming.
