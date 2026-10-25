# Design: Faz 5 — Standartlar & Interoperability (A2A Agent Card + MCP gate + Federation robustness)

**Date:** 2026-07-17
**Status:** Implemented (2026-07-17, dal `feat/faz5-interop`)
**Phase:** Faz 5 (opsiyonel büyük bahis) — odaklı kapsam
**Roadmap:** `~/.claude/plans/benim-i-in-ara-t-rma-yapman-tender-panda.md`, Bölüm 4 · Faz 5

## Problem

Faz 5 üç eksen içeriyor; keşif her üçünün de altyapısının **büyük ölçüde mevcut** olduğunu gösterdi:

- **A2A:** Gateway `/health`, `/run`, `/aliases`, OpenAI/Anthropic-compat sunuyor ama node'un kendini/agent'larını tanıttığı **standart bir "Agent Card"** yok — diğer A2A-uyumlu framework'ler bu node'u keşfedemez.
- **MCP:** Server (12 tool) var ama **config gate yok** (her zaman çalışıyor) ve tool'lar Agent Card'da ilan edilmiyor.
- **Federation:** CRDT + HTTP + client + summarizer **hepsi var, default-off**. Güvenle aktive etmek için iki kritik eksik: **delivery garantisi yok** (push transient hatada retry etmiyor) ve **TTL cleanup yok** (inbox/outbox summary'leri süresiz birikiyor).

## Goals

- **Block A:** Standart A2A Agent Card — gateway `GET /.well-known/agent.json` + pure builder + client fetch helper.
- **Block B:** MCP config gate (`mcp_enabled`, default on) + Agent Card MCP tool listesini ilan eder.
- **Block C:** Federation robustness — push retry/backoff (delivery garantisi) + bus TTL cleanup.
- Opt-in/gate default davranışı korunur; Agent Card read-only metadata (auth'suz, `/health` gibi).

## Non-Goals (bilerek ertelenen — gerçek network/infra gerektiriyor)

- **Tam external MCP client** (subprocess JSON-RPC ile başka MCP server'lara bağlanma) → ayrı faz.
- **Background sync daemon** (`sync_interval_seconds` loop) → deployment/Faz 6.
- **Token refresh/rotation** → deployment.
- Cross-instance embedding-based search (tasarım gereği keyword; değişmez).

## Proposed Solution

### Block A — A2A Agent Card

Yeni pure modül `engine/agent_card.py`:
```python
def build_agent_card(*, host: str, version: str, url: str,
                     aliases: list[str], mcp_tools: list[dict] | None = None) -> dict:
    card = {
        "name": f"orchestra@{host}",
        "description": "Orchestra multi-agent orchestration node",
        "url": url,
        "version": version,
        "protocolVersion": "0.2.0",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {"id": a, "name": a, "description": f"Orchestra agent alias '{a}'", "tags": ["agent"]}
            for a in aliases
        ],
    }
    if mcp_tools:
        card["mcpTools"] = [{"name": t["name"], "description": t.get("description", "")} for t in mcp_tools]
    return card
```
Gateway'e handler + route (auth'suz, discovery metadata):
```python
async def _agent_card(request):
    from orchestra.engine.agent_card import build_agent_card
    from orchestra import __version__, config
    mcp_tools = None
    if config.mcp_enabled():
        from orchestra.engine.mcp_registry import REGISTRY
        mcp_tools = REGISTRY.list_tools()
    return web.json_response(build_agent_card(
        host=socket.gethostname(), version=__version__,
        url=str(request.url.with_path("/")), aliases=available_aliases(), mcp_tools=mcp_tools))
# app.router.add_get("/.well-known/agent.json", _agent_card)
```
Client helper `agent_card.fetch_agent_card(base_url, token="") -> dict` (urllib, `remote_node` deseni; GET `/.well-known/agent.json`, Bearer opsiyonel).

### Block B — MCP config gate

`config.py`:
```python
def mcp_enabled() -> bool:
    """Expose Orchestra's MCP tool server + advertise tools in the A2A card (default on)."""
    return bool(get("mcp", "enabled", True))
```
Agent Card, `mcp_enabled()` iken `REGISTRY.list_tools()`'u `mcpTools` olarak ilan eder (Block A'da bağlı). `mcp_server.main()` de gate'i kontrol edebilir (opsiyonel; kapsam: card ilanı yeterli).

### Block C — Federation robustness

**C1 — Delivery garantisi (retry/backoff):** `federation_client.py`:
```python
async def push_with_retry(peer, summaries, *, max_retries=None, sleep_fn=None) -> dict:
    max_retries = max_retries if max_retries is not None else int(config.federation_config().get("max_push_retries", 3))
    sleep_fn = sleep_fn or asyncio.sleep
    attempt = 0
    while True:
        result = await push_to_peer(peer, summaries)
        if "error" not in result:
            return result
        if attempt >= max_retries:
            return result  # give up; caller keeps summaries in outbox (clear_outbox only removes accepted)
        await sleep_fn(min(2 ** attempt, 30))   # bounded exponential backoff
        attempt += 1
```
`sync_with_peer` `push_to_peer` yerine `push_with_retry` çağırır. `sleep_fn` DI → testte gerçek uyku yok. Başarısız summary'ler outbox'ta kalır (clear_outbox yalnız accepted'ı siler) → delivery garantisi.

**C2 — TTL cleanup:** `FederationBus.cleanup_expired(now: float) -> int`:
```python
def cleanup_expired(self, now: float) -> int:
    removed = 0
    for box in (self.outbox, self.inbox):
        for path in box.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if now > float(data.get("timestamp", 0.0)) + float(data.get("ttl_hours", 720)) * 3600:
                    path.unlink(); removed += 1
            except Exception:
                logger.warning("cleanup_expired: bad summary %s", path, exc_info=True)
    return removed
```
`now` parametre → deterministik test (sabit saat).

## Config (yeni accessor — `config.py`)

| Accessor | Anahtar | Default |
|----------|---------|---------|
| `mcp_enabled()` | `mcp.enabled` | `True` |

(`federation.max_push_retries` `federation_config()` dict'inden okunur, default 3 — yeni accessor gerekmez.)

## Değişecek Dosyalar

- `orchestra/engine/agent_card.py` — **YENİ** (builder + fetch helper).
- `orchestra/gateway.py` — `_agent_card` handler + route.
- `orchestra/config.py` — `mcp_enabled()`.
- `orchestra/federation_server/federation_client.py` — `push_with_retry`; `sync_with_peer` onu kullanır.
- `orchestra/engine/federation.py` — `FederationBus.cleanup_expired`.
- Testler: `tests/test_agent_card.py` (A/B), `tests/test_federation_robustness.py` (C) — YENİ.

## Testing (DI ile, gerçek network/sleep yok)

- A: `build_agent_card` pure → şema/skills/mcpTools doğrula. Gateway route: aiohttp `TestClient(TestServer(make_app()))` (mevcut test_gateway deseni) → `/.well-known/agent.json` 200 + card şeması.
- B: `mcp_enabled` default True; card `mcp_enabled=False` iken `mcpTools` içermez (monkeypatch).
- C1: fake `push_to_peer` (önce error, sonra success) + `sleep_fn` stub → `push_with_retry` retry sayısı + nihai başarı; hep-error → max_retries sonrası error döner, `sleep_fn` çağrı sayısı.
- C2: `tmp_path` bus, süresi geçmiş + taze summary yaz → `cleanup_expired(now)` doğru sayıda siler, tazeyi korur.
- Scratch dizininden: `.venv/bin/python -m pytest <dosya> -p no:cacheprovider --no-cov`.

## Risks

| Risk | Mitigation |
|------|------------|
| Agent Card auth'suz — bilgi sızıntısı | Yalnız alias adları + tool adları (zaten `/aliases` auth'lu ama card discovery için açık; hassas veri yok) |
| push retry sonsuz döngü | `max_retries` sınırı + bounded backoff (min(2^n,30)) |
| TTL cleanup taze summary siler | `now` parametreli deterministik eşik; test kapsar |
| mcp_enabled default True davranış değiştirir | Card sadece metadata ekler; MCP server zaten hep çalışıyordu → davranış aynı |

## Alternatives Considered

| Yaklaşım | Neden reddedildi |
|----------|------------------|
| Tam external MCP client | Network-stateful, subprocess; ayrı faza ertelendi (kullanıcı onayı) |
| Agent Card'ı `/aliases`'a gömmek | A2A standardı `/.well-known/agent.json` bekliyor — ayrı endpoint interop için gerekli |
| Background sync daemon | asyncio loop + gerçek infra; temiz unit-test edilemez, deployment fazına ait |

## Open Questions

Yok — kapsam (A2A card + MCP gate + federation retry/TTL) ve ertelenenler netleşti.
