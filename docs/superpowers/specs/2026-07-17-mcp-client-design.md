# Design: External MCP Client (Orchestra as MCP consumer)

**Date:** 2026-07-17
**Status:** Implemented (2026-07-17, dal `feat/faz5-mcp-client`; dogfood-verified against own mcp_server)
**Phase:** Faz 5 kalan — external MCP client (daha önce ertelenmişti; kullanıcı onayıyla alındı)
**Roadmap:** Bölüm 4 · Faz 5 (MCP tam entegrasyon)

## Problem

Orchestra bugün MCP **SERVER** (12 tool expose, `mcp_server.py`). Ama MCP **CLIENT** yok — başka bir MCP server'a bağlanıp onun araçlarını `tools/list` / `tools/call` ile kullanamıyor. Bu, MCP'nin (2026 de-facto standart, 97M+ aylık indirme) yarısı. Tam interop için Orchestra'nın external MCP server'ları **tüketebilmesi** gerekiyor.

Protokol zaten codebase'de mevcut (`mcp_server.py`): **newline-delimited JSON-RPC over stdio** (`json.dumps(msg)+"\n"`, `readline()`), metotlar `initialize` / `tools/list` / `tools/call`, protocolVersion `2024-11-05`.

## Goals

- `engine/mcp_client.py`: external MCP server'a bağlanan JSON-RPC client — `initialize`, `list_tools`, `call_tool`.
- **Transport-injected** tasarım: çekirdek (request/response id-korelasyonu, hata) gerçek subprocess olmadan test edilebilir.
- Gerçek subprocess için `StdioTransport` (Popen, newline-delimited JSON) — mevcut server framing'iyle birebir uyumlu.
- Config'den external server tanımı (`[mcp.client_servers]`) + factory.

## Non-Goals

- Fastlane'e external-MCP dispatch entegrasyonu (araçları Orchestra'nın tool registry'sine graftlamak) → ayrı, daha büyük iş; bu spec sadece client kütüphanesi + factory.
- HTTP/SSE MCP transport (sadece stdio; MCP'nin en yaygın yerel transport'u).
- Async client (senkron yeterli; provider subprocess deseni de senkron).

## Proposed Solution

### Transport abstraction (DI için)

```python
class Transport(Protocol):
    def send(self, obj: dict) -> None: ...
    def recv(self) -> dict | None: ...   # None = bağlantı kapandı
    def close(self) -> None: ...
```

`StdioTransport(command: list[str])`: `subprocess.Popen(command, stdin=PIPE, stdout=PIPE, text=True)`; `send` → `stdin.write(json.dumps(obj)+"\n"); flush`; `recv` → `line=stdout.readline(); None if not line else json.loads(line)`; `close` → terminate.

### McpClient (transport-injected)

```python
class McpClient:
    def __init__(self, transport): self._t = transport; self._id = 0
    def _request(self, method, params=None) -> dict:
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None: req["params"] = params
        self._t.send(req)
        while True:
            msg = self._t.recv()
            if msg is None:
                raise McpClientError("connection closed")
            if msg.get("id") == req["id"]:          # ignore notifications / other ids
                if "error" in msg:
                    raise McpClientError(msg["error"])
                return msg.get("result", {})
    def initialize(self) -> dict:
        return self._request("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "orchestra", "version": __version__}})
    def list_tools(self) -> list[dict]:
        return self._request("tools/list").get("tools", [])
    def call_tool(self, name, arguments=None) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments or {}})
    def close(self): self._t.close()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    @classmethod
    def from_command(cls, command: list[str]) -> "McpClient":
        t = StdioTransport(command); t.start(); return cls(t)
```

`McpClientError(Exception)`. Config factory: `connect_from_config(name)` reads `config.mcp_client_servers()[name]` (a command list) → `from_command`.

### Config

```python
def mcp_client_servers() -> dict:
    """Map of name -> command list for external MCP servers to consume."""
    return section("mcp").get("client_servers", {}) or {}
```

## Değişecek Dosyalar

- `orchestra/engine/mcp_client.py` — **YENİ** (Transport, StdioTransport, McpClient, McpClientError, from_command, connect_from_config).
- `orchestra/config.py` — `mcp_client_servers()`.
- Testler: `tests/test_mcp_client.py` — YENİ.

## Testing (DI ile, gerçek subprocess yok)

- **Core (FakeTransport):** scripted `recv` sırası → `initialize`/`list_tools`/`call_tool` doğru sonuç; gönderilen request'ler doğru method/params/id.
- **Id-korelasyon:** araya notification (id yok) / farklı id gelirse atlanır, doğru id yakalanır.
- **Error:** response'ta `error` → `McpClientError` raise.
- **Bağlantı kapanma:** `recv` None → `McpClientError`.
- **StdioTransport:** `subprocess.Popen`'ı mock'la → `send` doğru satır yazar (`json+"\n"`), `recv` satır parse eder, boş satır → None.
- **Dogfood entegrasyon (opsiyonel, işaretli):** `from_command([sys.executable,"-m","orchestra.mcp_server"])` → initialize + list_tools 12 tool döner. Yavaş/ortam-bağımlı olabilir → ayrı işaretli test.
- Scratch dizininden: `.venv/bin/python -m pytest <dosya> -p no:cacheprovider --no-cov`.

## Risks

| Risk | Mitigation |
|------|------------|
| `recv` blocking → deadlock | Yalnızca request sonrası recv; StdioTransport readline satır-tabanlı (server her yanıtı `\n` ile biter) |
| Subprocess sızıntısı | `close()` + context manager; `from_command` kullananlar `with` ile |
| Kötü niyetli external server | Yalnız trusted/config-tanımlı server'lar; komut config'den (kod değil) |

## Alternatives Considered

| Yaklaşım | Neden reddedildi |
|----------|------------------|
| Resmi `mcp` SDK'sını bağımlılık olarak ekle | Ekstra ağır bağımlılık; protokol basit (newline JSON-RPC), mevcut server deseniyle birebir; kendi client'ımız DI-test edilebilir |
| Async client | Senkron yeterli, provider deseniyle tutarlı; async karmaşıklık gereksiz |
| Fastlane'e tam graft | Daha büyük iş; bu spec client kütüphanesine odaklı (Non-Goal) |

## Open Questions

Yok.
