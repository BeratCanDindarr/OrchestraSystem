# Orchestra — Kurulum Kılavuzu

Multi-agent orchestration sistemi. Codex ve Gemini CLI'larını paralel çalıştırır,
artifact store (SQLite + JSONL) ve MCP server içerir.

## Gereksinimler

- Python 3.11+
- `codex` CLI kurulu (`npm install -g @openai/codex`)
- `gemini` CLI kurulu (`npm install -g @google/gemini-cli`)

## Kurulum

```bash
# 1. Bu klasörü projenin köküne kopyala
cp -r orchestra/      /your/project/orchestra/
cp -r .orchestra/     /your/project/.orchestra/

# 2. Bağımlılıkları yükle
pip install -r orchestra/requirements.txt

# 3. .orchestra/config.toml içindeki ayarları düzenle
#    (timeout, max_turns, redaction patterns vb.)

# 4. .mcp.json dosyasını projenin köküne koy,
#    /YOUR/PROJECT/ROOT kısmını gerçek path ile değiştir
```

## Kullanım

```bash
# CLI — tek agent
python3 -m orchestra ask cdx-deep "sorun ne?"

# CLI — paralel (cdx-deep + gmn-pro)
python3 -m orchestra run dual "mimari analiz yap"

# CLI — tam döngü (dual + cross-review)
python3 -m orchestra run critical "karar ver: A mı B mi?"

# Geçmiş
python3 -m orchestra ps
python3 -m orchestra logs <run_id>

# TUI — anlık izleme
python3 -m orchestra watch <run_id>
```

## MCP Olarak Kullanım (Claude Code)

`.mcp.json` dosyasını proje köküne koy, `/YOUR/PROJECT/ROOT` yerine
projenin gerçek absolute path'ini yaz, ardından Claude Code'u yeniden başlat.

MCP araçları: `orchestra_run`, `orchestra_list_runs`, `orchestra_get_logs`,
`orchestra_get_run`, `orchestra_cancel`, `orchestra_continue`

## Modlar

| Mod | Açıklama |
|-----|----------|
| `ask <alias>` | Tek agent, hızlı |
| `dual` | cdx-deep + gmn-pro paralel |
| `critical` | dual + çapraz inceleme (cross-review) |
| `auto` | Router keyword'e göre mod seçer |

## Aliaslar

| Alias | Provider | Model |
|-------|----------|-------|
| `cdx-fast` | Codex | gpt-5.4/low |
| `cdx-deep` | Codex | gpt-5.4/xhigh |
| `gmn-fast` | Gemini | gemini/flash |
| `gmn-pro` | Gemini | gemini/pro |

## Auto-Synthesis

`dual` modunda iki agent bitince otomatik sentez için:

```toml
# .orchestra/config.toml
[orchestra]
auto_synthesize = true
```

veya Python'dan:

```python
from orchestra.engine.runner import run_dual
run = run_dual("analiz et", synthesize=True)
```

## Circuit Breaker

```toml
[limits]
max_turns = 10       # bir run'da maksimum parallel çağrı sayısı
max_budget_usd = 5.0 # gelecekte cost tracking için
```
