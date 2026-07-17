# Design: Typed Memory (episodic/semantic/procedural) + Recency Retrieval

**Date:** 2026-07-17
**Status:** Draft
**Phase:** Faz 3 (Hafıza & RAG Entegrasyonu) — kalan iş
**Roadmap:** `~/.claude/plans/benim-i-in-ara-t-rma-yapman-tender-panda.md`, Bölüm 4 · Faz 3

## Problem

`memory.db` bugün **tek bir vektör tablosu** (`embeddings`) kullanıyor
(`orchestra/storage/memory.py:63-128`). Reflection dersleri, tamamlanan run
özetleri, elle eklenen bilgiler ve proje dokümanları hepsi aynı indekste karışıyor.
2026 agentic hafıza en iyi pratiği bunu bir sorun olarak işaret ediyor: **epizodik
loglar (bir kez yaşanmış olaylar) semantik indekse (zamansız bilgi) karışınca
retrieval bozuluyor** — bir görev için "ilgili genel bilgi" ararken eski çalıştırma
logları sonuçları kirletiyor.

Ayrıca retrieval skoru sadece `cosine + pinned_boost + citation_boost` — **recency
(yenilik) hiç kullanılmıyor** (sadece TTL/expiry mutlak silme için). Epizodik hafıza
için yenilik birinci-sınıf sinyal olmalı: son deneyim, üç hafta öncekinden değerli.

## Goals

- `memory.db` vektör deposunu **üç fiziksel ayrı tipe** böl: `episodic`, `semantic`,
  `procedural` — her biri kendi tablosu ve kendi HNSW indeks dosyası.
- Retrieval skoruna **üstel yarı-ömür recency decay** ekle (birinci-sınıf sinyal).
- Tümü **opt-in, config-gate default-off**. Gate kapalıyken davranış **birebir aynı**
  (sıfır regresyon) — mevcut `feat/agentic-core` kalite deseni.
- Kod tekrarı YOK (coding-style: no duplication) — üç depo tek kod yolunu paylaşır.

## Non-Goals

- Mevcut `embeddings` verisinin migration'ı YOK — o tablo olduğu gibi `semantic` sayılır.
- `procedural` tipi için henüz **yazan yok** (rezerve; ileride toolmaker / öğrenilen
  prosedürler). Sadece altyapı hazır olur.
- RAG'de tip başına sabit kota YOK — skor birleşimi karışımı belirler.
- `relations` ve `memory_events` tablolarına dokunulmaz (kesişen graf/denetim).
- `pin`/`unpin`/`delete`/`increment_citation` semantiğinde köklü değişiklik yok
  (tip-farkında hâle getirilir ama davranış korunur).

## Proposed Solution

### 1. `SemanticMemory` parametrikleştirme (behavior-preserving refactor)

`SemanticMemory` bugün `embeddings` tablosunu ve `memory_hnsw.idx` indeks yolunu
sabit (hardcoded) kullanıyor. Constructor'a iki parametre eklenir, **varsayılanları
bugünkü değerler**:

```python
class SemanticMemory:
    def __init__(self, db_path=..., *,
                 embeddings_table: str = "embeddings",
                 index_path: str | None = None,   # None -> mevcut default
                 now_fn=None):                     # recency testi için clock DI
```

Tüm SQL ifadeleri tablo adını parametreden okur. `now_fn` enjekte edilebilir saat
(recency yaşı hesaplamak ve testte deterministik olmak için — `sleep_fn` desenini
yansıtır; default `datetime.now`).

Bu adım **davranışı değiştirmez**; ayrı bir `refactor:` commit olarak yapılır ve
tüm mevcut testlerin yeşil kalmasıyla doğrulanır.

### 2. `storage/typed_memory.py` — yeni facade

Aynı `memory.db` dosyası üstünde **üç `SemanticMemory` örneği** tutan ince bir
yönlendirme katmanı:

| Tip | Tablo | HNSW indeks |
|-----|-------|-------------|
| `semantic` | `embeddings` (MEVCUT — migration yok) | `memory_hnsw.idx` |
| `episodic` | `embeddings_episodic` | `memory_hnsw_episodic.idx` |
| `procedural` | `embeddings_procedural` | `memory_hnsw_procedural.idx` |

`relations` ve `memory_events` tabloları tekil kalır — her örnek `CREATE TABLE IF
NOT EXISTS` çalıştırır (zararsız), denetim/graf paylaşılır. Yalnızca vektör tablosu
üçe bölünür.

`TypedMemory`, `SemanticMemory` ile **aynı arayüzü** sunar (add/search/pin/unpin/
delete/increment_citation/cleanup_expired) + bir `memory_type` parametresi:

- `add(content, ..., memory_type="semantic")` → doğru depoya route eder.
- `search(query, limit, memory_type=None)`:
  - Tip verilirse → sadece o deponun `search`'ü.
  - `None` → üç depoda ara, hepsi zaten skorlanmış+sıralı (recency dahil) döner;
    birleşik `similarity` skoruna göre **flat merge**, top-`limit` al. Dedup gerekmez
    (tablolar ayrık). Tip başına kota YOK.
- `pin/unpin/delete/increment_citation(item_id, ..., memory_type="semantic")` → o
  depoya route (ID'ler artık tablo başına).

### 3. `get_memory()` gate

```python
def get_memory():
    if config.memory_typed_enabled():        # default False
        return TypedMemory(...)              # facade
    return SemanticMemory(...)                # bugünkü tekil davranış
```

Düz `SemanticMemory.add/search` **`memory_type` parametresini kabul edip yok sayar**
(backwards-compat shim) — böylece çağıranlar (rag.py, reflection.py, common.py,
service.py) hiç dallanmaz, her zaman `memory_type` geçer. Gate kapalıyken tüm yazılar
tek `embeddings` tablosuna gider = bugünkü davranış.

### 4. Recency skoru (search içinde, tüm örneklere ortak)

```
score = raw_cosine
      + PINNED_BOOST            (pinned ise, mevcut)
      + citation_boost          (mevcut, cap 0.10)
      + recency_boost           (YENİ)

recency_boost = recency_weight * 0.5 ** (age_hours / half_life_hours)
                # gerçek yarı-ömür: boost her half_life_hours'ta yarılanır
                # sadece config.memory_recency_enabled() ise; değilse 0
age_hours = (now_fn() - created_at).total_seconds() / 3600
```

Kapalıyken `recency_boost = 0` → skor bugünküyle **birebir**.

### 5. Writer → tip eşlemesi (karar A)

| Yazan | Dosya | `memory_type` |
|-------|-------|---------------|
| Reflection dersi | `engine/reflection.py:_record_lesson` | `episodic` |
| Run finalize özeti | `engine/runners/common.py` (finalize indexing) | `episodic` |
| MCP `memory_add` | `service.py:memory_add_entry` | `semantic` (çağıran override edebilir) |
| Proje md indexleme | `service.py:memory_index_project_entries` | `semantic` |
| Procedural | — | (yazan yok; ileri faz) |

### 6. RAG kapsamı (karar A)

`engine/rag.py:retrieve_context` → `memory.search(prompt, limit, memory_type=None)`
(hepsinden ara, skorla merge). Gate kapalıyken düz memory `memory_type=None`'ı yok
sayıp tek tabloyu arar = bugünkü davranış.

## Config (yeni accessor'lar — `config.py`)

| Accessor | Anahtar | Default |
|----------|---------|---------|
| `memory_typed_enabled()` | `memory.typed_enabled` | `False` |
| `memory_recency_enabled()` | `memory.recency_enabled` | `False` |
| `memory_recency_half_life_hours()` | `memory.recency_half_life_hours` | `168.0` (1 hafta) |
| `memory_recency_weight()` | `memory.recency_weight` | `0.1` |

## Değişecek Dosyalar

- `orchestra/storage/memory.py` — `SemanticMemory` parametrikleştirme (table/index/
  now_fn), recency boost, `memory_type` shim (add/search).
- `orchestra/storage/typed_memory.py` — **YENİ** facade.
- `orchestra/storage/__init__.py` (veya `get_memory()` nerede tanımlıysa) — gate.
- `orchestra/config.py` — 4 accessor + varsayılanlar.
- `orchestra/engine/reflection.py` — `memory_type="episodic"`.
- `orchestra/engine/runners/common.py` — finalize indexing `memory_type="episodic"`.
- `orchestra/service.py` — iki writer'a `memory_type="semantic"`.
- `orchestra/engine/rag.py` — `memory_type=None` (açık).
- `tests/test_typed_memory.py` — **YENİ**.
- `tests/test_memory.py`, `tests/test_rag_unit.py` — genişlet.

## Testing (DI ile, gerçek model/DB olmadan — tekrarlanan kalite deseni)

- **Routing**: episodic add → sadece `embeddings_episodic` tablosunda; diğerleri boş.
- **Merged search**: üç depoya farklı içerik, `memory_type=None` skorla birleşik top-N.
- **Recency matematiği**: sabit `now_fn` clock ile `exp(-age/half_life)` beklenen boost;
  yaşlı vs yeni giriş sıralaması.
- **Gate-off parity**: `typed_enabled=false` → düz `SemanticMemory`, bugünkü testlerle
  birebir; `memory_type` param yok sayılıyor (shim).
- **Recency gate-off**: `recency_enabled=false` → boost 0, skor değişmez.
- `TypedMemory` inject edilebilir depo örnekleri alır → fake store ile hızlı test.
- Testler scratch dizininden: `.venv/bin/python -m pytest <dosya> -p no:cacheprovider --no-cov`.

## Risks

| Risk | Mitigation |
|------|------------|
| `SemanticMemory` parametrikleştirme mevcut çağıranları kırar | Varsayılanlar = bugünkü değerler; ayrı `refactor:` commit; tüm testler yeşil kalır |
| Üç HNSW indeks dosyası — disk + rebuild/cleanup ×3 | Mantık tek kod yolu, üç örnek; disk maliyeti opt-in (gate kapalıyken tek indeks) |
| `get_memory()` singleton cache toggle'ı yok sayar | Gate okuması cache invalidation ile; testte config override |
| Aynı DB'ye üç örnekten yazma (`database is locked`) | Mevcut per-op connection + WAL + busy_timeout deseni korunur |

## Alternatives Considered

| Yaklaşım | Neden reddedildi |
|----------|------------------|
| Tek tablo + `memory_type` kolonu + filtre | Daha az invaziv ama fiziksel izolasyon yok; kullanıcı plana sadık fiziksel ayrımı seçti (retrieval kirlenmesini kesin engeller) |
| RAG'de tip başına sabit kota | Config karmaşık, az fayda; skor birleşimi karışımı zaten belirliyor |
| Run finalize → semantic | Run özetleri "yaşanmış olay" = episodic; semantic kutuyu temiz tutmak ayrımın amacı |

## Open Questions

Yok — tüm tasarım kararları netleşti (şema mimarisi, recency modeli, writer→tip,
RAG kapsamı).
