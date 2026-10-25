# Design: Faz 2 — ask/dual Dayanıklı Yürütme (Durable Execution)

## Status
Accepted

## Date
2026-07-17

## Problem

Yol haritası Faz 2'nin ("Güvenilirlik & Dayanıklı Yürütme") kalan bloğu:
**`critical` modun state suspension'ını `ask` ve `dual` modlarına genişletmek.**

Kabul kriteri: *Run ortasında process kill → `orchestra resume` ile kaldığı yerden tamamlanır.*

Bugünkü durum:

- **Policy-based approval gate** (cost/confidence/complexity/mode) zaten dört modda
  (`ask`/`dual`/`critical`/`planned`) mevcut ve `WAITING_APPROVAL` durumunda run'ı
  `paused_runs` tablosuna suspend ediyor (`state_suspension.suspend_run`).
- **`_resume_critical`** (`critical_runner.py`) crash-recovery yapabiliyor: checkpoint
  status `RUNNING` ise son checkpoint'ten devam eder ve DB'deki paused state'i
  (agents/reviews/summary) geri yükler.
- **Ama** `resume_runner.py`'ın checkpoint fallback yolu (satır ~108-112) yalnızca
  `critical` modu `_resume_critical`'e yönlendiriyor. **`ask`/`dual` için crash-sonrası
  devam yolu yok** — run `from_manifest` ile kurulup devam ettirilmeden dönüyor.
- Ayrıca `ask`/`dual`/`critical` runner'ları `install_signal_handlers=False` yapıyor;
  mevcut `parallel.py` SIGINT handler'ı run'ı **iptal** ediyor (process'leri öldürüp
  `CANCELLED` işaretliyor) — suspend/persist etmiyor.

## Goals

- `ask` ve `dual` modlarında **checkpoint-boundary resume**: sert crash sonrası
  `orchestra resume <id>` son tamamlanan checkpoint'ten devam eder (`_resume_critical`
  ile aynı semantik).
- `ask` ve `dual` modlarında **signal-based live suspend**: config açıkken SIGINT/SIGTERM
  yakalanınca mevcut RUNNING state `paused_runs`'a yazılıp temiz çıkılır, sonra resume edilir.
- Opt-in, config-gate **varsayılan kapalı**; gate kapalıyken davranış **hiç değişmez**.
- Gerçek model/DB olmadan test edilebilir (dependency injection).
- Hassas `parallel.py`'ye **dokunmadan** (CLAUDE.md uyarısı: lock altında mutasyon).

## Non-Goals

- `parallel.py`'nin signal handler'ını değiştirmek veya `on_signal` modu eklemek
  (Yaklaşım 2 — reddedildi).
- `planned` modu (zaten kendi checkpoint akışına sahip; bu blok kapsamı dışında).
- Agent çağrısı **sırasında** kaybolan subprocess çıktısını kurtarmak (mümkün değil —
  aşağıdaki Dürüstlük Sınırı'na bakın).
- Yeni bir resume UX/komutu: `orchestra resume` CLI + MCP + web HITL zaten mevcut ve
  yeterli; yalnızca mode-özel devam mantığı eklenir.

## Dürüstlük Sınırı (açık sözleşme)

Agent subprocess'i çalışırken (LLM çağrısı in-flight) process öldürülürse, o agent'ın
kısmi çıktısı **kaybolur** — kurtarılamaz. Bu yüzden hem checkpoint-resume hem
signal-suspend yolları **son tamamlanan checkpoint'ten** devam garantisi verir; bu
`_resume_critical`'ın mevcut davranışıyla birebir aynıdır. Bu sınır spec'te, event
log'da (`durable_suspend` / `durable_resume` event'leri) ve modül docstring'inde açıkça
belirtilir.

## Proposed Solution (Yaklaşım 1)

Yeni saf modül `orchestra/engine/durable_execution.py` iki yeteneği kapsar; mevcut kod
minimal değişir.

### Yeni modül: `orchestra/engine/durable_execution.py`

Dependency injection ile — enjekte edilen fonksiyonlar sayesinde gerçek model/DB
olmadan test edilir.

```python
# Feature A — checkpoint-boundary resume (continuation)
def resume_ask(
    run, checkpoint, *,
    finalize_fn, load_paused_fn, delete_paused_fn, get_db_fn,
    force_continue=False, **kwargs,
) -> OrchestraRun: ...

def resume_dual(
    run, checkpoint, *,
    synthesize_fn, finalize_fn, load_paused_fn, delete_paused_fn, get_db_fn,
    force_continue=False, **kwargs,
) -> OrchestraRun: ...
```

Her ikisi de `_resume_critical` iskeletini yansıtır:

1. Checkpoint label `approval_gate` ve `approval_state == PENDING` ve `force_continue`
   değilse → `WAITING_APPROVAL` döndür (gate'te bekleyen run'a dokunma).
2. `force_continue` veya `approval_state ∈ {APPROVED, RESUMED}` veya
   `checkpoint_status == RUNNING` ise → `RUNNING`'e geç, `approval_state = RESUMED`,
   manifest yaz, `run_resumed` event'i.
3. DB'de paused state varsa (`load_paused_fn`) agents/reviews/summary restore et,
   `delete_paused_fn` ile sil, `paused_state_restored` event'i.
4. Devam: `dual` → `synthesize_fn(run, run.task, **kwargs)`; `ask` → devam yok
   (tek agent, sadece finalize).
5. `finalize_fn(run)` → run döndür.

```python
# Feature B — signal-based live suspend
@contextmanager
def suspend_on_signal(run, *, suspend_fn, get_db_fn, enabled_fn=durable_execution_enabled):
    # enabled_fn() False ise: no-op context (yield; return) — davranış değişmez.
    # True ise: SIGINT + SIGTERM için handler kur.
    #   Handler: get_db_fn() ile bağlan → suspend_fn(conn, run, paused_by="signal")
    #            → run.status = WAITING_APPROVAL, approval_state = RESUMED
    #            → durable_suspend event'i → önceki handler'ı restore et
    #            → sinyali yeniden yükselt (raise KeyboardInterrupt / re-signal).
    # finally: kurulan handler'ları her durumda eski haline getir.
```

`suspend_fn(conn, run, paused_by="signal")` mevcut `state_suspension.suspend_run`'ı
kullanır; RUNNING state'i `paused_runs`'a yazar. Resume tarafında bu row
`resume_run_from_db` ile yüklenir.

### `config.py` — yeni accessor

```python
def durable_execution_enabled() -> bool:
    """Persist run state on SIGINT/SIGTERM and enable ask/dual crash-recovery
    resume from the last checkpoint (default off)."""
    return bool(get("durable_execution", "enabled", False))
```

`.orchestra/config.toml` şablonuna yorumlu, varsayılan-kapalı bir `[durable_execution]`
bloğu eklenir.

### Mevcut kod entegrasyonu (minimal)

- **`resume_runner.py`** (checkpoint fallback yolu, ~satır 108-112): mevcut
  `if run.mode == "critical": return _resume_critical(...)` yanına:
  ```python
  if run.mode == "dual":
      return resume_dual(run, checkpoint, synthesize_fn=_synthesize_if_possible,
                         finalize_fn=_finalize, ...)
  if run.mode == "ask":
      return resume_ask(run, checkpoint, finalize_fn=_finalize, ...)
  ```
  Ayrıca DB yolu (`resume_run_from_db`) `WAITING_APPROVAL` guard'ı (satır ~45): sinyal
  ile suspend edilmiş run RUNNING durumda olabileceğinden guard `{WAITING_APPROVAL}`
  yerine `{WAITING_APPROVAL}` + signal-suspend işaretli RUNNING'i de kabul edecek. Bunu
  temiz tutmak için `suspend_on_signal` handler'ı suspend ederken status'u
  `WAITING_APPROVAL`'a alır (yukarıda), böylece guard değişmeden çalışır. **Karar:**
  guard'ı değiştirmiyoruz; suspend handler status'u `WAITING_APPROVAL` yapar.

- **`dual_runner.py`**: `run_dual` ana `try` bloğunu `with suspend_on_signal(run, ...):`
  ile sarmala.
- **`common.py:_run_ask_impl`**: aynı şekilde ana `try` bloğunu sarmala.

Gate kapalıyken `suspend_on_signal` no-op olduğundan iki runner'ın mevcut davranışı
birebir korunur.

## Architecture / Data Flow

```
Graceful kill (SIGTERM/SIGINT, gate açık):
  run_dual/_run_ask_impl  --with suspend_on_signal-->  sinyal yakalanır
     -> suspend_fn(conn, run, "signal")  [paused_runs'a RUNNING state, status=WAITING_APPROVAL]
     -> durable_suspend event -> temiz çıkış
  orchestra resume <id>
     -> resume_run_from_db  [WAITING_APPROVAL yükler]
     -> resume_dual/resume_ask  -> synthesize (dual) -> finalize -> paused sil

Sert crash (paused_runs kaydı yok):
  orchestra resume <id>
     -> resume_run_from_db  = None (kayıt yok)
     -> manifest + latest_checkpoint fallback
     -> mode==dual -> resume_dual ; mode==ask -> resume_ask
     -> checkpoint_status==RUNNING'den devam -> synthesize (dual) -> finalize
```

## Error Handling

- `load_paused_fn` / DB erişim hataları: `_resume_critical`'daki gibi `logger.warning`
  ile yut, devam et (paused state olmadan checkpoint'ten devam).
- `suspend_on_signal`'ın handler'ı içindeki hatalar (DB kilidi vb.): logla, yine de
  eski handler'ı restore edip sinyali yeniden yükselt (suspend başarısız olsa bile
  process asılı kalmaz).
- `finally` bloğu kurulan tüm signal handler'ları eski haline getirir (context manager
  sözleşmesi).

## Testing

`tests/test_durable_execution.py` — enjekte fonksiyonlar + in-memory sqlite, gerçek
model/DB yok:

- `resume_dual` / `resume_ask`: checkpoint_status=RUNNING'den devam → `synthesize_fn`
  (dual) çağrıldı, `finalize_fn` çağrıldı, paused kaydı silindi.
- `resume_*`: `approval_gate` + PENDING + `force_continue=False` → WAITING_APPROVAL
  döner, devam etmez.
- `resume_*`: DB'de paused state varsa agents/reviews/summary restore edildi.
- `suspend_on_signal`: `enabled_fn=lambda: False` → no-op (handler kurulmaz,
  `signal.getsignal` değişmez).
- `suspend_on_signal`: `enabled_fn=lambda: True` → handler kurulur; simüle edilen
  sinyalde `suspend_fn` çağrılır, status WAITING_APPROVAL olur, önceki handler restore
  edilir.

Regresyon: mevcut `ask`/`dual` testleri gate kapalıyken değişmeden geçmeli.
Test komutu (scratch dizininden): `.venv/bin/python -m pytest tests/test_durable_execution.py -p no:cacheprovider --no-cov`.

## Alternatives Considered

| Yaklaşım | Pros | Cons | Neden reddedildi |
|----------|------|------|------------------|
| 2: Yerinde genişletme + `parallel.py`'yi `on_signal` ile parametrele | Daha lokal, tek noktada signal mantığı | Hassas `parallel.py`'ye dokunur (lock altında mutasyon, CLAUDE.md uyarısı); `_resume_*` runner'lara dağılır | Risk/ödül dengesi kötü |
| 3: Sadece checkpoint-boundary resume | En basit, en düşük risk | Signal-suspend yok; kabul kriterinin "graceful kill" kısmı eksik | Kullanıcı "ikisi birden" istedi |

## Consequences

### Positive
- `ask`/`dual` crash-recovery kazanır; idle maliyeti düşer (suspend/resume).
- `parallel.py` dokunulmaz; risk izole.
- Repo'nun yerleşik kalite desenine (saf modül + DI + gate default-off) uyar.

### Negative
- İki runner'a `with` sarmalayıcı eklenir (küçük diff).
- Yeni config bölümü + accessor bakım yükü.

### Neutral
- Gate varsayılan kapalı: kullanıcı açıkça etkinleştirene kadar hiçbir davranış değişmez.
