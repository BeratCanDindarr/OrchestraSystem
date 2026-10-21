# Plan: Orchestra geliştirme yol haritası (2026-11)

## Status
Draft. Replaces `archive/roadmap-2026-10-engine.md`; its open items are carried over in §5.

## Goal
Orchestra'yı Claude Code için güvenilir bir "doğrulama altyapısı" yapmak: önce kanıtlanmış hataları
kapatmak, sonra yapıyı sadeleştirmek, en son rakiplerin önde olduğu ama Orchestra'nın farkını
(doğrulanmış inceleme, çapraz aile ikinci görüş) büyüten özellikleri eklemek.

## Nasıl çıkarıldı

Orchestra'nın kendisiyle, 2026-10-20'de yapılan beş run (toplam ≈ $3.1):

| Run | Ne | Ajanlar | Sonuç |
|---|---|---|---|
| `9f60e641` | Kod değerlendirmesi: artı / eksi / geliştirme (dual) | cld-deep + kimi-deep, doğrulayıcı oll-gemma | needs_attention (sıralama farkı), $1.20 |
| `3065be0e` | Rakip araştırması, web | kimi-deep | 9 araç, kaynaklı, $0.008 |
| `9f683eba` | Rakip araştırması, web, Ekim 2026 durumu doğrulanarak | agy-argon (Gemini 4 Argon) | 13 araç, sayfalar açılarak, $0.60 |
| `279890a6` | Yol haritası önerisinin tartışması (consensus) | cld-deep (lehte) + kimi-deep (aleyhte) + oll-coder (tarafsız) | passed, $0.44 |
| `a4422c76` | İlk Gemini denemesi | agy-argon → Opus'a düştü | 180 s zaman aşımı hatasını ortaya çıkardı (düzeltildi: `93d6989`) |

Değerlendirmedeki en kritik dört iddia ayrıca kodda elle doğrulandı (A1, A2, A3, A5).

---

## 1. Güçlü yönler (korunacak)

| # | Ne | Kanıt |
|---|---|---|
| G1 | **Bulgusu diff'te olmayan inceleme atılır.** Bir bulgu, alıntıladığı satır diff'te yoksa düşer. Araştırılan 13 aracın hiçbirinde yok. | `engine/code_review.py` |
| G2 | **Farklı model ailelerinden iki görüş + doğrulayıcı + sentez**, anlaşmazlığı ortalamadan korur; consensus modunda atanmış lehte/aleyhte/tarafsız roller. | `engine/runners/common.py`, `engine/stances.py` |
| G3 | **Dürüst yedek etiketi.** Bir alias cevap veremezse cevabı veren model adıyla bildirilir (`fallbacks`). | `providers/fallback.py`, `engine/mcp_registry.py` |
| G4 | **İptal sırası doğru.** İptal diske yazılır, sonra süreçler öldürülür; takılı kalan run'lar reaper ile kapanır; iş alma tek atomik `UPDATE`. | `service.py`, `storage/jobs.py:83-93` |
| G5 | **Bağlam ekonomisi.** MCP yanıtları kırpılmış (list_runs 68 KB → 2 KB), ajanlara dosya yapıştırılmıyor, Ollama ajanları `--bare`. | `engine/mcp_registry.py`, `providers/ollama_agent.py` |
| G6 | **Ölçülerek yönlendirme.** Hangi modelin hangi işi yapacağı benchmark'la belirleniyor. | `tools/model_bench/`, `docs/model-routing.md` |
| G7 | **Mevcut CLI girişleriyle çalışır**, API anahtarı gerekmez; 5 model ailesi, Ollama dahil hepsi ajan. | `providers/` |
| G8 | **Olay kaynaklı geçmiş**: replay, diff, timeline, forget, follow-up zinciri, çağrı başına maliyet. | `storage/event_log.py`, `engine/follow_up.py`, `engine/costing.py` |

## 2. Zayıf yönler (etkiye göre)

| # | Ne | Kanıt | Doğrulandı |
|---|---|---|---|
| Z1 | **MCP sunucusu kendi olay döngüsünü kilitliyor.** Async işleyici senkron `REGISTRY.invoke` çağırıyor; uzun bir run sürerken başka hiçbir araç çağrısı, **iptal dahil**, cevap alamıyor. `wait_for_run` döngüde `time.sleep` ile bekliyor. | `mcp_fastmcp.py:276`, `service.py` `wait_for_run` | ✔ elle |
| Z2 | **Senkron/async ajan koşucuları ikiz ve ayrışmış.** ~230 satırlık iki kopya; senkron olanda bütçe denetimi kilit içinde SQLite sorgusu yapıyor ve bir ajanın istisnası diğerlerinin sonucunu kaybettiriyor. MCP yolu, daha az test edilen senkron ikizi kullanıyor. | `engine/parallel/__init__.py:86-558` | ajanlar |
| Z3 | **Doğrulayıcı çökerse run çöküyor.** Çağrı korumasız; doğrulayıcı modeli kapalı/erişilemezse iyi bir run bile düşer. | `common.py:1058` | ✔ elle |
| Z4 | **Cevap önbelleği proje bilmiyor.** Anahtar = alias + prompt; proje kökü ve git HEAD yok, önbellek tüm projeler için ortak (`~/.orchestra/fts_cache.db`). Dosyalar değiştikten sonra 7 güne kadar eski cevap dönebilir. Önbellekten gelen cevaba güven 1.0 yazılıyor (kendi kuralına aykırı). `FtsSearchCache.search` içindeki BM25 karşılaştırması ters: en kötü eşleşmeleri tutuyor (şu an kullanılmıyor). | `storage/fts_cache.py:123`, `:253-279`, `common.py:1282` | ✔ elle (anahtar, 1.0) |
| Z5 | **Planlı run'da birleşme noktası tüm girdileri beklemiyor.** İki girdili bir düğüm, ilk girdi bitince başlıyor. | `engine/graph.py:95-106` | ✔ elle |
| Z6 | **`submit_run` run_id döndürmüyor.** İş arka planda başlayana kadar izlenemiyor ve iptal edilemiyor. | `service.py` `submit_run` | ajanlar |
| Z7 | **Config, import anında CWD'den bulunuyor.** Kullanıcı düzeyinde tek bir MCP sunucusu, bir projenin ayarını her projeye uygular. | `config.py:34, 96-97` | CLAUDE.md |
| Z8 | **CLI, service katmanını atlıyor.** `run auto` yönlendirmeyi kendisi yeniden yazıyor (outcome/cascade yok), `eval` ~210 satır orkestrasyon içeriyor; `cli.py` ~1.900 satır. | `cli.py:405-421` | ajanlar |
| Z9 | **4 ayrı SQLite dosyası, her işlemde yeni bağlantı.** Her olay ekleme bağlantı açıp kapatıyor; `memory.py` bağlantı sızdırıyor; manifest yazımı atomik değil. | `storage/event_log.py:144-227`, `storage/memory.py` | ajanlar |
| Z10 | **Yedek zinciri yetenek bilmiyor.** Salt okunur bir alias, izin bayrağı vermeyen bir ajana düşebiliyor. | `fallback.py:137-186`, `kimi.py:89` | açık soru |
| Z11 | **En sıcak yollarda kapsama en düşük:** `engine/parallel` %37, `service.py` %44, `cli.py` %32; 113 mypy hatası. | `docs/quality-audit.md` §7 | rapor |
| Z12 | **`orchestra undo` hiçbir zaman snapshot alınmadığı için hiçbir şey geri alamıyor.** | `engine/backtrack.py` | CLAUDE.md |
| Z13 | **İki fiyat kaynağı.** `_estimate_run_cost` kendi alias→fiyat tablosunu tutuyor, `costing.py` dışında. | `common.py:467-475` | ajanlar |

## 3. Rakiplerle karşılaştırma (Ekim 2026)

Kaynaklar: Kimi (`3065be0e`) ve Gemini (`9f683eba`); çeliştikleri yerde sayfayı açarak doğrulayan
Gemini esas alındı.

| Araç | Ne yapıyor | Lisans / durum |
|---|---|---|
| **PAL MCP** (ex zen-mcp) | Claude Code'dan diğer modellere/CLI'lara MCP köprüsü; `consensus`, `codereview`, `precommit`, `debug`, `clink` | Apache-2.0; **son commit 2025-12-15, durgun** (Kimi "çok aktif" dedi, Gemini sayfadan yalanladı) |
| **Claude Code subagents + Agent Teams** | Yerleşik paralel Claude ajanları, worktree, ortak görev listesi ve posta kutusu | Ticari; aktif (v2.1.304, 2026-10-19); Teams deneysel |
| **claude-squad** | tmux + worktree ile paralel ajan oturumları, TUI | AGPL-3.0; orta (2026-08) |
| **Conductor** | Mac uygulaması; paralel Claude/Codex/Cursor, worktree veya bulut microVM, PR görünümü | Ticari; aktif (2026-10-15) |
| **vibe-kanban** | Kanban kartı başına worktree'de ajan, satır içi diff yorumu | Apache-2.0; şirket kapandı (2026-04), topluluk sürdürüyor |
| **Sculptor (Imbue)** | Docker konteynerlerinde paralel ajanlar, Pairing Mode, CI Babysitter | MIT; aktif |
| **OpenHands** | Agent Canvas: birçok ajanı yerel/Docker/VM/bulutta çalıştıran kontrol merkezi, ACP | MIT; aktif |
| **Aider** | Tek oturum; AST repo haritası, architect/editor, her turda commit + undo | Apache-2.0; **durgun** (son sürüm 2025-08) |
| **Codex Cloud** | Bulut sandbox'larda paralel görev, PR incelemesi | Ticari; aktif |
| **ask-llm** | Giriş yapılmış CLI'ları (agy, ollama dahil) MCP ile ikinci görüş için bağlar; birden çok host'a kurulum sihirbazı | MIT; aktif |
| **Symphony (OpenAI)** *(2026)* | Linear panosunu izleyip izole uygulama run'ları başlatır, "proof-of-work" paketi | Apache-2.0 |
| **Orca (Stably)** *(2026)* | Bir prompt'u 5 worktree'ye dağıtıp kazananı birleştirir; mobil yönetim | MIT; çok aktif |
| **Composio Agent Orchestrator** *(2026)* | CI ve PR yorumlarını ajana geri besleyen kapalı döngü | Apache-2.0; aktif |

### Özellik matrisi

✓ var · ~ kısmen · – yok

| Özellik | Orchestra | PAL | CC Teams | claude-squad | Conductor | Sculptor | OpenHands | ask-llm | Orca |
|---|---|---|---|---|---|---|---|---|---|
| Birden çok sağlayıcının CLI ajanını sürer | ✓ (5 aile) | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Claude Code'a MCP olarak bağlanır | ✓ | ✓ | yerleşik | – | – | – | – | ✓ | – |
| İki ajan + doğrulayıcı + sentez | ✓ | ~ | – | – | – | – | – | – | – |
| Atanmış rollerle consensus | ✓ | ✓ | – | – | – | – | – | – | – |
| **Bulguları diff'e karşı doğrulanan inceleme** | ✓ | – | – | – | – | – | – | – | – |
| Ölçülerek yönlendirme | ✓ | ~ | – | – | – | – | – | – | – |
| Maliyet takibi | ✓ | – | ~ | – | – | – | ~ | – | – |
| Olay kaynaklı geçmiş / replay | ✓ | – | – | – | ~ | – | ✓ | – | – |
| Worktree / sandbox izolasyonu | ~ (worktree, kapalı) | – | ✓ | ✓ | ✓ (+microVM) | ✓ (Docker) | ✓ | – | ✓ |
| **Paralel yazan ajanlar / best-of-N** | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ |
| Diff inceleme yüzeyi | – | – | – | TUI | ✓ | ✓ | ✓ | – | ✓ |
| PR / CI entegrasyonu | – | – | ✓ (Actions) | – | ✓ | ✓ (CI) | ✓ | – | ~ |
| Çalışan ajana bağlanıp yön verme | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ |
| Birden çok host'a MCP kurulumu | – | ✓ | – | – | – | – | – | ✓ | – |
| Çalışan `undo` | – (snapshot yok) | – | ✓ | – | – | ✓ | – | – | – |

### Orchestra'nın önde olduğu yerler
- Diff'e karşı doğrulanmış inceleme bulguları: araştırılan hiçbir araçta yok.
- Farklı model ailelerinden iki görüşü bir doğrulayıcıdan geçirip anlaşmazlığı koruyan sentez, atanmış rollerle consensus.
- Ölçülerek yönlendirme ve dürüst yedek etiketi.
- Maliyet + olay kaynaklı geçmiş birlikte.
- **En yakın rakip PAL MCP durgunlaştı** (2025-12'den beri commit yok): "Claude Code'dan çok modelli doğrulama" alanı açık.

### Geride kaldığı yerler (masa standardı)
- Paralel yazan ajan / best-of-N yok (Orca, Conductor, Sculptor, CC Teams).
- Diff inceleme ve hunk bazında kabul yok.
- PR / CI döngüsü yok.
- Çalışan ajanın durumu yalnızca statusline'da; "dikkat gerektiriyor" durumu ve ilerleme bildirimi yok.
- Çalışan `undo` yok.

---

## 4. Karar verilen yol haritası

Consensus run `279890a6` (lehte Opus, aleyhte Kimi, tarafsız MiniMax; doğrulayıcı onayladı) öneriyi
**değişikliklerle kabul etti.** Sıra: **A → D0 → B → C1 → C3 → C4 → C2 (koşullu) → D1.**
Her madde için kabul ölçütü, düzeltmeden önce kırılan ve sonra geçen bir test.

### Phase A: Güvenilirlik (hepsi S, kanıtlanmış hatalar, hemen)

- [x] **A1: MCP çağrıları ayrı thread'de.** `_call_tool`, `REGISTRY.invoke`'u `anyio.to_thread.run_sync` ile çağırsın; `wait_for_run` olay döngüsünü uyutmasın.
  - Dosyalar: `orchestra/mcp_fastmcp.py:276`, `orchestra/service.py` (`wait_for_run`)
  - **Kabul:** 30 sn süren bir run devam ederken ikinci bir MCP çağrısı (`cancel_run` dahil) 1 sn içinde döner.
- [x] **A2: Doğrulayıcı hatası run'ı düşürmesin.** Doğrulayıcı istisna atarsa run `needs_attention` ile biter, aday cevaplar korunur.
  - Dosya: `engine/runners/common.py:1058`
  - **Kabul:** Doğrulayıcı istisna atınca run FAILED olmaz; aday cevaplar özette durur.
- [x] **A3: Önbellek proje ve sürüm bilsin.** Anahtar = proje kökü + git HEAD + alias + prompt; önbellekten gelen cevap, kaydedilmiş güven değerini taşısın; ters çalışan `FtsSearchCache.search` silinsin. *(Yapıldı: kapsam kök + HEAD + `git diff HEAD` hash'i; commit'lenmemiş düzenleme de yeni cevap ister.)*
  - Dosyalar: `storage/fts_cache.py:123`, `:253-279`, `engine/runners/common.py:1282`
  - **Kabul:** Aynı prompt iki ayrı repoda iki ayrı ıska; bir isabetin güveni kaydedilenle aynı.
- [x] **A4: `submit_run` hemen run_id döndürsün.**
  - Dosyalar: `service.py` (`submit_run`, `execute_job`), `storage/jobs.py`
  - **Kabul:** İlk ajan başlamadan verilen iptal, run'ı CANCELLED yapar.
- [x] **A5: Birleşme düğümü tüm girdileri beklesin.**
  - Dosyalar: `engine/graph.py:95-106`, `engine/runners/planned_runner.py`
  - **Kabul:** Elmas şeklindeki bir grafikte birleşme düğümü iki dal bittikten sonra bir kez çalışır; girdilerden biri başarısız olduğu durum da test edilir.
- [x] **A6 (tartışmada eklendi): Bir istisna run'ı sonsuza kadar RUNNING bırakmasın** (try/finally + reaper). *(Kod zaten vardı: üç koşucuda `_fail_crashed`, `reap_stale_runs`; eksik olan kabul testi eklendi.)*
  - **Kabul:** Run ortasında atılan istisnadan sonra run, son durumlardan birine ulaşır.
- [x] **A7 (tartışmada eklendi): Replay deposu budansın.** 48 saatten eski satırlar silinsin; arama tüm tabloyu taramasın.
  - **Kabul:** 48 saatten eski satır kalmaz; arama süresi satır sayısından bağımsız.

**Acceptance criteria (faz):** yukarıdaki yedi test geçer, hızlı paket yeşil, Claude Code'da uzun bir dual sürerken `orchestra_cancel_run` hemen cevap verir.

### Phase D0: Önce kapsama (yapı değişikliğinden önce)
- **Depends on**: Phase A
- [x] `engine/parallel` kapsamı ≥ %55-60, `service.py` ≥ %60. *(Ölçüm: `parallel/__init__` %39 → %63, `budget` %38 → %95, `service` %70; toplam %72 → %74.)*
- [x] Hiçbir şey doğrulamayan testleri düzelt ya da sil. *(14 güçlendirildi, 8 tekrar silindi; "hata atmamalı" smoke testleri bırakıldı.)*

Kimi'nin aleyhte argümanı burada kabul edildi: kapsaması düşük eşzamanlılık kodunu kapsamayı
yükseltmeden yeniden yapılandırmak güvenlik sırasını tersine çevirir. Test paketi
`orchestra.engine.parallel.<isim>`'i paket düzeyinde patch'lediği için, saf taşımayla bölme sınırlı.

**Acceptance criteria:** CI kapsama ölçümü yukarıdaki eşikleri, B başlamadan önce gösterir.

### Phase B: Yapı (M)
- **Depends on**: Phase D0
- [x] **B1: Tek async ajan koşucu çekirdeği**; senkron yol bunun üstünde ince bir sarmalayıcı (`asyncio.run`). Önce test patch hedeflerini taşı.
  - **Kabul:** İkizler kalkar, hızlı paket yeşil, mypy hata sayısı artmaz.
- [x] **B2: CLI yalnızca `service.py` üzerinden çalışsın.** *(Yolda bulunan: MCP `auto`, yönlendirici "planned" deyince hata veriyordu; artık planlı run olarak çalışıyor.)* `run auto` ve `eval` iş mantığını service'e taşı.
  - **Kabul:** Bir test, `cli.py`'nin `engine.runners`'ı import etmediğini doğrular.
- [x] **B3: Tek `storage.connect()` fabrikası** (WAL, busy_timeout, kapanan bağlantılar) ve **atomik manifest yazımı** (geçici dosya + rename).
  - **Kabul:** 20 yazıcılı bir stres testinde `database is locked` hatası çıkmaz.
- [~] **B4: Config proje başına ve tembel çözümlensin**, import anında değil. *(Kesildi, 2026-10-21: bunun çözdüğü durum, birden çok projeye hizmet eden tek süreç (gateway, daemon, dashboard), 2026-10-20'de silindi. Her MCP sunucusu ve iş worker'ı kendi projesinin klasöründe başlıyor; kodda `chdir` yok.)*
  - **Kabul:** Aynı süreçte iki proje kökü kendi config'ini okur.
- [x] **B5: Yetenek etiketli yedek zincirleri.** *(Açık soru kapandı: Kimi okuyucu profiliyle yazamaz; Codex artık `-s read-only`; bu projede Antigravity yazabilir ve salt okunur zincirlerden çıkarıldı.)* Salt okunur bir alias asla yazabilen bir ajana düşmez.
  - Önce açık soruyu kapat: Kimi'nin print modu işlemleri kendiliğinden onaylıyor mu?
  - **Kabul:** Özellik testi: hiçbir salt okunur alias'ın zincirinde yazan alias yok.
- [x] Ek: fiyatı tek kaynaktan al (`_estimate_run_cost` → `costing.py`). *(Eski tablo Kimi/Ollama/Antigravity'yi gpt-5.4/medium fiyatıyla sayıyordu; onay kapısının maliyet eşiği yanlış sayıya bakıyordu.)*

**Acceptance criteria:** beş madde ayrı commit, her biri sonrası hızlı paket yeşil.

### Phase C: Ürün (farkı büyüten özellikler)
- **Depends on**: Phase B (C1 ve C3, B3/B4 bitince başlayabilir)
- [x] **C1: Pre-commit / PR inceleme kapısı** (`orchestra hook install`). Commit öncesi `review` staged değişikliklere bakar, doğrulanmış yüksek önem dereceli bulguda commit'i durdurur. GitHub'a satır içi yorum seçenekli ve varsayılan kapalı.
  - Model: PAL `precommit`, Codex `@codex review`.
  - **Kabul:** Yalnızca alıntısı diff'te olan bulgular gönderilir; çıkış kodu commit'i engeller.
- [ ] **C3: MCP ilerleme bildirimleri + "dikkat" durumu.** Statusline "bekliyor / onay bekliyor / başarısız" durumunu gösterir. Polling azalır.
  - **Kabul:** Durum değişikliği 2 sn içinde statusline'da görünür.
- [ ] **C4: `orchestra diff <run_id>`: hunk bazında kabul**, GUI olmadan, mevcut `changes.patch` üzerine.
  - **Kabul:** Reddedilen hunk'lar run klasöründe kalır; kabul edilenler uygulanır.
- [ ] **C2 (koşullu): Best-of-N yazıcılar.** N worktree, her yama diff'e karşı doğrulanan incelemeden geçer, toplu iş başına bütçe sınırı. Kazanan bekleyen `changes.patch` olarak kalır ve **asla kendiliğinden uygulanmaz**.
  - Model: Orca, Codex cloud.
  - **Koşul:** C1 gerçekten kullanılınca.
  - **Kabul:** N yama, N inceleme, tek bekleyen yama; bütçe aşılınca toplu iş durur.

**Acceptance criteria:** her özellik kendi testiyle ve README / skill güncellemesiyle birlikte gelir.

### Phase D1: İnanılırlık
- [ ] `model_bench` sonuçlarını yönlendirme liderlik tablosu olarak yayımla: tarih, sürümler, yeniden üretme komutu.
- **Acceptance criteria:** `docs/` altında tek sayfa; komutu çalıştıran aynı tabloyu alır.

### Kesilenler (tartışmayla)
- **C5: Konteyner sandbox.** Toolmaker sandbox'ı 2026-10-13'te iyileştirilmedi, kaldırıldı; worktree gerçek tehdit modelini karşılıyor.
- **C6: OpenRouter / API sağlayıcı.** "Her alias projeyi okuyabilen bir ajandır" kuralı bugün uygulandı; geri eklemek yeni kapatılan maliyet/kullanım/yedek etiketi hatalarını geri açar. Dar istisna: bir sağlayıcı `ollama_agent.py` desenindeki gibi mevcut CLI sözleşmesinin arkasına sarılabiliyorsa, o yeni bir sağlayıcı değil, bir config girdisidir.
- **GUI.** Yalnızca CLI (`orchestra diff`) ve statusline.

### Rekabet mi entegrasyon mu
**Entegrasyon.** Orchestra zaten Claude Code'un içinden MCP ile kullanılıyor. Farkı (doğrulanmış
bulgular, dürüst yedek etiketi, ölçülerek yönlendirme) Agent Teams'i ve PAL'ı tamamlıyor, onların
yerini almıyor. Claude Code takım primitiflerini dışarı açarsa Orchestra onları kullanır, kopyalamaz.

### Kararı tersine çevirecek kanıt
1. Orchestra'nın Claude Code dışında kullanıldığını ya da paralel yazıcıların şimdi istendiğini gösteren kullanım verisi → C2 öne gelir, C6 yeniden değerlendirilir.
2. Worktree izolasyonunu aşan gösterilmiş bir tehdit → C5 geri gelir.
3. `engine/parallel` kapsaması zaten %60'ın üstündeyse → D0, B ile birleşir.

---

## 5. Değerlendirilecekler

Gemini'nin web araştırmasından çıkan ve eski plandan taşınan, consensus'a sunulmamış adaylar.
Phase C sonrası ayrı bir consensus ile karara bağlanmalı:

| Aday | Kaynak | Not |
|---|---|---|
| **Çalışan `undo`**: run öncesi git snapshot (stash ref) | Aider | Z12'yi kapatır; C4 ile doğal eşleşir. Küçük ve güvenlik artıran bir iş; C4'le birlikte ele alınabilir |
| **Worktree içinde test/lint döngüsü**: yama dönmeden `pytest`/`ruff` çalıştır, hatayı yazara geri besle | Sculptor CI Babysitter, Composio | C2'nin doğal parçası |
| **Dual/consensus'ta tek turluk çapraz sorgu** (rebuttal), sentezden önce | CC Agent Teams | Maliyeti ölçülmeli: bir tur = 2 ek çağrı |
| **Birden çok host'a MCP kurulumu** (`orchestra setup`: Codex, Gemini CLI, Cursor) | ask-llm | Kullanım şu an yalnızca Claude Code; talep olursa |
| **AST sembol haritası** (dosya yapıştırmak yerine sıralı sembol listesi) | Aider repo map | Ajanlar zaten okuyabiliyor; token kazancı ölçülmeden eklenmemeli |
| `ProviderResult`: sağlayıcılar `tuple[str, int]` + `last_usage()` yerine tek sonuç nesnesi döndürsün | eski plan 3.1b | B1 ile birlikte yapılması doğal |
| Tek eval modülü (`eval_harness` + `eval_judge`) | eski plan 5.2b | B2 ile birlikte |
| Follow-up'ta CLI oturumunu sürdürme (`claude --resume`) | eski plan 6.3 | Geçmişi her turda yeniden göndermemeyi sağlar: token kazancı ölçülmeli |

Eski plandaki 6.1 (MCP client) geçersiz: `engine/mcp_client.py` 2026-10-20'de silindi.

## Dependencies
- Internal: B1, D0'a bağlı. C2, B1 (tek koşucu) ve C1'e bağlı. C3, A1'e bağlı (bildirim için olay döngüsü serbest olmalı).
- External: `anyio` (A1 için, `mcp` paketiyle zaten geliyor). C1 için GitHub kısmı `gh` CLI ya da token ister (isteğe bağlı).

## Risks
- **B1 test patch hedeflerini kırar** → önce D0, sonra patch hedeflerini taşı, sonra birleştir.
- **A3 önbellek anahtarı eski girdileri geçersiz kılar** → kabul edilebilir: önbellek 7 günlük ve yeniden dolar.
- **C1 yanlış pozitifle commit'i durdurursa güven kaybolur** → yalnızca doğrulanmış ve yüksek önemli bulgularda durdur; `--no-verify` her zaman çalışır.
- **C2 maliyeti N katına çıkarır** → toplu iş başına zorunlu bütçe sınırı ve varsayılan N=2.
- **Araştırma bilgisi eskir** (rakiplerin durumu Ekim 2026) → Phase C öncesi rakip tablosunu yeniden doğrula.

## Progress Log
| Date | Phase | Notes |
|------|-------|-------|
| 2026-10-20 | - | Orchestra ile değerlendirildi, araştırıldı ve tartışıldı (run 9f60e641, 3065be0e, 9f683eba, 279890a6); A1-A3, A5 elle doğrulandı. Araştırma sırasında iki hata düzeltildi: `fc832b1` (senkron run'da yedek bilgisi), `93d6989` (Antigravity zaman aşımı) |
