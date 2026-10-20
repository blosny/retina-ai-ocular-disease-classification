# RETINA-AI: Derin Ogrenme Sorumlusu (Kisi 4) Teslim Notu

> **Durum (20 Ekim 2026):** v1 bolmesinde (veri_bolme.csv) kopya sizintisi ve cozunurluk kisayolu bulundu; v1 sonuclari (%91.2) gecersizdir. `veri_bolme_v2.csv` Kisi 1 tarafindan dogrulanip onaylandi (PR #8, #9). Model bu bolmeyle egitildi: **test dogrulugu %80.7** (171 goruntu), **5-katli capraz dogrulama %80.6 ± 3.0**. Kenar ve cozunurluk testleri v2 modeliyle tekrarlandi: v1'deki cozunurluk kisayolu yok.

---

### Ne yaptim:
1. `src/cnn_models.py`: EfficientNet-B0 transfer learning, egitim dongusu, erken durdurma, tahmin, Grad-CAM, hasta bazli k-katli capraz dogrulama (`capraz_dogrulama`).
2. `notebooks/04_derin_ogrenme_cnn.ipynb`: Colab'da egitim, test, Grad-CAM, kenar testi, cozunurluk testi ve capraz dogrulama.
3. Arayuz (Kisi 5) icin 3 fonksiyon: `model_yukle()`, `tahmin(goruntu)`, `gradcam(goruntu, sinif=None)`.
4. Test tahminlerini ekipte anlasilan formatta kaydeden fonksiyon: `reports/cnn_test_tahminleri_v2.csv`.
5. v1 modelinde kaynak/sizinti analizi (Kisi 5'in Grad-CAM sorusu uzerine) ve sonucunda `veri_bolme_v2.csv` taslagi (`src/veri_bolme_v2.py`). Kisi 1 `verify_splits.py` ile dogruladi ve onayladi (PR #8, #9).

---

### Neden yaptim:
- **Transfer learning:** Sinif basina birkac yuz egitim gorseliyle sifirdan CNN egitmek yetersiz kalir. ImageNet'te ogrenilmis kenar/doku filtreleri retinaya uyarlanir.
- **EfficientNet-B0:** Az parametreli (~5.3M), Colab'in ucretsiz GPU'sunda egitilebilir; gorev dagiliminda da bu model belirlendi.
- **Renkli (RGB) girdi:** `on_isle()` tek kanalli yesil + CLAHE dondurur. Kataraktin pusu ve DR'deki sari eksudalar renkte daha belirgin oldugu icin CNN'e kirpilmis renkli retina verildi. Kirpma ve boyutlandirma `on_isle()` ile ayni (`roi_kirp` + 512x512), boylece Grad-CAM haritasi on islenmis goruntuyle ayni koordinatlarda. Yesil + CLAHE ile denemek icin `girdi_modu="yesil_clahe"`.
- **Veri artirma:** Yatay cevirme + ±15° dondurme. Bulaniklastirma ve parlaklik/kontrast oynatma YOK, cunku bulaniklik kataraktin ayirt edici ipucu.
- **Erken durdurma:** Dogrulama kaybi iyilesmeyi birakinca durup asiri ogrenmeyi (overfitting) onler.
- **Grad-CAM, kenar ve cozunurluk testleri:** Modelin gercek bulgulara mi (optik disk, lezyon, bulaniklik) yoksa kaynak izlerine mi (kenar, cozunurluk, kamera) baktigini olcmek icin.
- **Capraz dogrulama:** v2 test seti sinif basina ~43 goruntu; tek bolmenin sonucu yaklasik ±6 puan oynayabilir. 5 katin ortalamasi ve standart sapmasi daha guvenilir.

---

### Hangi parametreleri kullandim:
- Model: EfficientNet-B0 (`IMAGENET1K_V1`), son katman Dropout(0.3) + Linear(1280 → 4)
- Girdi: 384x384 RGB, ImageNet normalizasyonu
- Optimizasyon: AdamW, lr=3e-4, weight_decay=1e-4, ReduceLROnPlateau (factor=0.5, patience=2)
- Not: Hiperparametreler ayarlanmadi; fine-tuning icin yaygin baslangic degerleri kullanildi.
- Kayip: CrossEntropy, label_smoothing=0.05
- Batch: 16, en fazla 30 epoch, erken durdurma sabri: 5, seed=42
- Karisik hassasiyet (AMP) GPU'da acik
- Grad-CAM hedef katmani: `model.features[-1]`
- Bolme: `data/splits/veri_bolme_v2.csv` (512x512, sinif basina 288, hasta bazli %70/15/15, seed=42)
- Capraz dogrulama: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grup = hasta; kat i test, kat i+1 dogrulama, kalan uc kat egitim.

---

### Arayuz sozlesmesi (Kisi 5 icin):
```python
from cnn_models import model_yukle, tahmin, gradcam

model = model_yukle()    # varsayilan: models/model_efficientnet_v2.pth
p = tahmin(bgr)          # {"normal": .., "diabetic_retinopathy": .., "glaucoma": .., "cataract": ..}
h = gradcam(bgr)         # 512x512, float32, 0-1 (tahmin edilen sinif icin)
h = gradcam(bgr, sinif="glaucoma")  # istenen sinif icin
```
- Girdi `on_isle()` ile ayni: dosya yolu veya BGR numpy dizisi. **`on_isle()` ciktisi degil.**
- Isi haritasi `boyutlandir(roi_kirp(bgr))` ile ayni koordinatlarda.
- **Degisiklik:** Varsayilan model dosyasi artik `models/model_efficientnet_v2.pth`. Eski `model_efficientnet.pth` (v1) kullanilmamali.

---

### Cikti/gorsel nerede:
- Kod: `src/cnn_models.py`, `src/veri_bolme_v2.py`, `notebooks/04_derin_ogrenme_cnn.ipynb`
- Bolme: `data/splits/veri_bolme_v2.csv` (eski `veri_bolme.csv` karsilastirma icin duruyor)
- Model (GitHub'a yuklenmez, ortak Drive): `models/model_efficientnet_v2.pth`
- Ortak Drive `reports/`: `cnn_test_tahminleri_v2.csv`, `cnn_egitim_gecmisi_v2.json`, `figures/cnn_egitim_grafigi_v2.png`, `figures/gradcam_ornekleri_v2.png`
- Kenar, cozunurluk ve capraz dogrulama: `cnn_kenar_testi_v2.csv`, `cnn_kenar_testi_ozet_v2.csv`, `cnn_cozunurluk_testi_v2.csv`, `cnn_capraz_dogrulama_v2.csv`, `cnn_capraz_tahminleri_v2.csv`
- v1 ciktilari (gecersiz, sadece karsilastirma; repoda `reports/` altinda ekleri olmadan): `cnn_test_tahminleri.csv`, `cnn_egitim_gecmisi.json`, `cnn_kenar_testi*.csv`, `figures/cnn_egitim_grafigi.png`, `figures/gradcam_ornekleri.png`, `model_efficientnet.pth`

---

### Sonuclar (v2, gecerli):

**Egitim** (`cnn_egitim_grafigi_v2.png`)
- 808 egitim / 173 dogrulama goruntusu. 12 epoch'ta erken durdurma (epoch basina ~12 sn).
- Ayni seed ile ikinci kez egitildiginde tum epoch degerleri ve test sonucu birebir ayni cikti (tekrarlanabilir).
- En iyi model: **epoch 7**, dogrulama kaybi 0.598, dogrulama dogrulugu %82.7.
- Egitim dogrulugu %99'a cikarken dogrulama %80 civarinda kaldi: veri azaldigi icin asiri ogrenme v1'e gore daha belirgin.

**Test seti** (171 goruntu; model egitimde hic gormedi)

| Sinif | Test goruntusu | v2 dogruluk | v1 (gecersiz) |
|---|---|---|---|
| Katarakt | 42 | %92.9 | %98.0 |
| Diyabetik retinopati | 43 | %90.7 | %97.4 |
| Normal | 43 | %72.1 | %88.1 |
| Glokom | 43 | %67.4 | %81.5 |
| **Genel** | **171** | **%80.7** | %91.2 |

- Her yanlis tahmin sinif dogrulugunu ~2.3 puan degistirir; genel sonucun %95 guven araligi yaklasik ±6 puan. Bu yuzden asil sonuc asagidaki capraz dogrulama ortalamasidir.
- Glokom ile normal en cok karisan ikili. Glokom tanisi klinikte de yalnizca fundus fotografina degil goz ici basinci ve gorme alanina dayanir. Arayuzde bazi fotograflarda gorulen "%40 normal / %40 glokom" kararsizligi bununla uyumlu.

**Grad-CAM** (`gradcam_ornekleri_v2.png`)
- Glokom: dogru ornekte isi tam optik diskte.
- Diyabetik retinopati: dogru ornekte isi lezyon bolgesinde (eksuda halkasi). v1'de bu odak yoktu.
- Normal: dogru ornekte damar kemeri; glokom sanilan normal ornekte optik disk.
- Katarakt: dogru ornekte isi ust kenarda. Kenar testi bunun kisayol olmadigini gosterdi (asagida).
- Karanlik/bulanik bir glokom goruntusu katarakt sanildi (isi ust kenarda): "detay yoksa katarakt" davranisi suruyor (kenar ve cozunurluk testleriyle de dogrulandi).

**5-katli capraz dogrulama** (`cnn_capraz_dogrulama_v2.csv`; 1152 goruntunun tamami, hasta bazli `StratifiedGroupKFold`, her katta ~690 egitim / ~230 dogrulama / ~230 test)

| Kat | Genel | Katarakt | DR | Glokom | Normal |
|---|---|---|---|---|---|
| 1 | 80.1 | 92.5 | 92.7 | 75.9 | 63.1 |
| 2 | 83.3 | 96.9 | 89.7 | 64.7 | 78.0 |
| 3 | 79.0 | 89.5 | 82.1 | 81.4 | 63.2 |
| 4 | 83.8 | 90.2 | 91.7 | 83.1 | 69.2 |
| 5 | 76.6 | 98.4 | 96.6 | 45.5 | 61.8 |
| **Ortalama ± std** | **80.6 ± 3.0** | 93.5 ± 4.0 | 90.6 ± 5.3 | 70.1 ± 15.5 | 67.0 ± 6.7 |

- Ortalama (%80.6) tek bolmenin sonucuyla (%80.7) ayni: v2 test seti temsil edici.
- Katarakt ve DR her katta %82'nin ustunde, kararli.
- Glokom en kararsiz sinif (%45.5 - %83.1). Hangi hastalarin teste dustugune gore cok degisiyor; glokom/normal ayrimi bu veri miktariyla guvenilir degil.
- Normal ortalamada en dusuk sinif (%67.0); hatalarin cogu glokom tarafina.
- Her katta egitim verisi tek bolmeden ~%15 az (690 / 808), bu yuzden sonuclar biraz karamsar olabilir.

**Kenar testi** (`cnn_kenar_testi_ozet_v2.csv`, test seti, sinif bazinda dogruluk %)

| Sinif | Orijinal | Kenar %15 kapali | Kenar %30 kapali | Sadece kenar | v1 kenar %30 / sadece kenar |
|---|---|---|---|---|---|
| Katarakt | 92.9 | 92.9 | 90.5 | 100.0 | 97.4 / 99.3 |
| Diyabetik retinopati | 90.7 | 88.4 | 72.1 | 32.6 | 53.6 / 88.1 |
| Glokom | 67.4 | 76.7 | 55.8 | 4.7 | 74.8 / 2.0 |
| Normal | 72.1 | 74.4 | 69.8 | 2.3 | 82.8 / 16.6 |

- Katarakt: kenar kapatilinca dusmuyor; Grad-CAM'deki ust kenar odagi kisayol degil.
- DR: kenar bagimliligi v1'e gore cok azaldi (sadece kenarla dogru tahmin %88.1 -> %32.6). %30 kapatma diskin yaklasik yarisini kapattigi icin cevredeki lezyonlarin da gitmesi beklenir.
- Glokom ve normal merkeze/optik diske dayaniyor.
- "Sadece kenar" kosulunda ortasi siyah goruntulerin cogu katarakt deniyor (glokom %90.7, normal %93.0, DR %53.5). Karanlik veya kotu cekilmis fotograf katarakt sanilabilir; arayuz uyarisinda belirtilmeli.
- Sinif basina ~43 goruntu oldugu icin ±5 puanlik farklar anlamli degil.

**Cozunurluk testi** (`cnn_cozunurluk_testi_v2.csv`; test goruntuleri 256x256'ya dusurulup JPEG q75 ile sikistirildi ve geri buyutuldu)

| Sinif | Orijinal | 256x256 | v1 256x256 |
|---|---|---|---|
| Katarakt | 92.9 | 100.0 | - |
| Diyabetik retinopati | 90.7 | 4.7 | 0 |
| Glokom | 67.4 | 58.1 | - |
| Normal | 72.1 | 72.1 | 1.3 |
| **Genel** | **80.7** | **58.5** | |

- **v1'deki cozunurluk kisayolu yok:** normal goruntuler 256'ya dusurulunce v1'de %1.3, v2'de %72.1 (hic degismiyor). Glokom da buyuk olcude yerinde.
- DR %4.7'ye dusuyor ama v1'den farkli olarak goruntuler **normal** deniyor (%67.4), katarakt/glokom degil. Mikroanevrizma ve kucuk kanamalar birkac piksellik bulgular; kucultme ve sikistirma bunlari siliyor. Bu klinik olarak beklenen bir dusus. DR'nin kaynak (EyePACS) dokusu da bu islemle siliniyor olabilir; bu test ikisini ayiramaz.
- Bulaniklasan goruntuler katarakta kayiyor (glokom %27.9, DR ve normal %16.3): "detay yoksa katarakt" davranisi.

---

### v1 bolmesinde bulunan sorunlar (analiz):

Kisi 5'in "katarakt Grad-CAM'i retina disina bakiyor" sorusu uzerine yapilan testler (`cnn_kaynak_testi.csv`):

1. **Kopya sizintisi:** 1223 adet 256x256 goruntu (`_N_xxxxxxx.jpg`, Keras augmentasyon isimlendirmesi) ~804 orijinalin dondurulmus/aynalanmis kopyalari ve sadece katarakt/glokomda var. Her kopya ayri hasta sayildigi icin hasta bazli bolme bunlari ayiramadi: test setindeki 178 kucuk goruntunun 77'sinin kopyasi egitim setindeydi. En yakin komsu analizinde test-egitim cift benzerligi 0.998'e kadar cikti.
2. **Cozunurluk kisayolu:** DR ve normal test goruntuleri 256x256'ya dusurulunce DR'nin %0'i, normalin %1.3'u dogru siniflandi; neredeyse hepsi glokom/katarakt dendi. Model "dusuk cozunurluk = katarakt/glokom" ogrenmisti (shortcut learning).
3. **Sisirilmis glokom basarisi:** v1 modelinde glokom dogrulugu 256x256 grupta %94.3, diger goruntulerde %63.5 idi.
4. **Onceki kenar testi yorumu eksikti:** Kenar testi sadece goruntu kenarina bagimliligi olcer; goruntunun tamamina yayilmis cozunurluk izini yakalayamaz. "Kamera/kaynak kisayolu yok" sonucu bu yuzden fazla genellemeydi.

---

### Sinirliliklar:
- **DR farkli kaynaktan:** DR goruntulerinin tamami `.jpeg` ve dosya kimlikleri 100-11598 araliginda (muhtemelen EyePACS); diger uc sinif ODIR-5K'dan. v2 bu kaynak farkini cozmez; DR basarisinin bir kismi kaynak izinden gelebilir. Cozum: DR goruntulerini de ODIR'den alarak dis test (Kisi 5 planinda "zaman kalirsa").
- **Veri miktari:** v2'de 1152 goruntu (v1: 4028). Test seti kucuk, sonuc belirsizligi yuksek.
- Hiperparametreler ayarlanmadi; tek seed ile egitildi.
- Dusuk cozunurluklu veya bulanik fotograflarda DR kacirilabilir, glokom/normal katarakt sanilabilir.

---

### Yapilanlar / kalanlar (v2 modeliyle):
- [x] Kenar testi (5b)
- [x] Cozunurluk testi (5c)
- [x] 5-katli capraz dogrulama (5d): %80.6 ± 3.0
- [x] Model ve raporlarin ortak Drive'a kopyalanmasi
- [ ] RGB ve yesil+CLAHE girdi karsilastirmasi (istege bagli)
- [ ] Dis test: DR goruntuleri ODIR'den (Kisi 5 planinda "zaman kalirsa")

---

### Karsilastigim sorun:
- v1 bolmesindeki kopya sizintisi ve cozunurluk kisayolu (yukarida). Sebep veri setinin yapisi; 256x256 kopyalarin dosya adlari hasta kimligi kalibina uymadigi icin her biri ayri hasta sayildi.
- `on_isle()` tek kanalli gri goruntu dondurur; CNN fonksiyonlari ham BGR goruntu alir.
