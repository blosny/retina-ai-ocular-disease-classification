# RETINA-AI: Derin Ogrenme Sorumlusu (Kisi 4) Teslim Notu

> Durum: model gercek veriyle egitildi ve test edildi (13 Ekim 2026, Colab T4). Test dogrulugu %91.2. Kenar testiyle kaynak/kamera kisayolu kontrol edildi.

---

### Ne yaptim:
1. `src/cnn_models.py`: EfficientNet-B0 transfer learning, egitim dongusu, erken durdurma, tahmin ve Grad-CAM.
2. `notebooks/04_derin_ogrenme_cnn.ipynb`: Colab'da tek tikla egitim, test ve figur uretimi.
3. Arayuz (Kisi 5) icin 3 fonksiyon: `model_yukle()`, `tahmin(goruntu)`, `gradcam(goruntu, sinif=None)`.
4. Test setindeki tum tahminleri `reports/cnn_test_tahminleri.csv` olarak kaydeden fonksiyon (Kisi 5 metrikleri buradan hesaplar).
5. Her siniftan 1 dogru + 1 yanlis ornekle Grad-CAM figuru.
6. Kenar testi: test goruntulerinin dis halkasi siyahla kapatilarak modelin kenar/kamera izlerine bagimliligi olculdu.

---

### Neden yaptim:
- **Transfer learning:** Sinif basina 705 egitim gorseli sifirdan CNN egitmek icin az. ImageNet'te ogrenilmis kenar/doku filtreleri retinaya uyarlanir.
- **EfficientNet-B0:** Az parametreli (~5.3M) oldugu icin Colab'in ucretsiz GPU'sunda egitilebilir; gorev dagiliminda da bu model belirlendi.
- **Renkli (RGB) girdi:** `on_isle()` tek kanalli yesil + CLAHE dondurur. Kataraktin pusu ve DR'deki sari eksudalar renkte daha belirgin oldugu icin CNN'e kirpilmis renkli retina verildi. Kirpma ve boyutlandirma `on_isle()` ile ayni (`roi_kirp` + 512x512), boylece Grad-CAM haritasi on islenmis goruntuyle ayni koordinatlarda. Yesil + CLAHE ile denemek icin `girdi_modu="yesil_clahe"`.
- **Veri artirma:** Yatay cevirme + ±15° dondurme. Bulaniklastirma ve parlaklik/kontrast oynatma YOK, cunku bulaniklik kataraktin ayirt edici ipucu.
- **Erken durdurma:** Dogrulama kaybi iyilesmeyi birakinca durup asiri ogrenmeyi (overfitting) onler.
- **Grad-CAM:** Modelin optik disk / lezyonlar gibi gercek bulgulara mi, yoksa kenar ve koseler gibi kamera izlerine mi baktigini gostermek icin.

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

---

### Arayuz sozlesmesi (Kisi 5 icin):
```python
from cnn_models import model_yukle, tahmin, gradcam

model = model_yukle("models/model_efficientnet.pth")  # bir kez, onbellekte tutulur
p = tahmin(bgr)          # {"normal": .., "diabetic_retinopathy": .., "glaucoma": .., "cataract": ..}
h = gradcam(bgr)         # 512x512, float32, 0-1 (tahmin edilen sinif icin)
h = gradcam(bgr, sinif="glaucoma")  # istenen sinif icin
```
- Girdi `on_isle()` ile ayni: dosya yolu veya BGR numpy dizisi. **`on_isle()` ciktisi degil.**
- Isi haritasi `boyutlandir(roi_kirp(bgr))` ile ayni koordinatlarda.

---

### Cikti/gorsel nerede:
- Kod: `src/cnn_models.py`, `notebooks/04_derin_ogrenme_cnn.ipynb`
- Model: `models/model_efficientnet.pth` (GitHub'a yuklenmez). Ortak Drive klasorunde yazma izni olmadigi icin simdilik Kisi 4'un kendi Drive'inda: `MyDrive/retina-ai-k4/models/`. Ortak klasore tasinacak.
- Egitim grafigi: `reports/figures/cnn_egitim_grafigi.png`
- Egitim gecmisi: `reports/cnn_egitim_gecmisi.json`
- Test tahminleri: `reports/cnn_test_tahminleri.csv`
- Grad-CAM: `reports/figures/gradcam_ornekleri.png`
- Kenar testi: `reports/cnn_kenar_testi.csv`, `reports/cnn_kenar_testi_ozet.csv`

---

### Sonuclar:

**Egitim** (`cnn_egitim_grafigi.png`)
- 13 epoch'ta erken durdurma ile durdu (epoch basina ~60 sn, toplam ~13 dk).
- En iyi model: **epoch 8**, dogrulama kaybi 0.395, dogrulama dogrulugu **%92.4**.
- Son epoch'ta egitim %99 / dogrulama %93: ~6 puanlik fark hafif asiri ogrenmeye isaret ediyor; erken durdurma bu yuzden devreye girdi.

**Test seti** (604 goruntu, sinif basina 151; model egitimde hic gormedi)

| Sinif | Dogruluk |
|---|---|
| Katarakt | %98.0 |
| Diyabetik retinopati | %97.4 |
| Normal | %88.1 |
| Glokom | %81.5 |
| **Genel** | **%91.2** |

- En cok karisan ikili glokom ve normal. Glokom tanisi klinikte de yalnizca fundus fotografina degil goz ici basinci ve gorme alanina dayandigi icin beklenen bir zayiflik.
- Precision, recall, F1 ve confusion matrix Kisi 5 tarafindan `cnn_test_tahminleri.csv` uzerinden hesaplanacak.

**Grad-CAM** (`gradcam_ornekleri.png`)
- Glokom: isi optik diskte yogunlasiyor (dogru ve yanlis ornekte de). Tibben beklenen bolge.
- Normal: optik disk ve damar kemeri. Glokom sanilan normal ornekte de model diske bakiyor.
- Diyabetik retinopati: dogru ornekte lezyon odagi net degil; yanlis ornekte isi goruntunun sag kenarinda.
- Katarakt: dogru ornek farkli bir kaynaktan gelmis gibi (yuvarlak degil, kenarlarinda beyaz yaylar) ve isi kenarlarda. Bu, kenar testiyle ayrica kontrol edildi.

**Kenar testi** (`cnn_kenar_testi_ozet.csv`, test seti, sinif bazinda dogruluk %)

| Sinif | Orijinal | Kenar %15 kapali | Kenar %30 kapali | Sadece kenar |
|---|---|---|---|---|
| Katarakt | 98.0 | 96.0 | 97.4 | 99.3 |
| Diyabetik retinopati | 97.4 | 89.4 | 53.6 | 88.1 |
| Glokom | 81.5 | 79.5 | 74.8 | 2.0 |
| Normal | 88.1 | 87.4 | 82.8 | 16.6 |

- Katarakt: dis halkanin %30'u kapatilinca bile %97.4. Model katarakti kenar/kamera izlerinden degil retina iceriginden taniyor; Grad-CAM'deki endise dogrulanmadi.
- Glokom: merkez kapatilinca %2'ye dusuyor. Model optik diske dayaniyor (Grad-CAM ile tutarli).
- "Sadece kenar" kosulunda ortasi siyah goruntulerin cogu (glokomlarin %88'i, normallerin %69'u) katarakt olarak siniflandiriliyor. Model "detay gorunmuyorsa katarakt" seklinde ogrenmis; karanlik veya kotu cekilmis bir fotograf katarakt sanilabilir. Arayuzdeki uyari notunda belirtilmeli.
- Diyabetik retinopati: kenarin %30'u kapaninca %97.4'ten %53.6'ya dusuyor. Lezyonlarin cevre retinaya yayilmasindan mi yoksa DR goruntulerinin farkli bir kaynaktan gelmesinden mi kaynaklandigi bu testle ayrilamiyor. Kesin cevap icin baska bir veri setiyle (ornegin ODIR-5K) dis test gerekir.

---

### Karsilastigim sorun:
- Grup mesajinda `on_isle()` ciktisi "512x512 RGB" yaziyor, ama kod tek kanalli (yesil + CLAHE) gri goruntu donduruyor. Bu yuzden CNN fonksiyonlari ham BGR goruntuyu aliyor; `app.py`'de `tahmin_et`/`gradcam` cagrilarina `islenmis` yerine `bgr` verilmeli.
- Ortak Drive klasorunde yazma iznim yok; model ve raporlar simdilik kendi Drive'imda.
- Henuz yapilmadi: RGB ve yesil+CLAHE girdi karsilastirmasi, farkli seed'lerle tekrar, dis veri setiyle test.
