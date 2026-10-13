# RETINA-AI: Derin Ogrenme Sorumlusu (Kisi 4) Teslim Notu

> Durum: kod hazir ve uctan uca test edildi. Gercek veriyle egitim sonuclari (dogruluk, grafikler) egitimden sonra bu nota eklenecek.

---

### Ne yaptim:
1. `src/cnn_models.py`: EfficientNet-B0 transfer learning, egitim dongusu, erken durdurma, tahmin ve Grad-CAM.
2. `notebooks/04_derin_ogrenme_cnn.ipynb`: Colab'da tek tikla egitim, test ve figur uretimi.
3. Arayuz (Kisi 5) icin 3 fonksiyon: `model_yukle()`, `tahmin(goruntu)`, `gradcam(goruntu, sinif=None)`.
4. Test setindeki tum tahminleri `reports/cnn_test_tahminleri.csv` olarak kaydeden fonksiyon (Kisi 5 metrikleri buradan hesaplar).
5. Her siniftan 1 dogru + 1 yanlis ornekle Grad-CAM figuru.

---

### Neden yaptim:
- **Transfer learning:** Sinif basina 705 egitim gorseli sifirdan CNN egitmek icin az. ImageNet'te ogrenilmis kenar/doku filtreleri retinaya uyarlanir.
- **EfficientNet-B0:** Az parametre (~5.3M), Colab T4'te hizli, ResNet50'ye yakin basari.
- **Renkli (RGB) girdi:** `on_isle()` tek kanalli yesil + CLAHE dondurur. Kataraktin pusu ve DR'deki sari eksudalar renkte daha belirgin oldugu icin CNN'e kirpilmis renkli retina verildi. Kirpma ve boyutlandirma `on_isle()` ile ayni (`roi_kirp` + 512x512), boylece Grad-CAM haritasi on islenmis goruntuyle ayni koordinatlarda. Yesil + CLAHE ile denemek icin `girdi_modu="yesil_clahe"`.
- **Veri artirma:** Yatay cevirme + ±15° dondurme. Bulaniklastirma ve parlaklik/kontrast oynatma YOK, cunku bulaniklik kataraktin ayirt edici ipucu.
- **Erken durdurma:** Dogrulama kaybi iyilesmeyi birakinca durup asiri ogrenmeyi (overfitting) onler.
- **Grad-CAM:** Modelin optik disk / lezyonlar gibi gercek bulgulara mi, yoksa kenar ve koseler gibi kamera izlerine mi baktigini gostermek icin.

---

### Hangi parametreleri kullandim:
- Model: EfficientNet-B0 (`IMAGENET1K_V1`), son katman Dropout(0.3) + Linear(1280 → 4)
- Girdi: 384x384 RGB, ImageNet normalizasyonu
- Optimizasyon: AdamW, lr=3e-4, weight_decay=1e-4, ReduceLROnPlateau (factor=0.5, patience=2)
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
- Model: Drive `models/model_efficientnet.pth` (GitHub'a yuklenmez)
- Egitim grafigi: `reports/figures/cnn_egitim_grafigi.png`
- Egitim gecmisi: `reports/cnn_egitim_gecmisi.json`
- Test tahminleri: `reports/cnn_test_tahminleri.csv`
- Grad-CAM: `reports/figures/gradcam_ornekleri.png`

---

### Karsilastigim sorun:
- Grup mesajinda `on_isle()` ciktisi "512x512 RGB" yaziyor, ama kod tek kanalli (yesil + CLAHE) gri goruntu donduruyor. Bu yuzden CNN fonksiyonlari ham BGR goruntuyu aliyor; `app.py`'de `tahmin_et`/`gradcam` cagrilarina `islenmis` yerine `bgr` verilmeli.
- Sonuclar: egitimden sonra eklenecek. Genel dogruluk %97 ustu cikarsa veri sizintisi / kamera kisayolu acisindan Grad-CAM ozellikle incelenecek.
