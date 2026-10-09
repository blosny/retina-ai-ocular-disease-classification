# RETINA-AI: Veri Dogrulama, Temizleme ve Bolme Sureci (Asama 1)

Bu dokuman, Veri ve On Isleme Sorumlusu tarafindan gerceklestirilen veri on hazirlik, dogrulama, dengeleme ve stratified bolme adimlarini aciklar.

---

## 1. Veri Kaynagi ve Mevcut Durum
Projede Kaggle 'eye_diseases_classification' veri seti kullanilmaktadir. Dizin taramasi sonucunda sinif bazinda ham gorsel sayilari asagidaki gibidir:

- Cataract (Katarakt): 1038 adet
- Diabetic Retinopathy (Diyabetik Retinopati): 1098 adet
- Glaucoma (Glokom): 1007 adet
- Normal (Saglikli Retina): 1074 adet
- Toplam Ham Gorsel: 4217 adet

---

## 2. Kritik Bulgu: Capraz Sinif Kopya Kesfi (Cross-Class Leakage)

MD5 kriptografik tarama sonucunda siniflarin kendi icinde kopya olmadigi (sinif ici kopya: 0) dogrulanmistir. Ancak Kaggle veri setinde **2 adet gorselin hem Katarakt hem de Glokom klasorunde ayni anda bulundugu** tespit edilmistir:

| Gorsel Adi | MD5 Hash Degeri | Bulundugu Siniflar |
| :--- | :--- | :--- |
| `625_left.jpg` | `06f4c9b0835adcdddc48252f45f6480e` | Cataract & Glaucoma |
| `1415_right.jpg` | `e2d70a8954bcdd55ef63989a9a5b84c3` | Cataract & Glaucoma |

### Cozum ve Temizleme Stratejisi:
Ayni goz dibi fotografinin iki farkli hastalik etiketiyle bulunmasi, model egitiminde **etiket gurultusune (label noise)** ve siniflar arasi veri sizintisina (leakage) yol acar. 
Glokom sinifi toplamda tam 1007 ornege sahip oldugu icin bu 2 cakisik ornek, ornek havuzu genis olan Katarakt sinifindan elenmistir (`1038 -> 1036`). Boylece Glokom sinifindaki 1007 benzersiz gorsel korunmus ve siniflar arasindaki etiket kirliligi tamamen sifirlanmistir.

---

## 3. Stratified Bolme Oranlari ve Bilimsel Gerekcesi

Veri seti, her siniftan 1007 ornek secilerek dengelenmis (toplam 4028 ornek) ve **%70 Train, %15 Validation, %15 Test** oraninda stratified olarak bolunmustur:

| Bolum (Split) | Oran | Sinif Basina Ornek | Toplam Ornek | Rol ve Amac |
| :--- | :--- | :--- | :--- | :--- |
| **Train (Egitim)** | %70 | 705 | 2820 | EfficientNet-B0 ve SVM modellerinin ogrenmesi |
| **Validation (Dogrulama)** | %15 | 151 | 604 | Erken durdurma (early stopping) ve hiperparametre ayari |
| **Test (Bagimsiz Test)** | %15 | 151 | 604 | Tarafsiz nihai metriklerin ve hata matrisinin olcumu |
| **TOPLAM** | **%100** | **1007** | **4028** | Tam dengeli 4 sinifli okuler veri seti |

### Neden %70 / %15 / %15 Orani Optimaldir?
1. **Asiri Ogrenmeyi (Overfitting) Engelleme:** Egitim kumesinde sinif basina 705 ornek (toplam 2820), medikal transfer learning ve klasik ML modellerinin genellenebilirligi icin yeterli temsiliyeti saglar.
2. **Guvenilir Dogrulama:** Dogrulama kumesindeki 604 ornek (sinif basina 151), egitim sirasinda kayip (loss) dalgalanmalarini onler ve hiperparametre seciminde guclu istatistiksel guvence sunar.
3. **Guclu Test Gucu:** Test kumesinde her siniftan tam 151 ornek bulunmasi; Sensitivity (Recall), Specificity, ROC-AUC ve Karmaşıklik Matrisi (Confusion Matrix) hesaplamalarinda tek bir yanlis tahminin sonuclari asiri saptirmasini (varyans sapmasini) onler.

---

## 4. Referans Dosyasi ve Gorsel Ciktilar

1. **`data/splits/veri_bolme.csv`**:
   - Tum ekibin kullanacagi ana indeks dosyasidir.
   - Sutunlar: `image_name`, `file_path`, `relative_path`, `class_name`, `split`, `width`, `height`, `channels`, `md5_hash`.
2. **`reports/figures/sinif_dagilimi.png`**:
   - Siniflarin train/val/test kumelerine dengeli dagilimini gosteren bar grafigi.
3. **`reports/figures/cozunurluk_dagilimi.png`**:
   - Veri setindeki ham gorsellerin cozunurluk (genislik x yukseklik) varyasyonunu gosteren dagilim grafigi.
