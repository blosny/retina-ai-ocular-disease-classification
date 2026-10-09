# RETINA-AI: Veri Dogrulama, Temizleme ve Hasta Bazli Bolme Sureci (Asama 1)

Bu dokuman, Veri ve On Isleme Sorumlusu tarafindan gerceklestirilen veri on hazirlik, dogrulama, dengeleme ve hasta bazli (patient-level) stratified bolme adimlarini aciklar.

---

## 1. Veri Kaynagi ve Mevcut Durum
Projede Kaggle 'eye_diseases_classification' veri seti kullanilmaktadir. Dizin taramasi sonucunda sinif bazinda ham gorsel sayilari asagidaki gibidir:

- Cataract (Katarakt): 1038 adet
- Diabetic Retinopathy (Diyabetik Retinopati): 1098 adet
- Glaucoma (Glokom): 1007 adet
- Normal (Saglikli Retina): 1074 adet
- Toplam Ham Gorsel: 4217 adet

---

## 2. Kritik Bulgu 1: Capraz Sinif Kopya Kesfi (Cross-Class Leakage)

MD5 kriptografik tarama sonucunda siniflarin kendi icinde kopya olmadigi (sinif ici kopya: 0) dogrulanmistir. Ancak Kaggle veri setinde **2 adet gorselin hem Katarakt hem de Glokom klasorunde ayni anda bulundugu** tespit edilmistir:

| Gorsel Adi | MD5 Hash Degeri | Bulundugu Siniflar |
| :--- | :--- | :--- |
| `625_left.jpg` | `06f4c9b0835adcdddc48252f45f6480e` | Cataract & Glaucoma |
| `1415_right.jpg` | `e2d70a8954bcdd55ef63989a9a5b84c3` | Cataract & Glaucoma |

### Cozum:
Ayni goz dibi fotografinin iki farkli hastalik etiketiyle bulunmasi, model egitiminde **etiket gurultusune (label noise)** ve veri sizintisina yol acar. Glokom sinifi toplamda tam 1007 ornege sahip oldugu icin bu 2 cakisik ornek, havuzu genis olan Katarakt sinifindan elenmistir (`1038 -> 1036`). Boylece Glokom sinifindaki 1007 benzersiz gorsel korunmus ve siniflar arasindaki etiket kirliligi tamamen sifirlanmistir.

---

## 3. Kritik Bulgu 2: Hasta Bazli Veri Sizintisi (Patient Leakage) ve Cozumu

### Problem (Ekip Geri Bildirimi):
Retina goruntulerinde cogu hasta icin hem sag hem sol goz fotoğraflari bulunmaktadir (`1083_left.jpg`, `1083_right.jpg`). Rastgele ornek bazli bolme yapildiginda, **test setindeki hastalarin yaklasik %36-38'inin diger gozu egitim kumesine (Train)** dusmektedir. Bu durum agin hastanin bireysel vaskuler yapisini ezberlemesine ve sonuclari yapay olarak yuksek gostermesine (Data Leakage) sebep olur.

### Cozum: Hasta Duzeyinde Grup Tabakali Bolme (Patient-Level Group Stratified Split)
- Her gorselin dosya adindan `patient_id` cikarilmistir (Toplam 2941 benzersiz hasta).
- Bolme islemi ornek bazinda degil, **hasta kimligi (patient_id) bazinda** yapilmistir.
- **Kural:** Bir hastanin tum gozleri (sag ve sol) istisnasiz YALNIZCA BIR kumeye (Train, Val veya Test) atanir.
- **Sonuc:**
  - Train ve Test arasindaki hasta kesisimi: **0 hasta**
  - Train ve Val arasindaki hasta kesisimi: **0 hasta**
  - Val ve Test arasindaki hasta kesisimi: **0 hasta**
  - **Hasta Sizinti Orani: %0.0**

---

## 4. Stratified Bolme Oranlari ve Sayilari

Veri seti, her siniftan 1007 ornek secilerek dengelenmis (toplam 4028 ornek) ve **%70 Train, %15 Validation, %15 Test** oraninda hasta bazli bolunmustur:

| Bolum (Split) | Oran | Sinif Basina Ornek | Toplam Ornek | Benzersiz Hasta Sayisi |
| :--- | :--- | :--- | :--- | :--- |
| **Train (Egitim)** | %70 | 705 | 2820 | 2059 |
| **Validation (Dogrulama)** | %15 | 151 | 604 | 443 |
| **Test (Bagimsiz Test)** | %15 | 151 | 604 | 439 |
| **TOPLAM** | **%100** | **1007** | **4028** | **2941** |

---

## 5. Referans Dosyasi: `data/splits/veri_bolme.csv`

Tabloda artik `patient_id` sutunu da mevcuttur:
- `image_name`: Gorsel dosya adi.
- `file_path`: Tam yerel yol.
- `relative_path`: Goreli yol (`glaucoma/100_right.jpg`).
- `class_name`: Hastalik sinifi (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`).
- `split`: Ait oldugu kume (`train`, `val`, `test`).
- `patient_id`: Hasta kimlik numarasi (ayni hastanin tum gozleri ayni kumede).
- `width`, `height`, `channels`: Cozunurluk bilgileri.
- `md5_hash`: Dogrulama hash degeri.
