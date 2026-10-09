# RETINA-AI: Veri Dogrulama, Temizleme ve Bolme Sureci (Asama 1)

Bu dokuman, Kisi 1 (Veri ve On Isleme Sorumlusu) tarafindan gerceklestirilen veri on hazirlik, dogrulama, dengeleme ve stratified bolme adimlarini aciklar.

---

## 1. Veri Kaynagi ve Mevcut Durum
Projede Kaggle 'eye_diseases_classification' veri seti kullanilmaktadir. Dizin taramasi sonucunda sinif bazinda gorsel sayilari asagidaki gibidir:

- Cataract (Katarakt): 1038 adet
- Diabetic Retinopathy (Diyabetik Retinopati): 1098 adet
- Glaucoma (Glokom): 1007 adet
- Normal (Saglikli Retina): 1074 adet
- Toplam Ham Gorsel: 4217 adet

---

## 2. Veri Temizleme ve Dogrulama Adimlari

### 2.1. Bozuk Dosya Kontrolu
Tum gorseller OpenCV (cv2.imread) ile bayt seviyesinde okunarak kontrol edilmistir. Okunamayan veya basligi bozulmus gorseller pipeline'dan ayiklanmistir.

### 2.2. MD5 Ozetleme ile Mukerrer (Kopya) Gorsel Taramasi
Veri setinde ayni gorselin farkli dosya isimleriyle kaydedilmis olma olasiligina karsi her dosyanin 128-bit MD5 kriptografik hash degeri hesaplanmistir. Mukerrer hash'e sahip gorsellerden yalnizca ilk kopya korunmus, digerleri elenmistir.

### 2.3. Sinif Dengeleme (Balancing)
Sinif dengesizliginin makine ogrenmesi ve derin ogrenme modellerinde yanlilik (bias) yaratmasini engellemek icin veri seti, en az ornege sahip sinif olan Glokom (1007 ornek) sayisina esitlenmistir. 
Her siniftan seed=42 tohum degeriyle rastgele 1007 ornek secilmis ve toplam veri seti 4028 gorsele sabitlenmistir.

---

## 3. Stratified Bolme Oranlari ve Ayrintilari
Veri seti, hastalik siniflarinin temsil oranlarini egitim, dogrulama ve test kumelerinde tam olarak korumak adina `Stratified Split` yontemiyle asagidaki gibi bolunmustur:

| Bolum (Split) | Yuzde | Sinif Basina Ornek | Toplam Ornek |
| :--- | :--- | :--- | :--- |
| **Train (Egitim)** | %70 | 705 | 2820 |
| **Validation (Dogrulama)** | %15 | 151 | 604 |
| **Test (Test)** | %15 | 151 | 604 |
| **TOPLAM** | **%100** | **1007** | **4028** |

---

## 4. Cikti Dosyasi: `data/splits/veri_bolme.csv`
Tum ekip uyelerinin (Kisi 2, 3, 4, 5) erisecegi referans tablo `data/splits/veri_bolme.csv` olarak uretilmistir. Tabloda su sutunlar bulunmaktadir:

1. `image_name`: Gorselin dosya adi (ornek: `100_right.jpg`).
2. `file_path`: Dosyanin yerel tam dosya yolu.
3. `relative_path`: Sinif klasoruyle birlikte goreli yol (ornek: `glaucoma/100_right.jpg`). Hem yerel hem Colab ortaminda `data_root / relative_path` seklinde dogrudan cagrilabilir.
4. `class_name`: Hastalik sinifi (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`).
5. `split`: Verinin ait oldugu kume (`train`, `val`, `test`).
6. `width`: Gorselin piksel genisligi.
7. `height`: Gorselin piksel yuksekligi.
8. `channels`: Renk kanali sayisi (3).
9. `md5_hash`: Dogrulama hash degeri.
