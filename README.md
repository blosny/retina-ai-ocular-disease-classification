# RETINA-AI: Fundus Görüntülerinden Göz Hastalığı Sınıflandırma

Fırat Üniversitesi Bilgisayar Mühendisliği, BMÜ469 Medikal Görüntü İşleme dersi grup projesi.

RETINA-AI, göz dibi (fundus) fotoğrafını dört sınıftan birine ayırır: **normal**, **diyabetik retinopati**, **glokom** ve **katarakt**. Projenin amacı, aynı test kümesi üzerinde iki yaklaşımı karşılaştırmaktır:

- **Klasik yaklaşım:** Görüntüden ölçülen el yapımı özellikler (netlik, kontrast, damar oranı, lezyon ve kanama ölçümleri) ile eğitilmiş bir **SVM**.
- **Derin öğrenme:** Transfer öğrenme ile eğitilmiş **EfficientNet-B0 (CNN)** ve kararını açıklayan **Grad-CAM** ısı haritaları.

> Bu proje akademik bir çalışmadır. Tıbbi teşhis amacıyla kullanılamaz.

## Sonuçlar

İki model de aynı 171 görüntülük test kümesinde değerlendirilmiştir.

| | CNN (EfficientNet-B0) | SVM (7 özellik) |
|---|---|---|
| Doğruluk | **%80,7** | %61,4 |
| Makro F1 | **0,806** | 0,595 |
| Makro AUC | **0,941** | 0,826 |

Sınıf bazında doğruluk (recall):

| Sınıf | CNN | SVM |
|---|---|---|
| Normal | %72,1 | %32,6 |
| Diyabetik retinopati | %90,7 | %79,1 |
| Glokom | %67,4 | %46,5 |
| Katarakt | %92,9 | %88,1 |

- CNN dört sınıfın hepsinde SVM'den daha başarılıdır.
- İki modelin de en çok zorlandığı sınıflar normal ve glokomdur. CNN, 9 glokom görüntüsünü normal olarak sınıflandırmıştır.
- SVM, bulanıklığın netlik ölçümüyle kolayca yakalandığı katarakt sınıfında başarılıdır. Özellikleri arasında optik disk ölçümü bulunmadığı için glokomda zayıftır.

Bütün tablolar ve grafikler [`reports/karne`](reports/karne) klasöründedir.

### Veri sızıntısı bulgusu

İlk veri bölmesinde (v1) CNN %91,2 doğruluk göstermiştir. İnceleme sırasında katarakt ve glokom sınıflarındaki 1223 adet 256x256 görüntünün, orijinal fotoğrafların döndürülmüş ve çevrilmiş kopyaları olduğu ortaya çıkmıştır. Bu kopyalar ayrı hasta sayıldığı için hem eğitim hem test kümesine düşmüş, model de hastalığı değil görüntü boyutunu öğrenmiştir (kısa yol öğrenme).

Kopyalar çıkarılıp yalnızca 512x512 görüntülerle hasta bazlı yeni bir bölme (v2) yapıldığında gerçek başarı %80,7 olarak ölçülmüştür. Bu raporda yer alan bütün sonuçlar v2 bölmesine aittir.

| Sınıf | v1 (sızıntılı, geçersiz) | v2 (temiz) |
|---|---|---|
| Genel | %91,2 | %80,7 |
| Normal | %88,1 | %72,1 |
| Diyabetik retinopati | %97,4 | %90,7 |
| Glokom | %81,5 | %67,4 |
| Katarakt | %98,0 | %92,9 |

## Veri seti

- Kaynak: Kaggle `eye_diseases_classification` veri seti (ODIR ve EyePACS kaynaklı fundus fotoğrafları).
- v2 bölmesi: Her sınıftan 288 görüntü, toplam **1152** görüntü, yalnızca 512x512 çözünürlük.
- Bölme hasta bazlıdır, yani aynı hastanın iki gözü her zaman aynı kümede yer alır. MD5 ile kopya kontrolü yapılmıştır.

| Küme | Görüntü |
|---|---|
| Eğitim (train) | 808 |
| Doğrulama (val) | 173 |
| Test | 171 |

Bölme tablosu: [`data/splits/veri_bolme_v2.csv`](data/splits/veri_bolme_v2.csv). Görüntüler boyutları nedeniyle repoda tutulmaz.

**Sınırlılık:** Diyabetik retinopati görüntülerinin tamamı EyePACS'ten, diğer üç sınıf ise ODIR'den gelmektedir. Model, hastalık yerine kaynağa özgü özellikler (kamera, renk tonu) öğrenmiş olabilir.

## Sistem nasıl çalışır?

```
Fundus fotoğrafı
   │
   ├─ 1. Ön işleme ............ siyah kenarlar kırpılır, 512x512'ye getirilir, yeşil kanal + CLAHE
   ├─ 2. Anatomi .............. damar haritası, damar yoğunluğu, optik disk ve çukur/disk oranı
   │
   ├─ 3a. CNN ................. fotoğrafın tamamı EfficientNet-B0'a verilir → 4 sınıf olasılığı + Grad-CAM
   └─ 3b. SVM ................. 7 özellik ölçülür → RBF çekirdekli SVM → 4 sınıf olasılığı
   │
   └─ 4. Arayüz ............... tahmin, güven değeri, olasılıklar ve işleme adımları gösterilir
```

En yüksek olasılık %60'ın altında kalırsa arayüz "model kararsız" uyarısı verir.

## Kurulum ve çalıştırma

Python 3.10 veya üstü gerekir.

```bash
git clone https://github.com/blosny/retina-ai-ocular-disease-classification.git
cd retina-ai-ocular-disease-classification
pip install -r requirements.txt
```

CNN model dosyası boyutu nedeniyle GitHub'da değildir. `model_efficientnet_v2.pth` dosyasını ekibin ortak Drive klasöründen indirip `models/` klasörüne koyun. SVM modeli (`models/svm_model_v2.pkl`) repoda hazırdır.

Arayüzü başlatmak için:

```bash
streamlit run src/app.py
```

Windows'ta `streamlit` komutu tanınmazsa `py -m streamlit run src/app.py` kullanın.

Değerlendirme karnesini yeniden üretmek için:

```bash
python src/degerlendirme.py
```

## Klasör yapısı

```
├── data/
│   ├── splits/veri_bolme_v2.csv    hasta bazlı train/val/test bölmesi (v2)
│   └── oznitelikler_v2.csv         SVM için çıkarılan 7 özellik
├── docs/                           teslim notları, makale taslağı, ekip kuralları
├── models/                         SVM modeli (CNN modeli Drive'dan indirilir)
├── notebooks/                      veri hazırlama ve CNN eğitimi defterleri
├── reports/
│   ├── karne/                      final metrikler, karışıklık matrisleri, ROC, karşılaştırma
│   └── figures/                    eğitim grafiği, Grad-CAM örnekleri, özellik önemi
└── src/
    ├── preprocessing.py            ön işleme
    ├── veri_bolme_v2.py            v2 bölmesi
    ├── verify_splits.py            sızıntı ve bölme kontrolü
    ├── anatomy.py                  damar haritası ve optik disk
    ├── k3_oznitelik.py             SVM özellik çıkarımı ve tahmin
    ├── model_egitimi.py            SVM eğitimi
    ├── cnn_models.py               CNN eğitimi, tahmin ve Grad-CAM
    ├── degerlendirme.py            final değerlendirme karnesi
    └── app.py                      Streamlit arayüzü
```

## Ekip

| Görev | Sorumlu |
|---|---|
| Veri hazırlama ve ön işleme | Taha Buğra Çiçek |
| Damar haritası ve optik disk analizi | Berçem Papatya |
| Özellik çıkarımı ve SVM | Ahmet Şamil Şahin |
| Derin öğrenme (CNN) ve Grad-CAM | Zilan Korkmaz |
| Değerlendirme ve Streamlit arayüzü | Ahmet Ethem Akyol |
| Literatür ve makale | Yunus Emre Aydın |

Ayrıntılı açıklamalar için [`docs`](docs) klasöründeki teslim notlarına bakabilirsiniz.
