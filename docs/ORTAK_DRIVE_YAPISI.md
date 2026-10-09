# RETINA-AI: Ortak Google Drive Klasor Yapisi Rehberi

Bu dokuman, 6 kisilik proje ekibinin bulut uzerinde (Google Drive ve Google Colab) calisirken dosyalara ortak erisebilmesi ve cikti klasorlerini standart tutabilmesi icin hazirlanmistir.

---

## 1. Ortak Google Drive Klasor Baglantisi
Ekip calismasi icin acilan resmi ortak klasor:
- **Baglanti:** [Google Drive Ortak Klasoru](https://drive.google.com/drive/folders/1nwjYPcgwE-nXCPeCYR1B6o8ucXRCKT6O?usp=drive_link)
- **Klasor Adi:** `retina-ai-ocular-disease-classification`

> **Ekip Uyeleri Icin Onemli Not:** 
> Bu baglantiya girdikten sonra klasor basligina tiklayip **"Drive'a kisayol ekle" (Add shortcut to Drive) -> "Drive'im" (MyDrive)** secenegini seciniz. Boylece Colab uzerinde `/content/drive/MyDrive/retina-ai-ocular-disease-classification` yolundan dogrudan erisebilirsiniz.

---

## 2. Drive Klasor Agaci Yapisi

Klasor altinda tutulacak standart hiyerarsi:

```text
Google Drive/
└── retina-ai-ocular-disease-classification/
    ├── data/
    │   ├── raw/                        # Kaggle veri seti gorselleri
    │   └── splits/
    │       └── veri_bolme.csv          # Ortak referans tablosu (4028 ornek)
    │
    ├── src/
    │   ├── preprocessing.py            # on_isle() fonksiyonu
    │   ├── anatomy.py                  # Kisi 2 (Damar ve Optik Disk)
    │   ├── lesions.py                  # Kisi 3 (Lezyon ve Netlik)
    │   ├── cnn_models.py               # Kisi 4 (EfficientNet-B0)
    │   └── app.py                      # Kisi 5 (Streamlit Arayuzu)
    │
    ├── notebooks/
    │   ├── 01_veri_on_isleme.ipynb     # Ortak sablon (Veri ve On Isleme)
    │   ├── 02_damar_ve_disk.ipynb      # Kisi 2
    │   ├── 03_lezyon_ve_svm.ipynb      # Kisi 3
    │   ├── 04_derin_ogrenme_cnn.ipynb  # Kisi 4
    │   └── 05_degerlendirme_ui.ipynb   # Kisi 5
    │
    ├── models/                         # Agir model dosyalari (Git'e atilmaz, Drive'da tutulur)
    │   ├── model_efficientnet.pth      # Kisi 4
    │   └── model_svm.pkl               # Kisi 3
    │
    ├── reports/
    │   └── figures/                    # Makale ve sunum grafikleri
    │       ├── sinif_dagilimi.png
    │       ├── cozunurluk_dagilimi.png
    │       └── on_isleme_asamalari.png
    │
    └── notlar/                         # Haftalik teslim notlari klasoru
        ├── teslim_01_veri_ve_on_isleme.md
        ├── teslim_02_damar_ve_disk.md
        ├── teslim_03_lezyon_ve_svm.md
        └── teslim_04_derin_ogrenme.md
```

---

## 3. Google Colab Uzerinde Standart Baglanti Kodu
Tum ekip uyeleri Colab notebook'larinin basinda su kodu kullanarak projeye erisebilir:

```python
from google.colab import drive
import sys
from pathlib import Path

# Drive mount
drive.mount('/content/drive')

# Ortak proje kok dizini
proje_koku = Path('/content/drive/MyDrive/retina-ai-ocular-disease-classification')

if str(proje_koku) not in sys.path:
    sys.path.append(str(proje_koku))

print(f"Baglanti basarili: {proje_koku}")
```
