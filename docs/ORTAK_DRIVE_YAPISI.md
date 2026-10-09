# RETINA-AI: Ortak Google Drive Klasor Yapisi Rehberi

Bu dokuman, 6 kisilik proje ekibinin bulut uzerinde (Google Drive ve Google Colab) calisirken dosyalarini duzenli, karismadan ve tek bir ortak agac yapisinda tutabilmesi icin hazirlanmistir.

---

## 1. Ana Dizin Yapisi: `My Drive/RETINA-AI/`

Google Drive'inizda ana klasorun adi tam olarak **`RETINA-AI`** olmalidir. Tum kodlar ve notebook'lar bu kok dizine gore goreli (relative) yollar kullanir.

```text
Google Drive/
└── RETINA-AI/
    ├── data/
    │   ├── raw/                        # Kaggle dataset klasoru (cataract, glaucoma vb.)
    │   └── splits/
    │       └── veri_bolme.csv          # Veri ve On Isleme Sorumlusu'nun teslim ettigi ana tablo
    │
    ├── src/
    │   ├── preprocessing.py            # on_isle() pipeline modulu
    │   ├── anatomy.py                  # Kisi 2 (Damar ve Optik Disk)
    │   ├── lesions.py                  # Kisi 3 (Lezyon ve Netlik)
    │   ├── cnn_models.py               # Kisi 4 (EfficientNet-B0)
    │   └── app.py                      # Kisi 5 (Streamlit Dashboard)
    │
    ├── notebooks/
    │   ├── 01_veri_on_isleme.ipynb     # Ortak sablon ve on isleme defteri
    │   ├── 02_damar_ve_disk.ipynb      # Kisi 2
    │   ├── 03_lezyon_ve_svm.ipynb      # Kisi 3
    │   ├── 04_derin_ogrenme_cnn.ipynb  # Kisi 4
    │   └── 05_degerlendirme_ui.ipynb   # Kisi 5
    │
    ├── models/                         # Agir model agirliklari (Git'e atilmaz, Drive'da tutulur)
    │   ├── model_efficientnet.pth      # Kisi 4 ciktisi
    │   └── model_svm.pkl               # Kisi 3 ciktisi
    │
    ├── reports/
    │   └── figures/                    # Makale ve sunum icin yuksek cozunurluklu grafikler
    │       ├── sinif_dagilimi.png
    │       ├── cozunurluk_dagilimi.png
    │       └── on_isleme_asamalari.png
    │
    └── notlar/                         # Haftalik teslim notlari (Her teknik sorumlu buraya koyar)
        ├── teslim_01_veri_ve_on_isleme.md
        ├── teslim_02_damar_ve_disk.md
        ├── teslim_03_lezyon_ve_svm.md
        └── teslim_04_derin_ogrenme.md
```

---

## 2. Google Colab Uzerinde Baglanti Kodu
Ekiptekilerin tek yapmasi gereken notebook basinda su kodu calistirmaktir:

```python
from google.colab import drive
import sys
from pathlib import Path

drive.mount('/content/drive')
proje_koku = Path('/content/drive/MyDrive/RETINA-AI')

if str(proje_koku) not in sys.path:
    sys.path.append(str(proje_koku))
```
