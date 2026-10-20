"""
Veri Seti Bolme ve Hasta Sizintisi (Leakage) Dogrulama Araci
------------------------------------------------------------
Bu script, 'data/splits/veri_bolme.csv' dosyasini denetler:
1. Ayn hastanin sol ve sag gozlerinin farkli kumelere dagilip dagilmadigini kontrol eder.
2. Train, Val ve Test arasindaki hasta kesisimini hesaplar (Hedef: 0).
3. Sinif basina ornek sayilarinin dengesini dogrular.
"""

import sys
from pathlib import Path
import pandas as pd

def veri_bolmeyi_dogrula(csv_adi="veri_bolme_v2.csv"):
    proje_koku = Path(__file__).resolve().parent.parent
    csv_yolu = proje_koku / "data" / "splits" / csv_adi

    if not csv_yolu.exists():
        print(f"[HATA] Dosya bulunamadi: {csv_yolu}")
        return

    df = pd.read_csv(csv_yolu)
    hasta_sutunu = "hasta_anahtari" if "hasta_anahtari" in df.columns else "patient_id"

    print("+" + "-" * 72 + "+")
    print(f"|  RETINA-AI: VERI BOLME VE SIZINTI (LEAKAGE) RAPORU - {csv_adi:<17} |")
    print("+" + "-" * 72 + "+")
    print(f"Toplam Gorsel Sayisi   : {len(df)}")
    print(f"Toplam Hasta Sayisi    : {df[hasta_sutunu].nunique()}")
    print(f"Kullanilan Hasta Kolonu: {hasta_sutunu}")

    # Cozunurluk analizi
    cozunurlukler = df[["width", "height"]].value_counts().to_dict()
    print(f"Cozunurluk Dagilimi    : {cozunurlukler}")

    # MD5 Tekrar Kontrolu
    if "md5_hash" in df.columns:
        md5_tekrar = int(df["md5_hash"].duplicated().sum())
        print(f"MD5 Tekrar (Kopya)     : {md5_tekrar}")

    # 1. Hasta Kümeleri
    train_hastalar = set(df[df["split"] == "train"][hasta_sutunu])
    val_hastalar = set(df[df["split"] == "val"][hasta_sutunu])
    test_hastalar = set(df[df["split"] == "test"][hasta_sutunu])

    train_test_kesisim = train_hastalar.intersection(test_hastalar)
    train_val_kesisim = train_hastalar.intersection(val_hastalar)
    val_test_kesisim = val_hastalar.intersection(test_hastalar)

    print("\n--- HASTA SIZINTISI (PATIENT LEAKAGE) KONTROLU ---")
    print(f"Train - Test Ortak Hasta Sayisi : {len(train_test_kesisim)}")
    print(f"Train - Val  Ortak Hasta Sayisi : {len(train_val_kesisim)}")
    print(f"Val   - Test Ortak Hasta Sayisi : {len(val_test_kesisim)}")

    if len(train_test_kesisim) == 0 and len(train_val_kesisim) == 0 and len(val_test_kesisim) == 0:
        print("[BASARILI] HASTA SIZINTISI %0.0! Tum hastalar yalnizca tek bir kumede.")
    else:
        print("[UYARI] SIZINTI TESPIT EDILDI! Bazi hastalar birden fazla kumede yer aliyor.")

    # 2. Sınıf ve Kümeler Dağılım Tablosu
    print("\n--- SINIF VE BOLUM DAGILIM TABLOSU ---")
    tablo = pd.crosstab(df["class_name"], df["split"], margins=True)
    print(tablo)

    # 3. Hasta Dağılım Özeti
    print("\n--- KUMELERE GORE HASTA SAYILARI ---")
    print(f"Train Kumesi Hasta Sayisi : {len(train_hastalar)}")
    print(f"Val   Kumesi Hasta Sayisi : {len(val_hastalar)}")
    print(f"Test  Kumesi Hasta Sayisi : {len(test_hastalar)}")
    print("=" * 74)

if __name__ == "__main__":
    hedef = sys.argv[1] if len(sys.argv) > 1 else "veri_bolme_v2.csv"
    veri_bolmeyi_dogrula(hedef)

