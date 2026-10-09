"""
Veri Seti Bolme ve Hasta Sizintisi (Leakage) Dogrulama Araci
------------------------------------------------------------
Bu script, 'data/splits/veri_bolme.csv' dosyasini denetler:
1. Ayn hastanin sol ve sag gozlerinin farkli kumelere dagilip dagilmadigini kontrol eder.
2. Train, Val ve Test arasindaki hasta kesisimini hesaplar (Hedef: 0).
3. Sinif basina ornek sayilarinin dengesini dogrular.
"""

from pathlib import Path
import pandas as pd

def veri_bolmeyi_dogrula():
    proje_koku = Path(__file__).resolve().parent.parent
    csv_yolu = proje_koku / "data" / "splits" / "veri_bolme.csv"

    if not csv_yolu.exists():
        print(f"[HATA] Dosya bulunamadi: {csv_yolu}")
        return

    df = pd.read_csv(csv_yolu)

    print("+" + "-" * 68 + "+")
    print("|      RETINA-AI: VERI BOLME VE SIZINTI (LEAKAGE) DOGRULAMA RAPORU     |")
    print("+" + "-" * 68 + "+")
    print(f"Toplam Gorsel Sayisi : {len(df)}")
    print(f"Toplam Hasta Sayisi  : {df['patient_id'].nunique()}")

    # 1. Hasta Kümeleri
    train_hastalar = set(df[df["split"] == "train"]["patient_id"])
    val_hastalar = set(df[df["split"] == "val"]["patient_id"])
    test_hastalar = set(df[df["split"] == "test"]["patient_id"])

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
    print("=" * 70)

if __name__ == "__main__":
    veri_bolmeyi_dogrula()
