"""
RETINA-AI: Klasik Makine Ogrenmesi Model Egitimi ve Degerlendirme (Kisi 3)
--------------------------------------------------------------------------
Bu modul, `data/oznitelikler.csv` verisi uzerinden:
1. `split` sutununa gore Train / Val / Test kumelerini ayirir.
2. StandardScaler + SVC(probability=True) SVM Pipeline modelini egitir.
3. RandomForestClassifier(n_estimators=100, random_state=42) modelini egitir.
4. Test kumesinde Accuracy ve Macro F1 skorlarini hesaplayip karsilastirir.
5. Random Forest oznitelik onem grafigini cizerek `reports/figures/feature_importance.png` dosyasina kaydeder.
6. Nihai SVM modelini `models/svm_model.pkl` dosyasina kaydeder.
"""

from pathlib import Path
import os
import sys

# Windows konsol UTF-8 uyumlulugu
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Proje kok dizini
PROJE_KOKU = Path(__file__).resolve().parent.parent

OZNITELIK_KOLONLARI = [
    "laplacian_varyansi",
    "ortalama_kontrast",
    "damar_orani",
    "sari_lezyon_alani",
    "sari_lezyon_sayisi",
    "kanama_alani",
    "kanama_sayisi"
]

OZNITELIK_ISIMLERI_TR = {
    "laplacian_varyansi": "Laplacian Varyansi (Netlik)",
    "ortalama_kontrast": "Ortalama Kontrast (RMS)",
    "damar_orani": "Gorunur Damar Orani",
    "sari_lezyon_alani": "Sari Eksuda Alani",
    "sari_lezyon_sayisi": "Sari Eksuda Sayisi",
    "kanama_alani": "Kanama Alani",
    "kanama_sayisi": "Kanama Sayisi"
}


def veriyi_yukle_ve_bol(csv_yolu: Path):
    """
    oznitelikler.csv dosyasini yukler ve split sutununa gore ayirir.
    """
    if not csv_yolu.exists():
        raise FileNotFoundError(f"Oznitelik CSV dosyasi bulunamadi: {csv_yolu}")

    df = pd.read_csv(csv_yolu)
    print(f"[1/5] Veri yuklendi: {csv_yolu.name} (Toplam {len(df)} ornek)")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train = train_df[OZNITELIK_KOLONLARI].values
    y_train = train_df["class_name"].values

    X_val = val_df[OZNITELIK_KOLONLARI].values
    y_val = val_df["class_name"].values

    X_test = test_df[OZNITELIK_KOLONLARI].values
    y_test = test_df["class_name"].values

    print(f"      Egitim kumesi    (Train) : {len(X_train)} ornek")
    print(f"      Dogrulama kumesi (Val)   : {len(X_val)} ornek")
    print(f"      Test kumesi      (Test)  : {len(X_test)} ornek\n")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def modelleri_egit(X_train, y_train):
    """
    SVM (StandardScaler + SVC) ve Random Forest modellerini egitir.
    """
    print("[2/5] Modeller egitiliyor...")

    # 1. SVM Pipeline
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, C=1.5, gamma="scale", random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    print("      [OK] SVM (StandardScaler + RBF SVC) modeli egitildi.")

    # 2. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12)
    rf_model.fit(X_train, y_train)
    print("      [OK] Random Forest (100 agac) modeli egitildi.\n")

    return svm_pipeline, rf_model


def modelleri_degerlendir(svm_model, rf_model, X_test, y_test):
    """
    Test kumesinde Accuracy ve Macro F1 metriklerini hesaplar.
    """
    print("[3/5] Test kumesi uzerinde modeller degerlendiriliyor...")

    # SVM Tahminleri
    svm_tahminler = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_tahminler)
    svm_f1 = f1_score(y_test, svm_tahminler, average="macro")

    # Random Forest Tahminleri
    rf_tahminler = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_tahminler)
    rf_f1 = f1_score(y_test, rf_tahminler, average="macro")

    print("=" * 65)
    print("                 MODEL TEST PERFORMANS TABLOSU")
    print("=" * 65)
    print(f"{'Model':<25} | {'Accuracy (Dogruluk)':<20} | {'Macro F1':<15}")
    print("-" * 65)
    print(f"{'SVM (RBF Kernel)':<25} | %{svm_acc * 100:<19.2f} | {svm_f1:<15.4f}")
    print(f"{'Random Forest (100)':<25} | %{rf_acc * 100:<19.2f} | {rf_f1:<15.4f}")
    print("=" * 65 + "\n")

    print("--- SVM Detayli Siniflandirma Raporu ---")
    print(classification_report(y_test, svm_tahminler, digits=3))

    print("--- Random Forest Detayli Siniflandirma Raporu ---")
    print(classification_report(y_test, rf_tahminler, digits=3))

    return {
        "svm": {"accuracy": svm_acc, "macro_f1": svm_f1},
        "rf": {"accuracy": rf_acc, "macro_f1": rf_f1}
    }


def oznitelik_onem_grafigi_ciz(rf_model, cikti_yolu: Path):
    """
    Random Forest modelinin feature_importances_ degerlerini grafik olarak kaydeder.
    """
    print(f"[4/5] Oznitelik onem grafigi olusturuluyor: {cikti_yolu.name}...")
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)

    onemler = rf_model.feature_importances_
    etiketler = [OZNITELIK_ISIMLERI_TR.get(col, col) for col in OZNITELIK_KOLONLARI]

    df_onem = pd.DataFrame({
        "Oznitelik": etiketler,
        "Onem": onemler
    }).sort_values(by="Onem", ascending=True)

    plt.figure(figsize=(10, 6))
    palet = sns.color_palette("viridis", len(df_onem))
    bars = plt.barh(df_onem["Oznitelik"], df_onem["Onem"], color=palet, edgecolor="black", height=0.65)

    plt.title("RETINA-AI: Random Forest Oznitelik Onem Dereceleri (Feature Importance)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Gini Onem Skoru (Importance Score)", fontsize=11)
    plt.ylabel("Medikal Oznitelik", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Cubuklarin uzerine yuzde/deger yazdir
    for bar in bars:
        genislik = bar.get_width()
        plt.text(
            genislik + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"%{genislik * 100:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold"
        )

    plt.xlim(0, max(df_onem["Onem"]) * 1.15)
    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=200)
    plt.close()
    print(f"      [OK] Grafik kaydedildi: {cikti_yolu}\n")


def svm_modelini_kaydet(svm_model, model_yolu: Path):
    """
    Nihai SVM modelini disk uzerine kaydeder.
    """
    print(f"[5/5] Nihai SVM modeli kaydediliyor: {model_yolu.name}...")
    model_yolu.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_model, model_yolu)
    print(f"      [OK] Model basariyla kaydedildi: {model_yolu}\n")


def main():
    csv_yolu = PROJE_KOKU / "data" / "oznitelikler.csv"
    fig_yolu = PROJE_KOKU / "reports" / "figures" / "feature_importance.png"
    model_yolu = PROJE_KOKU / "models" / "svm_model.pkl"

    print("+" + "-" * 63 + "+")
    print("|      RETINA-AI: K3 MODEL EGITIMI VE PERFORMANS ANALIZI        |")
    print("+" + "-" * 63 + "+\n")

    # 1. Veri Bolme
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = veriyi_yukle_ve_bol(csv_yolu)

    # 2. Modelleri Egit
    svm_model, rf_model = modelleri_egit(X_train, y_train)

    # 3. Degerlendir
    sonuclar = modelleri_degerlendir(svm_model, rf_model, X_test, y_test)

    # 4. Feature Importance Grafigi
    oznitelik_onem_grafigi_ciz(rf_model, fig_yolu)

    # 5. Modeli Kaydet
    svm_modelini_kaydet(svm_model, model_yolu)

    print("[TAMAMLANDI] Tum egitim, analiz ve kayit islemleri basariyla sonuclandi.")


if __name__ == "__main__":
    main()
