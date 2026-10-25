"""
RETINA-AI: Klasik Makine Ogrenmesi Model Egitimi ve Degerlendirme (v2 - Gercek Veri)
-----------------------------------------------------------------------------------
Bu modul, `data/oznitelikler_v2.csv` dosyasindaki gercek medikal retina ozniteliklerini kullanarak:
1. `split` sutununa gore Train (808), Val (173) ve Test (171) kumelerine ayirir.
2. Data leakage (veri sizintisi) olmamasi icin StandardScaler olceklendirme islemini
   SADECE Train kumesi uzerinde fit eder; Val ve Test kumelerine yalnizca transform uygular.
3. SVM modelini (SVC(kernel='rbf', probability=True)) sadece train kumesiyle egitir.
4. Val (dogrulama) kumesinde basarimi olcer. Sonucun yapay %100 degil, gercekci tibbi
   veri araliginda (%60-%85 bandinda) cikip cikmadigini denetler.
5. Random Forest modelini egitir ve guncel oznitelik onem grafigini olusturur.
6. Egitilen guncel SVM Pipeline modelini `models/svm_model_v2.pkl` olarak kaydeder.
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
    "laplacian_varyansi": "Laplacian Varyansı (Netlik)",
    "ortalama_kontrast": "Ortalama Kontrast (RMS)",
    "damar_orani": "Görünür Damar Oranı",
    "sari_lezyon_alani": "Sarı Eksüda Alanı",
    "sari_lezyon_sayisi": "Sarı Eksüda Sayısı",
    "kanama_alani": "Kanama Alanı",
    "kanama_sayisi": "Kanama Sayısı"
}


def veriyi_yukle_ve_bol(csv_yolu: Path):
    """
    oznitelikler_v2.csv dosyasini yukler ve split sutununa gore ayirir.
    """
    if not csv_yolu.exists():
        raise FileNotFoundError(f"Öznitelik CSV dosyası bulunamadı: {csv_yolu}")

    df = pd.read_csv(csv_yolu)
    print(f"[1/5] Veri yuklendi: {csv_yolu.name} (Toplam {len(df)} gercek ornek)")

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


def sizintisiz_olcekleme_ve_egitim(X_train, y_train):
    """
    Data leakage (veri sizintisi) OLMAMASI icin:
    - StandardScaler SADECE X_train uzerinde fit edilir.
    - SVM modeli sadece egitim kumesiyle egitilir.
    """
    print("[2/5] Data Leakage onleme ve Model Egitimi...")

    # SADECE train verisinde fit et
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("      [OK] StandardScaler SADECE train verisinde fit_transform edildi (Val/Test dahil edilmedi).")

    # SVM modelini train uzerinde egit
    svm_model = SVC(kernel="rbf", probability=True, C=2.0, gamma="scale", random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    print("      [OK] SVM (RBF Kernel, C=2.0) sadece train kumesiyle egitildi.")

    # Random Forest modelini egit (Oznitelik onem analizi icin)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    print("      [OK] Random Forest (100 agac) train kumesiyle egitildi.\n")

    # Uctan uca boru hatti (Pipeline) olustur
    svm_pipeline = Pipeline([
        ("scaler", scaler),
        ("svm", svm_model)
    ])

    return scaler, svm_model, svm_pipeline, rf_model


def dogrulama_ve_test_degerlendir(scaler, svm_model, rf_model, X_val, y_val, X_test, y_test):
    """
    Val ve Test verilerine SADECE transform uygular ve basarimi olcer.
    Val basariminin gercekci tibbi veri bandinda (%60-%85) olup olmadigini denetler.
    """
    print("[3/5] Dogrulama (Val) ve Test kumelerinde basarim olcumu...")

    # Val ve Test verilerine ASLA fit uygulanmaz, sadece transform uygulanir
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("      [OK] Val ve Test kumelerine SADECE transform uygulandi.")

    # 1. VAL DEGERLENDIRMESI
    val_preds = svm_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")

    # 2. TEST DEGERLENDIRMESI
    test_preds = svm_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    # Random Forest Test
    rf_test_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_test_preds)
    rf_f1 = f1_score(y_test, rf_test_preds, average="macro")

    print("\n" + "=" * 70)
    print("                  SVM MODELI PERFORMANS TABLOSU (v2)")
    print("=" * 70)
    print(f"{'Kume':<20} | {'Accuracy (Dogruluk)':<22} | {'Macro F1':<18}")
    print("-" * 70)
    print(f"{'Validation (Val)':<20} | %{val_acc * 100:<21.2f} | {val_f1:<18.4f}")
    print(f"{'Test':<20} | %{test_acc * 100:<21.2f} | {test_f1:<18.4f}")
    print(f"{'Random Forest (Test)':<20} | %{rf_acc * 100:<21.2f} | {rf_f1:<18.4f}")
    print("=" * 70)

    # Gercekci aralik kontrolu: %60 - %85 bandi
    aralikta_mi = (0.60 <= val_acc <= 0.85)
    print("\n[KLINIK DOGRULAMA KONTROLU]")
    if aralikta_mi:
        print(f"  -> DURUM: BASARILI!")
        print(f"  -> Val Dogruluk Orani: %{val_acc * 100:.2f}")
        print(f"  -> Model basarimi sentetik/yapay %100 degildir.")
        print(f"  -> Gercekci tibbi veri araligi olan %60 - %85 bandi icerisindedir.")
    else:
        print(f"  -> UYARI: Val dogruluk orani (%{val_acc * 100:.2f}) beklenen %60-%85 bandinin disinda.")

    print("\n--- SVM Dogrulama (Val) Kumesi Detayli Raporu ---")
    print(classification_report(y_val, val_preds, digits=3))

    return val_acc, test_acc


def oznitelik_onem_grafigi_ciz(rf_model, cikti_yolu: Path):
    """
    Random Forest feature_importances_ degerlerini grafik olarak kaydeder.
    """
    print(f"[4/5] Oznitelik onem grafigi guncelleniyor: {cikti_yolu.name}...")
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)

    onemler = rf_model.feature_importances_
    etiketler = [OZNITELIK_ISIMLERI_TR.get(col, col) for col in OZNITELIK_KOLONLARI]

    df_onem = pd.DataFrame({
        "Oznitelik": etiketler,
        "Onem": onemler
    }).sort_values(by="Onem", ascending=True)

    plt.figure(figsize=(10, 6))
    palet = sns.color_palette("mako", len(df_onem))
    bars = plt.barh(df_onem["Oznitelik"], df_onem["Onem"], color=palet, edgecolor="black", height=0.65)

    plt.title("RETINA-AI: Gercek Veri Random Forest Oznitelik Onem Dereceleri (v2)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Gini Onem Skoru (Importance Score)", fontsize=11)
    plt.ylabel("Medikal Oznitelik", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

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


def test_tahminlerini_kaydet(
    svm_pipeline,
    csv_yolu: Path = PROJE_KOKU / "data" / "oznitelikler_v2.csv",
    cikti_yolu: Path = PROJE_KOKU / "reports" / "svm_test_tahminleri_v2.csv",
    bolum: str = "test"
) -> pd.DataFrame:
    """
    Test kumesindeki verileri modele vererek tahmin raporunu uretir.
    Kisi 4'un urettigi test raporu dosyasi (reports/cnn_test_tahminleri_v2.csv) ile
    birebir uyumlu sutun yapisinda kaydeder.
    """
    print(f"[5/6] {bolum.capitalize()} kumesi tahmin raporu uretiliyor: {cikti_yolu.name}...")
    df_tum = pd.read_csv(csv_yolu)
    df_bolum = df_tum[df_tum["split"] == bolum].reset_index(drop=True)

    X_bolum = df_bolum[OZNITELIK_KOLONLARI].values
    y_bolum = df_bolum["class_name"].values

    preds = svm_pipeline.predict(X_bolum)
    probas = svm_pipeline.predict_proba(X_bolum)
    classes = list(svm_pipeline.classes_)

    satirlar = []
    for i in range(len(df_bolum)):
        tahmin_sinif = preds[i]
        gercek_sinif = y_bolum[i]
        prob_dict = {cls: probas[i, idx] for idx, cls in enumerate(classes)}
        conf = prob_dict[tahmin_sinif]
        satirlar.append({
            "image_name": df_bolum.loc[i, "image_name"],
            "relative_path": df_bolum.loc[i, "relative_path"],
            "gercek": gercek_sinif,
            "tahmin": tahmin_sinif,
            "dogru": (tahmin_sinif == gercek_sinif),
            "confidence_score": round(float(conf), 5),
            "p_normal": round(float(prob_dict.get("normal", 0.0)), 5),
            "p_diabetic_retinopathy": round(float(prob_dict.get("diabetic_retinopathy", 0.0)), 5),
            "p_glaucoma": round(float(prob_dict.get("glaucoma", 0.0)), 5),
            "p_cataract": round(float(prob_dict.get("cataract", 0.0)), 5)
        })

    df_sonuc = pd.DataFrame(satirlar)
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    df_sonuc.to_csv(cikti_yolu, index=False)
    dogruluk = df_sonuc["dogru"].mean()
    print(f"      [OK] Rapor kaydedildi: {cikti_yolu} (Dogruluk: %{dogruluk * 100:.2f})\n")
    return df_sonuc


def modeli_kaydet(svm_pipeline, model_yolu: Path):
    """
    Nihai SVM modelini `models/svm_model_v2.pkl` olarak kaydeder.
    """
    print(f"[5/5] Egitilmis SVM Pipeline modeli kaydediliyor: {model_yolu.name}...")
    model_yolu.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_pipeline, model_yolu)
    print(f"      [OK] Model basariyla kaydedildi: {model_yolu}")

    # Geriye donuk uyumluluk icin svm_model.pkl olarak da kopyala
    yedek_yol = model_yolu.parent / "svm_model.pkl"
    joblib.dump(svm_pipeline, yedek_yol)
    print(f"      [OK] Arayuz uyumlulugu icin yedeklendi: {yedek_yol}\n")


def main():
    csv_yolu = PROJE_KOKU / "data" / "oznitelikler_v2.csv"
    fig_yolu = PROJE_KOKU / "reports" / "figures" / "feature_importance.png"
    model_v2_yolu = PROJE_KOKU / "models" / "svm_model_v2.pkl"

    print("+" + "-" * 68 + "+")
    print("|      RETINA-AI: K3 MODEL EGITIMI VE PERFORMANS ANALIZI (v2)        |")
    print("+" + "-" * 68 + "+\n")

    # 1. Veri Yukleme ve Bolme
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = veriyi_yukle_ve_bol(csv_yolu)

    # 2. Veri Sizintisini Onleyerek Olcekleme ve Egitim
    scaler, svm_model, svm_pipeline, rf_model = sizintisiz_olcekleme_ve_egitim(X_train, y_train)

    # 3. Dogrulama (Val) ve Test Basarimi Olcumu
    val_acc, test_acc = dogrulama_ve_test_degerlendir(
        scaler, svm_model, rf_model, X_val, y_val, X_test, y_test
    )

    # 4. Feature Importance Grafigi
    oznitelik_onem_grafigi_ciz(rf_model, fig_yolu)

    # 5. Test Tahmin Raporu Uretimi (reports/svm_test_tahminleri_v2.csv)
    rapor_yolu = PROJE_KOKU / "reports" / "svm_test_tahminleri_v2.csv"
    test_tahminlerini_kaydet(svm_pipeline, csv_yolu=csv_yolu, cikti_yolu=rapor_yolu, bolum="test")

    # 6. Modeli svm_model_v2.pkl Olarak Kaydet
    modeli_kaydet(svm_pipeline, model_v2_yolu)

    print("[TAMAMLANDI] Model egitimi, basarim kontrolu, test tahmin raporu ve model kayit adimlari basariyla tamamlandi.")


if __name__ == "__main__":
    main()
