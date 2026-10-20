"""
RETINA-AI - Final karnesi (Kisi 5: Degerlendirme)

Ne yapar?
    Modellerin test seti tahmin dosyalarini okuyup metrikleri, tablolari ve grafikleri uretir:
      - Genel dogruluk, sinif bazinda precision / recall / F1, makro ortalama
      - Karisiklik matrisi (confusion matrix)
      - ROC egrileri ve AUC (her sinif icin bire-karsi-hepsi)
      - Once-sonra karsilastirmasi (v1 sizintili model vs v2 temiz model)
      - Iki model (CNN, SVM) ayni test setinde varsa model karsilastirmasi

Girdi dosyalari (reports/ altinda, ekipte anlasilan format):
    image_name, relative_path, gercek, tahmin, dogru, p_normal, p_diabetic_retinopathy, p_glaucoma, p_cataract

    cnn_test_tahminleri_v2.csv   -> zorunlu (Kisi 4, v2 modeli)
    cnn_test_tahminleri.csv      -> varsa once-sonra icin (Kisi 4, v1 modeli, gecersiz)
    svm_test_tahminleri_v2.csv   -> varsa CNN ile karsilastirma icin (Kisi 3)

Ciktilar: reports/karne/
    metrikler_<model>.csv, karisiklik_<model>.csv, ozet.csv, once_sonra.csv
    figures/karisiklik_<model>.png, roc_<model>.png, once_sonra.png, model_karsilastirma.png

Calistirma (proje ana klasorunden):
    py src/degerlendirme.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, precision_recall_fscore_support,
                             roc_curve)

PROJE_KOKU = Path(__file__).resolve().parent.parent
RAPOR = PROJE_KOKU / "reports"
CIKTI = RAPOR / "karne"
SINIFLAR = ["normal", "diabetic_retinopathy", "glaucoma", "cataract"]
ADLAR = {"normal": "Normal", "diabetic_retinopathy": "Diyabetik retinopati",
         "glaucoma": "Glokom", "cataract": "Katarakt"}

# Renkler (dogrulanmis referans paleti, acik tema)
YUZEY, MUREKKEP, IKINCIL, SOLUK, IZGARA, EKSEN = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
KATEGORIK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # sabit sira: sinif 1..4
SIRALI = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

DOSYALAR = {
    "CNN v2": "cnn_test_tahminleri_v2.csv",
    "SVM v2": "svm_test_tahminleri_v2.csv",
    "CNN v1 (geçersiz)": "cnn_test_tahminleri.csv",
}


def stil():
    """Turkce karakterli yazi tipi ve sade grafik stili."""
    adaylar = [f.name for f in font_manager.fontManager.ttflist]
    for yazi in ["Segoe UI", "DejaVu Sans", "Arial"]:
        if yazi in adaylar:
            plt.rcParams["font.family"] = yazi
            break
    plt.rcParams.update({
        "figure.facecolor": YUZEY, "axes.facecolor": YUZEY, "savefig.facecolor": YUZEY,
        "axes.edgecolor": EKSEN, "axes.labelcolor": IKINCIL, "text.color": MUREKKEP,
        "xtick.color": SOLUK, "ytick.color": SOLUK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })


def oku(dosya: Path) -> pd.DataFrame:
    df = pd.read_csv(dosya)
    eksik = {"gercek", "tahmin", *[f"p_{s}" for s in SINIFLAR]} - set(df.columns)
    if eksik:
        raise ValueError(f"{dosya.name} dosyasinda eksik sutun: {eksik}")
    return df


def metrikler(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sinif bazinda precision/recall/F1/AUC ve genel ozet."""
    p, r, f, n = precision_recall_fscore_support(df["gercek"], df["tahmin"], labels=SINIFLAR, zero_division=0)
    auclar = []
    for s in SINIFLAR:
        fpr, tpr, _ = roc_curve(df["gercek"] == s, df[f"p_{s}"])
        auclar.append(auc(fpr, tpr))
    tablo = pd.DataFrame({"sinif": [ADLAR[s] for s in SINIFLAR], "goruntu": n,
                          "precision": p, "recall (dogruluk)": r, "f1": f, "auc": auclar})
    ozet = {"goruntu": len(df), "dogruluk": accuracy_score(df["gercek"], df["tahmin"]),
            "makro_precision": p.mean(), "makro_recall": r.mean(), "makro_f1": f.mean(),
            "makro_auc": float(np.mean(auclar))}
    return tablo, ozet


def karisiklik_ciz(df: pd.DataFrame, model: str, yol: Path) -> pd.DataFrame:
    km = confusion_matrix(df["gercek"], df["tahmin"], labels=SINIFLAR)
    oran = km / km.sum(axis=1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("mavi", SIRALI)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(oran, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(SINIFLAR)):
        for j in range(len(SINIFLAR)):
            koyu = oran[i, j] > 0.55
            ax.text(j, i, f"{km[i, j]}\n%{oran[i, j] * 100:.0f}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if koyu else MUREKKEP, fontweight="bold" if i == j else "normal")
    etiket = [ADLAR[s].replace("Diyabetik retinopati", "Diyabetik\nretinopati") for s in SINIFLAR]
    ax.set_xticks(range(4), etiket)
    ax.set_yticks(range(4), etiket)
    ax.set_xlabel("Modelin tahmini")
    ax.set_ylabel("Gerçek sınıf")
    ax.set_title(f"Karışıklık matrisi · {model}", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(yol, dpi=200)
    plt.close(fig)
    return pd.DataFrame(km, index=[ADLAR[s] for s in SINIFLAR], columns=[ADLAR[s] for s in SINIFLAR])


def roc_ciz(df: pd.DataFrame, model: str, yol: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], color=EKSEN, lw=1, ls="--")
    for renk, s in zip(KATEGORIK, SINIFLAR):
        fpr, tpr, _ = roc_curve(df["gercek"] == s, df[f"p_{s}"])
        ax.plot(fpr, tpr, color=renk, lw=2, label=f"{ADLAR[s]} (AUC {auc(fpr, tpr):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Yanlış pozitif oranı")
    ax.set_ylabel("Doğru pozitif oranı (duyarlılık)")
    ax.set_title(f"ROC eğrileri · {model}", loc="left")
    ax.grid(color=IZGARA, lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(yol, dpi=200)
    plt.close(fig)


def gruplu_cubuk(tablo: pd.DataFrame, seriler: list[str], baslik: str, yol: Path, renkler: list[str]) -> None:
    """Satirlar: Genel + siniflar; sutunlar: seriler (yuzde)."""
    x = np.arange(len(tablo))
    gen = 0.8 / len(seriler)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (seri, renk) in enumerate(zip(seriler, renkler)):
        degerler = tablo[seri].values
        cubuk = ax.bar(x + (k - (len(seriler) - 1) / 2) * gen, degerler, gen * 0.92, color=renk, label=seri,
                       edgecolor=YUZEY, linewidth=1)
        for b, v in zip(cubuk, degerler):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"%{v:.1f}", ha="center", fontsize=8.5, color=IKINCIL)
    ax.set_xticks(x, tablo.index)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Doğruluk (%)")
    ax.yaxis.grid(color=IZGARA, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(baslik, loc="left")
    ax.legend(frameon=False, ncol=len(seriler), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(yol, dpi=200)
    plt.close(fig)


def sinif_dogruluklari(df: pd.DataFrame) -> pd.Series:
    d = df.assign(ok=df["gercek"] == df["tahmin"])
    s = d.groupby("gercek")["ok"].mean().reindex(SINIFLAR) * 100
    s.index = [ADLAR[i] for i in s.index]
    return pd.concat([pd.Series({"Genel": d["ok"].mean() * 100}), s])


def main():
    stil()
    (CIKTI / "figures").mkdir(parents=True, exist_ok=True)

    modeller = {ad: oku(RAPOR / dosya) for ad, dosya in DOSYALAR.items() if (RAPOR / dosya).exists()}
    if "CNN v2" not in modeller:
        print("[HATA] reports/cnn_test_tahminleri_v2.csv bulunamadi. Ortak Drive'dan indirip reports/ altina koyun.")
        return
    for ad, dosya in DOSYALAR.items():
        print(f"  {'[VAR] ' if ad in modeller else '[YOK] '} {ad:<20} {dosya}")

    ozetler = {}
    for ad, df in modeller.items():
        kisa = ad.split(" (")[0].lower().replace(" ", "_")
        tablo, ozet = metrikler(df)
        ozetler[ad] = ozet
        tablo.round(4).to_csv(CIKTI / f"metrikler_{kisa}.csv", index=False)
        karisiklik_ciz(df, ad, CIKTI / "figures" / f"karisiklik_{kisa}.png").to_csv(CIKTI / f"karisiklik_{kisa}.csv")
        roc_ciz(df, ad, CIKTI / "figures" / f"roc_{kisa}.png")
        print(f"\n=== {ad} · {ozet['goruntu']} test goruntusu ===")
        print(tablo.round(3).to_string(index=False))
        print(f"Genel dogruluk %{ozet['dogruluk'] * 100:.1f} | makro F1 {ozet['makro_f1']:.3f} | makro AUC {ozet['makro_auc']:.3f}")

    pd.DataFrame(ozetler).T.round(4).to_csv(CIKTI / "ozet.csv")

    # Once-sonra: v1 (sizintili) vs v2 (temiz). Farkli test setleri; v1 sayilari gecersizdir.
    if "CNN v1 (geçersiz)" in modeller:
        os_tablo = pd.DataFrame({"v1 (kopyalı, geçersiz)": sinif_dogruluklari(modeller["CNN v1 (geçersiz)"]),
                                 "v2 (temiz)": sinif_dogruluklari(modeller["CNN v2"])})
        os_tablo["fark (puan)"] = os_tablo["v2 (temiz)"] - os_tablo["v1 (kopyalı, geçersiz)"]
        os_tablo.round(1).to_csv(CIKTI / "once_sonra.csv")
        gruplu_cubuk(os_tablo, ["v1 (kopyalı, geçersiz)", "v2 (temiz)"],
                     "CNN doğruluğu: veri temizliği öncesi ve sonrası", CIKTI / "figures" / "once_sonra.png",
                     [EKSEN, KATEGORIK[0]])
        print("\n=== Once-sonra (CNN) ===")
        print(os_tablo.round(1).to_string())

    # Model karsilastirmasi: ayni v2 test setinde CNN ve SVM
    if "SVM v2" in modeller:
        mk = pd.DataFrame({"CNN": sinif_dogruluklari(modeller["CNN v2"]),
                           "SVM": sinif_dogruluklari(modeller["SVM v2"])})
        mk.round(1).to_csv(CIKTI / "model_karsilastirma.csv")
        gruplu_cubuk(mk, ["CNN", "SVM"], "Aynı test setinde CNN ve SVM",
                     CIKTI / "figures" / "model_karsilastirma.png", KATEGORIK[:2])
        print("\n=== CNN vs SVM ===")
        print(mk.round(1).to_string())

    print(f"\n[TAMAM] Tablolar ve grafikler: {CIKTI}")


if __name__ == "__main__":
    main()
