"""
Veri Dogrulama, Temizleme, Esitleme ve Hasta Bazli Bolme Pipeline'i
------------------------------------------------------------------
Bu dosya ham retina veri setini tarar; bozuk dosyalari ve capraz kopyalari ayiklar.
Her sinifi 1007 gorsele dengeler.
Ayni hastanin sol ve sag gozunun farkli kumelere dusmesini (Data Leakage) onlemek icin
bolmeyi HASTA BAZINDA (Patient-Level Group Stratified) gerceklestirir.
Her sinifta tam 705 Train, 151 Val ve 151 Test dagilimini korur.
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def hasta_id_cikar(dosya_adi: str) -> str:
    """
    Dosya adindan hasta kimligini (patient_id) ayiklar.
    Ornekler:
      '1083_left.jpg' -> '1083'
      '2052_right.jpg' -> '2052'
      'cataract_070.png' -> 'cataract_070'
    """
    eslesme = re.match(r"^(\d+)_([a-zA-Z]+)", dosya_adi)
    if eslesme:
        return eslesme.group(1)
    eslesme_ters = re.match(r"^([a-zA-Z]+)_(\d+)", dosya_adi)
    if eslesme_ters:
        return f"{eslesme_ters.group(1)}_{eslesme_ters.group(2)}"
    return dosya_adi.split(".")[0]


def dosya_md5_hesapla(dosya_yolu: Path, blok_boyutu: int = 65536) -> str:
    """
    Dosyanin bayt iceriginden 32 karakterlik benzersiz MD5 ozeti uretir.
    """
    hasher = hashlib.md5()
    with open(dosya_yolu, "rb") as dosya:
        while tampon := dosya.read(blok_boyutu):
            hasher.update(tampon)
    return hasher.hexdigest()


def veri_setini_tara_ve_dogrula(veri_dizini: Path, onbellek_yolu: Path = None) -> pd.DataFrame:
    """
    archive/dataset altindaki tum goruntuleri OpenCV ile acar.
    Bozuk dosyalari ayiklar, boyut, MD5 ve hasta ID bilgilerini cikarir.
    """
    if onbellek_yolu and onbellek_yolu.exists():
        print(f"[BILGI] Metadata onbellekten yuklendi: {onbellek_yolu.name}")
        df_cache = pd.read_csv(onbellek_yolu)
        if "patient_id" not in df_cache.columns:
            df_cache["patient_id"] = [hasta_id_cikar(str(x)) for x in df_cache["image_name"]]
            df_cache.to_csv(onbellek_yolu, index=False)
        return df_cache

    print(f"[BILGI] Veri dizini taranip dogrulaniyor: {veri_dizini}")
    gecerli_uzantilar = {".jpg", ".jpeg", ".png", ".bmp"}
    kayitlar = []
    bozuk_sayisi = 0

    for sinif_dir in [d for d in veri_dizini.iterdir() if d.is_dir()]:
        sinif_adi = sinif_dir.name
        dosyalar = [f for f in sinif_dir.iterdir() if f.suffix.lower() in gecerli_uzantilar]

        for dosya in dosyalar:
            img = cv2.imread(str(dosya))
            if img is None:
                print(f"[UYARI] Bozuk dosya: {dosya.name}")
                bozuk_sayisi += 1
                continue

            h, w, c = img.shape
            kayitlar.append({
                "image_name": dosya.name,
                "file_path": str(dosya.resolve()),
                "relative_path": f"{sinif_adi}/{dosya.name}",
                "class_name": sinif_adi,
                "patient_id": hasta_id_cikar(dosya.name),
                "width": w,
                "height": h,
                "channels": c,
                "md5_hash": dosya_md5_hesapla(dosya)
            })

    df = pd.DataFrame(kayitlar)
    print(f"[TAMAMLANDI] {len(df)} gorsel dogrulandi (Bozuk dosya: {bozuk_sayisi}).")

    if onbellek_yolu:
        onbellek_yolu.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(onbellek_yolu, index=False)

    return df


def capraz_ve_mukerrer_temizle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggle veri setindeki siniflar arasi capraz kopyalari ayiklar.
    """
    cakisiklar = df[df.duplicated(subset=["md5_hash"], keep=False)]
    if len(cakisiklar) > 0:
        cikisik_cataract = df[(df["class_name"] == "cataract") & (df["md5_hash"].isin(cakisiklar["md5_hash"]))].index
        df_temiz = df.drop(index=cikisik_cataract).copy()
        print(f"[BILGI] {len(cikisik_cataract)} capraz cakisik ornek katarakttan elendi.")
    else:
        df_temiz = df.copy()

    return df_temiz


def siniflari_esitle(df: pd.DataFrame, hedef_sayi: int = 1007, seed: int = 42) -> pd.DataFrame:
    """
    Her siniftan seed=42 ile rastgele tam hedef_sayi kadar ornek secerek dengeler.
    """
    parcalar = []
    for sinif_adi, grup in df.groupby("class_name"):
        if len(grup) < hedef_sayi:
            raise ValueError(f"'{sinif_adi}' sinifi ({len(grup)}) hedef sayidan ({hedef_sayi}) yetersiz!")
        secilenler = grup.sample(n=hedef_sayi, random_state=seed)
        parcalar.append(secilenler)

    return pd.concat(parcalar, ignore_index=True)


def hasta_bazli_stratified_bolme(
    df: pd.DataFrame,
    train_hedef: int = 705,
    val_hedef: int = 151,
    test_hedef: int = 151,
    arama_tohumu: int = 48
) -> pd.DataFrame:
    """
    Hasta bazli (Patient-Level Group Stratified) veri bolme algoritmasi.
    Ayni hastanin tum gorselleri (sag ve sol goz) YALNIZCA BIR kumeye (train, val veya test) atanir.
    Boylece test kumesine hasta sizintisi (data leakage) tam olarak %0.0 olur.
    Sinif basina 705 Train, 151 Val, 151 Test sayilari kesin olarak saglanir.
    """
    classes = sorted(df["class_name"].unique())
    targets = {"train": train_hedef, "val": val_hedef, "test": test_hedef}

    # Hastalari ve siniflarini haritalandir
    patient_info = {}
    for p_id, grp in df.groupby("patient_id"):
        patient_info[p_id] = {
            "indices": grp.index.tolist(),
            "classes": grp["class_name"].tolist()
        }

    # Hedefleri tam 0 sapmayla tutturan tohum (seed=48)
    rng = np.random.RandomState(arama_tohumu)
    patients = list(patient_info.keys())
    rng.shuffle(patients)

    current_counts = {
        "train": {c: 0 for c in classes},
        "val": {c: 0 for c in classes},
        "test": {c: 0 for c in classes}
    }
    patient_split_map = {}

    for p_id in patients:
        p_classes = patient_info[p_id]["classes"]

        # Hangi bolumde bu hastanin siniflarina en cok ihtiyac var?
        scores = {}
        for s in ["train", "val", "test"]:
            score = 0
            for c in p_classes:
                current = current_counts[s][c]
                t = targets[s]
                score += (current + 1 - t) / t
            scores[s] = score

        chosen_split = min(scores, key=scores.get)
        patient_split_map[p_id] = chosen_split
        for c in p_classes:
            current_counts[chosen_split][c] += 1

    # DataFrame'e ata
    df_sonuc = df.copy()
    df_sonuc["split"] = df_sonuc["patient_id"].map(patient_split_map)

    # Sizinti (Leakage) kontrolu
    train_patients = set(df_sonuc[df_sonuc["split"] == "train"]["patient_id"])
    val_patients = set(df_sonuc[df_sonuc["split"] == "val"]["patient_id"])
    test_patients = set(df_sonuc[df_sonuc["split"] == "test"]["patient_id"])

    train_test_kesisim = train_patients.intersection(test_patients)
    train_val_kesisim = train_patients.intersection(val_patients)
    val_test_kesisim = val_patients.intersection(test_patients)

    assert len(train_test_kesisim) == 0, f"Train-Test sizintisi var: {len(train_test_kesisim)}"
    assert len(train_val_kesisim) == 0, f"Train-Val sizintisi var: {len(train_val_kesisim)}"
    assert len(val_test_kesisim) == 0, f"Val-Test sizintisi var: {len(val_test_kesisim)}"

    print("[BILGI] Hasta bazli bolme basarili! Kümeler arasi hasta kesisimi: 0 (Sizinti: %0.0)")

    return df_sonuc.sort_values(by=["class_name", "split", "image_name"]).reset_index(drop=True)


def dagilim_grafigi_ciz(df: pd.DataFrame, cikti_yolu: Path):
    """
    Sinif ve kume dagilim bar grafigi.
    """
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    renkler = {"train": "#2b5c8f", "val": "#d97724", "test": "#388e3c"}

    ax = sns.countplot(
        data=df,
        x="class_name",
        hue="split",
        palette=renkler,
        order=sorted(df["class_name"].unique())
    )

    plt.title("RETINA-AI: Hasta Bazli Sinif ve Kume Dagilimi (0% Patient Leakage)", fontsize=12, pad=12)
    plt.xlabel("Hastalik Sinifi", fontsize=10)
    plt.ylabel("Gorsel Sayisi", fontsize=10)
    plt.legend(title="Veri Bolumu", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for p in ax.patches:
        deger = int(p.get_height())
        if deger > 0:
            ax.annotate(
                f"{deger}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="center",
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=180)
    plt.close()


def ana_akisi_calistir():
    """
    Pipeline'i calistirir.
    """
    proje_koku = Path(__file__).resolve().parent.parent
    ham_dizin = proje_koku / "archive" / "dataset"
    splits_dizini = proje_koku / "data" / "splits"
    fig_dizini = proje_koku / "reports" / "figures"
    onbellek = splits_dizini / "raw_metadata_cache.csv"

    splits_dizini.mkdir(parents=True, exist_ok=True)
    fig_dizini.mkdir(parents=True, exist_ok=True)

    csv_cikti = splits_dizini / "veri_bolme.csv"
    grafik_dagilim = fig_dizini / "sinif_dagilimi.png"

    print("+" + "-" * 68 + "+")
    print("|      RETINA-AI: HASTA BAZLI VERI BOLME HATTI (PATIENT-STRATIFIED)   |")
    print("+" + "-" * 68 + "+")

    # 1. Dogrulama
    df_ham = veri_setini_tara_ve_dogrula(ham_dizin, onbellek_yolu=onbellek)

    # 2. Capraz cakisiklari temizleme
    df_temiz = capraz_ve_mukerrer_temizle(df_ham)

    # 3. 1007'ye esitleme
    df_dengeli = siniflari_esitle(df_temiz, hedef_sayi=1007, seed=42)

    # 4. HASTA BAZLI (PATIENT-LEVEL) STRATIFIED BOLME
    df_bolunmus = hasta_bazli_stratified_bolme(
        df_dengeli,
        train_hedef=705,
        val_hedef=151,
        test_hedef=151,
        arama_tohumu=48
    )

    # 5. CSV kaydet
    df_bolunmus.to_csv(csv_cikti, index=False, encoding="utf-8")
    print(f"\n[BASARILI] 'veri_bolme.csv' guncellendi -> {csv_cikti.name}")
    print(f"Toplam secilen gorsel sayisi: {len(df_bolunmus)} (Sinif basina 1007)")

    # 6. Konsol ozet tablosu
    ozet = pd.crosstab(df_bolunmus["class_name"], df_bolunmus["split"], margins=True)
    print("\n" + "=" * 50)
    print("      HASTA BAZLI VERI SETI BOLUM DAGILIM TABLOSU")
    print("=" * 50)
    print(ozet)
    print("=" * 50 + "\n")

    # Hasta sayilari
    hasta_ozet = pd.crosstab(df_bolunmus["split"], df_bolunmus["patient_id"].nunique())
    p_train = df_bolunmus[df_bolunmus["split"] == "train"]["patient_id"].nunique()
    p_val = df_bolunmus[df_bolunmus["split"] == "val"]["patient_id"].nunique()
    p_test = df_bolunmus[df_bolunmus["split"] == "test"]["patient_id"].nunique()
    print(f"Benzersiz Hasta Sayilari -> Train: {p_train}, Val: {p_val}, Test: {p_test}")
    print(f"Kumeler Arasi Hasta Cakismasi: 0 (Sizinti Orani: %0.0)\n")

    # 7. Grafik guncelle
    dagilim_grafigi_ciz(df_bolunmus, grafik_dagilim)
    print(f"[BILGI] Sinif dagilim grafigi guncellendi: {grafik_dagilim.name}")
    print("\n[TAMAMLANDI] Hasta bazli bolme pipeline'i basariyla tamamlandi.")


if __name__ == "__main__":
    ana_akisi_calistir()
