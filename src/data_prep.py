"""
Veri Dogrulama, Temizleme, Esitleme ve Bolme Pipeline'i
------------------------------------------------------
Bu dosya ham retina veri setini tarar; bozuk dosyalari ve siniflar arasi 
kopya gorselleri ayiklar. Her sinifi 1007 gorsele esitleyerek %70 egitim,
%15 dogrulama ve %15 test kumelerine (stratified) boler.
Cikti olarak 'veri_bolme.csv' ve analiz grafiklerini uretir.
"""

import hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def dosya_md5_hesapla(dosya_yolu: Path, blok_boyutu: int = 65536) -> str:
    """
    Dosyanin bayt iceriginden 32 karakterlik benzersiz MD5 ozeti uretir.
    Gorsel isimleri farkli olsa bile ayni fotograflari tespit etmek icin kullanilir.
    """
    hasher = hashlib.md5()
    with open(dosya_yolu, "rb") as dosya:
        while tampon := dosya.read(blok_boyutu):
            hasher.update(tampon)
    return hasher.hexdigest()


def veri_setini_tara_ve_dogrula(veri_dizini: Path, onbellek_yolu: Path = None) -> pd.DataFrame:
    """
    archive/dataset altindaki tum goruntuleri OpenCV ile tek tek acar.
    Okunamayan (bozuk) dosyalari raporlar; gecerli olanlarin boyut ve MD5 bilgilerini cikarir.
    Hiz icin metadata onbellek dosyasindan (raw_metadata_cache.csv) okunabilir.
    """
    if onbellek_yolu and onbellek_yolu.exists():
        print(f"[BILGI] Metadata onbellekten yuklendi: {onbellek_yolu.name}")
        return pd.read_csv(onbellek_yolu)

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
    Kaggle veri setindeki siniflar arasi capraz kopya (label contamination) sorununu cozer.
    Ayni hash'e sahip '625_left.jpg' ve '1415_right.jpg' gorselleri hem katarakt hem glokomda yer alir.
    Glokomun ornek sayisi kritik (1007) oldugundan, bu cakisiklar fazla ornegi olan katarakttan elenir.
    """
    cakisiklar = df[df.duplicated(subset=["md5_hash"], keep=False)]
    if len(cakisiklar) > 0:
        cikisik_cataract = df[(df["class_name"] == "cataract") & (df["md5_hash"].isin(cakisiklar["md5_hash"]))].index
        df_temiz = df.drop(index=cikisik_cataract).copy()
        print(f"[BILGI] {len(cikisik_cataract)} capraz cakisik ornek katarakttan elendi (Veri sizintisi onlendi).")
    else:
        df_temiz = df.copy()

    return df_temiz


def siniflari_esitle(df: pd.DataFrame, hedef_sayi: int = 1007, seed: int = 42) -> pd.DataFrame:
    """
    Her siniftan seed=42 ile rastgele tam 1007 ornek secerek veri setini dengeler.
    Boylece model egitiminde cok sayida ornegi olan siniflara kayma (bias) onlenir.
    """
    parcalar = []
    for sinif_adi, grup in df.groupby("class_name"):
        if len(grup) < hedef_sayi:
            raise ValueError(f"'{sinif_adi}' sinifi ({len(grup)}) hedef sayidan ({hedef_sayi}) yetersiz!")
        secilenler = grup.sample(n=hedef_sayi, random_state=seed)
        parcalar.append(secilenler)

    return pd.concat(parcalar, ignore_index=True)


def stratified_veri_bolme(
    df: pd.DataFrame,
    train_sayi: int = 705,
    val_sayi: int = 151,
    test_sayi: int = 151,
    seed: int = 42
) -> pd.DataFrame:
    """
    Her sinifi orantisal olarak 705 Egitim (%70), 151 Dogrulama (%15), 151 Test (%15) olarak ayirir.
    Sinif dengesi her uc kumedede birebir korunur (705 + 151 + 151 = 1007).
    """
    bolunmus_listeler = []
    for _, grup in df.groupby("class_name"):
        karisik = grup.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_part = karisik.iloc[:train_sayi].copy()
        train_part["split"] = "train"

        val_part = karisik.iloc[train_sayi : train_sayi + val_sayi].copy()
        val_part["split"] = "val"

        test_part = karisik.iloc[train_sayi + val_sayi : train_sayi + val_sayi + test_sayi].copy()
        test_part["split"] = "test"

        bolunmus_listeler.extend([train_part, val_part, test_part])

    sonuc = pd.concat(bolunmus_listeler, ignore_index=True)
    return sonuc.sort_values(by=["class_name", "split", "image_name"]).reset_index(drop=True)


def dagilim_grafigi_ciz(df: pd.DataFrame, cikti_yolu: Path):
    """
    Sinif ve kume (train/val/test) ornek sayilarini gosteren bar grafigi uretir.
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

    plt.title("RETINA-AI: Veri Seti Sinif ve Kume Dagilimi (%70 Train, %15 Val, %15 Test)", fontsize=12, pad=12)
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
    plt.savefig(cikti_yolu, dpi=300)
    plt.close()


def cozunurluk_dagilimi_ciz(df: pd.DataFrame, cikti_yolu: Path):
    """
    Ham retina gorsellerinin genislik ve yukseklik (cozunurluk) dagilimini
    sinif bazinda gosteren akademik sacilim grafigi cizer.
    """
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="width",
        y="height",
        hue="class_name",
        alpha=0.6,
        s=30,
        palette="tab10"
    )

    plt.title("RETINA-AI: Ham Gorsellerin Cozunurluk (Genislik x Yukseklik) Dagilimi", fontsize=12, pad=12)
    plt.xlabel("Genislik (Piksel)", fontsize=10)
    plt.ylabel("Yukseklik (Piksel)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Sinif", frameon=True)
    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=300)
    plt.close()


def ana_akisi_calistir():
    """
    Tum pipeline adimlarini calistirir ve konsola formatli terminal ciktisi verir.
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
    grafik_cozunurluk = fig_dizini / "cozunurluk_dagilimi.png"

    print("+" + "-" * 68 + "+")
    print("|      RETINA-AI: VERI VE ON ISLEME SORUMLUSU - VERI BOLME HATTI      |")
    print("+" + "-" * 68 + "+")

    # 1. Dogrulama
    df_ham = veri_setini_tara_ve_dogrula(ham_dizin, onbellek_yolu=onbellek)

    # 2. Capraz cakisiklari temizleme
    df_temiz = capraz_ve_mukerrer_temizle(df_ham)

    # 3. 1007'ye esitleme
    df_dengeli = siniflari_esitle(df_temiz, hedef_sayi=1007, seed=42)

    # 4. Stratified bolme
    df_bolunmus = stratified_veri_bolme(
        df_dengeli,
        train_sayi=705,
        val_sayi=151,
        test_sayi=151,
        seed=42
    )

    # 5. CSV kaydet
    df_bolunmus.to_csv(csv_cikti, index=False, encoding="utf-8")
    print(f"\n[BASARILI] 'veri_bolme.csv' olusturuldu -> {csv_cikti.name}")
    print(f"Toplam secilen gorsel sayisi: {len(df_bolunmus)} (Sinif basina 1007)")

    # 6. Konsol ozet tablosu
    ozet = pd.crosstab(df_bolunmus["class_name"], df_bolunmus["split"], margins=True)
    print("\n" + "=" * 50)
    print("           VERI SETI BOLUM DAGILIM TABLOSU")
    print("=" * 50)
    print(ozet)
    print("=" * 50 + "\n")

    # 7. Grafikler
    dagilim_grafigi_ciz(df_bolunmus, grafik_dagilim)
    print(f"[BILGI] Grafik kaydedildi: {grafik_dagilim.name}")

    cozunurluk_dagilimi_ciz(df_bolunmus, grafik_cozunurluk)
    print(f"[BILGI] Grafik kaydedildi: {grafik_cozunurluk.name}")

    print("\n[TAMAMLANDI] Pipeline basariyla calisti. Ekran goruntusu alinabilir.")


if __name__ == "__main__":
    ana_akisi_calistir()
