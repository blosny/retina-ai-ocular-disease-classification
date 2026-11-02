"""
RETINA-AI (Kisi 2): Anatomi testi - v2 veri bolmesi

data/splits/veri_bolme_v2.csv dosyasinin TEST kismindan her siniftan 5 goruntu
(random_state=42) secer; her biri icin damar yogunlugu, optik disk ve CDR hesaplar.

Cikti:
  reports/anatomi_sonuclari_v2.csv   (dosya, sinif, orijinal_boyut, damar_yogunlugu, disk_bulundu, cdr)
  reports/figures/anatomi_ornekleri_v2.png
Eski reports/anatomi_sonuclari.csv'ye dokunulmaz.

Calistirma (proje kokunden):
  python src/anatomi_testi.py
  python src/anatomi_testi.py --goruntu_dizin "C:/projects/.../archive/dataset"   # goruntuler baska yerdeyse
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import damar_haritasi, damar_yogunlugu, disk_isaretle, optik_disk  # noqa: E402
from preprocessing import boyutlandir, on_isle, roi_kirp  # noqa: E402

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOLME_CSV = os.path.join(KOK, "data", "splits", "veri_bolme_v2.csv")
SONUC_CSV = os.path.join(KOK, "reports", "anatomi_sonuclari_v2.csv")
GORSEL_YOLU = os.path.join(KOK, "reports", "figures", "anatomi_ornekleri_v2.png")
SINIF_SIRASI = ["normal", "diabetic_retinopathy", "glaucoma", "cataract"]


def goruntu_yolu_bul(satir, ek_dizin=None):
    """Goruntuyu bilinen yerlerde arar, bulunan ilk yolu dondurur (yoksa None)."""
    adaylar = []
    if ek_dizin:
        adaylar.append(os.path.join(ek_dizin, satir["relative_path"]))
    adaylar.append(os.path.join(KOK, "data", "splits", "test", satir["class_name"], satir["image_name"]))
    adaylar.append(os.path.join(KOK, "archive", "dataset", satir["relative_path"]))
    adaylar.append(str(satir["file_path"]))  # CSV'deki tam yol (kendi bilgisayarinda calisir)
    for yol in adaylar:
        if os.path.isfile(yol):
            return yol
    return None


def ornekleri_sec(df, sinif_basina=5, sadece_mevcut=False, ek_dizin=None):
    """Test kismindan her siniftan sinif_basina goruntuyu sabit tohumla secer."""
    test = df[df["split"] == "test"].copy()
    if test.empty:
        sys.exit("HATA: CSV'de split == 'test' satiri yok.")
    if sadece_mevcut:  # sadece dosyasi bilgisayarda olanlar arasindan sec (deneme icin)
        test = test[test.apply(lambda s: goruntu_yolu_bul(s, ek_dizin) is not None, axis=1)]
    test = test.sort_values(["class_name", "image_name"]).reset_index(drop=True)  # siralama sabit olsun
    parcalar = []
    for _, grup in test.groupby("class_name"):  # apply yerine dongu: pandas surumunden bagimsiz
        parcalar.append(grup.sample(n=min(sinif_basina, len(grup)), random_state=42))
    secilen = pd.concat(parcalar, ignore_index=True)
    secilen["_sira"] = secilen["class_name"].map({s: i for i, s in enumerate(SINIF_SIRASI)})
    return secilen.sort_values(["_sira", "image_name"]).drop(columns="_sira").reset_index(drop=True)


def goruntu_analiz(yol):
    """Tek goruntu icin damar yogunlugu, disk ve gorsel kutulari (uygulamadaki akisla ayni)."""
    bgr = cv2.imread(yol)
    if bgr is None:
        raise ValueError(f"Goruntu okunamadi: {yol}")
    orijinal_boyut = int(bgr.shape[1])

    islenmis = on_isle(bgr)
    renkli = cv2.cvtColor(boyutlandir(roi_kirp(bgr)), cv2.COLOR_BGR2RGB)

    damarlar = damar_haritasi(islenmis)
    yogunluk = damar_yogunlugu(damarlar, islenmis)
    disk = optik_disk(renkli)
    return orijinal_boyut, yogunluk, disk, damarlar, renkli


def ozet_gorsel_ciz(kayitlar, yol):
    """Her goruntu icin yan yana: damar haritasi | disk isareti. Sinif basina bir satir grubu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(kayitlar)
    sutun = 5
    satir = int(np.ceil(n / sutun))
    fig, eksenler = plt.subplots(satir * 2, sutun, figsize=(3.2 * sutun, 3.4 * satir * 2))
    eksenler = np.atleast_2d(eksenler)
    for ax in eksenler.ravel():
        ax.axis("off")

    for i, k in enumerate(kayitlar):
        r, c = (i // sutun) * 2, i % sutun
        ax_damar, ax_disk = eksenler[r, c], eksenler[r + 1, c]
        ax_damar.imshow(k["damarlar"], cmap="gray", vmin=0, vmax=255)
        ax_damar.set_title(f"{k['sinif']}\n{k['dosya']}\nyogunluk={k['yogunluk']:.3f}", fontsize=7)
        ax_disk.imshow(disk_isaretle(k["renkli"], k["disk"]))
        durum = "disk yok" if k["disk"] is None else f"CDR={k['disk']['cukur_disk_orani']:.2f}"
        ax_disk.set_title(durum, fontsize=8, color="crimson" if k["disk"] is None else "black")

    fig.suptitle("RETINA-AI anatomi testi (v2 bolmesi, test kismi)  -  ust: damar haritasi, alt: optik disk (yesil) ve cukur (sari)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    try:
        fig.savefig(yol, dpi=130)
    except OSError:  # dosya acik/kilitli (goruntuleyici, OneDrive) olabilir: yan dosyaya yaz
        yol = yol.replace(".png", "_yeni.png")
        print(f"UYARI: ilk dosya yazilamadi (acik/kilitli olabilir), su dosyaya kaydedildi: {yol}")
        fig.savefig(yol, dpi=130)
    plt.close(fig)
    return yol


def ozet_yazdir(sonuc):
    print("\n=== Sinif bazinda ozet ===")
    print(f"{'sinif':24s} {'goruntu':>8s} {'disk bulunan':>13s} {'bulma orani':>12s} {'ort. CDR':>9s} {'ort. damar yog.':>16s}")
    for sinif in SINIF_SIRASI:
        g = sonuc[sonuc["sinif"] == sinif]
        if g.empty:
            continue
        bulunan = int(g["disk_bulundu"].sum())
        cdr = g["cdr"].dropna()
        cdr_metin = f"{cdr.mean():.2f}" if len(cdr) else "-"
        print(f"{sinif:24s} {len(g):8d} {bulunan:13d} {bulunan / len(g):12.0%} {cdr_metin:>9s} {g['damar_yogunlugu'].mean():16.4f}")
    toplam = int(sonuc["disk_bulundu"].sum())
    print(f"{'TOPLAM':24s} {len(sonuc):8d} {toplam:13d} {toplam / len(sonuc):12.0%}")


def main():
    ap = argparse.ArgumentParser(description="Anatomi testi (v2 bolmesi)")
    ap.add_argument("--sinif_basina", type=int, default=5)
    ap.add_argument("--goruntu_dizin", default=None, help="Goruntu klasoru (icinde cataract/, glaucoma/ ... olan)")
    ap.add_argument("--sadece_mevcut", action="store_true",
                    help="Sadece dosyasi bilgisayarda bulunan goruntulerden sec (deneme icin)")
    ap.add_argument("--cikti_csv", default=SONUC_CSV)
    ap.add_argument("--cikti_gorsel", default=GORSEL_YOLU)
    args = ap.parse_args()

    if not os.path.isfile(BOLME_CSV):
        sys.exit(f"HATA: {BOLME_CSV} bulunamadi.")
    df = pd.read_csv(BOLME_CSV)
    secilen = ornekleri_sec(df, args.sinif_basina, args.sadece_mevcut, args.goruntu_dizin)
    print(f"{len(secilen)} goruntu secildi (random_state=42).")

    satirlar, kayitlar, eksik = [], [], []
    for _, s in secilen.iterrows():
        yol = goruntu_yolu_bul(s, args.goruntu_dizin)
        if yol is None:
            eksik.append(s["image_name"])
            continue
        boyut, yogunluk, disk, damarlar, renkli = goruntu_analiz(yol)
        bulundu = disk is not None
        cdr = disk["cukur_disk_orani"] if bulundu else np.nan
        satirlar.append({
            "dosya": s["image_name"], "sinif": s["class_name"], "orijinal_boyut": boyut,
            "damar_yogunlugu": round(float(yogunluk), 4), "disk_bulundu": bulundu, "cdr": cdr,
        })
        kayitlar.append({"dosya": s["image_name"], "sinif": s["class_name"], "yogunluk": yogunluk,
                         "disk": disk, "damarlar": damarlar, "renkli": renkli})
        print(f"  {s['class_name']:22s} {s['image_name']:22s} yogunluk={yogunluk:.4f} "
              f"disk={'var' if bulundu else 'YOK'} cdr={cdr if bulundu else '-'}")

    if eksik:
        print(f"\nUYARI: {len(eksik)} goruntu bulunamadi (atlandi): {eksik}")
        print("       --goruntu_dizin ile goruntu klasorunu belirtebilirsin.")
    if not satirlar:
        sys.exit("HATA: hic goruntu islenemedi.")

    sonuc = pd.DataFrame(satirlar)
    os.makedirs(os.path.dirname(args.cikti_csv), exist_ok=True)
    sonuc.to_csv(args.cikti_csv, index=False)
    print(f"\nKaydedildi: {args.cikti_csv}")
    ozet_yazdir(sonuc)  # tablo, gorsel hata verse bile ekrana ciksin diye once yazdirilir
    gorsel = ozet_gorsel_ciz(kayitlar, args.cikti_gorsel)
    print(f"\nKaydedildi: {gorsel}")


if __name__ == "__main__":
    main()
