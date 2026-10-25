"""
RETINA-AI: Toplu Oznitelik Cikarma Betigi (Batch Feature Extraction v2)
----------------------------------------------------------------------
Bu betik, `data/splits/veri_bolme_v2.csv` dosyasindaki tum gercek retinayi
tarayarak guncellenmis `oznitelik_cikar` fonksiyonu ile medikal oznitelikleri
cikarir ve sonuclari `data/oznitelikler_v2.csv` dosyasina kaydeder.

Kural:
    - Gorsel bulunamazsa sahte/sentetik resim cizilmez; dogrudan FileNotFoundError firlatilir.
    - `file_path` yerine `relative_path` kullanilarak platform bagimsiz arama yapilir.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pandas as pd

# Windows konsol UTF-8 uyumlulugu
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Proje kokunu sys.path'e ekle
proje_koku = Path(__file__).resolve().parent.parent
if str(proje_koku) not in sys.path:
    sys.path.append(str(proje_koku))

from src.k3_oznitelik import oznitelik_cikar


def veri_kokunu_tespit_et(proje_koku: Path, ozel_kök: Optional[str] = None) -> Optional[Path]:
    """
    Yerel dizinde, masaustunde veya Drive yapisinda ham goruntulerin bulundugu kok dizini arar.
    """
    if ozel_kök:
        p = Path(ozel_kök)
        if p.exists():
            return p

    adaylar = [
        Path(r"C:\Users\samil\Desktop\raw-20261025T183432Z-1-001\raw"),
        proje_koku / "data" / "raw",
        proje_koku / "archive" / "dataset",
        proje_koku / "dataset",
        proje_koku / "data" / "raw" / "dataset",
        proje_koku,
        Path("/content/dataset"),
        Path("/content/drive/MyDrive/retina-ai-ocular-disease-classification/data/raw"),
        Path("/content/drive/MyDrive/retina-ai-ocular-disease-classification"),
    ]

    for aday in adaylar:
        if aday.exists() and ((aday / "cataract").is_dir() or (aday / "normal").is_dir()):
            return aday
    return None


def toplu_oznitelik_cikar(
    csv_yolu: Path,
    cikti_yolu: Path,
    veri_koku: Optional[Path] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    veri_bolme_v2.csv tablosundaki tum gercek goruntuleri isleyerek oznitelikler_v2.csv tablosunu uretir.
    Eger herhangi bir fotograf bulunamazsa kesinlikle sahte resim uretilmez, dogrudan FileNotFoundError firlatilir.
    """
    if not csv_yolu.exists():
        raise FileNotFoundError(f"Veri bolme referans tablosu bulunamadi: {csv_yolu}")

    print(f"[BASLANGIC] CSV yukleniyor: {csv_yolu}")
    df = pd.read_csv(csv_yolu)
    toplam_kayit = len(df) if limit is None else min(limit, len(df))
    if limit:
        df = df.head(limit)

    print(f"[BILGI] Islenecek hedef ornek sayisi: {toplam_kayit}")

    # Veri kokunu tespit et
    if veri_koku is None:
        veri_koku = veri_kokunu_tespit_et(proje_koku)

    if veri_koku is None or not veri_koku.exists():
        raise FileNotFoundError(
            f"Ham gorsel kok dizini bulunamadi! Lutfen '--veri-koku' argumani ile gorsel klasorunu belirtin."
        )

    print(f"[BILGI] Ham veri kok dizini: {veri_koku}")

    kayitlar = []
    baslangic_zamani = time.time()
    basarili_sayisi = 0

    for idx, (_, satir) in enumerate(df.iterrows()):
        rel_yol = satir["relative_path"]
        sinif = satir["class_name"]

        # Gorsel dosyasini relative_path ile ara
        gorsel_yolu = veri_koku / rel_yol

        # file_path mevcutsa alternatif olarak kontrol et
        if not gorsel_yolu.exists() and "file_path" in satir and pd.notna(satir["file_path"]):
            aday_fp = Path(satir["file_path"])
            if aday_fp.exists():
                gorsel_yolu = aday_fp

        # Fotograf bulunamazsa kesin kontrol: HATA FIRLAT
        if not gorsel_yolu.exists():
            raise FileNotFoundError(
                f"[HATA] Fotograf bulunamadi: '{rel_yol}'\n"
                f"Aranan tam yol: {gorsel_yolu}\n"
                f"Sistem gercek veri seti ile calismaya ayarlanmistir; sentetik cizim yapilamaz."
            )

        # Gorseli oku
        img = cv2.imread(str(gorsel_yolu))
        if img is None:
            raise FileNotFoundError(
                f"[HATA] Fotograf dosyasi acilamadi veya bozuk: {gorsel_yolu}"
            )

        # Gercek medikal oznitelikleri hesapla
        # laplacian_varyansi, ortalama_kontrast, damar_orani, sari_lezyon_alani, sari_lezyon_sayisi, kanama_alani, kanama_sayisi
        oz = oznitelik_cikar(img, bgr_mi=True)
        basarili_sayisi += 1

        # Sonuclari topla
        kayit = {
            "image_name": satir["image_name"],
            "relative_path": rel_yol,
            "class_name": sinif,
            "patient_id": satir.get("patient_id", ""),
            "split": satir.get("split", ""),
            "laplacian_varyansi": oz["laplacian_varyansi"],
            "ortalama_kontrast": oz["ortalama_kontrast"],
            "damar_orani": oz["damar_orani"],
            "sari_lezyon_alani": oz["sari_lezyon_alani"],
            "sari_lezyon_sayisi": oz["sari_lezyon_sayisi"],
            "kanama_alani": oz["kanama_alani"],
            "kanama_sayisi": oz["kanama_sayisi"],
        }
        kayitlar.append(kayit)

        if (idx + 1) % 200 == 0 or (idx + 1) == toplam_kayit:
            gecen = time.time() - baslangic_zamani
            hiz = (idx + 1) / gecen
            print(f"  -> [{idx + 1}/{toplam_kayit}] islendi (%{(idx + 1) / toplam_kayit * 100:.1f}) - Hiz: {hiz:.1f} gorsel/sn")

    df_oznitelikler = pd.DataFrame(kayitlar)

    # Hedef klasoru olustur ve CSV kaydet
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    df_oznitelikler.to_csv(cikti_yolu, index=False, encoding="utf-8")

    toplam_sure = time.time() - baslangic_zamani
    print("\n" + "=" * 65)
    print(f"[TAMAMLANDI] Oznitelikler basariyla kaydedildi:")
    print(f"  Hedef dosya      : {cikti_yolu}")
    print(f"  Basariyla islenen: {basarili_sayisi} / {toplam_kayit} gercek gorsel")
    print(f"  Toplam sure      : {toplam_sure:.2f} saniye ({basarili_sayisi/toplam_sure:.1f} gorsel/sn)")
    print("=" * 65)

    # Ozet dagilim
    print("\n--- OZNITELIK ORTALAMALARI (SINIF BAZINDA GERCEK VERI) ---")
    ozet = df_oznitelikler.groupby("class_name")[
        ["laplacian_varyansi", "ortalama_kontrast", "damar_orani", "sari_lezyon_alani", "kanama_alani"]
    ].mean()
    print(ozet.round(3))
    print("=" * 65 + "\n")

    return df_oznitelikler


def main():
    parser = argparse.ArgumentParser(description="RETINA-AI Toplu Oznitelik Cikarma Betigi (v2)")
    parser.add_argument(
        "--csv-yolu",
        type=Path,
        default=proje_koku / "data" / "splits" / "veri_bolme_v2.csv",
        help="Girdi referans tablosu (varsayilan: data/splits/veri_bolme_v2.csv)"
    )
    parser.add_argument(
        "--cikti-yolu",
        type=Path,
        default=proje_koku / "data" / "oznitelikler_v2.csv",
        help="Cikti CSV dosyasi (varsayilan: data/oznitelikler_v2.csv)"
    )
    parser.add_argument(
        "--veri-koku",
        type=Path,
        default=None,
        help="Ham gorsellerin kok dizini (varsayilan: otomatik tespit)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Test icin islenecek maksimum ornek sayisi"
    )

    args = parser.parse_args()
    toplu_oznitelik_cikar(
        csv_yolu=args.csv_yolu,
        cikti_yolu=args.cikti_yolu,
        veri_koku=args.veri_koku,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
