"""
Goruntu On Isleme Modulu (on_isle Pipeline)
------------------------------------------
Bu modul, goz dibi (fundus) goruntulerine medikal on isleme adimlarini uygular:
1. Otomatik ROI kirpma (Siyah kenarliklari atma ve retinayi merkezleme).
2. Standart boyutlandirma (512x512).
3. Yesil kanal izolasyonu (Damar ve lezyon kontrastini maksimize etme).
4. CLAHE (Kontrast sinirli adaptif histogram esitleme).
5. Median filtre (Gurultu azaltma).
6. [0, 1] piksel normalizasyonu.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def roi_kirp(gorsel: np.ndarray, esik: int = 10, dairesel_maske: bool = True) -> np.ndarray:
    """
    Retina gorselinin etrafindaki gereksiz siyah kenarliklari otomatik kirpar.
    Parametreler:
        gorsel: BGR formatinda giris gorseli.
        esik: Siyah piksel siniri (varsayilan: 10).
        dairesel_maske: Koselerdeki kalan siyah kalintilari yuvarlatilmis maskeyle temizler.
    """
    gri = cv2.cvtColor(gorsel, cv2.COLOR_BGR2GRAY) if gorsel.ndim == 3 else gorsel
    maske = gri > esik

    if not np.any(maske):
        return gorsel

    # Maske sinirlarini bul
    satirlar = np.any(maske, axis=1)
    sutunlar = np.any(maske, axis=0)
    ymin, ymax = np.where(satirlar)[0][[0, -1]]
    xmin, xmax = np.where(sutunlar)[0][[0, -1]]

    kirpilmis = gorsel[ymin:ymax + 1, xmin:xmax + 1]

    if dairesel_maske:
        h, w = kirpilmis.shape[:2]
        cember_maske = np.zeros((h, w), dtype=np.uint8)
        merkez = (w // 2, h // 2)
        yari_cap = min(w, h) // 2
        cv2.circle(cember_maske, merkez, yari_cap, 255, -1)

        if kirpilmis.ndim == 3:
            kirpilmis = cv2.bitwise_and(kirpilmis, kirpilmis, mask=cember_maske)
        else:
            kirpilmis = cv2.bitwise_and(kirpilmis, cember_maske)

    return kirpilmis


def boyutlandir(gorsel: np.ndarray, hedef_boyut: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Gorseli belirlenen piksel boyutuna (varsayilan: 512x512) boyutlandirir.
    """
    return cv2.resize(gorsel, hedef_boyut, interpolation=cv2.INTER_AREA)


def yesil_kanali_izole_et(gorsel: np.ndarray) -> np.ndarray:
    """
    BGR renk uzayindaki retinadan 1. indeksteki Yesil (Green) kanali ayirir.
    Yesil kanal, hemoglobin emilimi nedeniyle damar ve lezyonlarin en belirgin oldugu kanaldir.
    """
    if gorsel.ndim == 3:
        return gorsel[:, :, 1]
    return gorsel


def clahe_uygula(
    tek_kanal: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Kontrast Sinirli Adaptif Histogram Esitleme (CLAHE) uygular.
    Asiri parlak bolgelerde doygunlugu (over-amplification) onleyerek lokal kontrast saglar.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(tek_kanal)


def gurultu_azalt(tek_kanal: np.ndarray, cekirdek_boyutu: int = 3) -> np.ndarray:
    """
    Median filtre ile kenar keskinligini bozmadan tuz-biber ve sensor gurultulerini temizler.
    """
    return cv2.medianBlur(tek_kanal, cekirdek_boyutu)


def normalize_et(gorsel: np.ndarray) -> np.ndarray:
    """
    Piksel degerlerini [0, 255] araligindan makine ogrenmesine uygun [0.0, 1.0] araligina olcekler.
    """
    return gorsel.astype(np.float32) / 255.0


def on_isle(
    girdi: Union[str, Path, np.ndarray],
    hedef_boyut: Tuple[int, int] = (512, 512),
    normalize: bool = True,
    ara_adimlari_ver: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Tum on isleme adimlarini uctan uca calistiran ana fonksiyon.
    
    Parametreler:
        girdi: Gorsel dosya yolu (str/Path) veya yuklenmis BGR numpy matrisi.
        hedef_boyut: Cikti piksel cozunurlugu (varsayilan: 512x512).
        normalize: Ciktinin [0.0, 1.0] araligina normalize edilip edilmeyecegi.
        ara_adimlari_ver: True ise raporlama icin ara asamalar sozluk olarak dondurulur.
        
    Dondurur:
        islenmis_gorsel (np.ndarray) veya (islenmis_gorsel, ara_adimlar_sozlugu)
    """
    if isinstance(girdi, (str, Path)):
        dosya_yolu = Path(girdi)
        if not dosya_yolu.exists():
            raise FileNotFoundError(f"Gorsel bulunamadi: {dosya_yolu}")
        bgr = cv2.imread(str(dosya_yolu))
        if bgr is None:
            raise ValueError(f"Gorsel okunamadi veya bozuk: {dosya_yolu}")
    elif isinstance(girdi, np.ndarray):
        bgr = girdi.copy()
    else:
        raise TypeError("Girdi dosya yolu veya numpy ndarray olmalidir.")

    # 1. Adim: ROI Kirpma
    kirpilmis = roi_kirp(bgr, esik=10, dairesel_maske=True)

    # 2. Adim: Boyutlandirma
    boyutlu = boyutlandir(kirpilmis, hedef_boyut=hedef_boyut)

    # 3. Adim: Yesil Kanal Izolasyonu
    yesil = yesil_kanali_izole_et(boyutlu)

    # 4. Adim: CLAHE ile Kontrast Iyilestirme
    kontrastli = clahe_uygula(yesil, clip_limit=2.0, tile_grid_size=(8, 8))

    # 5. Adim: Median Filtre ile Gurultu Azaltma
    filtrelenmis = gurultu_azalt(kontrastli, cekirdek_boyutu=3)

    # 6. Adim: Normalizasyon
    nihai = normalize_et(filtrelenmis) if normalize else filtrelenmis

    if ara_adimlari_ver:
        adimlar = {
            "1_orijinal": bgr,
            "2_roi_kirpilmis": kirpilmis,
            "3_yesil_kanal": yesil,
            "4_clahe": kontrastli,
            "5_nihai_islenmis": nihai
        }
        return nihai, adimlar

    return nihai


def on_isleme_rapor_grafigi_olustur(csv_yolu: Path, cikti_yolu: Path):
    """
    Her hastalik sinifindan 1'er ornek alarak on isleme adimlarini gosteren
    yayin kalitesinde (4 satir x 5 sutun) karsilastirma figuru olusturur.
    """
    df = pd.read_csv(csv_yolu)
    siniflar = sorted(df["class_name"].unique())

    fig, axes = plt.subplots(nrows=len(siniflar), ncols=5, figsize=(15, 12))
    sutun_basliklari = [
        "Orijinal Gorsel",
        "ROI Kirpilmis",
        "Yesil Kanal (G)",
        "CLAHE Kontrast",
        "Nihai Islenmis (Median)"
    ]

    for satir_idx, sinif in enumerate(siniflar):
        ornek_satir = df[df["class_name"] == sinif].iloc[0]
        dosya_yolu = Path(ornek_satir["file_path"])

        _, adimlar = on_isle(dosya_yolu, ara_adimlari_ver=True)

        # 1. Orijinal (BGR -> RGB)
        axes[satir_idx, 0].imshow(cv2.cvtColor(adimlar["1_orijinal"], cv2.COLOR_BGR2RGB))
        axes[satir_idx, 0].set_ylabel(sinif.upper(), fontsize=11, fontweight="bold")

        # 2. ROI Kırpılmış (BGR -> RGB)
        axes[satir_idx, 1].imshow(cv2.cvtColor(adimlar["2_roi_kirpilmis"], cv2.COLOR_BGR2RGB))

        # 3. Yeşil Kanal (Gri ton)
        axes[satir_idx, 2].imshow(adimlar["3_yesil_kanal"], cmap="gray")

        # 4. CLAHE (Gri ton)
        axes[satir_idx, 3].imshow(adimlar["4_clahe"], cmap="gray")

        # 5. Nihai (Gri ton)
        axes[satir_idx, 4].imshow(adimlar["5_nihai_islenmis"], cmap="gray")

        for col_idx in range(5):
            axes[satir_idx, col_idx].set_xticks([])
            axes[satir_idx, col_idx].set_yticks([])

    for col_idx, baslik in enumerate(sutun_basliklari):
        axes[0, col_idx].set_title(baslik, fontsize=11, pad=10, fontweight="bold")

    plt.suptitle("RETINA-AI: Hastalik Siniflarina Gore On Isleme Pipeline Asamalari", fontsize=14, y=0.99)
    plt.tight_layout()
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cikti_yolu, dpi=180)
    plt.close()
    print(f"[BILGI] On isleme karsilastirma figuru kaydedildi: {cikti_yolu.name}")


if __name__ == "__main__":
    proje_koku = Path(__file__).resolve().parent.parent
    csv_dosyasi = proje_koku / "data" / "splits" / "veri_bolme.csv"
    fig_dosyasi = proje_koku / "reports" / "figures" / "on_isleme_asamalari.png"

    print("+" + "-" * 68 + "+")
    print("|      RETINA-AI: VERI VE ON ISLEME SORUMLUSU - ON ISLEME HATTI       |")
    print("+" + "-" * 68 + "+")

    if csv_dosyasi.exists():
        on_isleme_rapor_grafigi_olustur(csv_dosyasi, fig_dosyasi)
        print("[TAMAMLANDI] on_isle pipeline testi ve karsilastirma figuru basariyla uretildi.")
    else:
        print("[HATA] 'veri_bolme.csv' bulunamadi. Once src/data_prep.py calistirilmalidir.")
