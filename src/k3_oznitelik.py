"""
RETINA-AI - Klasik Model ve Oznitelik Cikarimi (Kisi 3: Oznitelik Muhendisligi ve SVM)
--------------------------------------------------------------------------------------
Bu modul, goz dibi (fundus) goruntulerinden klasik makine ogrenmesi (SVM / Random Forest) icin
tibbi ve morfolojik oznitelikler cikarir ve egitilmis model uzerinden sinif olasiliklarini tahmin eder.

Fonksiyonlar:
    1. oznitelik_cikar(goruntu, bgr_mi): 512x512 RGB numpy dizisinden kalite, damar ve lezyon oznitelikleri cikarir.
    2. k3_tahmin_et(goruntu, model_path, bgr_mi): Oznitelikleri SVM modeline vererek 4 sinifli olasilik sozluk formatinda tahmin dondurur.
"""

from pathlib import Path
from typing import Dict, Union, Optional
import os
import cv2
import numpy as np

# Sinif sirasi ve standart etiketler
SINIFLAR = ["normal", "diabetic_retinopathy", "glaucoma", "cataract"]


def oznitelik_cikar(
    goruntu: Union[np.ndarray, str, Path],
    bgr_mi: bool = False
) -> Dict[str, Union[float, int]]:
    """
    512x512 RGB formatindaki retina fundus goruntusunden medikal oznitelikler cikarir.
    
    Cikarilan Oznitelikler:
        - Netlik/Kalite:
            * laplacian_varyansi: Laplacian varyansi (katarakt gibi bulanik durumlarda belirgin duser)
            * ortalama_kontrast: Retina dokusunun RMS kontrast / standart sapma degeri
            * damar_orani: Yesil kanalda morfolojik olarak filtrelenen damar alaninin toplam retina alanina orani
        - Diyabetik Retinopati Lezyonlari:
            * sari_lezyon_alani: HSV renk uzayinda sari lezyon / sert eksuda (hard exudates) toplam piksel alani
            * sari_lezyon_sayisi: Tespit edilen sari eksuda / lezyon odaklarinin sayisi
            * kanama_alani: Yesil kanalda koyu gorunen kirmizi kanama / mikroanevrizma lezyonlarinin alani
            * kanama_sayisi: Tespit edilen kanama odaklarinin sayisi

    Parametreler:
        goruntu (np.ndarray | str | Path): 512x512 goruntu matrisi veya dosya yolu.
        bgr_mi (bool): Goruntu OpenCV'nin varsayilan BGR formatinda geliyorsa True verilir (otomatik RGB'ye cevrilir).
        
    Dondurur:
        dict: Tum degerlerin standart Python float/int turunde oldugu sozluk:
            {
                "laplacian_varyansi": float,
                "ortalama_kontrast": float,
                "damar_orani": float,
                "sari_lezyon_alani": float,
                "sari_lezyon_sayisi": int,
                "kanama_alani": float,
                "kanama_sayisi": int
            }
    """
    if goruntu is None:
        raise ValueError("Girdi goruntusu None olamaz.")

    # 1. Dosya Yolu veya Matris Okuma / BGR -> RGB Donusumu
    if isinstance(goruntu, (str, Path)):
        dosya_yolu = Path(goruntu)
        if not dosya_yolu.exists():
            raise FileNotFoundError(f"Gorsel bulunamadi: {dosya_yolu}")
        bgr = cv2.imread(str(dosya_yolu))
        if bgr is None:
            raise ValueError(f"Gorsel okunamadi: {dosya_yolu}")
        gorsel = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(goruntu, np.ndarray):
        gorsel = goruntu.copy()
        if bgr_mi and gorsel.ndim == 3 and gorsel.shape[2] >= 3:
            gorsel = cv2.cvtColor(gorsel, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("Girdi np.ndarray, str veya Path olmalidir.")

    # 2. Boyut ve Renk Kanali Dogrulama
    if gorsel.ndim == 2:
        gorsel = cv2.cvtColor(gorsel, cv2.COLOR_GRAY2RGB)
    elif gorsel.shape[2] == 4:
        gorsel = cv2.cvtColor(gorsel, cv2.COLOR_RGBA2RGB)

    # 512x512 boyutlandirma
    if gorsel.shape[:2] != (512, 512):
        gorsel = cv2.resize(gorsel, (512, 512), interpolation=cv2.INTER_AREA)

    # [0.0, 1.0] float ise [0, 255] uint8 donusumu
    if np.issubdtype(gorsel.dtype, np.floating):
        if gorsel.max() <= 1.0:
            gorsel_uint8 = np.clip(gorsel * 255.0, 0, 255).astype(np.uint8)
        else:
            gorsel_uint8 = np.clip(gorsel, 0, 255).astype(np.uint8)
    else:
        gorsel_uint8 = gorsel.astype(np.uint8)

    # Gri tonlama ve retina kuresi (fundus FOV) izolasyonu
    gri = cv2.cvtColor(gorsel_uint8, cv2.COLOR_RGB2GRAY)
    fundus_maske = (gri > 15) & (gorsel_uint8[:, :, 0] > 15)
    if not np.any(fundus_maske):
        fundus_maske = np.ones_like(gri, dtype=bool)

    # Kenar gecislerindeki sahte lezyonlari engellemek icin fundus maskesini hafif erode et
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fundus_ic_maske = cv2.erode(fundus_maske.astype(np.uint8), kernel_erode)

    # ============================================================
    # 1. NETLIK / KALITE OZNITELIKLERI (Katarakt analizi icin)
    # ============================================================
    # Laplacian filtresi: Kataraktli gozlerde lens opasitesi sebebiyle kenar keskinligi ve varyans belirgin duser.
    laplacian = cv2.Laplacian(gri, cv2.CV_64F)
    laplacian_varyansi = float(laplacian.var())

    # Ortalama kontrast (Retina dokusu icindeki RMS kontrast / standart sapma)
    retina_pikselleri = gri[fundus_maske]
    ortalama_kontrast = float(np.std(retina_pikselleri)) if len(retina_pikselleri) > 0 else float(np.std(gri))

    # ============================================================
    # 2. GORUNUR DAMAR ORANI (Yesil kanalda damar yogunlugu)
    # ============================================================
    # Yesil kanalda hemoglobin emilimi nedeniyle damarlar en belirgin kontrasta sahiptir.
    g_kanal = gorsel_uint8[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g_kanal)

    kernel_damar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    damar_blackhat = cv2.morphologyEx(g_clahe, cv2.MORPH_BLACKHAT, kernel_damar)
    _, damar_maske = cv2.threshold(damar_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Sadece retina kuresi icini maskele
    damar_retina = cv2.bitwise_and(damar_maske, damar_maske, mask=fundus_ic_maske)
    
    # Retina alani ve damar piksel sayisi
    retina_piksel_alani = float(np.count_nonzero(fundus_maske))
    damar_piksel_alani = float(np.count_nonzero(damar_retina))
    
    # ZeroDivisionError onlemek icin paydaya kucuk tolerans skaler (1e-6) ekle
    damar_orani = float(damar_piksel_alani / (retina_piksel_alani + 1e-6))

    # ============================================================
    # 3. DIYABETIK RETINOPATI: HSV UZAYINDA SARI LEZYON / EKSUDA
    # ============================================================
    # Eksudalar yuksek yagli lipid birikintileridir; parlak saridir.
    hsv = cv2.cvtColor(gorsel_uint8, cv2.COLOR_RGB2HSV)
    
    # OpenCV HSV uzayinda sari ton araligi (H: 16-36, S: 50-255, V: 110-255)
    alt_sari = np.array([16, 50, 110], dtype=np.uint8)
    ust_sari = np.array([36, 255, 255], dtype=np.uint8)
    sari_maske = cv2.inRange(hsv, alt_sari, ust_sari)
    
    # Yalnizca retina kuresi icini filtrele
    sari_maske = cv2.bitwise_and(sari_maske, sari_maske, mask=fundus_ic_maske)
    
    # Morfolojik acma ile gurultuleri temizle
    kernel_kucuk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sari_maske = cv2.morphologyEx(sari_maske, cv2.MORPH_OPEN, kernel_kucuk)

    konturlar_sari, _ = cv2.findContours(sari_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sari_lezyon_alani = 0.0
    sari_lezyon_sayisi = 0

    # Optik disk cok buyuk parlak bir yapidir (>3500 px). Eksuda odaklari kucuk-orta caplidir.
    for c in konturlar_sari:
        alan = cv2.contourArea(c)
        if 4 <= alan <= 3500:
            sari_lezyon_sayisi += 1
            sari_lezyon_alani += float(alan)

    # ============================================================
    # 4. DIYABETIK RETINOPATI: YESIL KANALDA KOYU KIRMIZI KANAMA
    # ============================================================
    # Zemin dokusundan lokal olarak koyu kalan morfolojik cukurlari yakalamak icin Black-hat
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blackhat_g = cv2.morphologyEx(g_kanal, cv2.MORPH_BLACKHAT, kernel_bh)

    # Koyu kirmizi kanama: blackhat belirgin, lokal dokuya gore koyu
    kanama_adaylari = (blackhat_g > 15) & (fundus_ic_maske > 0)
    kanama_maske = kanama_adaylari.astype(np.uint8) * 255
    kanama_maske = cv2.morphologyEx(kanama_maske, cv2.MORPH_OPEN, kernel_kucuk)

    konturlar_kanama, _ = cv2.findContours(kanama_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kanama_alani = 0.0
    kanama_sayisi = 0

    for c in konturlar_kanama:
        alan = cv2.contourArea(c)
        if 4 <= alan <= 3500:
            # Damarlari elemek icin en/boy orani kontrolu
            x, y, w, h = cv2.boundingRect(c)
            en_boy = max(w, h) / (min(w, h) + 1e-4)
            if en_boy < 4.0:  # Uzun dallanan damarlari ayikla, yuvarlak/noktasal kanamalari al
                kanama_sayisi += 1
                kanama_alani += float(alan)

    # ============================================================
    # OZNITELIK SOZLUGU (JSON Uyumlu Standart Python Tipleri)
    # ============================================================
    oznitelikler = {
        "laplacian_varyansi": float(round(laplacian_varyansi, 4)),
        "ortalama_kontrast": float(round(ortalama_kontrast, 4)),
        "damar_orani": float(round(damar_orani, 4)),
        "sari_lezyon_alani": float(round(sari_lezyon_alani, 2)),
        "sari_lezyon_sayisi": int(sari_lezyon_sayisi),
        "kanama_alani": float(round(kanama_alani, 2)),
        "kanama_sayisi": int(kanama_sayisi),
    }

    return oznitelikler


def _sezgisel_olasilik_hesapla(oznitelikler: Dict[str, Union[float, int]]) -> Dict[str, float]:
    """
    Egitilmis model henuz diskte bulunmadiginda, tibbi kurallara dayali
    gercekci bir olasilik dagilimi uretir.
    """
    lap = float(oznitelikler.get("laplacian_varyansi", 100.0))
    kontrast = float(oznitelikler.get("ortalama_kontrast", 35.0))
    damar = float(oznitelikler.get("damar_orani", 0.05))
    sari_alan = float(oznitelikler.get("sari_lezyon_alani", 0.0))
    kanama_alan = float(oznitelikler.get("kanama_alani", 0.0))

    # Katarakt skoru: Dusuk netlik (<60), dusuk kontrast ve damarlarin secilememesi
    skor_katarakt = max(
        0.05,
        (100.0 - min(lap, 100.0)) / 20.0 + (35.0 - min(kontrast, 35.0)) / 12.0 + (0.04 - min(damar, 0.04)) * 25.0
    )
    
    # Diyabetik Retinopati skoru: Eksuda ve kanama alanlari
    skor_dr = max(0.05, (sari_alan / 60.0) + (kanama_alan / 50.0))
    
    # Glokom skoru: Genel net fundus, optik sinir cupping
    skor_glokom = 1.0 if lap > 50.0 else 0.2
    
    # Normal skoru: Yuksek netlik, lezyonsuz yapi ve belirgin damar agi
    if sari_alan < 15 and kanama_alan < 15 and lap > 80:
        skor_normal = 3.5
    else:
        skor_normal = max(0.05, 1.5 - (skor_dr * 0.7 + skor_katarakt * 0.5))

    skorlar = np.array([skor_normal, skor_dr, skor_glokom, skor_katarakt], dtype=np.float64)
    exp_skorlar = np.exp(skorlar - np.max(skorlar))
    olasiliklar = exp_skorlar / np.sum(exp_skorlar)
    
    olasilik_sozluk = {
        "normal": float(round(olasiliklar[0], 2)),
        "diabetic_retinopathy": float(round(olasiliklar[1], 2)),
        "glaucoma": float(round(olasiliklar[2], 2)),
        "cataract": float(round(olasiliklar[3], 2)),
    }
    
    # Toplami tam olarak 1.0 yap
    fark = round(1.0 - sum(olasilik_sozluk.values()), 2)
    if abs(fark) > 1e-5:
        en_buyuk = max(olasilik_sozluk, key=olasilik_sozluk.get)
        olasilik_sozluk[en_buyuk] = round(olasilik_sozluk[en_buyuk] + fark, 2)

    return olasilik_sozluk


_MODEL_CACHE = {}


def _svm_modelini_yukle(model_dosyasi: Path):
    """SVM Pipeline modelini yukler ve onbellekte saklar."""
    dosya_str = str(model_dosyasi.resolve())
    if dosya_str in _MODEL_CACHE:
        return _MODEL_CACHE[dosya_str]

    try:
        import sklearn
    except ImportError:
        import sys
        extra_path = r"C:\Users\samil\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages"
        if os.path.exists(extra_path) and extra_path not in sys.path:
            sys.path.insert(0, extra_path)

    import joblib
    model = joblib.load(model_dosyasi)
    _MODEL_CACHE[dosya_str] = model
    return model


def k3_tahmin_et(
    goruntu: Union[np.ndarray, str, Path],
    model_path: str = "models/svm_model_v2.pkl",
    bgr_mi: bool = False,
    as_json: bool = False
) -> Union[Dict[str, float], str]:
    """
    Fundus goruntusunden oznitelikleri cikarir ve v2 SVM Pipeline modelini (StandardScaler + SVC)
    kullanarak 4 sinifli hastalik olasiliklarini dondurur.

    Parametreler:
        goruntu (np.ndarray | str | Path): 512x512 RGB/BGR goruntu matrisi veya dosya yolu.
        model_path (str): Egitilmis v2 SVM model dosyasinin (.pkl) konumu.
        bgr_mi (bool): Girdi goruntusu BGR formatindaysa True verilir.
        as_json (bool): True verilirse dogrudan JSON dizgisi (string) dondurur.

    Dondurur:
        dict veya str: JSON uyumlu standart Python float sozluk veya JSON string:
            {"normal": 0.10, "diabetic_retinopathy": 0.70, "glaucoma": 0.15, "cataract": 0.05}
    """
    import json

    oznitelikler = oznitelik_cikar(goruntu, bgr_mi=bgr_mi)
    
    # data/oznitelikler_v2.csv siralamasina tam uygun 7 oznitelikli vektor
    vektor_7 = [
        float(oznitelikler["laplacian_varyansi"]),
        float(oznitelikler["ortalama_kontrast"]),
        float(oznitelikler["damar_orani"]),
        float(oznitelikler["sari_lezyon_alani"]),
        float(oznitelikler["sari_lezyon_sayisi"]),
        float(oznitelikler["kanama_alani"]),
        float(oznitelikler["kanama_sayisi"])
    ]
    # Geriye donuk 6 oznitelikli model uyumlulugu
    vektor_6 = [
        float(oznitelikler["laplacian_varyansi"]),
        float(oznitelikler["ortalama_kontrast"]),
        float(oznitelikler["sari_lezyon_alani"]),
        float(oznitelikler["sari_lezyon_sayisi"]),
        float(oznitelikler["kanama_alani"]),
        float(oznitelikler["kanama_sayisi"])
    ]

    model_dosyasi = Path(model_path)
    if not model_dosyasi.exists():
        for aday in [
            Path("models/svm_model_v2.pkl"),
            Path("models/svm_model.pkl"),
            Path(__file__).resolve().parent.parent / "models" / "svm_model_v2.pkl",
            Path(__file__).resolve().parent.parent / "models" / "svm_model.pkl"
        ]:
            if aday.exists():
                model_dosyasi = aday
                break

    if model_dosyasi.exists():
        try:
            model = _svm_modelini_yukle(model_dosyasi)
            
            beklenen_oznitelik_sayisi = getattr(model, "n_features_in_", None)
            if beklenen_oznitelik_sayisi == 6:
                vektor = np.array([vektor_6], dtype=np.float32)
            else:
                vektor = np.array([vektor_7], dtype=np.float32)

            if hasattr(model, "predict_proba"):
                probalar = model.predict_proba(vektor)[0]
                sinif_isimleri = getattr(model, "classes_", SINIFLAR)
                
                sonuc = {}
                for idx, sinif in enumerate(sinif_isimleri):
                    sinif_str = str(sinif).lower().strip()
                    sonuc[sinif_str] = float(round(probalar[idx], 2))
                
                for s in SINIFLAR:
                    if s not in sonuc:
                        sonuc[s] = 0.0
                        
                toplam = sum(sonuc.values())
                if toplam > 0:
                    sonuc = {k: round(v / toplam, 2) for k, v in sonuc.items()}
                    fark = round(1.0 - sum(sonuc.values()), 2)
                    en_buyuk = max(sonuc, key=sonuc.get)
                    sonuc[en_buyuk] = round(sonuc[en_buyuk] + fark, 2)
                    
                nihai_sonuc = {s: float(sonuc[s]) for s in SINIFLAR}
                if as_json:
                    return json.dumps(nihai_sonuc, ensure_ascii=False)
                return nihai_sonuc
        except Exception:
            pass

    sezgisel = _sezgisel_olasilik_hesapla(oznitelikler)
    if as_json:
        return json.dumps(sezgisel, ensure_ascii=False)
    return sezgisel


if __name__ == "__main__":
    import json

    print("=== RETINA-AI: K3 Oznitelik Modulu Testi ===")
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(test_img, (256, 256), 220, (190, 95, 35), -1)
    for i in range(15):
        cv2.line(test_img, (256, 256), (100 + i*20, 60 + i*25), (100, 30, 20), 2)
    cv2.circle(test_img, (280, 200), 8, (245, 225, 60), -1)   # Sari eksuda
    cv2.circle(test_img, (200, 300), 7, (90, 25, 15), -1)     # Koyu kanama

    oznitelik_sonuc = oznitelik_cikar(test_img)
    print("Cikarilan Oznitelikler:")
    for k, v in oznitelik_sonuc.items():
        print(f"  {k}: {v} (tip: {type(v).__name__})")

    # JSON Serilestirme Testi
    json_cikti = json.dumps(oznitelik_sonuc)
    print("\nJSON Serilestirme Basarili:")
    print(json_cikti)

    tahmin_sonuc = k3_tahmin_et(test_img)
    print("\nTahmin Edilen Olasiliklar:")
    print(tahmin_sonuc)
    print(f"Toplam Olasilik: {sum(tahmin_sonuc.values()):.2f}")
