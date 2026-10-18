import cv2
import numpy as np


def _gri_uint8(goruntu):
    """Gelen görüntüyü 0-255 arası tek kanallı hâle getirir."""
    if goruntu is None:
        raise ValueError("Görüntü boş olamaz.")
    g = goruntu[:, :, 1] if goruntu.ndim == 3 else goruntu  # renkliyse yeşil kanal
    if g.dtype != np.uint8:
        if g.max() <= 1.0:
            g = g * 255
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def _retina_alani(gri, kenar_payi=10):
    """Retinanın olduğu yuvarlak bölgeyi bulur (siyah kenarlar dışarıda kalır)."""
    ikili = (gri > 5).astype(np.uint8) * 255
    ikili = cv2.morphologyEx(ikili, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    konturlar, _ = cv2.findContours(ikili, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konturlar:
        return np.full_like(gri, 255)
    alan = np.zeros_like(gri)
    cv2.drawContours(alan, [max(konturlar, key=cv2.contourArea)], -1, 255, -1)
    # Kenardan içeri çek. Görüntü sınırının dışı da "boş" sayılır (kenar kıymıklarını önler)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kenar_payi + 1,) * 2)
    return cv2.erode(alan, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)


def damar_haritasi(goruntu, kernel_boyutu=15, min_alan=30, min_uzunluk=25):
    """Damarlar beyaz (255), geri kalan siyah (0) olan maske döndürür."""
    gri = _gri_uint8(goruntu)
    alan = _retina_alani(gri)
    ic = alan > 0
    if not ic.any():
        return np.zeros_like(gri)

    # Dışarıyı retinanın ortalama parlaklığıyla doldur
    dolgulu = gri.copy()
    dolgulu[~ic] = int(np.median(gri[ic]))

    # Koyu ince yapıları (damarları) öne çıkar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyutu, kernel_boyutu))
    tophat = cv2.morphologyEx(dolgulu, cv2.MORPH_BLACKHAT, kernel)
    tophat[~ic] = 0

    # Eşik değerini sadece retina içinden hesapla
    esik, _ = cv2.threshold(tophat[ic].reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    esik = max(esik, 12)   # zayıf kontrastlı (bulanık) görüntülerde gürültüyü damar sayma
    maske = ((tophat > esik) & ic).astype(np.uint8) * 255

    # Parçaları ele: küçükleri, kısa olanları ve yuvarlak leke gibi duranları sil.
    # Damarlar uzun ve ince, lekeler (kanama vb.) kısa ve dolgundur.
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)
    tut = np.zeros(n, dtype=bool)
    for i in range(1, n):
        a = ist[i, cv2.CC_STAT_AREA]
        w = ist[i, cv2.CC_STAT_WIDTH]
        h = ist[i, cv2.CC_STAT_HEIGHT]
        uzunluk = max(w, h)
        doluluk = a / float(w * h)          # 1'e yakınsa dolgun leke
        if a >= min_alan and uzunluk >= min_uzunluk and not (doluluk > 0.55 and uzunluk < 50):
            tut[i] = True
    return tut[etiket].astype(np.uint8) * 255


def damar_yogunlugu(maske, goruntu):
    """Damar piksellerinin retina alanına oranı (0-1 arası)."""
    alan = _retina_alani(_gri_uint8(goruntu)) > 0
    if not alan.any():
        return 0.0
    return float(((maske > 0) & alan).sum() / alan.sum())


def _parca_maskesi(maske, nokta=None):
    """Maskedeki bir beyaz parçayı seçip onu tek başına maske olarak döndürür.
    nokta verilirse o noktayı içeren parçayı, yoksa en büyüğünü alır."""
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)
    if n <= 1:
        return None
    secilen = 0
    if nokta is not None:
        secilen = etiket[nokta[1], nokta[0]]
    if secilen == 0:
        secilen = 1 + int(np.argmax(ist[1:, cv2.CC_STAT_AREA]))
    return (etiket == secilen).astype(np.uint8) * 255


def _dikey_cap(parca):
    """Bir parçanın en üst ve en alt noktası arasındaki yükseklik (piksel)."""
    satirlar = np.where(parca.any(axis=1))[0]
    if len(satirlar) == 0:
        return 0
    return int(satirlar[-1] - satirlar[0] + 1)


def optik_disk(goruntu, min_daireselik=0.7, min_kontrast=25):
    """Renkli 512x512 RGB görüntüden optik diski ve dikey çukur/disk oranını (CDR) bulur.
    Döner: {"merkez": (x, y), "yaricap": r, "cukur_disk_orani": 0.4}
    Güvenilir bir disk bulunamazsa None."""
    if goruntu is None or goruntu.ndim != 3:
        raise ValueError("Renkli (RGB) görüntü vermelisin.")
    if goruntu.dtype != np.uint8:
        goruntu = np.clip(goruntu * 255 if goruntu.max() <= 1.0 else goruntu, 0, 255).astype(np.uint8)

    gri = cv2.cvtColor(goruntu, cv2.COLOR_RGB2GRAY)
    gri_yumusak = cv2.GaussianBlur(gri, (5, 5), 0)
    yesil_yumusak = cv2.GaussianBlur(goruntu[:, :, 1], (5, 5), 0)
    alan = _retina_alani(gri, kenar_payi=25)
    ic = alan > 0
    if not ic.any():
        return None

    # 1) Retinanın en parlak bölgesi: disk adayı
    bulanik = cv2.GaussianBlur(gri, (0, 0), 12)
    _, _, _, (bx, by) = cv2.minMaxLoc(bulanik, mask=alan)

    # 2) O noktanın çevresinde parlak bölgeyi ayır
    pencere = np.zeros_like(gri)
    cv2.circle(pencere, (bx, by), 90, 255, -1)
    bolge = (pencere > 0) & ic
    esik, _ = cv2.threshold(gri_yumusak[bolge].reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    aday = ((gri_yumusak > esik) & bolge).astype(np.uint8) * 255
    aday = cv2.morphologyEx(aday, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    disk_parca = _parca_maskesi(aday, (bx, by))
    if disk_parca is None:
        return None

    # Merkez ve yarıçap
    m = cv2.moments(disk_parca, binaryImage=True)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    yaricap = float(np.sqrt(m["m00"] / np.pi))

    # Güvenilirlik 1: şekil yuvarlak mı? (damar çıkıntılarını yok saymak için dış zarf üzerinden)
    kontur, _ = cv2.findContours(disk_parca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zarf = cv2.convexHull(max(kontur, key=cv2.contourArea))
    zarf_alan = cv2.contourArea(zarf)
    zarf_cevre = cv2.arcLength(zarf, True)
    daireselik = 4 * np.pi * zarf_alan / (zarf_cevre ** 2) if zarf_cevre > 0 else 0

    # Güvenilirlik 2: çevresinden parlak mı?
    genis = cv2.dilate(disk_parca, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    dar = cv2.dilate(disk_parca, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    halka = (genis > 0) & (dar == 0) & ic
    if halka.sum() > 0:
        kontrast = float(gri[disk_parca > 0].mean()) - float(gri[halka].mean())
    else:
        kontrast = -999.0

    if not (15 <= yaricap <= 70) or daireselik < min_daireselik or kontrast < min_kontrast:
        return None

    # 3) Çukur: diskin içindeki EN parlak kısım (yeşil kanalda)
    degerler = yesil_yumusak[disk_parca > 0]
    alt, ust = np.percentile(degerler, [10, 98])
    cukur_esik = alt + 0.6 * (ust - alt)
    cukur = ((yesil_yumusak > cukur_esik) & (disk_parca > 0)).astype(np.uint8) * 255
    cukur = cv2.morphologyEx(cukur, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cukur_parca = _parca_maskesi(cukur)

    # 4) CDR = çukurun dikey çapı / diskin dikey çapı
    disk_dikey = _dikey_cap(disk_parca)
    cukur_dikey = _dikey_cap(cukur_parca) if cukur_parca is not None else 0
    oran = min(cukur_dikey / disk_dikey, 1.0) if disk_dikey > 0 else 0.0

    return {"merkez": (int(cx), int(cy)),
            "yaricap": round(yaricap, 1),
            "cukur_disk_orani": round(float(oran), 2)}

def disk_isaretle(goruntu, sonuc):
    """Bulunan diski (yeşil) ve çukuru (sarı) görüntü üzerine çizer. Kontrol ve makale görseli için."""
    cikti = goruntu.copy()
    if sonuc is None:
        return cikti
    merkez = sonuc["merkez"]
    cv2.circle(cikti, merkez, int(sonuc["yaricap"]), (0, 255, 0), 2)
    cv2.circle(cikti, merkez, int(sonuc["yaricap"] * sonuc["cukur_disk_orani"]), (255, 255, 0), 2)
    return cikti