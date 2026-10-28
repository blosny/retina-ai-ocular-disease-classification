"""anatomy.py için otomatik testler.
Çalıştırma (proje kökünden):  python -m pytest src/test_anatomy.py -q
pytest yoksa:                 python src/test_anatomy.py
Test görüntüleri data/splits/test/ altındaki örneklerdir."""
import contextlib
import glob
import io
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import damar_haritasi, damar_yogunlugu, disk_isaretle, optik_disk  # noqa: E402
from preprocessing import on_isle  # noqa: E402

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIZIN = os.path.join(KOK, "data", "splits", "test")


def _yukle(goreli):
    yol = os.path.join(TEST_DIZIN, goreli)
    bgr = cv2.imread(yol)
    assert bgr is not None, f"Görüntü yok: {yol}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _hepsi():
    yollar = sorted(glob.glob(os.path.join(TEST_DIZIN, "*", "*.jp*g")))
    assert yollar, "Test görüntüsü bulunamadı"
    return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in yollar]


def test_optik_disk_donus_bicimi():
    sonuc = optik_disk(_yukle("glaucoma/1218_right.jpg"))
    assert set(sonuc) == {"merkez", "yaricap", "cukur_disk_orani"}
    assert isinstance(sonuc["merkez"], tuple) and len(sonuc["merkez"]) == 2
    assert isinstance(sonuc["yaricap"], float)
    assert 0.0 <= sonuc["cukur_disk_orani"] <= 1.0


def test_disk_bulma_orani_artti():
    bulunan = sum(optik_disk(g) is not None for g in _hepsi())
    assert bulunan >= 13, f"Disk bulma düştü: {bulunan}"


def test_parlama_disk_sanilmaz():
    # glaucoma/1218_left: sol üstteki yeşilimsi parlama disk değil.
    # Gerçek disk (153, 283) civarında görünüyor: ya None ya da bu gerçek disk bulunmalı, parlama asla.
    sonuc = optik_disk(_yukle("glaucoma/1218_left.jpg"))
    assert sonuc is None or (abs(sonuc["merkez"][0] - 153) <= 20 and abs(sonuc["merkez"][1] - 283) <= 20), sonuc


def test_kenara_yakin_kucuk_disk_bulunur():
    sonuc = optik_disk(_yukle("diabetic_retinopathy/10010_right.jpeg"))
    assert sonuc is not None and sonuc["merkez"][0] < 120


def test_buyuk_daire_duzeltmesi():
    for dosya in ("cataract/1083_left.jpg", "normal/2329_right.jpg"):
        sonuc = optik_disk(_yukle(dosya))
        assert sonuc is not None and sonuc["yaricap"] < 50, dosya


def test_boyut_bagimsizligi():
    im = _yukle("normal/2329_left.jpg")
    ref = optik_disk(im)
    for w, h in ((256, 256), (1024, 1024), (1000, 700)):
        s = optik_disk(cv2.resize(im, (w, h)))
        assert s is not None, (w, h)
        x512, y512 = s["merkez"][0] * 512 / w, s["merkez"][1] * 512 / h
        assert abs(x512 - ref["merkez"][0]) <= 4 and abs(y512 - ref["merkez"][1]) <= 4


def test_bos_goruntuler_none_doner():
    assert optik_disk(np.zeros((512, 512, 3), np.uint8)) is None
    assert optik_disk(np.full((512, 512, 3), 255, np.uint8)) is None


def test_gecersiz_girdi_hata_verir():
    for kotu in (None, np.zeros((512, 512), np.uint8)):
        try:
            optik_disk(kotu)
        except ValueError:
            continue
        raise AssertionError("ValueError bekleniyordu")


def test_ayrinti_varsayilan_sessiz_ve_aciksa_yazar():
    siyah = np.zeros((512, 512, 3), np.uint8)
    sessiz, sesli = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(sessiz):
        optik_disk(siyah)
    with contextlib.redirect_stdout(sesli):
        optik_disk(siyah, ayrinti=True)
    assert sessiz.getvalue() == "" and "[optik_disk]" in sesli.getvalue()


def test_damar_haritasi_ve_yogunluk():
    for im in _hepsi():
        maske = damar_haritasi(on_isle(im))
        assert maske.shape == im.shape[:2] and maske.dtype == np.uint8
        assert set(np.unique(maske)) <= {0, 255}
        assert 0.0 <= damar_yogunlugu(maske, im) <= 1.0


def test_yeni_ayar_ince_damar_kaybettirmez():
    im = _yukle("normal/2329_left.jpg")
    g = on_isle(im)
    eski = damar_haritasi(g, min_alan=30, min_uzunluk=25)
    yeni = damar_haritasi(g)
    assert int(((eski > 0) & (yeni == 0)).sum()) == 0


def test_disk_isaretle_none_ve_cizim():
    im = _yukle("normal/2329_left.jpg")
    assert np.array_equal(disk_isaretle(im, None), im)
    assert not np.array_equal(disk_isaretle(im, optik_disk(im)), im)


if __name__ == "__main__":
    hatalar = 0
    for ad, f in sorted(globals().items()):
        if ad.startswith("test_") and callable(f):
            try:
                f(); print("OK   ", ad)
            except Exception as e:
                hatalar += 1; print("HATA ", ad, "->", repr(e))
    sys.exit(1 if hatalar else 0)
