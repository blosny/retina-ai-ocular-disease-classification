"""
RETINA-AI - Streamlit arayuzu (Kisi 5: Degerlendirme ve Streamlit)

Calistirma (proje ana klasorunden):
    streamlit run src/app.py

Gercek kod: on_isle (Kisi 1, src/preprocessing.py)
Gecici dubluler: damar_haritasi (Kisi 2), tahmin_et (Kisi 3 + 4), gradcam (Kisi 4)
Arkadaslarin kodu geldikce dublorleri onlarin fonksiyonlariyla degistir.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# KISI 1 - Gercek on isleme kodu (src/preprocessing.py)
from preprocessing import on_isle, boyutlandir

SINIFLAR = {
    "normal": "🟢 Normal",
    "diabetic_retinopathy": "🟡 Diyabetik Retinopati",
    "glaucoma": "🔵 Glokom",
    "cataract": "⚪ Katarakt",
}


# ============================================================
# GECICI DUBLORLER - sahipleri bitirince degistir
# ============================================================

def damar_haritasi(islenmis: np.ndarray) -> np.ndarray:
    """KISI 2 (src/anatomy.py) - Damar segmentasyonu.
    Girdi: on_isle ciktisi (512x512, 0-1 arasi gri). Cikti: siyah-beyaz maske."""
    gri = (islenmis * 255).astype(np.uint8)
    cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    siyah_sapka = cv2.morphologyEx(gri, cv2.MORPH_BLACKHAT, cekirdek)
    _, maske = cv2.threshold(siyah_sapka, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    goz_alani = cv2.erode((gri > 10).astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=3)
    return cv2.bitwise_and(maske, goz_alani)


def tahmin_et(islenmis: np.ndarray, model_turu: str) -> dict:
    """KISI 3 (SVM) ve KISI 4 (CNN) - Sinif olasiliklari, toplami 1.
    SU AN SAHTE: ayni goruntu icin hep ayni, rastgele sayilar."""
    tohum = int((islenmis * 1000).astype(np.int64).sum() % (2**32))
    if model_turu.startswith("SVM"):
        tohum += 1
    olasiliklar = np.random.default_rng(tohum).dirichlet(np.ones(len(SINIFLAR)) * 0.6)
    return {sinif: float(p) for sinif, p in zip(SINIFLAR, olasiliklar)}


def gradcam(islenmis: np.ndarray) -> np.ndarray:
    """KISI 4 (src/cnn_models.py) - Grad-CAM, 0-1 arasi, 512x512.
    SU AN SAHTE: tek bir sicak nokta."""
    h, w = islenmis.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - w * 0.6) ** 2 + (y - h * 0.5) ** 2) / (2 * (w * 0.12) ** 2)))


# ============================================================
# YARDIMCI (Kisi 5 - kalici)
# ============================================================

def isi_haritasi_bindir(renkli: np.ndarray, harita: np.ndarray) -> np.ndarray:
    """Isi haritasini renkli retinanin ustune yari saydam bindirir."""
    renkli_harita = cv2.applyColorMap((harita * 255).astype(np.uint8), cv2.COLORMAP_JET)
    renkli_harita = cv2.cvtColor(renkli_harita, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(renkli, 0.6, renkli_harita, 0.4, 0)


# ============================================================
# ARAYUZ
# ============================================================

def main():
    st.set_page_config(page_title="RETINA-AI", page_icon="👁️", layout="wide")
    st.title("👁️ RETINA-AI")
    st.caption("Fundus görüntüsünden 4 sınıflı göz hastalığı sınıflandırma · Akademik simülasyon")

    with st.sidebar:
        st.header("Ayarlar")
        model_turu = st.radio("Model", ["CNN (derin öğrenme)", "SVM (klasik)"])
        st.info("Sınıflar: Normal, Diyabetik Retinopati, Glokom, Katarakt")
        st.warning("⚠️ Bu sistem teşhis koymaz. Yalnızca eğitim amaçlıdır.")

    dosya = st.file_uploader("Retina (fundus) görüntüsünü yükleyin", type=["jpg", "jpeg", "png"])
    if dosya is None:
        st.info("Başlamak için bir fundus görüntüsü yükleyin.")
        return

    ham = np.array(Image.open(dosya).convert("RGB"))
    bgr = cv2.cvtColor(ham, cv2.COLOR_RGB2BGR)  # Kisi 1'in kodu BGR bekliyor

    with st.spinner("İşleniyor..."):
        islenmis, adimlar = on_isle(bgr, ara_adimlari_ver=True)
        renkli = cv2.cvtColor(boyutlandir(adimlar["2_roi_kirpilmis"]), cv2.COLOR_BGR2RGB)
        damarlar = damar_haritasi(islenmis)
        olasiliklar = tahmin_et(islenmis, model_turu)
        isi = isi_haritasi_bindir(renkli, gradcam(islenmis))

    st.subheader("1. Görüntünün yolculuğu")
    k1, k2, k3, k4 = st.columns(4)
    k1.image(renkli, caption="Kırpılmış retina")
    k2.image(islenmis, caption="Ön işlenmiş (yeşil kanal + CLAHE)", clamp=True)
    k3.image(damarlar, caption="Damar haritası")
    k4.image(isi, caption="Grad-CAM: modelin baktığı yer")

    st.subheader("2. Sonuç")
    en_iyi = max(olasiliklar, key=olasiliklar.get)
    sol, sag = st.columns([1, 2])
    with sol:
        st.markdown(f"#### Tahmin: {SINIFLAR[en_iyi]}")
        st.metric("Model güven değeri", f"%{olasiliklar[en_iyi] * 100:.1f}")
        st.caption(f"Kullanılan model: {model_turu}")
    with sag:
        for sinif, p in sorted(olasiliklar.items(), key=lambda kv: kv[1], reverse=True):
            st.write(f"{SINIFLAR[sinif]} · %{p * 100:.1f}")
            st.progress(p)

    st.warning("⚠️ Bu sonuç yalnızca akademik simülasyon amaçlıdır. Tıbbi teşhis yerine geçmez.")
    st.caption("Not: Tahminler, damar haritası ve Grad-CAM şu an örnek (geçici) verilerdir.")


if __name__ == "__main__":
    main()
