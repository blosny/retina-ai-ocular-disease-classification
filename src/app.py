"""
RETINA-AI - Streamlit arayuzu (Kisi 5: Degerlendirme ve Streamlit)

Calistirma (proje ana klasorunden):
    streamlit run src/app.py

Gercek kod:
    on_isle            -> Kisi 1 (src/preprocessing.py)
    CNN tahmin+Grad-CAM -> Kisi 4 (src/cnn_models.py)
                          Model dosyasi: models/model_efficientnet.pth (Drive'dan indirilir)
Gecici dublorler:
    damar_haritasi     -> Kisi 2 (src/anatomy.py gelince degisecek)
    svm_tahmin         -> Kisi 3 (src/lesions.py + SVM gelince degisecek)
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# KISI 1 - Gercek on isleme kodu
from preprocessing import boyutlandir, on_isle, roi_kirp

SINIFLAR = {
    "normal": "🟢 Normal",
    "diabetic_retinopathy": "🟡 Diyabetik Retinopati",
    "glaucoma": "🔵 Glokom",
    "cataract": "⚪ Katarakt",
}


# ============================================================
# KISI 4 - CNN (gercek)
# ============================================================

@st.cache_resource(show_spinner="Model yükleniyor...")
def cnn_modelini_yukle():
    """Modeli bir kez yukler. Dosya yoksa None + hata mesaji dondurur."""
    try:
        from cnn_models import VARSAYILAN_MODEL_YOLU, model_yukle
    except ImportError as hata:
        return None, f"PyTorch kurulu değil ({hata}). Kurulum: py -m pip install torch torchvision"
    if not VARSAYILAN_MODEL_YOLU.exists():
        return None, (
            f"Model dosyası bulunamadı: {VARSAYILAN_MODEL_YOLU}\n\n"
            "Drive'daki model_efficientnet.pth dosyasını proje klasöründeki "
            "models klasörüne koyun."
        )
    return model_yukle(VARSAYILAN_MODEL_YOLU), None


# ============================================================
# GECICI DUBLORLER - sahipleri bitirince degistir
# ============================================================

def damar_haritasi(islenmis: np.ndarray) -> np.ndarray:
    """KISI 2 (src/anatomy.py) - GECICI. Girdi: on_isle ciktisi. Cikti: siyah-beyaz maske."""
    gri = (islenmis * 255).astype(np.uint8)
    cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    siyah_sapka = cv2.morphologyEx(gri, cv2.MORPH_BLACKHAT, cekirdek)
    _, maske = cv2.threshold(siyah_sapka, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    goz_alani = cv2.erode((gri > 10).astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=3)
    return cv2.bitwise_and(maske, goz_alani)


def svm_tahmin(islenmis: np.ndarray) -> dict:
    """KISI 3 (SVM) - GECICI: ayni goruntu icin hep ayni, rastgele sayilar."""
    tohum = int((islenmis * 1000).astype(np.int64).sum() % (2**32))
    olasiliklar = np.random.default_rng(tohum).dirichlet(np.ones(len(SINIFLAR)) * 0.6)
    return {sinif: float(p) for sinif, p in zip(SINIFLAR, olasiliklar)}


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

    cnn_secili = model_turu.startswith("CNN")
    model, model_hatasi = cnn_modelini_yukle() if cnn_secili else (None, None)
    if model_hatasi:
        st.error(model_hatasi)
        return

    dosya = st.file_uploader("Retina (fundus) görüntüsünü yükleyin", type=["jpg", "jpeg", "png"])
    if dosya is None:
        st.info("Başlamak için bir fundus görüntüsü yükleyin.")
        return

    ham = np.array(Image.open(dosya).convert("RGB"))
    bgr = cv2.cvtColor(ham, cv2.COLOR_RGB2BGR)  # Kisi 1 ve Kisi 4'un kodu BGR bekliyor

    with st.spinner("İşleniyor..."):
        islenmis = on_isle(bgr)
        renkli = cv2.cvtColor(boyutlandir(roi_kirp(bgr)), cv2.COLOR_BGR2RGB)
        damarlar = damar_haritasi(islenmis)

        if cnn_secili:
            from cnn_models import gradcam, tahmin
            # CNN on_isle ciktisini degil, ham BGR goruntuyu alir (renk bilgisi icin)
            olasiliklar = tahmin(bgr, model)
            isi = isi_haritasi_bindir(renkli, gradcam(bgr, model=model))
            isi_aciklama = "Grad-CAM: modelin baktığı yer"
        else:
            olasiliklar = svm_tahmin(islenmis)
            isi = None

    st.subheader("1. Görüntünün yolculuğu")
    k1, k2, k3, k4 = st.columns(4)
    k1.image(renkli, caption="Kırpılmış retina")
    k2.image(islenmis, caption="Ön işlenmiş (yeşil kanal + CLAHE)", clamp=True)
    k3.image(damarlar, caption="Damar haritası (geçici)")
    if isi is not None:
        k4.image(isi, caption=isi_aciklama)
    else:
        k4.info("Grad-CAM yalnızca CNN modelinde gösterilir.")

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

    st.warning(
        "⚠️ Bu sonuç yalnızca akademik simülasyon amaçlıdır. Tıbbi teşhis yerine geçmez. "
        "Karanlık, bulanık veya kötü çekilmiş fotoğraflar katarakt olarak sınıflandırılabilir."
    )
    if not cnn_secili:
        st.caption("Not: SVM tahminleri şu an örnek (geçici) verilerdir. Kişi 3'ün modeli gelince gerçek olacak.")


if __name__ == "__main__":
    main()
