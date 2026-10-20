"""
RETINA-AI - Streamlit arayuzu (Kisi 5: Degerlendirme ve Streamlit)

Calistirma (proje ana klasorunden):
    streamlit run src/app.py

Gercek kod:
    on_isle            -> Kisi 1 (src/preprocessing.py)
    damar + optik disk -> Kisi 2 (src/anatomy.py)
    CNN tahmin+Grad-CAM -> Kisi 4 (src/cnn_models.py)
                          Model dosyasi: models/model_efficientnet_v2.pth (ortak Drive'dan indirilir)
                          Eski v1 modeli (model_efficientnet.pth) sizinti icerdigi icin kullanilmaz.
Gecici dublorler:
    svm_tahmin         -> Kisi 3 (gercek goruntulerle ve v2 bolmesiyle egitilmis SVM gelince degisecek)
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# KISI 1 - Gercek on isleme kodu
from preprocessing import boyutlandir, on_isle, roi_kirp

# KISI 2 - Gercek damar ve optik disk kodu
from anatomy import damar_haritasi, damar_yogunlugu, disk_isaretle, optik_disk

# En yuksek olasilik bu esigin altindaysa arayuz "kararsiz tahmin" uyarisi gosterir
KARARSIZLIK_ESIGI = 0.60

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
            f"Ortak Drive'daki models/{VARSAYILAN_MODEL_YOLU.name} dosyasını proje klasöründeki "
            "models klasörüne koyun. Eski model_efficientnet.pth (v1) kullanılmamalı."
        )
    return model_yukle(VARSAYILAN_MODEL_YOLU), None


def cnn_model_adi() -> str:
    """Arayuzde gosterilecek model dosyasinin adi (orn. model_efficientnet_v2.pth)."""
    from cnn_models import VARSAYILAN_MODEL_YOLU
    return VARSAYILAN_MODEL_YOLU.name


# ============================================================
# GECICI DUBLORLER - sahipleri bitirince degistir
# ============================================================

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

        # KISI 2: damar haritasi, damar yogunlugu ve optik disk
        damarlar = damar_haritasi(islenmis)
        yogunluk = damar_yogunlugu(damarlar, islenmis)
        disk = optik_disk(renkli)  # renkli 512x512 RGB alir, bulamazsa None doner
        disk_gorsel = disk_isaretle(renkli, disk)

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
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.image(renkli, caption="Kırpılmış retina")
    k2.image(islenmis, caption="Ön işlenmiş (yeşil kanal + CLAHE)", clamp=True)
    k3.image(damarlar, caption=f"Damar haritası (yoğunluk: {yogunluk:.3f})")
    if isi is not None:
        k4.image(isi, caption=isi_aciklama)
    else:
        k4.info("Grad-CAM yalnızca CNN modelinde gösterilir.")
    if disk is None:
        k5.warning("Optik disk güvenilir şekilde bulunamadı (bulanık, karanlık veya disk görüntü dışında olabilir).")
    else:
        k5.image(
            disk_gorsel,
            caption=f"Optik disk (çukur/disk oranı ≈ {disk['cukur_disk_orani']}, yaklaşık değer)",
        )

    st.subheader("2. Sonuç")
    en_iyi = max(olasiliklar, key=olasiliklar.get)
    sol, sag = st.columns([1, 2])
    with sol:
        st.markdown(f"#### Tahmin: {SINIFLAR[en_iyi]}")
        st.metric("Model güven değeri", f"%{olasiliklar[en_iyi] * 100:.1f}")
        st.caption(f"Kullanılan model: {model_turu}" + (f" · {cnn_model_adi()}" if cnn_secili else ""))
    with sag:
        for sinif, p in sorted(olasiliklar.items(), key=lambda kv: kv[1], reverse=True):
            st.write(f"{SINIFLAR[sinif]} · %{p * 100:.1f}")
            st.progress(p)

    if olasiliklar[en_iyi] < KARARSIZLIK_ESIGI:
        ilk_iki = sorted(olasiliklar.items(), key=lambda kv: kv[1], reverse=True)[:2]
        st.warning(
            f"⚠️ Model bu görüntüde kararsız ({SINIFLAR[ilk_iki[0][0]]} %{ilk_iki[0][1] * 100:.1f} / "
            f"{SINIFLAR[ilk_iki[1][0]]} %{ilk_iki[1][1] * 100:.1f}). Sonuç güvenilir değil."
        )

    st.warning(
        "⚠️ Bu sonuç yalnızca akademik simülasyon amaçlıdır. Tıbbi teşhis yerine geçmez. "
        "Karanlık, bulanık veya kötü çekilmiş fotoğraflar katarakt olarak sınıflandırılabilir."
    )
    if not cnn_secili:
        st.caption("Not: SVM tahminleri şu an örnek (geçici) verilerdir. Kişi 3'ün gerçek görüntülerle ve v2 ile eğitilmiş modeli gelince gerçek olacak.")


if __name__ == "__main__":
    main()