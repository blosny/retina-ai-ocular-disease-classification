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
    SVM tahmin         -> Kisi 3 (src/k3_oznitelik.py, v2 SVM Pipeline)

Gorunum:
    Tasarim (renkler, yazi tipi, masaustu/telefon yerlesimi) asagidaki STIL blogunda.
    Sonuc karti ve goruntu adimlari HTML olarak cizilir; boylece masaustunde yan yana,
    telefonda alt alta / ikili izgara olarak dizilir.
"""

import base64
import html

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
    "normal": "Normal",
    "diabetic_retinopathy": "Diyabetik retinopati",
    "glaucoma": "Glokom",
    "cataract": "Katarakt",
}
# Her sinifin arayuzdeki rengi (cubuk ve etiket noktasi)
SINIF_RENK = {
    "normal": "#2E8B57",
    "diabetic_retinopathy": "#C98A0B",
    "glaucoma": "#2F6FB5",
    "cataract": "#7A6C93",
}
SINIF_ACIKLAMA = {
    "normal": "Belirgin bir hastalık bulgusu görülmedi.",
    "diabetic_retinopathy": "Şeker hastalığının retina damarlarında bıraktığı lekeler ve kanamalar.",
    "glaucoma": "Optik sinir başındaki çukurun büyümesiyle ilişkili sinir hasarı.",
    "cataract": "Göz merceğindeki bulanıklık; fotoğraf puslu ve detaysız görünür.",
}
MODEL_SECENEKLERI = ["CNN (derin öğrenme)", "SVM (klasik)"]


# ============================================================
# STIL
# ============================================================

STIL = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --zemin: #EEF2F4;
  --kart: #FFFFFF;
  --murekkep: #16263D;
  --ikincil: #4E5D70;
  --soluk: #6F7D8C;
  --cizgi: #DCE3E8;
  --ana: #0B6E79;
  --ana-koyu: #08555E;
  --ana-acik: #E3F1F2;
  --uyari: #8A5200;
  --uyari-zemin: #FFF4E0;
}

html, body, .stApp, button, input, label,
[data-testid="stMarkdownContainer"] *, [data-testid="stSidebar"] *, .ra-kart *, .ra-ust * {
  font-family: 'Manrope', system-ui, sans-serif !important;
}
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: var(--zemin); color: var(--murekkep); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 3rem; }

/* ---------- Ust bolum ---------- */
.ra-ust { display: flex; align-items: center; gap: 26px; margin-bottom: 24px; }
.ra-retina { flex: 0 0 auto; width: 92px; height: 92px; }
.ra-baslik { font-size: 2.6rem !important; font-weight: 800 !important; letter-spacing: -0.02em; line-height: 1.05 !important;
  margin: 0 !important; padding: 0 !important; color: var(--murekkep) !important; }
.ra-alt { font-size: 1.12rem; color: var(--ikincil); margin: 8px 0 0; max-width: 62ch; line-height: 1.55; }

/* ---------- Kartlar ---------- */
.ra-kart { background: var(--kart); border: 1px solid var(--cizgi); border-radius: 18px; padding: 26px 28px; margin-bottom: 22px; }
.ra-kart h2 { font-size: 1.35rem !important; font-weight: 700 !important; margin: 0 0 4px !important; padding: 0 !important; color: var(--murekkep) !important; }
.ra-not { color: var(--soluk); font-size: 0.98rem; margin: 0 0 18px; }

/* Yukleme bolumunun Streamlit kutusu */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--kart); border-color: var(--cizgi) !important; border-radius: 18px !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 10px 12px; }
.ra-yukle-baslik { font-size: 1.35rem; font-weight: 700; color: var(--murekkep); margin: 0; }
.ra-yukle-not { color: var(--soluk); font-size: 0.98rem; margin: 2px 0 6px; }

/* ---------- Model secimi ---------- */
div[data-testid="stButtonGroup"] button {
  padding: 10px 20px !important; min-height: 46px; border-radius: 12px !important; border-color: var(--cizgi) !important;
  background: #fff !important; color: var(--murekkep) !important;
}
div[data-testid="stButtonGroup"] button p { font-size: 1.02rem !important; font-weight: 600; color: inherit !important; }
div[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--ana) !important; color: #fff !important; border-color: var(--ana) !important;
}
div[data-testid="stButtonGroup"] button:focus-visible { outline: 3px solid #7CC3CB; outline-offset: 2px; }

/* ---------- Yukleme alani ---------- */
[data-testid="stFileUploaderDropzone"] {
  min-height: 180px; border: 2px dashed #9FB7C0 !important; border-radius: 16px !important;
  background: #F7FAFB !important; padding: 28px 32px !important; align-items: center; gap: 24px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--ana) !important; background: var(--ana-acik) !important; }
/* Streamlit'in Ingilizce yazilarini Turkce ile degistir */
[data-testid="stFileUploaderDropzoneInstructions"] { display: flex; flex-direction: column; gap: 4px; }
[data-testid="stFileUploaderDropzoneInstructions"] * { font-size: 0 !important; line-height: 0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"]::before {
  content: "Fotoğrafı buraya sürükleyin"; font-size: 1.15rem; font-weight: 700; color: var(--murekkep); line-height: 1.4;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
  content: "JPG veya PNG, en fazla 200 MB"; font-size: 0.95rem; color: var(--soluk); line-height: 1.4;
}
[data-testid="stFileUploaderDropzone"] > span > button[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] > button[data-testid="stBaseButton-secondary"] {
  background: var(--ana) !important; border: none !important; border-radius: 12px !important;
  padding: 12px 28px !important; min-height: 52px; font-size: 0 !important; color: #fff !important;
}
[data-testid="stFileUploaderDropzone"] > span > button[data-testid="stBaseButton-secondary"] *,
[data-testid="stFileUploaderDropzone"] > button[data-testid="stBaseButton-secondary"] * { display: none !important; }
[data-testid="stFileUploaderDropzone"] > span > button[data-testid="stBaseButton-secondary"]::after,
[data-testid="stFileUploaderDropzone"] > button[data-testid="stBaseButton-secondary"]::after {
  content: "Fotoğraf seç"; font-size: 1.05rem; font-weight: 700; color: #fff;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover { background: var(--ana-koyu) !important; }
[data-testid="stFileUploaderDropzone"] button:focus-visible { outline: 3px solid #7CC3CB; outline-offset: 2px; }
[data-testid="stFileChip"] { font-size: 1rem; }

/* ---------- Sonuc karti ---------- */
.ra-sonuc { display: grid; grid-template-columns: 300px 1fr; gap: 32px; align-items: start; }
.ra-sonuc figure { margin: 0; }
.ra-sonuc img { width: 100%; border-radius: 50%; aspect-ratio: 1; object-fit: cover; background: #000; display: block; }
.ra-sonuc figcaption { text-align: center; color: var(--soluk); font-size: 0.88rem; margin-top: 8px; }
.ra-etiket { font-size: 0.98rem; color: var(--soluk); margin: 0; }
.ra-tani { display: flex; align-items: center; gap: 12px; margin: 2px 0 6px; }
.ra-tani .ra-nokta { width: 16px; height: 16px; border-radius: 50%; flex: 0 0 auto; }
.ra-tani .ra-ad { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.1; color: var(--murekkep); }
.ra-guven { font-size: 1.05rem; color: var(--ikincil); margin: 0 0 4px; }
.ra-guven b { color: var(--murekkep); font-size: 1.3rem; }
.ra-aciklama { color: var(--ikincil); margin: 0 0 18px; font-size: 1rem; line-height: 1.5; }
.ra-satir { display: grid; grid-template-columns: 190px 1fr 64px; align-items: center; gap: 14px; margin: 11px 0; font-size: 1rem; }
.ra-isim { display: flex; align-items: center; gap: 9px; color: var(--murekkep); }
.ra-isim i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }
.ra-cubuk { height: 12px; background: #E8EDF0; border-radius: 6px; overflow: hidden; }
.ra-cubuk div { height: 100%; border-radius: 6px; }
.ra-yuzde { text-align: right; font-variant-numeric: tabular-nums; color: var(--ikincil); font-weight: 600; }
.ra-secili .ra-isim, .ra-secili .ra-yuzde { font-weight: 800; color: var(--murekkep); }
.ra-meta { margin: 16px 0 0; color: var(--soluk); font-size: 0.9rem; }

.ra-kutu { border-radius: 14px; padding: 14px 18px; margin: 16px 0 0; font-size: 0.98rem; line-height: 1.5; }
.ra-kutu.uyari { background: var(--uyari-zemin); color: var(--uyari); border: 1px solid #F1D29A; }
.ra-kutu.bilgi { background: var(--ana-acik); color: var(--ana-koyu); border: 1px solid #BFDDE0; margin-bottom: 22px; }

/* ---------- Isleme adimlari (gercek bir sira oldugu icin numarali) ---------- */
.ra-adimlar { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }
.ra-adim figure { margin: 0; }
.ra-resim { position: relative; border-radius: 14px; overflow: hidden; background: #000; aspect-ratio: 1; }
.ra-resim img { width: 100%; height: 100%; object-fit: cover; display: block; }
.ra-bos { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; text-align: center;
  padding: 16px; color: #C9D2DA; font-size: 0.88rem; line-height: 1.45; background: #1D2733; }
.ra-resim img.ra-soluk { opacity: 0.55; filter: grayscale(0.5); }
.ra-etiket { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; padding: 10px; text-align: center; }
.ra-etiket b { background: rgba(22, 38, 61, 0.88); color: #FFFFFF; font-size: 0.8rem; font-weight: 700; line-height: 1.3;
  padding: 6px 10px; border-radius: 8px; }
.ra-olcum-kap { position: absolute; inset: 0; background: #1D2733; container-type: size; }
.ra-olcum { list-style: none; margin: 0; height: 100%; box-sizing: border-box;
  padding: 24cqh 9cqw 5cqh; display: flex; flex-direction: column; justify-content: space-between; }
.ra-olcum li { display: flex; justify-content: space-between; gap: 4cqw; font-size: 7.4cqh; line-height: 1.2; color: #C9D2DA;
  margin: 0 !important; padding: 0 !important; }
.ra-olcum li span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ra-olcum li b { color: #FFFFFF; font-weight: 700; font-variant-numeric: tabular-nums; }
.ra-no { position: absolute; top: 8px; left: 8px; width: 26px; height: 26px; border-radius: 50%; background: rgba(255,255,255,0.92);
  color: var(--murekkep); font-weight: 800; font-size: 0.85rem; display: flex; align-items: center; justify-content: center; }
.ra-adim figcaption { margin-top: 10px; }
.ra-adim figcaption b { display: block; font-size: 1rem; color: var(--murekkep); }
.ra-adim figcaption span { display: block; font-size: 0.88rem; color: var(--soluk); line-height: 1.4; margin-top: 2px; }

/* ---------- Bos durum ---------- */
.ra-siniflar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.ra-siniflar div { background: #F7FAFB; border: 1px solid var(--cizgi); border-radius: 14px; padding: 16px; }
.ra-siniflar b { display: flex; align-items: center; gap: 8px; font-size: 1rem; color: var(--murekkep); }
.ra-siniflar b i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.ra-siniflar span { display: block; color: var(--ikincil); font-size: 0.9rem; margin-top: 6px; line-height: 1.45; }
.ra-alt-bilgi { color: var(--soluk); font-size: 0.88rem; text-align: center; margin-top: 6px; }

/* ---------- Kenar cubugu ---------- */
[data-testid="stSidebar"] { background: #F7FAFB; border-right: 1px solid var(--cizgi); }
[data-testid="stSidebar"] h3 { font-size: 1.1rem !important; font-weight: 700 !important; color: var(--murekkep) !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { font-size: 0.95rem; color: var(--ikincil); line-height: 1.5; }

/* ---------- Tablet ---------- */
@media (max-width: 960px) {
  .ra-adimlar { grid-template-columns: repeat(3, 1fr); }
  .ra-sonuc { grid-template-columns: 220px 1fr; gap: 22px; }
  .ra-siniflar { grid-template-columns: 1fr 1fr; }
}
/* ---------- Telefon ---------- */
@media (max-width: 640px) {
  .block-container { padding: 3.2rem 0.9rem 2.4rem !important; }
  .ra-ust { gap: 14px; margin-bottom: 18px; align-items: flex-start; }
  .ra-retina { width: 60px; height: 60px; }
  .ra-baslik { font-size: 2rem !important; }
  .ra-alt { font-size: 1rem; }
  .ra-kart { padding: 20px 18px; border-radius: 16px; }
  .ra-sonuc { grid-template-columns: 1fr; gap: 18px; }
  .ra-sonuc figure { max-width: 230px; margin: 0 auto; }
  .ra-tani .ra-ad { font-size: 1.85rem; }
  .ra-satir { grid-template-columns: 1fr 64px; row-gap: 6px; margin: 14px 0; }
  .ra-satir .ra-cubuk { grid-column: 1 / -1; grid-row: 2; height: 14px; }
  .ra-adimlar { grid-template-columns: 1fr 1fr; gap: 12px; }
  .ra-siniflar { grid-template-columns: 1fr; }
  [data-testid="stFileUploaderDropzone"] { flex-direction: column; min-height: 210px; text-align: center; gap: 16px; padding: 24px 18px !important; }
  [data-testid="stFileUploaderDropzone"] > span, [data-testid="stFileUploaderDropzone"] > span > button[data-testid="stBaseButton-secondary"] { width: 100%; }
  [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] { min-height: 56px; }
  div[data-testid="stButtonGroup"] > div { width: 100%; }
  div[data-testid="stButtonGroup"] button { flex: 1 1 0; min-height: 52px; padding: 8px 8px !important; }
  div[data-testid="stButtonGroup"] button p { font-size: 0.95rem !important; white-space: normal !important;
    overflow: visible !important; text-overflow: clip !important; line-height: 1.25; }
  div[data-testid="stButtonGroup"] button div, div[data-testid="stButtonGroup"] button span { overflow: visible !important; max-width: none !important; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""

# Ustteki soyut retina cizimi (turuncu goz dibi, optik disk ve damarlar)
RETINA_SVG = (
    '<svg class="ra-retina" viewBox="0 0 100 100" role="img" aria-label="Retina çizimi">'
    '<defs><radialGradient id="rg" cx="45%" cy="45%" r="60%">'
    '<stop offset="0" stop-color="#F0A04B"/><stop offset="1" stop-color="#B4471F"/></radialGradient></defs>'
    '<circle cx="50" cy="50" r="48" fill="#16263D"/><circle cx="50" cy="50" r="42" fill="url(#rg)"/>'
    '<g fill="none" stroke="#7A1F12" stroke-width="1.6" stroke-linecap="round" opacity="0.85">'
    '<path d="M68 46 C58 34 44 28 26 30"/><path d="M68 46 C56 40 42 40 30 46"/>'
    '<path d="M68 50 C58 60 44 66 28 66"/><path d="M68 50 C60 58 50 72 44 84"/>'
    '<path d="M68 46 C66 32 60 22 52 14"/></g>'
    '<circle cx="69" cy="48" r="7.5" fill="#FBE3A6"/><circle cx="69" cy="48" r="3.2" fill="#FFF6DC"/>'
    '<circle cx="44" cy="50" r="3.6" fill="#8C2A14" opacity="0.7"/></svg>'
)


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

def svm_tahmin(gorsel: np.ndarray) -> dict:
    """
    KISI 3 (SVM) - Gercek v2 SVM Pipeline modeli ile tahmin.
    Gorselden medikal oznitelikleri cikarir ve egitilmis SVM modeli uzerinden
    sinif olasiliklarini dondurur.
    """
    try:
        from k3_oznitelik import k3_tahmin_et
    except ImportError:
        from src.k3_oznitelik import k3_tahmin_et
    return k3_tahmin_et(gorsel, bgr_mi=True)


# ============================================================
# YARDIMCI (Kisi 5 - kalici)
# ============================================================

def isi_haritasi_bindir(renkli: np.ndarray, harita: np.ndarray) -> np.ndarray:
    """Isi haritasini renkli retinanin ustune yari saydam bindirir."""
    renkli_harita = cv2.applyColorMap((harita * 255).astype(np.uint8), cv2.COLORMAP_JET)
    renkli_harita = cv2.cvtColor(renkli_harita, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(renkli, 0.6, renkli_harita, 0.4, 0)


def resim_veri(goruntu: np.ndarray, boyut: int = 420) -> str:
    """Goruntuyu HTML icinde gosterilebilecek kucuk bir JPEG'e (base64) cevirir."""
    if goruntu.dtype != np.uint8:
        goruntu = np.clip(goruntu * 255 if goruntu.max() <= 1.0 else goruntu, 0, 255).astype(np.uint8)
    if goruntu.ndim == 2:
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_GRAY2RGB)
    goruntu = cv2.resize(goruntu, (boyut, boyut), interpolation=cv2.INTER_AREA)
    tamam, tampon = cv2.imencode(".jpg", cv2.cvtColor(goruntu, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(tampon.tobytes()).decode() if tamam else ""


def html_yaz(icerik: str):
    """HTML parcasini sayfaya basar (Markdown bosluklarindan etkilenmesin diye satirlari birlestirir)."""
    st.markdown(" ".join(satir.strip() for satir in icerik.splitlines()), unsafe_allow_html=True)


def ust_bolum():
    html_yaz(f"""<div class="ra-ust">{RETINA_SVG}<div>
        <h1 class="ra-baslik">RETINA-AI</h1>
        <p class="ra-alt">Göz dibi (fundus) fotoğrafını yükleyin. Sistem fotoğrafı işler ve normal, diyabetik
        retinopati, glokom ya da katarakt sınıflarından hangisine benzediğini gösterir.</p></div></div>""")


def sonuc_karti(olasiliklar: dict, gorsel: np.ndarray, gorsel_aciklama: str, model_etiketi: str):
    en_iyi = max(olasiliklar, key=olasiliklar.get)
    guven = olasiliklar[en_iyi]
    sirali = sorted(olasiliklar.items(), key=lambda kv: kv[1], reverse=True)
    satirlar = "".join(
        f'<div class="ra-satir{" ra-secili" if s == en_iyi else ""}">'
        f'<div class="ra-isim"><i style="background:{SINIF_RENK[s]}"></i>{SINIFLAR[s]}</div>'
        f'<div class="ra-cubuk"><div style="width:{p * 100:.1f}%;background:{SINIF_RENK[s]}"></div></div>'
        f'<div class="ra-yuzde">%{p * 100:.1f}</div></div>'
        for s, p in sirali
    )
    kararsizlik = ""
    if guven < KARARSIZLIK_ESIGI:
        (s1, p1), (s2, p2) = sirali[0], sirali[1]
        kararsizlik = (f'<div class="ra-kutu uyari"><b>Model bu görüntüde kararsız.</b> {SINIFLAR[s1]} %{p1 * 100:.1f}, '
                       f"{SINIFLAR[s2]} %{p2 * 100:.1f}. Bu sonuç güvenilir değil.</div>")

    html_yaz(f"""<div class="ra-kart"><div class="ra-sonuc">
        <figure><img src="{resim_veri(gorsel)}" alt="{html.escape(gorsel_aciklama)}">
        <figcaption>{html.escape(gorsel_aciklama)}</figcaption></figure>
        <div><p class="ra-etiket">Tahmin</p>
        <div class="ra-tani"><span class="ra-nokta" style="background:{SINIF_RENK[en_iyi]}"></span>
        <span class="ra-ad">{SINIFLAR[en_iyi]}</span></div>
        <p class="ra-guven">Model güveni <b>%{guven * 100:.1f}</b></p>
        <p class="ra-aciklama">{SINIF_ACIKLAMA[en_iyi]}</p>
        {satirlar}{kararsizlik}
        <p class="ra-meta">Kullanılan model: {html.escape(model_etiketi)}</p>
        </div></div></div>""")


def adimlar_karti(adimlar: list):
    """adimlar: [(baslik, aciklama, goruntu veya None, bos_mesaj)]
    bos_mesaj yerine ("html", "<...>") verilirse kutuya hazır HTML yazılır (SVM ölçümleri için)."""
    hucreler = ""
    for i, (baslik, aciklama, goruntu, bos) in enumerate(adimlar, start=1):
        if goruntu is not None:
            icerik = f'<img src="{resim_veri(goruntu)}" alt="{html.escape(baslik)}">'
        elif isinstance(bos, tuple) and bos[0] == "html":
            icerik = bos[1]
        else:
            icerik = f'<div class="ra-bos">{html.escape(bos)}</div>'
        hucreler += (f'<div class="ra-adim"><figure><div class="ra-resim">{icerik}<span class="ra-no">{i}</span></div>'
                     f"<figcaption><b>{html.escape(baslik)}</b><span>{html.escape(aciklama)}</span></figcaption>"
                     "</figure></div>")
    html_yaz(f"""<div class="ra-kart"><h2>Görüntü işleme adımları</h2>
        <p class="ra-not">Fotoğrafın sisteme girdikten sonra geçtiği aşamalar.</p>
        <div class="ra-adimlar">{hucreler}</div></div>""")


SVM_OLCUMLERI = [  # (anahtar, arayuzde gorunen ad, ondalik basamak)
    ("laplacian_varyansi", "Netlik", 0),
    ("ortalama_kontrast", "Kontrast", 1),
    ("damar_orani", "Damar skoru", 3),
    ("sari_lezyon_sayisi", "Sarı leke", 0),
    ("sari_lezyon_alani", "Leke alanı", 0),
    ("kanama_sayisi", "Kanama", 0),
    ("kanama_alani", "Kanama alanı", 0),
]


def svm_olcum_kutusu(oz: dict) -> str:
    """SVM'nin kararını verirken baktığı 7 ölçümü küçük bir liste olarak döndürür."""
    satirlar = "".join(
        f"<li><span>{ad}</span><b>{oz[anahtar]:.{basamak}f}</b></li>"
        for anahtar, ad, basamak in SVM_OLCUMLERI if anahtar in oz)
    return f'<div class="ra-olcum-kap"><ul class="ra-olcum">{satirlar}</ul></div>'


def soluk_kutu(goruntu: np.ndarray, mesaj: str) -> str:
    """Bulunamayan adımlar için: fotoğrafı soluk gösterip üstüne kısa bir etiket koyar."""
    return (f'<img class="ra-soluk" src="{resim_veri(goruntu)}" alt="">'
            f'<div class="ra-etiket"><b>{html.escape(mesaj)}</b></div>')


def bos_durum():
    hucreler = "".join(f'<div><b><i style="background:{SINIF_RENK[s]}"></i>{SINIFLAR[s]}</b>'
                       f"<span>{SINIF_ACIKLAMA[s]}</span></div>" for s in SINIFLAR)
    html_yaz(f"""<div class="ra-kart"><h2>Sistemin tanıdığı sınıflar</h2>
        <p class="ra-not">Bir fotoğraf yüklediğinizde tahmin ve işleme adımları burada görünür.</p>
        <div class="ra-siniflar">{hucreler}</div></div>
        <p class="ra-alt-bilgi">Bu sistem teşhis koymaz; yalnızca eğitim amaçlı bir simülasyondur.</p>""")


def model_secimi() -> str:
    if hasattr(st, "segmented_control"):
        secim = st.segmented_control("Model", MODEL_SECENEKLERI, default=MODEL_SECENEKLERI[0],
                                     label_visibility="collapsed")
        return secim or MODEL_SECENEKLERI[0]
    return st.radio("Model", MODEL_SECENEKLERI, horizontal=True, label_visibility="collapsed")


def kenar_cubugu():
    with st.sidebar:
        st.markdown("### Proje hakkında")
        st.markdown("Medikal Görüntü İşleme dersi grup projesi. Kaggle fundus veri seti, temizlenmiş v2 bölmesi "
                    "(sınıf başına 288 görüntü) ile eğitildi.")
        st.markdown("### Modeller")
        st.markdown("- **CNN:** EfficientNet-B0, fotoğrafa doğrudan bakar. Test doğruluğu %80.7.\n"
                    "- **SVM:** Damar, lezyon ve netlik ölçümlerine bakar. Test doğruluğu %61.4.")
        st.markdown("### Sınırlılıklar")
        st.markdown("Glokom ve normal en çok karışan sınıflar. Karanlık veya bulanık fotoğraflar katarakt "
                    "sanılabilir. Eğitim verisine benzemeyen fotoğraflarda sonuç güvenilir değildir.")


# ============================================================
# ARAYUZ
# ============================================================

def main():
    st.set_page_config(page_title="RETINA-AI", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(STIL, unsafe_allow_html=True)
    kenar_cubugu()
    ust_bolum()

    with st.container(border=True):
        html_yaz('<p class="ra-yukle-baslik">Fotoğraf yükle</p>'
                 '<p class="ra-yukle-not">Önce modeli seçin, sonra göz dibi fotoğrafını sürükleyin ya da seçin.</p>')
        model_turu = model_secimi()
        dosya = st.file_uploader("Retina (fundus) fotoğrafı", type=["jpg", "jpeg", "png"],
                                 label_visibility="collapsed")

    cnn_secili = model_turu.startswith("CNN")
    model, model_hatasi = cnn_modelini_yukle() if cnn_secili else (None, None)
    if model_hatasi:
        st.error(model_hatasi)
        return

    if dosya is None:
        bos_durum()
        return

    ham = np.array(Image.open(dosya).convert("RGB"))
    bgr = cv2.cvtColor(ham, cv2.COLOR_RGB2BGR)  # Kisi 1 ve Kisi 4'un kodu BGR bekliyor

    with st.spinner("Fotoğraf işleniyor..."):
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
            model_etiketi = f"CNN (EfficientNet-B0), {cnn_model_adi()}"
        else:
            olasiliklar = svm_tahmin(bgr)
            isi = None
            try:
                from k3_oznitelik import oznitelik_cikar
            except ImportError:
                from src.k3_oznitelik import oznitelik_cikar
            svm_olcumleri = oznitelik_cikar(bgr, bgr_mi=True)  # SVM'e giden 7 sayi (egitimdekiyle ayni girdi)
            model_etiketi = "SVM (RBF Kernel v2, %61.4 test doğruluğu)"

    if isi is not None:
        sonuc_karti(olasiliklar, isi, "Grad-CAM: kırmızı bölgeler modelin en çok baktığı yerler", model_etiketi)
    else:
        sonuc_karti(olasiliklar, renkli, "Yüklenen fotoğraf", model_etiketi)
        html_yaz('<div class="ra-kutu bilgi">SVM fotoğrafın kendisine değil, ondan ölçülen 7 sayıya bakar '
                 "(netlik, kontrast, damar oranı, sarı leke ve kanama ölçümleri). Bu sayılar aşağıdaki 5. kutuda görünür; SVM Grad-CAM üretmez.</div>")

    disk_aciklama = (f"Çukur/disk oranı yaklaşık {disk['cukur_disk_orani']}." if disk is not None
                     else "Fotoğraf bulanık ya da karanlık; sistem yanlış yer göstermek yerine aramayı bıraktı.")
    adimlar_karti([
        ("Kırpılmış retina", "Siyah kenarlar kesildi, 512×512'ye getirildi.", renkli, ""),
        ("Ön işleme", "Yeşil kanal ve CLAHE ile damarlar belirginleşti.", islenmis, ""),
        ("Damar haritası", f"Damarların retina alanına oranı {yogunluk:.3f}.", damarlar, ""),
        ("Optik disk", disk_aciklama, disk_gorsel if disk is not None else None,
         ("html", soluk_kutu(renkli, "Disk bulunamadı"))),
        ("Grad-CAM", "Modelin karar verirken en çok baktığı bölgeler kırmızı.", isi,
         "Grad-CAM yalnızca CNN modelinde gösterilir.") if cnn_secili else
        ("SVM ölçümleri", "SVM fotoğrafa değil, bu 7 sayıya bakarak karar verdi.", None,
         ("html", svm_olcum_kutusu(svm_olcumleri))),
    ])

    html_yaz('<div class="ra-kutu uyari">Bu sonuç yalnızca akademik simülasyon amaçlıdır ve tıbbi teşhis yerine '
             "geçmez. Karanlık, bulanık veya kötü çekilmiş fotoğraflar katarakt olarak sınıflandırılabilir.</div>")


if __name__ == "__main__":
    main()
