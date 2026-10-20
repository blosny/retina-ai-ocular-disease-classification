"""
Derin Ogrenme Modulu (Kisi 4) - EfficientNet-B0 + Grad-CAM
----------------------------------------------------------
Arayuz (Kisi 5) icin disariya acilan 3 fonksiyon:

    model_yukle(yol=None)            -> egitilmis model (bir kez yuklenir, onbellekte tutulur)
    tahmin(goruntu)                  -> {"normal": 0.10, "diabetic_retinopathy": 0.70,
                                         "glaucoma": 0.15, "cataract": 0.05}
    gradcam(goruntu, sinif=None)     -> 512x512, float32, 0-1 arasi isi haritasi

`goruntu`: on_isle() ile AYNI girdi -> dosya yolu (str/Path) veya BGR numpy dizisi
(cv2.imread ciktisi). Kirpma, boyutlandirma ve ImageNet normalizasyonu fonksiyonun
icinde yapilir. Isi haritasi on_isle() ciktisiyla ayni koordinatlardadir
(ayni roi_kirp + 512x512 boyutlandirma), yani dogrudan ustune bindirilebilir.

Neden on_isle() ciktisi degil de ham goruntu?
    on_isle() tek kanalli (yesil + CLAHE) goruntu dondurur. CNN ise renkli goruntuyle
    daha iyi ogrenir: kataraktin pusu ve DR'deki sari eksudalar renkte belirgindir.
    Yesil kanal + CLAHE ile egitmek istenirse: egit(..., girdi_modu="yesil_clahe").
    Secilen mod model dosyasina kaydedilir, tahmin/gradcam otomatik olarak ona uyar.

Egitim (Colab):
    from src.cnn_models import egit
    egit(proje_koku, veri_koku=Path('/content/dataset'))

Komut satiri:
    python src/cnn_models.py egit --veri-koku /content/dataset
    python src/cnn_models.py test --veri-koku /content/dataset
    python src/cnn_models.py tahmin ornek.jpg
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:  # src klasoru sys.path'teyse (app.py boyle calisiyor)
    from preprocessing import boyutlandir, on_isle, roi_kirp
except ImportError:  # proje koku sys.path'teyse (notebook'lar boyle calisiyor)
    from src.preprocessing import boyutlandir, on_isle, roi_kirp


# ============================================================
# ORTAK SABITLER (ekipte anlasilan isimler)
# ============================================================

SINIFLAR = ["normal", "diabetic_retinopathy", "glaucoma", "cataract"]
CIKTI_BOYUTU = 512          # on_isle() ile ayni
GORUNTU_BOYUTU = 384        # CNN'e giren boyut (Colab'da yavas kalirsa 224 yapin)
IMAGENET_ORT = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJE_KOKU = Path(__file__).resolve().parent.parent
# v2: veri_bolme_v2.csv ile egitilmis model. Eski (v1) model sizinti/kisayol icerdigi icin kullanilmamali.
VARSAYILAN_MODEL_YOLU = PROJE_KOKU / "models" / "model_efficientnet_v2.pth"
# Hangi bolme dosyasi kullanilacak (data/splits/ altinda). Ekip karariyla varsayilan v2.
# Eski bolmeyle karsilastirma icin: cnn.BOLME_DOSYASI = "veri_bolme.csv"; cnn.DOSYA_EKI = ""
BOLME_DOSYASI = "veri_bolme_v2.csv"
# Uretilen rapor dosyalarinin adina eklenir (cnn_test_tahminleri_v2.csv gibi)
DOSYA_EKI = "_v2"

Girdi = Union[str, Path, np.ndarray]


# ============================================================
# MODEL
# ============================================================

def model_olustur(sinif_sayisi: int = len(SINIFLAR), onegitimli: bool = True,
                  dropout: float = 0.3) -> nn.Module:
    """ImageNet'te egitilmis EfficientNet-B0, son katmani 4 sinifa uyarlanmis."""
    agirliklar = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if onegitimli else None
    model = models.efficientnet_b0(weights=agirliklar)
    giris = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(giris, sinif_sayisi))
    return model


# ============================================================
# GIRDI HAZIRLAMA
# ============================================================

def _bgr_oku(girdi: Girdi) -> np.ndarray:
    if isinstance(girdi, (str, Path)):
        bgr = cv2.imread(str(girdi))
        if bgr is None:
            raise FileNotFoundError(f"Gorsel okunamadi: {girdi}")
        return bgr
    if isinstance(girdi, np.ndarray):
        if girdi.ndim != 3:
            raise ValueError(
                "CNN renkli (BGR, 3 kanal) goruntu bekler. on_isle() ciktisini degil, "
                "ham goruntuyu (cv2.imread ciktisi) verin."
            )
        return girdi
    raise TypeError("Girdi dosya yolu veya BGR numpy dizisi olmalidir.")


def cnn_girdisi_hazirla(girdi: Girdi, girdi_modu: str = "rgb") -> np.ndarray:
    """
    Ham goruntuyu 512x512x3 uint8 RGB diziye cevirir (on_isle ile ayni kirpma).
    girdi_modu="rgb"         : kirpilmis renkli retina
    girdi_modu="yesil_clahe" : on_isle() ciktisi, 3 kanala kopyalanmis
    """
    bgr = _bgr_oku(girdi)
    if girdi_modu == "rgb":
        kirpilmis = roi_kirp(bgr, esik=10, dairesel_maske=True)
        boyutlu = boyutlandir(kirpilmis, (CIKTI_BOYUTU, CIKTI_BOYUTU))
        return cv2.cvtColor(boyutlu, cv2.COLOR_BGR2RGB)
    if girdi_modu == "yesil_clahe":
        gri = on_isle(bgr, hedef_boyut=(CIKTI_BOYUTU, CIKTI_BOYUTU), normalize=False)
        return np.repeat(gri[:, :, None], 3, axis=2)
    raise ValueError(f"Bilinmeyen girdi_modu: {girdi_modu}")


def donusum_olustur(boyut: int, egitim: bool) -> transforms.Compose:
    """
    Veri artirma: yatay cevirme + hafif dondurme.
    BULANIKLASTIRMA ve parlaklik/kontrast oynatma YOK: bulaniklik kataraktin ipucu.
    """
    adimlar = [transforms.ToPILImage(), transforms.Resize((boyut, boyut))]
    if egitim:
        adimlar += [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15)]
    adimlar += [transforms.ToTensor(), transforms.Normalize(IMAGENET_ORT, IMAGENET_STD)]
    return transforms.Compose(adimlar)


# ============================================================
# VERI SETI
# ============================================================

def dosya_yolunu_bul(satir: pd.Series, veri_koku: Optional[Path]) -> Path:
    """Once veri_koku/relative_path, olmazsa CSV'deki tam yol (file_path) denenir."""
    if veri_koku is not None:
        aday = Path(veri_koku) / satir["relative_path"]
        if aday.exists():
            return aday
    aday = Path(satir["file_path"])
    if aday.exists():
        return aday
    raise FileNotFoundError(
        f"{satir['relative_path']} bulunamadi. veri_koku altinda "
        f"<sinif>/<dosya>.jpg yapisi olmali (verilen: {veri_koku})."
    )


def veri_kokunu_tahmin_et(proje_koku: Path) -> Optional[Path]:
    """Drive / yerel klasor yapisinda goruntulerin oldugu yeri arar."""
    adaylar = [
        Path("/content/dataset"),                        # Colab'a kopyalanmis hali (hizli)
        proje_koku / "data" / "raw",
        proje_koku / "archive" / "dataset",
        proje_koku / "dataset",
        proje_koku / "data" / "raw" / "dataset",
    ]
    for aday in adaylar:
        if (aday / "glaucoma").is_dir():
            return aday
    return None


class FundusVeriSeti(Dataset):
    def __init__(self, df: pd.DataFrame, veri_koku: Optional[Path], boyut: int,
                 egitim: bool, girdi_modu: str = "rgb"):
        self.df = df.reset_index(drop=True)
        self.veri_koku = veri_koku
        self.girdi_modu = girdi_modu
        self.donusum = donusum_olustur(boyut, egitim)
        self.etiketler = [SINIFLAR.index(s) for s in self.df["class_name"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        yol = dosya_yolunu_bul(self.df.iloc[i], self.veri_koku)
        rgb = cnn_girdisi_hazirla(yol, self.girdi_modu)
        return self.donusum(rgb), self.etiketler[i]


# ============================================================
# EGITIM
# ============================================================

def _tohum_ayarla(tohum: int) -> None:
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    torch.cuda.manual_seed_all(tohum)


def _bir_tur(model, yukleyici, kriter, cihaz, optimizer=None, olcekleyici=None):
    """Bir epoch calistirir. optimizer verilirse egitim, verilmezse dogrulama."""
    egitim = optimizer is not None
    model.train(egitim)
    toplam_kayip, dogru, adet = 0.0, 0, 0
    with torch.set_grad_enabled(egitim):
        for x, y in yukleyici:
            x, y = x.to(cihaz, non_blocking=True), y.to(cihaz, non_blocking=True)
            with torch.autocast(device_type=cihaz.type, enabled=cihaz.type == "cuda"):
                cikti = model(x)
                kayip = kriter(cikti, y)
            if egitim:
                optimizer.zero_grad(set_to_none=True)
                olcekleyici.scale(kayip).backward()
                olcekleyici.step(optimizer)
                olcekleyici.update()
            toplam_kayip += kayip.item() * y.size(0)
            dogru += (cikti.argmax(1) == y).sum().item()
            adet += y.size(0)
    return toplam_kayip / adet, dogru / adet


def egit(
    proje_koku: Path = PROJE_KOKU,
    veri_koku: Optional[Path] = None,
    model_yolu: Optional[Path] = None,
    epoch: int = 30,
    sabir: int = 5,
    batch: int = 16,
    lr: float = 3e-4,
    boyut: int = GORUNTU_BOYUTU,
    girdi_modu: str = "rgb",
    num_workers: int = 2,
    tohum: int = 42,
    onegitimli: bool = True,
    ornek_sayisi: Optional[int] = None,
    df: Optional[pd.DataFrame] = None,
    rapor_yaz: bool = True,
) -> Dict:
    """
    EfficientNet-B0'i transfer learning ile egitir.
    - Erken durdurma: dogrulama kaybi `sabir` epoch iyilesmezse durur.
    - En iyi model (en dusuk val kaybi) models/model_efficientnet.pth olarak kaydedilir.
    - Kayip/dogruluk grafigi: reports/figures/cnn_egitim_grafigi.png
    - ornek_sayisi: hizli deneme icin her bolumden en fazla bu kadar goruntu kullanilir.
    """
    proje_koku = Path(proje_koku)
    _tohum_ayarla(tohum)
    veri_koku = Path(veri_koku) if veri_koku else veri_kokunu_tahmin_et(proje_koku)
    model_yolu = Path(model_yolu) if model_yolu else proje_koku / "models" / "model_efficientnet.pth"
    model_yolu.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = pd.read_csv(proje_koku / "data" / "splits" / BOLME_DOSYASI)
    df_train, df_val = df[df["split"] == "train"], df[df["split"] == "val"]
    if ornek_sayisi:
        df_train = df_train.groupby("class_name").head(ornek_sayisi // len(SINIFLAR) + 1)
        df_val = df_val.groupby("class_name").head(max(ornek_sayisi // 4, 4) // len(SINIFLAR) + 1)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BILGI] Cihaz: {cihaz} | veri: {veri_koku} | train={len(df_train)} val={len(df_val)}")

    yukle = dict(batch_size=batch, num_workers=num_workers, pin_memory=cihaz.type == "cuda")
    train_dl = DataLoader(FundusVeriSeti(df_train, veri_koku, boyut, True, girdi_modu),
                          shuffle=True, drop_last=len(df_train) > batch, **yukle)
    val_dl = DataLoader(FundusVeriSeti(df_val, veri_koku, boyut, False, girdi_modu),
                        shuffle=False, **yukle)

    model = model_olustur(onegitimli=onegitimli).to(cihaz)
    kriter = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    zamanlayici = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    olcekleyici = torch.amp.GradScaler(enabled=cihaz.type == "cuda")

    gecmis = {"train_kayip": [], "val_kayip": [], "train_dogruluk": [], "val_dogruluk": [], "lr": []}
    en_iyi_kayip, iyilesmeyen = float("inf"), 0

    for ep in range(1, epoch + 1):
        t0 = time.time()
        tr_k, tr_d = _bir_tur(model, train_dl, kriter, cihaz, optimizer, olcekleyici)
        va_k, va_d = _bir_tur(model, val_dl, kriter, cihaz)
        zamanlayici.step(va_k)

        for anahtar, deger in zip(gecmis, [tr_k, va_k, tr_d, va_d, optimizer.param_groups[0]["lr"]]):
            gecmis[anahtar].append(deger)
        print(f"Epoch {ep:02d} | train kayip {tr_k:.4f} dogr {tr_d:.3f} | "
              f"val kayip {va_k:.4f} dogr {va_d:.3f} | {time.time() - t0:.0f} sn")

        if va_k < en_iyi_kayip:
            en_iyi_kayip, iyilesmeyen = va_k, 0
            torch.save({
                "model_durumu": model.state_dict(),
                "siniflar": SINIFLAR,
                "girdi_modu": girdi_modu,
                "goruntu_boyutu": boyut,
                "epoch": ep,
                "val_kayip": va_k,
                "val_dogruluk": va_d,
            }, model_yolu)
            print(f"   -> en iyi model kaydedildi ({model_yolu.name})")
        else:
            iyilesmeyen += 1
            if iyilesmeyen >= sabir:
                print(f"[BILGI] Erken durdurma: {sabir} epoch boyunca val kaybi iyilesmedi.")
                break

    if rapor_yaz:
        rapor = proje_koku / "reports"
        (rapor / "figures").mkdir(parents=True, exist_ok=True)
        (rapor / f"cnn_egitim_gecmisi{DOSYA_EKI}.json").write_text(json.dumps(gecmis, indent=2))
        egitim_grafigi_ciz(gecmis, rapor / "figures" / f"cnn_egitim_grafigi{DOSYA_EKI}.png")
    _onbellek.clear()
    return gecmis


def egitim_grafigi_ciz(gecmis: Dict, cikti_yolu: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochlar = range(1, len(gecmis["train_kayip"]) + 1)
    en_iyi = int(np.argmin(gecmis["val_kayip"])) + 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(epochlar, gecmis["train_kayip"], "o-", label="Egitim")
    a1.plot(epochlar, gecmis["val_kayip"], "o-", label="Dogrulama")
    a1.set(title="Kayip (Loss)", xlabel="Epoch", ylabel="Cross-entropy")
    a2.plot(epochlar, gecmis["train_dogruluk"], "o-", label="Egitim")
    a2.plot(epochlar, gecmis["val_dogruluk"], "o-", label="Dogrulama")
    a2.set(title="Dogruluk (Accuracy)", xlabel="Epoch", ylabel="Dogruluk", ylim=(0, 1))
    for a in (a1, a2):
        a.axvline(en_iyi, color="gray", ls="--", lw=1, label=f"En iyi model (epoch {en_iyi})")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("RETINA-AI: EfficientNet-B0 Egitim Sureci")
    fig.tight_layout()
    fig.savefig(cikti_yolu, dpi=150)
    plt.close(fig)
    print(f"[BILGI] Egitim grafigi kaydedildi: {cikti_yolu}")


# ============================================================
# ARAYUZ FONKSIYONLARI (Kisi 5 bunlari cagirir)
# ============================================================

_onbellek: Dict[str, nn.Module] = {}


def model_yukle(yol: Optional[Union[str, Path]] = None,
                cihaz: Optional[str] = None) -> nn.Module:
    """Egitilmis modeli yukler. Ayni yol ikinci kez istenirse onbellekten dondurur."""
    yol = Path(yol) if yol else VARSAYILAN_MODEL_YOLU
    anahtar = str(yol.resolve())
    if anahtar in _onbellek:
        return _onbellek[anahtar]
    if not yol.exists():
        raise FileNotFoundError(
            f"Model dosyasi yok: {yol}. Ortak Drive'daki models/{yol.name} "
            "dosyasini bu konuma kopyalayin."
        )
    cihaz = torch.device(cihaz or ("cuda" if torch.cuda.is_available() else "cpu"))
    kayit = torch.load(yol, map_location=cihaz, weights_only=False)
    model = model_olustur(len(kayit["siniflar"]), onegitimli=False)
    model.load_state_dict(kayit["model_durumu"])
    model.to(cihaz).eval()
    model.bilgi = {k: v for k, v in kayit.items() if k != "model_durumu"}  # type: ignore[attr-defined]
    _onbellek[anahtar] = model
    return model


def _tensor_hazirla(goruntu: Girdi, model: nn.Module) -> torch.Tensor:
    bilgi = model.bilgi  # type: ignore[attr-defined]
    rgb = cnn_girdisi_hazirla(goruntu, bilgi["girdi_modu"])
    x = donusum_olustur(bilgi["goruntu_boyutu"], egitim=False)(rgb).unsqueeze(0)
    return x.to(next(model.parameters()).device)


@torch.no_grad()
def tahmin(goruntu: Girdi, model: Optional[nn.Module] = None) -> Dict[str, float]:
    """Sinif olasiliklari (toplami 1), ekipte anlasilan formatta."""
    model = model or model_yukle()
    olasiliklar = F.softmax(model(_tensor_hazirla(goruntu, model)), dim=1)[0].cpu().numpy()
    siniflar = model.bilgi["siniflar"]  # type: ignore[attr-defined]
    return {s: float(olasiliklar[siniflar.index(s)]) for s in SINIFLAR}


def gradcam(goruntu: Girdi, sinif: Optional[str] = None,
            model: Optional[nn.Module] = None) -> np.ndarray:
    """
    Grad-CAM isi haritasi: 512x512, float32, 0-1 arasi.
    sinif=None ise modelin tahmin ettigi sinif icin hesaplanir.
    Hedef katman: EfficientNet-B0'in son konvolusyon blogu (model.features[-1]).
    """
    model = model or model_yukle()
    katman = model.features[-1]
    kayitlar: Dict[str, torch.Tensor] = {}

    def ileri(_m, _g, cikti):
        kayitlar["aktivasyon"] = cikti
        cikti.register_hook(lambda grad: kayitlar.__setitem__("gradyan", grad))

    tutucu = katman.register_forward_hook(ileri)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = model(_tensor_hazirla(goruntu, model))
            siniflar = model.bilgi["siniflar"]  # type: ignore[attr-defined]
            hedef = siniflar.index(sinif) if sinif else int(logit.argmax(1))
            logit[0, hedef].backward()
    finally:
        tutucu.remove()

    aktivasyon, gradyan = kayitlar["aktivasyon"].detach(), kayitlar["gradyan"].detach()
    agirlik = gradyan.mean(dim=(2, 3), keepdim=True)            # her kanalin onemi
    harita = F.relu((agirlik * aktivasyon).sum(dim=1, keepdim=True))
    harita = F.interpolate(harita, size=(CIKTI_BOYUTU, CIKTI_BOYUTU),
                           mode="bilinear", align_corners=False)[0, 0]
    harita = harita - harita.min()
    harita = harita / (harita.max() + 1e-8)
    return harita.cpu().numpy().astype(np.float32)


def isi_haritasi_bindir(rgb: np.ndarray, harita: np.ndarray, alfa: float = 0.4) -> np.ndarray:
    """Isi haritasini 512x512 RGB retinanin ustune bindirir (gorsel raporlar icin)."""
    renkli = cv2.applyColorMap((harita * 255).astype(np.uint8), cv2.COLORMAP_JET)
    renkli = cv2.cvtColor(renkli, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alfa, renkli, alfa, 0)


# ============================================================
# TEST SETI CIKTILARI (Kisi 5'in degerlendirmesi icin)
# ============================================================

def test_tahminlerini_kaydet(proje_koku: Path = PROJE_KOKU, veri_koku: Optional[Path] = None,
                             model_yolu: Optional[Path] = None,
                             bolum: str = "test") -> pd.DataFrame:
    """
    Test setindeki her goruntu icin tahmin + olasiliklari CSV'ye yazar:
    reports/cnn_test_tahminleri.csv  (Kisi 5 metrikleri buradan hesaplar)
    """
    proje_koku = Path(proje_koku)
    veri_koku = Path(veri_koku) if veri_koku else veri_kokunu_tahmin_et(proje_koku)
    model = model_yukle(model_yolu)
    df = pd.read_csv(proje_koku / "data" / "splits" / BOLME_DOSYASI)
    df = df[df["split"] == bolum].reset_index(drop=True)

    satirlar = []
    for _, satir in df.iterrows():
        p = tahmin(dosya_yolunu_bul(satir, veri_koku), model)
        tahmin_sinif = max(p, key=p.get)
        satirlar.append({"image_name": satir["image_name"],
                         "relative_path": satir["relative_path"],
                         "gercek": satir["class_name"], "tahmin": tahmin_sinif,
                         "dogru": tahmin_sinif == satir["class_name"],
                         **{f"p_{s}": round(v, 5) for s, v in p.items()}})
    sonuc = pd.DataFrame(satirlar)
    cikti = proje_koku / "reports" / f"cnn_{bolum}_tahminleri{DOSYA_EKI}.csv"
    cikti.parent.mkdir(parents=True, exist_ok=True)
    sonuc.to_csv(cikti, index=False)
    print(f"[BILGI] {bolum} dogrulugu: {sonuc['dogru'].mean():.3f} -> {cikti}")
    return sonuc


def gradcam_figuru_uret(tahminler: pd.DataFrame, proje_koku: Path = PROJE_KOKU,
                        veri_koku: Optional[Path] = None,
                        model_yolu: Optional[Path] = None) -> Path:
    """
    Her siniftan 1 dogru + 1 yanlis tahmin icin Grad-CAM figuru (4 satir x 4 sutun).
    Dogru ornek: en emin dogru tahmin. Yanlis ornek: en emin yanlis tahmin.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proje_koku = Path(proje_koku)
    veri_koku = Path(veri_koku) if veri_koku else veri_kokunu_tahmin_et(proje_koku)
    model = model_yukle(model_yolu)
    tahminler = tahminler.copy()
    tahminler["guven"] = [r[f"p_{r['tahmin']}"] for _, r in tahminler.iterrows()]

    fig, eksenler = plt.subplots(len(SINIFLAR), 4, figsize=(14, 14))
    basliklar = ["Dogru: orijinal", "Dogru: Grad-CAM", "Yanlis: orijinal", "Yanlis: Grad-CAM"]
    for i, sinif in enumerate(SINIFLAR):
        alt = tahminler[tahminler["gercek"] == sinif].sort_values("guven", ascending=False)
        for j, durum in enumerate([True, False]):
            secim = alt[alt["dogru"] == durum]
            e_org, e_cam = eksenler[i, 2 * j], eksenler[i, 2 * j + 1]
            for e in (e_org, e_cam):
                e.set_xticks([]); e.set_yticks([])
            if secim.empty:
                e_org.text(0.5, 0.5, "ornek yok", ha="center", va="center")
                continue
            satir = secim.iloc[0]
            yol = (Path(veri_koku) / satir["relative_path"]) if veri_koku else Path(satir["relative_path"])
            rgb = cnn_girdisi_hazirla(yol, "rgb")
            harita = gradcam(yol, model=model)
            e_org.imshow(rgb)
            e_cam.imshow(isi_haritasi_bindir(rgb, harita))
            e_cam.set_xlabel(f"tahmin: {satir['tahmin']} (%{satir['guven'] * 100:.0f})", fontsize=9)
        eksenler[i, 0].set_ylabel(sinif.upper(), fontsize=11, fontweight="bold")
    for j, b in enumerate(basliklar):
        eksenler[0, j].set_title(b, fontsize=11, fontweight="bold")
    fig.suptitle("RETINA-AI: Grad-CAM - modelin baktigi bolgeler", fontsize=14)
    fig.tight_layout()
    cikti = proje_koku / "reports" / "figures" / f"gradcam_ornekleri{DOSYA_EKI}.png"
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=150)
    plt.close(fig)
    print(f"[BILGI] Grad-CAM figuru kaydedildi: {cikti}")
    return cikti


# ============================================================
# K-KATLI CAPRAZ DOGRULAMA (hasta bazli)
# ============================================================

def capraz_dogrulama(proje_koku: Path = PROJE_KOKU, veri_koku: Optional[Path] = None,
                     k: int = 5, gecici_klasor: Path = Path("/tmp/retina_cv"),
                     **egit_ayarlari) -> pd.DataFrame:
    """
    Bolme dosyasindaki TUM goruntuleri (train+val+test) hasta bazli k kata boler.
    Kat i icin: test = kat i, val = kat i+1, train = kalan katlar.
    Boylece her goruntu tam bir kez test edilir. Test seti kucuk oldugunda (sinif basina ~43)
    tek bir bolmeden daha guvenilir bir ortalama ve standart sapma verir.

    Cikti:
      reports/cnn_capraz_dogrulama{EKI}.csv  -> kat x sinif dogrulugu
      reports/cnn_capraz_tahminleri{EKI}.csv -> her goruntunun kat disi tahmini
    """
    from sklearn.model_selection import StratifiedGroupKFold

    proje_koku = Path(proje_koku)
    veri_koku = Path(veri_koku) if veri_koku else veri_kokunu_tahmin_et(proje_koku)
    df = pd.read_csv(proje_koku / "data" / "splits" / BOLME_DOSYASI).reset_index(drop=True)
    gruplar = df["hasta_anahtari"] if "hasta_anahtari" in df.columns else df["patient_id"].astype(str)

    katlar = np.zeros(len(df), dtype=int)
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    for i, (_, idx) in enumerate(skf.split(df, df["class_name"], gruplar)):
        katlar[idx] = i
    gecici_klasor.mkdir(parents=True, exist_ok=True)

    tum_tahminler, ozet = [], []
    for i in range(k):
        kume = np.where(katlar == i, "test", np.where(katlar == (i + 1) % k, "val", "train"))
        df_kat = df.assign(split=kume)
        print(f"\n========== KAT {i + 1}/{k} | "
              f"train={int((kume == 'train').sum())} val={int((kume == 'val').sum())} "
              f"test={int((kume == 'test').sum())} ==========")
        model_yolu = gecici_klasor / f"kat{i + 1}.pth"
        egit(proje_koku, veri_koku=veri_koku, model_yolu=model_yolu, df=df_kat,
             rapor_yaz=False, **egit_ayarlari)

        model = model_yukle(model_yolu)
        test = df_kat[df_kat["split"] == "test"]
        for _, satir in test.iterrows():
            p = tahmin(dosya_yolunu_bul(satir, veri_koku), model)
            t = max(p, key=p.get)
            tum_tahminler.append({"kat": i + 1, "image_name": satir["image_name"],
                                  "relative_path": satir["relative_path"],
                                  "gercek": satir["class_name"], "tahmin": t,
                                  "dogru": t == satir["class_name"],
                                  **{f"p_{s}": round(v, 5) for s, v in p.items()}})
        _onbellek.clear()

        kat_df = pd.DataFrame([r for r in tum_tahminler if r["kat"] == i + 1])
        satir_ozet = {"kat": i + 1, "genel": kat_df["dogru"].mean()}
        satir_ozet.update(kat_df.groupby("gercek")["dogru"].mean().to_dict())
        ozet.append(satir_ozet)
        print(f"KAT {i + 1} test dogrulugu: {satir_ozet['genel']:.3f}")

    ozet_df = pd.DataFrame(ozet).set_index("kat")
    ozet_df.loc["ortalama"] = ozet_df.mean()
    ozet_df.loc["std"] = ozet_df.iloc[:-1].std()
    rapor = proje_koku / "reports"
    rapor.mkdir(parents=True, exist_ok=True)
    ozet_df.round(4).to_csv(rapor / f"cnn_capraz_dogrulama{DOSYA_EKI}.csv")
    pd.DataFrame(tum_tahminler).to_csv(rapor / f"cnn_capraz_tahminleri{DOSYA_EKI}.csv", index=False)
    print("\n[BILGI] Capraz dogrulama ozeti (dogruluk):")
    print((ozet_df * 100).round(1))
    return ozet_df


# ============================================================
# KOMUT SATIRI
# ============================================================

def _ana() -> None:
    ayr = argparse.ArgumentParser(description="RETINA-AI Kisi 4: EfficientNet-B0")
    alt = ayr.add_subparsers(dest="komut", required=True)

    e = alt.add_parser("egit", help="Modeli egit")
    e.add_argument("--veri-koku", type=Path)
    e.add_argument("--epoch", type=int, default=30)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--boyut", type=int, default=GORUNTU_BOYUTU)
    e.add_argument("--girdi-modu", choices=["rgb", "yesil_clahe"], default="rgb")
    e.add_argument("--ornek-sayisi", type=int, help="Hizli deneme icin kucuk alt kume")

    t = alt.add_parser("test", help="Test seti tahminleri + Grad-CAM figuru")
    t.add_argument("--veri-koku", type=Path)

    p = alt.add_parser("tahmin", help="Tek goruntu icin tahmin")
    p.add_argument("goruntu", type=Path)

    a = ayr.parse_args()
    if a.komut == "egit":
        egit(veri_koku=a.veri_koku, epoch=a.epoch, batch=a.batch, boyut=a.boyut,
             girdi_modu=a.girdi_modu, ornek_sayisi=a.ornek_sayisi)
    elif a.komut == "test":
        tahminler = test_tahminlerini_kaydet(veri_koku=a.veri_koku)
        gradcam_figuru_uret(tahminler, veri_koku=a.veri_koku)
    else:
        print(json.dumps(tahmin(a.goruntu), indent=2))


if __name__ == "__main__":
    _ana()
