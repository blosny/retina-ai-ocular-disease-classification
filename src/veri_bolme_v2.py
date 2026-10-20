"""
Veri Bolme v2 (TASLAK - Kisi 1 onayi bekliyor)
----------------------------------------------
veri_bolme.csv'den kaynak/kopya sorunlarini ayiklayarak yeni bir bolme uretir:

1. Sadece 512x512 goruntuler alinir.
   - 256x256 goruntuler (1223 adet, `_N_xxxxxxx.jpg`) ayni orijinallerin Keras augmentasyon
     kopyalari ve sadece katarakt/glokomda var -> kopya sizintisi + cozunurluk kisayolu.
   - Yuksek cozunurluklu goruntuler (1848/2464/2592 px) da sadece katarakt/glokomda var.
2. Her siniftan 288 goruntu (en kucuk sinif: katarakt, 288).
3. Hasta bazli %70/15/15 bolme (seed=42).
   - ODIR'de ayni hastanin bir gozu katarakt digeri glokom olabilir; hasta tum siniflarda
     ayni kumeye atanir.
   - DR goruntuleri ayri bir kaynaktan (.jpeg, id 100-11598) geldigi icin hasta anahtari
     "eyepacs_<id>", digerleri "odir_<id>" olarak tutulur (ayni sayi = ayni kisi degil).

Bilinen sinirlilik: DR ayri bir veri setinden geliyor; bu bolme DR'deki kaynak farkini cozmez.

Calistirma (proje kokunden):  python src/veri_bolme_v2.py
Cikti: data/splits/veri_bolme_v2.csv  (eski veri_bolme.csv'ye dokunulmaz)
"""

from pathlib import Path
import numpy as np
import pandas as pd

TOHUM = 42
SINIF_BASINA = 288
ORANLAR = {"train": 0.70, "val": 0.15, "test": 0.15}

proje = Path(__file__).resolve().parent.parent
df = pd.read_csv(proje / "data" / "splits" / "veri_bolme.csv")

d = df[(df["width"] == 512) & (df["height"] == 512)].copy()
d["kaynak"] = np.where(d["class_name"] == "diabetic_retinopathy", "eyepacs", "odir")
d["hasta_anahtari"] = d["kaynak"] + "_" + d["patient_id"].astype(str)

rng = np.random.default_rng(TOHUM)
secilen, hasta_kumesi = [], {}

for sinif in ["cataract", "glaucoma", "normal", "diabetic_retinopathy"]:
    alt = d[d["class_name"] == sinif]
    tum_hastalar = sorted(alt["hasta_anahtari"].unique())
    hastalar = [tum_hastalar[i] for i in rng.permutation(len(tum_hastalar))]

    # 1) 288 goruntuye ulasana kadar butun hastalari al (son hastadan fazlasi atilir)
    parcalar, toplam = [], 0
    for h in hastalar:
        if toplam >= SINIF_BASINA:
            break
        p = alt[alt["hasta_anahtari"] == h]
        parcalar.append(p)
        toplam += len(p)
    s = pd.concat(parcalar).head(SINIF_BASINA).copy()

    # 2) Hastalari kumelere dagit; baska sinifta zaten atanmis hasta ayni kumede kalir
    hedef = {k: round(v * SINIF_BASINA) for k, v in ORANLAR.items()}
    hedef["train"] = SINIF_BASINA - hedef["val"] - hedef["test"]
    sayac = {k: 0 for k in ORANLAR}
    for h in s["hasta_anahtari"].unique():
        n = int((s["hasta_anahtari"] == h).sum())
        if h not in hasta_kumesi:
            eksik = {k: hedef[k] - sayac[k] for k in ORANLAR}
            hasta_kumesi[h] = max(eksik, key=eksik.get)
        sayac[hasta_kumesi[h]] += n
    s["split"] = s["hasta_anahtari"].map(hasta_kumesi)
    secilen.append(s)

yeni = pd.concat(secilen).reset_index(drop=True)

# Kontroller
kesisim = yeni.groupby("hasta_anahtari")["split"].nunique()
assert (kesisim == 1).all(), "Bir hasta birden fazla kumede!"
assert yeni["image_name"].str.match(r"^_\d+_\d+\.").sum() == 0, "256'lik kopya kaldi!"

cikti = proje / "data" / "splits" / "veri_bolme_v2.csv"
yeni.to_csv(cikti, index=False)
print(pd.crosstab(yeni["class_name"], yeni["split"], margins=True))
print("\nBenzersiz hasta:", yeni["hasta_anahtari"].nunique(), "| kumeler arasi hasta kesisimi: 0")
print("Kaydedildi:", cikti)
