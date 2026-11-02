"""
RETINA-AI (Kisi 5): Sunum icin demo klasoru hazirlar.

v2 bolmesinin TEST kumesinden (modellerin egitimde hic gormedigi fotograflar)
sunumda gosterilecek ornekleri secer ve data/processed/demo/ klasorune kopyalar:

  1_dogru/      her siniftan, iki modelin de dogru bildigi 2 ornek
  2_cnn_dogru_svm_yanlis/   CNN'in bilip SVM'in yanildigi 2 ornek
  3_svm_dogru_cnn_yanlis/   SVM'in bilip CNN'in yanildigi 2 ornek
  4_ikisi_de_yanlis/        iki modelin de yanildigi 2 ornek (glokom <-> normal karisikligi)

Ayrica data/processed/demo/demo_listesi.csv dosyasina her fotografin gercek sinifini,
iki modelin tahminini ve guvenini yazar. Sunumda bu listeye bakarak anlatabilirsin.

Calistirma (proje kokunden):
  py src/demo_klasoru_hazirla.py --goruntu_dizin "C:/.../dataset"

--goruntu_dizin: icinde cataract/, diabetic_retinopathy/, glaucoma/, normal/ klasorleri olan yer.
Fotograflar GitHub'a gitmez (data/processed/ .gitignore'da).
"""
import argparse
import os
import shutil
import sys

import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOLME = os.path.join(KOK, "data", "splits", "veri_bolme_v2.csv")
CNN = os.path.join(KOK, "reports", "cnn_test_tahminleri_v2.csv")
SVM = os.path.join(KOK, "reports", "svm_test_tahminleri_v2.csv")
CIKTI = os.path.join(KOK, "data", "processed", "demo")
SINIFLAR = ["normal", "diabetic_retinopathy", "glaucoma", "cataract"]
OLASILIK = [f"p_{s}" for s in SINIFLAR]


def tablo_hazirla():
    bolme = pd.read_csv(BOLME)
    test = bolme[bolme["split"] == "test"][["image_name", "relative_path", "class_name", "file_path"]]
    cnn = pd.read_csv(CNN)
    svm = pd.read_csv(SVM)
    cnn["cnn_guven"] = cnn[OLASILIK].max(axis=1)
    svm["svm_guven"] = svm[OLASILIK].max(axis=1)
    t = (test.merge(cnn[["image_name", "tahmin", "dogru", "cnn_guven"]]
                    .rename(columns={"tahmin": "cnn_tahmin", "dogru": "cnn_dogru"}), on="image_name")
             .merge(svm[["image_name", "tahmin", "dogru", "svm_guven"]]
                    .rename(columns={"tahmin": "svm_tahmin", "dogru": "svm_dogru"}), on="image_name"))
    if len(t) != len(test):
        print(f"UYARI: test kumesinde {len(test)} fotograf var, tahminlerle eslesen {len(t)}.")
    return t


def ornek_sec(t):
    secim = []
    # 1) Her siniftan, ikisinin de dogru ve CNN'in en emin oldugu 2 ornek
    for s in SINIFLAR:
        g = t[(t.class_name == s) & t.cnn_dogru & t.svm_dogru].sort_values("cnn_guven", ascending=False)
        secim += [("1_dogru", r) for _, r in g.head(2).iterrows()]
    # 2-4) Iki modelin ayristigi ve ikisinin de yanildigi ornekler
    gruplar = [
        ("2_cnn_dogru_svm_yanlis", t.cnn_dogru & ~t.svm_dogru, "cnn_guven"),
        ("3_svm_dogru_cnn_yanlis", ~t.cnn_dogru & t.svm_dogru, "svm_guven"),
        ("4_ikisi_de_yanlis", ~t.cnn_dogru & ~t.svm_dogru, "cnn_guven"),
    ]
    for klasor, kosul, sirala in gruplar:
        g = t[kosul].sort_values(sirala, ascending=False)
        if klasor == "4_ikisi_de_yanlis":  # once en cok karisan cift: glokom <-> normal
            gn = g[g.class_name.isin(["glaucoma", "normal"]) & g.cnn_tahmin.isin(["glaucoma", "normal"])]
            g = pd.concat([gn, g.drop(gn.index)])
        secim += [(klasor, r) for _, r in g.head(2).iterrows()]
    return secim


def dosya_bul(r, dizin):
    adaylar = []
    if dizin:
        adaylar.append(os.path.join(dizin, r["relative_path"]))
    adaylar += [os.path.join(KOK, "data", "splits", "test", r["class_name"], r["image_name"]),
                os.path.join(KOK, "archive", "dataset", r["relative_path"]),
                str(r["file_path"])]
    return next((a for a in adaylar if os.path.isfile(a)), None)


def main():
    ap = argparse.ArgumentParser(description="Sunum icin demo klasoru (v2 test kumesi)")
    ap.add_argument("--goruntu_dizin", default=None,
                    help="Fotograf klasoru (icinde cataract/, glaucoma/ ... olan)")
    args = ap.parse_args()

    for yol in (BOLME, CNN, SVM):
        if not os.path.isfile(yol):
            sys.exit(f"HATA: {yol} bulunamadi.")

    t = tablo_hazirla()
    secim = ornek_sec(t)
    if os.path.isdir(CIKTI):
        shutil.rmtree(CIKTI)

    satirlar, eksik = [], []
    for klasor, r in secim:
        kaynak = dosya_bul(r, args.goruntu_dizin)
        hedef_ad = f"{r['class_name']}_{r['image_name']}"
        if kaynak is None:
            eksik.append(r["image_name"])
        else:
            os.makedirs(os.path.join(CIKTI, klasor), exist_ok=True)
            shutil.copy2(kaynak, os.path.join(CIKTI, klasor, hedef_ad))
        satirlar.append({
            "klasor": klasor, "dosya": hedef_ad, "gercek": r["class_name"],
            "cnn_tahmin": r["cnn_tahmin"], "cnn_guven": round(float(r["cnn_guven"]), 3),
            "svm_tahmin": r["svm_tahmin"], "svm_guven": round(float(r["svm_guven"]), 3),
            "kopyalandi": kaynak is not None,
        })

    liste = pd.DataFrame(satirlar)
    os.makedirs(CIKTI, exist_ok=True)
    liste.to_csv(os.path.join(CIKTI, "demo_listesi.csv"), index=False)

    pd.set_option("display.width", 160)
    print(liste.drop(columns="kopyalandi").to_string(index=False))
    print(f"\n{int(liste.kopyalandi.sum())}/{len(liste)} fotograf kopyalandi: {CIKTI}")
    if eksik:
        print(f"UYARI: {len(eksik)} fotograf bulunamadi. --goruntu_dizin ile fotograf klasorunu ver.")


if __name__ == "__main__":
    main()
