"""
RETINA-AI: Veri Dogrulama, Temizleme, Esitleme ve Bolme Modulu
-----------------------------------------------------------------------
Gorevli: Kisi 1 (Veri ve On Isleme Sorumlusu)
Ders: Firat Universitesi Bilgisayar Muhendisligi - Medikal Goruntu Isleme

Aciklama:
Bu modul, ham retina goruntulerini tarayarak su islemleri gerceklestirir:
1. Gorsellerin OpenCV ile okunabilirlik kontrolu (Bozuk dosya tespiti).
2. MD5 ozetleme (hashing) ile mukerrer ve capraz-sinif cakisimlarinin tespiti:
   - Kaggle veri setinde 2 gorsel (1415_right.jpg ve 625_left.jpg) hem 'cataract'
     hem de 'glaucoma' klasorunde bulunmaktadir (etiket karismasi / capraz kopya).
   - Glokom sinifi tam 1007 gorsele sahip oldugu icin bu 2 cakisik gorsel 
     ornek sayisi yuksek olan 'cataract' sinifindan elenerek veri sizintisi (leakage)
     tamamen onlenir.
3. Her sinifin 1007 ornege dengelenmesi (Toplam: 4028 gorsel).
4. Stratified Split yontemiyle %70 Train (705), %15 Validation (151), %15 Test (151) bolunmesi.
5. Tum ekibin ortak referans alacagi 'veri_bolme.csv' dosyasinin uretilmesi.
6. Sinif dagilimini gosteren rapor grafiginin olusturulmasi.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def dosya_md5_hesapla(dosya_yolu: Path, blok_boyutu: int = 65536) -> str:
    """
    Bir dosyanin MD5 hash ozetini hesaplar.
    
    Parametreler:
        dosya_yolu (Path): Hash degeri hesaplanacak dosyanin yolu.
        blok_boyutu (int): Okunacak tampon bayt boyutu.
        
    Dondurur:
        str: 32 karakterlik MD5 onaltilik dizgi.
    """
    hasher = hashlib.md5()
    with open(dosya_yolu, "rb") as dosya:
        tampon = dosya.read(blok_boyutu)
        while len(tampon) > 0:
            hasher.update(tampon)
            tampon = dosya.read(blok_boyutu)
    return hasher.hexdigest()


def veri_setini_tara_ve_dogrula(veri_dizini: Path, onbellek_dosyasi: Path = None) -> pd.DataFrame:
    """
    Ham veri seti dizinini tarar, bozuk dosyalari ayiklar, MD5 hashlerini
    ve gorsel boyutlarini cikarip bir DataFrame olarak dondurur.
    Onbellek dosyasi varsa hizli yukleme yapar.
    
    Parametreler:
        veri_dizini (Path): Ana veri seti dizini.
        onbellek_dosyasi (Path): Taranan sonuclarin saklandigi ara CSV.
        
    Dondurur:
        pd.DataFrame: Dogrulanmis gorsellere ait metadata tablosu.
    """
    if onbellek_dosyasi and onbellek_dosyasi.exists():
        print(f"[BILGI] Onbellekteki metadata yukleniyor: {onbellek_dosyasi}")
        return pd.read_csv(onbellek_dosyasi)

    print(f"[BILGI] Veri dizini taraniyor: {veri_dizini}")
    gecerli_uzantilar = {".jpg", ".jpeg", ".png", ".bmp"}
    kayitlar: List[Dict] = []
    bozuk_dosya_sayisi = 0
    
    sinif_klasorleri = [d for d in veri_dizini.iterdir() if d.is_dir()]
    
    for sinif_dir in sinif_klasorleri:
        sinif_adi = sinif_dir.name
        dosyalar = [f for f in sinif_dir.iterdir() if f.suffix.lower() in gecerli_uzantilar]
        print(f"  -> Sinif '{sinif_adi}': {len(dosyalar)} dosya bulundu.")
        
        for dosya in dosyalar:
            img = cv2.imread(str(dosya))
            if img is None:
                print(f"[UYARI] Bozuk dosya tespit edildi: {dosya.name}")
                bozuk_dosya_sayisi += 1
                continue
                
            yukseklik, genislik, kanal = img.shape
            md5_hash = dosya_md5_hesapla(dosya)
            
            kayitlar.append({
                "image_name": dosya.name,
                "file_path": str(dosya.resolve()),
                "relative_path": f"{sinif_adi}/{dosya.name}",
                "class_name": sinif_adi,
                "width": genislik,
                "height": yukseklik,
                "channels": kanal,
                "md5_hash": md5_hash
            })
            
    df = pd.DataFrame(kayitlar)
    print(f"[TAMAMLANDI] Toplam {len(df)} gecerli gorsel dogrulandi. (Bozuk dosya: {bozuk_dosya_sayisi})")
    
    if onbellek_dosyasi:
        onbellek_dosyasi.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(onbellek_dosyasi, index=False)
        
    return df


def capraz_ve_mukerrer_temizle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri setinde ayni hash degerine sahip cakisik veya mukerrer gorselleri cozer.
    Sinif ici kopya yoktur. Siniflar arasi (cataract vs glaucoma) 2 cakisik gorsel 
    sayisi bol olan cataract sinifindan cikarilarak her iki sinifin da 
    baskinligi ve temizligi korunur.
    
    Parametreler:
        df (pd.DataFrame): Dogrulanmis metadata tablosu.
        
    Dondurur:
        pd.DataFrame: Cakisiklardan arindirilmis tablo.
    """
    ciftler = df[df.duplicated(subset=["md5_hash"], keep=False)]
    if len(ciftler) > 0:
        print("[BILGI] Siniflar arasi capraz kopya analizi yapiliyor:")
        for hash_val, grup in ciftler.groupby("md5_hash"):
            siniflar = grup["class_name"].tolist()
            adlar = grup["image_name"].tolist()
            print(f"  -> Cakisik Gorsel: {adlar} - Siniflar: {siniflar}")
            
        # Cataract sinifindaki cakisik ornekleri filtrele (Cataract'ta 1038 ornek oldugu icin rahat elenir)
        cikisik_cataract_indeksler = df[(df["class_name"] == "cataract") & (df["md5_hash"].isin(ciftler["md5_hash"]))].index
        df_temiz = df.drop(index=cikisik_cataract_indeksler).copy()
        print(f"[BILGI] {len(cikisik_cataract_indeksler)} cakisik ornek cataract sinifindan elendi.")
    else:
        df_temiz = df.copy()
        
    return df_temiz


def siniflari_esitle(df: pd.DataFrame, hedef_sayi: int = 1007, seed: int = 42) -> pd.DataFrame:
    """
    Her siniftan tam olarak belirlenen sayi kadar ornek rastgele secer.
    
    Parametreler:
        df (pd.DataFrame): Temizlenmis gorsel tablosu.
        hedef_sayi (int): Her siniftan secilecek gorsel sayisi (varsayilan: 1007).
        seed (int): Rastgele secim icin tekrarlanabilir tohum degeri.
        
    Dondurur:
        pd.DataFrame: Her siniftan tam 1007 ornek iceren dengeli tablo.
    """
    dengeli_parcalar = []
    
    for sinif_adi, grup in df.groupby("class_name"):
        mevcut_sayi = len(grup)
        if mevcut_sayi < hedef_sayi:
            raise ValueError(
                f"Sinif '{sinif_adi}' icin mevcut sayi ({mevcut_sayi}) hedef sayidan ({hedef_sayi}) kucuk!"
            )
            
        secilenler = grup.sample(n=hedef_sayi, random_state=seed)
        dengeli_parcalar.append(secilenler)
        print(f"  -> Sinif '{sinif_adi}': {mevcut_sayi} ornek arasindan {hedef_sayi} ornek secildi.")
        
    dengeli_df = pd.concat(dengeli_parcalar, ignore_index=True)
    return dengeli_df


def stratified_veri_bolme(
    df: pd.DataFrame, 
    train_sayi_sinif_basina: int = 705, 
    val_sayi_sinif_basina: int = 151, 
    test_sayi_sinif_basina: int = 151, 
    seed: int = 42
) -> pd.DataFrame:
    """
    Veriyi her siniftan tam olarak 705 Train (%70), 151 Val (%15), 151 Test (%15)
    olacak sekilde kesin oranda boler (705 + 151 + 151 = 1007).
    
    Parametreler:
        df (pd.DataFrame): Her siniftan 1007 ornek iceren dengeli tablo.
        train_sayi_sinif_basina (int): Egitim ornek sayisi (705).
        val_sayi_sinif_basina (int): Dogrulama ornek sayisi (151).
        test_sayi_sinif_basina (int): Test ornek sayisi (151).
        seed (int): Rastgele karistirma tohum degeri.
        
    Dondurur:
        pd.DataFrame: 'split' sutunu atanmis nihai veri tablosu.
    """
    bolunmus_parcalar = []
    
    for sinif_adi, grup in df.groupby("class_name"):
        # Karistir
        karisik = grup.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        train_parca = karisik.iloc[:train_sayi_sinif_basina].copy()
        train_parca["split"] = "train"
        
        val_parca = karisik.iloc[train_sayi_sinif_basina : train_sayi_sinif_basina + val_sayi_sinif_basina].copy()
        val_parca["split"] = "val"
        
        test_parca = karisik.iloc[train_sayi_sinif_basina + val_sayi_sinif_basina : train_sayi_sinif_basina + val_sayi_sinif_basina + test_sayi_sinif_basina].copy()
        test_parca["split"] = "test"
        
        bolunmus_parcalar.extend([train_parca, val_parca, test_parca])
        
    sonuc_df = pd.concat(bolunmus_parcalar, ignore_index=True)
    sonuc_df = sonuc_df.sort_values(by=["class_name", "split", "image_name"]).reset_index(drop=True)
    return sonuc_df


def dagilim_grafigi_olustur(df: pd.DataFrame, cikti_yolu: Path):
    """
    Sinif ve bolum (train/val/test) dagilimini gosteren yayin kalitesinde bar grafigi cizer.
    
    Parametreler:
        df (pd.DataFrame): 'split' sutununu iceren veri tablosu.
        cikti_yolu (Path): Grafigin kaydedilecegi PNG dosyasi yolu.
    """
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    renkler = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    
    ax = sns.countplot(
        data=df,
        x="class_name",
        hue="split",
        palette=renkler,
        order=sorted(df["class_name"].unique())
    )
    
    plt.title("RETINA-AI: Veri Seti Sinif ve Bolum Dagilimi (Stratified %70/15/15)", fontsize=13, pad=15)
    plt.xlabel("Hastalik Sinifi", fontsize=11)
    plt.ylabel("Gorsel Sayisi", fontsize=11)
    plt.legend(title="Veri Bolumu", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for p in ax.patches:
        deger = int(p.get_height())
        if deger > 0:
            ax.annotate(
                f"{deger}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=9
            )
            
    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=300)
    plt.close()
    print(f"[BILGI] Sinif dagilim grafigi kaydedildi: {cikti_yolu}")


def ana_akisi_calistir():
    """
    Veri hazirlama ve bolme surecinin tum adimlarini yoneten ana fonksiyon.
    """
    proje_koku = Path(__file__).resolve().parent.parent
    ham_veri_dizini = proje_koku / "archive" / "dataset"
    cikti_dizini = proje_koku / "data" / "splits"
    rapor_dizini = proje_koku / "reports" / "figures"
    onbellek_yolu = proje_koku / "data" / "splits" / "raw_metadata_cache.csv"
    
    cikti_dizini.mkdir(parents=True, exist_ok=True)
    rapor_dizini.mkdir(parents=True, exist_ok=True)
    
    csv_cikti_yolu = cikti_dizini / "veri_bolme.csv"
    grafik_cikti_yolu = rapor_dizini / "sinif_dagilimi.png"
    
    print("=" * 70)
    print("RETINA-AI: KISI 1 VERI VE ON ISLEME - ASAMA 1 (VERI BOLME)")
    print("=" * 70)
    
    # 1. Dosyalari tara ve dogrula
    df_ham = veri_setini_tara_ve_dogrula(ham_veri_dizini, onbellek_dosyasi=onbellek_yolu)
    
    # 2. Capraz kopya (cross-class duplicate) temizle
    df_temiz = capraz_ve_mukerrer_temizle(df_ham)
    
    # 3. Siniflari esitle (Her biri 1007 adet)
    df_dengeli = siniflari_esitle(df_temiz, hedef_sayi=1007, seed=42)
    
    # 4. Kesin Stratified Bolme: 705 Train, 151 Val, 151 Test
    df_bolunmus = stratified_veri_bolme(
        df_dengeli, 
        train_sayi_sinif_basina=705, 
        val_sayi_sinif_basina=151, 
        test_sayi_sinif_basina=151, 
        seed=42
    )
    
    # 5. CSV olarak kaydet
    df_bolunmus.to_csv(csv_cikti_yolu, index=False, encoding="utf-8")
    print(f"\n[BASARILI] 'veri_bolme.csv' basariyla kaydedildi: {csv_cikti_yolu}")
    print(f"Toplam secilen gorsel sayisi: {len(df_bolunmus)}")
    
    # 6. Dagilim ozeti yazdir
    ozet_tablo = pd.crosstab(df_bolunmus["class_name"], df_bolunmus["split"], margins=True)
    print("\n--- SINIF VE BOLUM DAGILIM TABLOSU ---")
    print(ozet_tablo)
    print("--------------------------------------\n")
    
    # 7. Grafik ciz
    dagilim_grafigi_olustur(df_bolunmus, grafik_cikti_yolu)
    print("[TAMAMLANDI] Asama 1 basariyla tamamlandi!")


if __name__ == "__main__":
    ana_akisi_calistir()
