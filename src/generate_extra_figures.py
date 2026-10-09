"""
Ek Rapor ve Makale Figürleri Üretici Betik
-----------------------------------------
Veri ve Ön İşleme görevimizin görsel zenginliğini artırmak için 
makale ve sunum kalitesinde 4 yeni figür üretir:
1. sinif_ornekleri_4x4.png (Her sınıftan 4'er ham örnek)
2. rgb_kanallari_karsilastirma.png (Neden Yeşil Kanal? RGB ayrıştırma)
3. histogram_kontrast_analizi.png (Orijinal vs CLAHE yoğunluk histogramı)
4. sinif_netlik_laplacian_kutusu.png (Bulanıklık / netlik boxplot analizi)
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

proje_koku = Path(__file__).resolve().parent.parent
if str(proje_koku) not in sys.path:
    sys.path.append(str(proje_koku))

from src.preprocessing import on_isle, roi_kirp, clahe_uygula, yesil_kanali_izole_et

csv_yolu = proje_koku / "data" / "splits" / "veri_bolme.csv"
fig_dizini = proje_koku / "reports" / "figures"
fig_dizini.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_yolu)
siniflar = sorted(df["class_name"].unique())

print("[1/4] 4x4 Sınıf Örnek Galerisi üretiliyor...")
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for row_idx, sinif in enumerate(siniflar):
    ornekler = df[df["class_name"] == sinif].head(4)
    for col_idx, (_, satir) in enumerate(ornekler.iterrows()):
        dosya = Path(satir["file_path"])
        img = cv2.imread(str(dosya))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax = axes[row_idx, col_idx]
        ax.imshow(img_rgb)
        ax.axis("off")
        if col_idx == 0:
            ax.set_title(f"{sinif.upper()}", fontsize=11, fontweight="bold", loc="left")

plt.suptitle("RETINA-AI: Veri Seti Siniflarina Gore Ham Retina Gorselleri (4x4)", fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(fig_dizini / "sinif_ornekleri_4x4.png", dpi=180)
plt.close()

print("[2/4] RGB Kanal Karşılaştırması üretiliyor...")
ornek_normal = df[df["class_name"] == "normal"].iloc[0]
img_norm = cv2.imread(ornek_normal["file_path"])
img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
r_kanal = img_rgb[:, :, 0]
g_kanal = img_rgb[:, :, 1]
b_kanal = img_rgb[:, :, 2]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_rgb)
axes[0].set_title("Orijinal Renkli (RGB)", fontsize=11, fontweight="bold")
axes[1].imshow(r_kanal, cmap="gray")
axes[1].set_title("Kirmizi (R) - Asiri Parlak / Dusuk Kontrast", fontsize=10)
axes[2].imshow(g_kanal, cmap="gray")
axes[2].set_title("Yesil (G) - En Yuksek Damar Kontrasti", fontsize=10, fontweight="bold")
axes[3].imshow(b_kanal, cmap="gray")
axes[3].set_title("Mavi (B) - Dusuk Isik / Gurultulu", fontsize=10)

for ax in axes:
    ax.axis("off")

plt.suptitle("RETINA-AI: Renk Kanallari Analizi ve Yesil Kanalin Tercih Nedeni", fontsize=13, y=1.05)
plt.tight_layout()
plt.savefig(fig_dizini / "rgb_kanallari_karsilastirma.png", dpi=180)
plt.close()

print("[3/4] Histogram ve Kontrast Analizi üretiliyor...")
g_kirpilmis = yesil_kanali_izole_et(roi_kirp(img_norm))
g_clahe = clahe_uygula(g_kirpilmis)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].imshow(g_kirpilmis, cmap="gray")
axes[0, 0].set_title("Ham Yesil Kanal", fontsize=11)
axes[0, 0].axis("off")

axes[0, 1].imshow(g_clahe, cmap="gray")
axes[0, 1].set_title("CLAHE Uygulanmis Yesil Kanal", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

# Histogramlar (0 piksel arka planını hariç tutarak)
mask_ham = g_kirpilmis > 10
mask_clahe = g_clahe > 10

axes[1, 0].hist(g_kirpilmis[mask_ham].ravel(), bins=64, color="#2b5c8f", alpha=0.75, range=(10, 255))
axes[1, 0].set_title("Ham Goruntu Piksel Yogunlugu Histogrami", fontsize=10)
axes[1, 0].set_xlabel("Piksel Degeri [0-255]")
axes[1, 0].set_ylabel("Frekans")
axes[1, 0].grid(True, linestyle="--", alpha=0.4)

axes[1, 1].hist(g_clahe[mask_clahe].ravel(), bins=64, color="#388e3c", alpha=0.75, range=(10, 255))
axes[1, 1].set_title("CLAHE Sonrasi Piksel Yogunlugu (Esitlenmis)", fontsize=10, fontweight="bold")
axes[1, 1].set_xlabel("Piksel Degeri [0-255]")
axes[1, 1].set_ylabel("Frekans")
axes[1, 1].grid(True, linestyle="--", alpha=0.4)

plt.suptitle("RETINA-AI: CLAHE Kontrast Iyilestirme ve Histogram Esitleme Etkisi", fontsize=13)
plt.tight_layout()
plt.savefig(fig_dizini / "histogram_kontrast_analizi.png", dpi=180)
plt.close()

print("[4/4] Laplacian Bulanıklık/Netlik Analizi üretiliyor (örnek 200 görsel)...")
orneklem = df.groupby("class_name").sample(n=50, random_state=42)
laplacian_skorlar = []

for _, satir in orneklem.iterrows():
    img = cv2.imread(satir["file_path"], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        skor = cv2.Laplacian(img, cv2.CV_64F).var()
        laplacian_skorlar.append({"class_name": satir["class_name"], "laplacian_var": skor})

df_lap = pd.DataFrame(laplacian_skorlar)

plt.figure(figsize=(9, 5))
sns.boxplot(data=df_lap, x="class_name", y="laplacian_var", palette="Set2")
plt.title("RETINA-AI: Siniflar Bazinda Gorsel Netlik/Bulaniklik (Laplacian Varyansi)", fontsize=12)
plt.xlabel("Hastalik Sinifi", fontsize=10)
plt.ylabel("Laplacian Varyansi (Yuksek = Daha Net, Dusuk = Bulanik)", fontsize=10)
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(fig_dizini / "sinif_netlik_laplacian_kutusu.png", dpi=180)
plt.close()

print("[TAMAMLANDI] Tum ek figurler basariyla kaydedildi.")
