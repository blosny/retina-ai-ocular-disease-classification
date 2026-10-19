"""
RETINA-AI: Test Veri Setinden Rastgele Orneklerle Model Dogrulama
-----------------------------------------------------------------
Bu betik, `data/splits/test` klasorundeki 4 farkli siniftan (cataract,
diabetic_retinopathy, glaucoma, normal) rastgele 1'er gorsel secer,
`k3_tahmin_et` ile tahmin alir ve gercek sinif ile karsilastirir.
"""

import sys
from pathlib import Path

PROJE_KOKU = Path(__file__).resolve().parent.parent
if str(PROJE_KOKU) not in sys.path:
    sys.path.append(str(PROJE_KOKU))

import random
from src.k3_oznitelik import k3_tahmin_et, oznitelik_cikar

TEST_KOKU = PROJE_KOKU / "data" / "splits" / "test"
SINIFLAR = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

def main():
    print("=" * 78)
    print("      RETINA-AI: GERCEK TEST GORSELLERI ILE MODEL TAHMIN DOGRULAMASI")
    print("=" * 78 + "\n")

    dogru_sayisi = 0

    for sinif in SINIFLAR:
        klasor = TEST_KOKU / sinif
        dosyalar = sorted(list(klasor.glob("*.*")))
        if not dosyalar:
            print(f"[UYARI] {sinif} klasorunde gorsel bulunamadi.")
            continue

        secilen_dosya = random.choice(dosyalar)

        # 1. Gercek sinif ve dosya yolu
        print(f"1. Gercek Sinif : {sinif.upper()}")
        print(f"   Dosya Yolu   : {secilen_dosya.relative_to(PROJE_KOKU).as_posix()}")

        # 2 & 3. Tahmin et ve olasilik sozlugunu al
        olasiliklar = k3_tahmin_et(secilen_dosya)
        print(f"2. Tahmin Sozlugu:")
        print(f"   {olasiliklar}")

        # Oznitelikleri de gosterelim
        oz = oznitelik_cikar(secilen_dosya)
        print(f"   Cikarilan Oznitelikler:")
        print(f"   * Laplacian: {oz['laplacian_varyansi']} | Kontrast: {oz['ortalama_kontrast']} | Damar: {oz['damar_orani']}")
        print(f"   * Sari Eksuda: {oz['sari_lezyon_alani']} (sayi: {oz['sari_lezyon_sayisi']}) | Kanama: {oz['kanama_alani']} (sayi: {oz['kanama_sayisi']})")

        # 4. En yuksek olasilikli sinif ile eslesme kontrolu
        en_yuksek_sinif = max(olasiliklar, key=olasiliklar.get)
        guven = olasiliklar[en_yuksek_sinif]
        eslesme = (en_yuksek_sinif == sinif)

        if eslesme:
            dogru_sayisi += 1
            durum_metni = "[BASARILI] Model tahmini gercek sinif ile TAM ESLESTI."
        else:
            durum_metni = f"[UYUSMADI] Model '{en_yuksek_sinif}' tahmin etti (Beklenen: '{sinif}')."

        print(f"3. En Yuksek Tahmin : {en_yuksek_sinif} (%{guven * 100:.1f})")
        print(f"4. Eslesme Durumu   : {durum_metni}")
        print("-" * 78 + "\n")

    print(f"[SONUC] Toplam Basari: {dogru_sayisi} / {len(SINIFLAR)} dogru eslesme.")


if __name__ == "__main__":
    main()
