# RETINA-AI: Goruntu On Isleme Pipeline Dokumantasyonu (Asama 2)

Bu dokuman, Veri ve On Isleme Sorumlusu tarafindan gelistirilen `on_isle()` fonksiyonunun medikal gerekcelerini, matematiksel temellerini ve adimlarini aciklar.

---

## 1. On Isleme Modulunun Rolu ve Amaci
Fundus fotografciliginda farkli kameralar, degisken aydinlatma kosullari, hastanin goz kirpmasi ve optik sapmalar nedeniyle goruntuler arasinda buyuk kalite farkliliklari olusur. Bu modul, ham goruntuleri temizleyerek hem klasik makine ogrenmesi (oznitelik cikarimi) hem de derin ogrenme modelleri icin standart, yuksek kontrastli bir girdi haline getirir.

---

## 2. Pipeline Adimlari ve Medikal Gerekceleri

### Adim 1: Otomatik ROI Kirpma (Auto-Crop ve Dairesel Maskeleme)
- **Sorun:** Ham retina fotograflarinin etrafinda retinaya ait olmayan buyuk siyah bosluklar ve bazen kenar parazitleri bulunur.
- **Yontem:** Gri tonlama uzerinde esikleme (`gri > 10`) uygulanarak retinanin gecerli piksellerini iceren minimum cevreleyen dortgen (`bounding box`) kesilir. Ardindan retinayi tam kavrayan dairesel bir maske ile koselerdeki yapay parazitler sifirlanir.
- **Fayda:** Sinir aglarinin veya doku analiz algoritmalarinin anlamsiz siyah arka plana odaklanmasi (false feature) engellenir.

### Adim 2: Standart Cozunurluk (512x512)
- **Yontem:** Tum kirpilmis retina fotograflari `cv2.INTER_AREA` interpolasyonu ile 512x512 piksele boyutlandirilir.
- **Fayda:** Kucultme sirasinda piksellerin ortalamasini alarak kucuk damar yapilarinin ve mikroanevrizmalarin kaybolmasi onlenir.

### Adim 3: Yesil Kanal Izolasyonu (Green Channel Isolation)
- **Medikal Gerekce:** Fundus goruntulerinde Kirmizi (R), Yesil (G) ve Mavi (B) kanallari farkli optik ozellikler gosterir:
  - *Kirmizi Kanal:* Koryoretinal dokular tarafindan asiri yansitildigi icin doygunluga (over-saturation) ugrar, kontrast cok dusuktur.
  - *Mavi Kanal:* Goz ici ortam tarafindan cok emildigi icin gurultuludur ve karanliktir.
  - *Yesil Kanal:* Retina damarlarindaki ve kanamalardaki hemoglobin tarafindan en guclu emilen dalga boyudur. Damarlar, mikroanevrizmalar, eksudalar ve optik disk sinirlari en yuksek kontrastini yesil kanalda verir.
- **Uygulama:** BGR formatindaki goruntuden 1. indeks (Green) ayristirilarak tek kanalli medikal girdi elde edilir.

### Adim 4: Kontrast Sinirli Adaptif Histogram Esitleme (CLAHE)
- **Yontem:** Standart histogram esitleme goruntunun tamamina uygulandiginda parlak bolgelerde patlamalara (over-amplification) yol acar. CLAHE ise goruntuyu 8x8 kucuk bloklara (`tileGridSize=(8, 8)`) bolerek yerel kontrast uygular. `clipLimit=2.0` parametresi ile histogram tepe degeri sinirlandirilarak gurultu yukselmesi onlenir.
- **Fayda:** Ozellikle kataraktli bulanik gozlerde ve karanlik retina kenarlarinda gizli kalan lezyon ve damar detaylari aciga cikar.

### Adim 5: Gurultu Azaltma (Median Filter)
- **Yontem:** 3x3 cekirdek boyutlu Median Filtre uygulanir.
- **Medikal Gerekce:** Gaussian filtresi keskin damar kenarlarini ve kucuk mikro-kanamalari bulandirabilirken, Median filtre tuz-biber gurultulerini temizlerken damar sinirlarini ve lezyon kenarlarini (edge preservation) korur.

### Adim 6: Piksel Normalizasyonu
- Piksel degerleri `[0, 255]` araligindan `[0.0, 1.0]` araligina float32 tipinde olceklenir. Bu adim model egitiminde gradyan patlamalarini ve kaymasini onler.

---

## 3. Ekip Uyeleri Icin Fonksiyon Kullanimi

Ekipteki herhangi bir kisi (ozellikle Kisi 2, 3 ve 4) bu fonksiyonu tek satirda cagirabilir:

```python
from src.preprocessing import on_isle

# 1. Standart kullanim (512x512 normalize cikti)
islenmis_img = on_isle("dosya/yolu/ornek.jpg")

# 2. Ara adimlari gorsellestirmek icin kullanim
islenmis_img, adimlar = on_isle("dosya/yolu/ornek.jpg", ara_adimlari_ver=True)
# adimlar: "1_orijinal", "2_roi_kirpilmis", "3_yesil_kanal", "4_clahe", "5_nihai_islenmis"
```

---

## 4. Gorsel Cikti: `reports/figures/on_isleme_asamalari.png`
Bu figur, 4 sinifin her birinden secilen ornekler uzerinde yukaridaki 5 adimi adim adim karsilastirmali olarak gosterir. Makalenin "Yontem (Methodology)" bolumune dogrudan sekil olarak eklenebilir.
