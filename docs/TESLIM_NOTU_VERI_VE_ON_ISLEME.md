# RETINA-AI: Veri ve On Isleme Sorumlusu Teslim Notu

Bu rapor, proje kurallarinda belirlenen teslim sablonuna tam olarak uygun sekilde hazirlanmistir.

---

### Ne yaptim:
1. Kaggle 'eye_diseases_classification' veri setindeki 4217 gorselin tamamini OpenCV ile tarayip bozuk dosya kontrolu yaptim.
2. MD5 kriptografik hash taramasiyla mukerrer ve capraz sinif cakisiklarini tespit ettim.
3. Glokom sinifinin 1007 olan boyutunu baz alarak her siniftan seed=42 ile 1007 ornek sectim ve toplam 4028 dengeli gorsel elde ettim.
4. Hasta Bazli Stratified Split (Patient-Level Group Stratified) yontemiyle ayni hastanin sag ve sol gozlerinin ayni kumede kalmasini sagladim (%0.0 sizinti). Her sinifi kesin olarak %70 Train (705), %15 Validation (151), %15 Test (151) olarak boldum ve `veri_bolme.csv` dosyasini urettim.
5. Tum ekibin kullanacagi `on_isle()` fonksiyonunu yazdim (ROI kirpma, 512x512 boyutlandirma, yesil kanal izolasyonu, CLAHE, median filtre ve [0.0, 1.0] normalizasyonu).
6. Ekip arkadaslarimizin hemen kod yazmaya baslayabilmesi icin ortak Drive klasor agacini ve calisir durumda `01_veri_on_isleme.ipynb` Colab sablonunu hazirladim.

---

### Neden yaptim:
- Siniflardaki dengesizlik (imbalance), modellerin cogunluk sinifina kaymasina (bias) sebep olur. 1007'ye esitlemek bu sorunu kokten cozdu.
- Ham retinadaki devasa siyah kenarliklar yapay zeka modellerinin gereksiz arka plana odaklanmasina sebep olur; ROI kirpma ile sadece retina dokusuna odaklanildi.
- Yesil kanal, kandaki hemoglobinin en cok absorbe ettigi dalga boyu oldugu icin damarlarin ve kanama/eksudalarin en net goruldugu yerdir.
- CLAHE, yerel aydinlatma farklarini ve kataraktli gozlerdeki puslu goruntuleri dengeler.
- Median filtre ise Gauss filtresinin aksine damar ve lezyon kenarlarini bulandirmadan sensor gurultusunu yok eder.

---

### Hangi parametreleri kullandim:
- Bolme tohum degeri: `seed=42`
- Bolme oranlari: `%70 Train (705)`, `%15 Val (151)`, `%15 Test (151)`
- Hedef boyut: `512x512` piksel (`interpolation=cv2.INTER_AREA`)
- ROI esik degeri: `piksel > 10` ve dairesel merkez maskesi
- CLAHE parametreleri: `clipLimit=2.0`, `tileGridSize=(8, 8)`
- Gurultu filtresi: `MedianBlur (cekirdek_boyutu=3)`
- Normalizasyon: `[0.0, 1.0]` float32

---

### Cikti/gorsel nerede:
- Ortak Drive Klasoru: https://drive.google.com/drive/folders/1nwjYPcgwE-nXCPeCYR1B6o8ucXRCKT6O?usp=drive_link
- Referans Bolme Tablosu: `data/splits/veri_bolme.csv`
- On Isleme Modulu: `src/preprocessing.py` (fonksiyon: `on_isle`)
- Sinif Dagilim Grafigi: `reports/figures/sinif_dagilimi.png`
- Cozunurluk Dagilim Grafigi: `reports/figures/cozunurluk_dagilimi.png`
- On Isleme Asamalari Karsilastirma Figuru: `reports/figures/on_isleme_asamalari.png`
- Ortak Colab Sablonu: `notebooks/01_veri_on_isleme.ipynb`
- Detayli Dokumanlar: `docs/VERI_BOLME_VE_METADATA.md`, `docs/ON_ISLEME_PIPELINE.md`, `docs/ORTAK_DRIVE_YAPISI.md`

---

### Karsilastigim sorun:
Kaggle veri setinde `625_left.jpg` ve `1415_right.jpg` adli iki gorselin ayni MD5 hash degeriyle hem 'cataract' hem de 'glaucoma' klasorunde yer aldigi ortaya cikti (capraz sinif cakisimi). Bu durum ayni hastanin iki farkli sinifta gorunmesi sebebiyle etiket kirliligine ve veri sizintisina yol acabilirdi. Glokom sinifi tam 1007 ornege sahip oldugu icin, bu 2 cakisik ornek sayisi fazla olan Katarakt sinifindan ayiklanarak veri sizintisi sifirlandi.

---

### Ek Not: Veri Seti v2 Revizyonu ve Cozum Analizi (20 Ekim 2026)

#### 1. v1 Bolmesindeki Sorun ve Kisayol Ogrenme (Shortcut Learning)
Kaggle veri setindeki katarakt ve glokom siniflarinda yer alan 1223 adet 256x256 boyutundaki gorselin, orijinal fotograflarin dondurulmus ve cevrilmis kopyalari oldugu ortaya cikmistir. Bu kopyalar dosya adlarindaki farklilik sebebiyle ayri hasta gibi algilanmis ve hem egitim hem test kumelerine dagilarak hasta bazli sizintiya neden olmustur. Model hastalik morfolojisi yerine goruntu boyutunu ogrenerek (shortcut learning) kucuk goruntuleri dogrudan katarakt veya glokom olarak tahmin etmistir.

#### 2. v2 Bolmesinde Nasil Cozuldu?
- Cozunurluk Filtresi: Tum 256x256 kopyalar ve asiri buyuk goruntuler elendi. Yalnizca standart 512x512 cozunurlukteki orijinal goruntuler secildi.
- Sinif Dengesi: 512x512 cozunurlukte en az sayiya sahip olan Katarakt sinifi (288 adet) referans alinarak her 4 siniftan tam 288'er adet (toplam 1152 gorsel) ile tam dengeli kume kuruldu.
- Hasta Yonetimi ve Stratified Split: ODIR (`odir_<id>`) ve EyePACS (`eyepacs_<id>`) kaynaklari hasta anahtarinda birlestirilerek ayni hastanin iki gozunun ayni kumede kalmasi saglandi (seed=42).
- Dogrulama Sonuclari: `src/verify_splits.py` ile `data/splits/veri_bolme_v2.csv` tablosu dogrulanmis, kumeler arasi hasta kesisimi 0 (%0.0 sizinti) ve MD5 kopya sayisi 0 olarak kesinlesmistir.
- Karsilastirma Guvencesi: Eski `veri_bolme.csv` tablosu silinmeyip korunmus, makalede once/sonra basari karsilastirmasi icin referans birakilmistir.

