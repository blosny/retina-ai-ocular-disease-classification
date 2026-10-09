# RETINA-AI: Veri ve On Isleme Sorumlusu Teslim Notu

Bu rapor, proje kurallarinda belirlenen teslim sablonuna tam olarak uygun sekilde hazirlanmistir.

---

### Ne yaptim:
1. Kaggle 'eye_diseases_classification' veri setindeki 4217 gorselin tamamini OpenCV ile tarayip bozuk dosya kontrolu yaptim.
2. MD5 kriptografik hash taramasiyla mukerrer ve capraz sinif cakisiklarini tespit ettim.
3. Glokom sinifinin 1007 olan boyutunu baz alarak her siniftan seed=42 ile 1007 ornek sectim ve toplam 4028 dengeli gorsel elde ettim.
4. Stratified Split ile her sinifi kesin olarak %70 Train (705), %15 Validation (151), %15 Test (151) olarak boldum ve tum ekibin ortak kullanacagi `veri_bolme.csv` dosyasini urettim.
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
