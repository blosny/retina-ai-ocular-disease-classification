# RETINA-AI: Kişi 2 (Damar + Optik Disk) Teslim Notu

### Ne yaptım:
1. `src/anatomy.py` dosyasında damar segmentasyonu, damar yoğunluğu ve optik disk/çukur tespiti fonksiyonlarını yazdım:
   - `damar_haritasi(islenmis)`: `on_isle` çıktısından 0/255 damar maskesi üretir (damar beyaz).
   - `damar_yogunlugu(maske, islenmis)`: damar piksellerinin retina alanına oranı (0-1). İki argüman alır.
   - `optik_disk(renkli)`: renkli 512x512 RGB görüntüden `{"merkez": (x, y), "yaricap": r, "cukur_disk_orani": oran}` döndürür; güvenilir disk bulamazsa `None`.
   - `disk_isaretle(goruntu, sonuc)`: diski ve çukuru görüntü üzerine çizer.
2. Her sınıftan 5 görüntüyle (toplam 20) toplu test yaptım, sonuçları `anatomi_sonuclari.csv` dosyasına kaydettim.
3. Veri setindeki görüntü çözünürlüklerini sınıflara göre saydım (aşağıda).
4. Fonksiyonlarımı K5'in Streamlit arayüzüne (`src/app.py`) bağladım: geçici damar fonksiyonu silindi, 5. sütuna optik disk eklendi. Arayüzde bir normal ve bir katarakt görüntüsüyle uçtan uca denedim.

---

### Neden yaptım:
- Damarlar yeşil kanalda koyu ve ince yapılar olduğu için black top-hat ile öne çıkardım.
- Retina dışındaki siyah alan ve daire kenarı damar sanılmasın diye retina alanını görüntüden buldum, tüm hesapları bu alanla sınırladım.
- Yoğunluğu retina alanına bölerek köşelerdeki siyah piksellerin sonucu küçültmesini önledim.
- Diski, retinanın en parlak bölgesi olarak aradım. Yanlış yere daire çizmek yerine "bulamadım" demesi için yuvarlaklık ve çevresinden parlaklık kontrolü ekledim.
- Çukur/disk oranını tıpta kullanılan dikey çap oranıyla hesapladım.

---

### Hangi parametreleri kullandım:
- Retina alanı: gri > 5 eşiği, 15x15 closing, en büyük kontur. Damar için kenardan 10 piksel, disk için 25 piksel içeri çekildi.
- Damar: 15x15 elips çekirdekle black top-hat, Otsu eşiği (sadece retina içinden hesaplandı), eşik alt sınırı 12.
- Damar parça filtresi: alan en az 30 piksel, uzunluk en az 25 piksel, dolgun ve kısa (doluluk > 0.55, uzunluk < 50) parçalar silinir.
- Disk: Gauss bulanıklaştırma (sigma 12) ile en parlak nokta, çevresinde 90 piksel yarıçaplı pencerede Otsu eşiği, 7x7 closing.
- Disk kabul koşulları: yarıçap 15-70 piksel, yuvarlaklık en az 0.7 (dış zarf üzerinden), çevresinden parlaklık farkı en az 25.
- Çukur: disk içinde yeşil kanalın %10-%98 aralığında %60 eşiği, 5x5 closing. Oran = çukurun dikey çapı / diskin dikey çapı.

---

### Çıktı/görsel nerede:
- Kod: `src/anatomy.py` 
- 20 görüntülük sonuç tablosu: `anatomi_sonuclari.csv` // eklenecek

**20 görüntü sonuçları:**

| Sınıf | Damar yoğunluğu (ort.) | Disk bulunma | Çukur/disk oranı (ort.) |
|---|---|---|---|
| normal | 0.072 | 5/5 | 0.46 |
| diabetic | 0.072 | 5/5 | 0.56 |
| glaucoma | 0.065 | 4/5 | 0.69 |
| cataract | 0.005 | 0/5 | - |

**Arayüz denemesi:** Normal görüntüde (`8_left.jpg`) damar yoğunluğu 0.073 ve çukur/disk oranı 0.62 çıktı, toplu testteki değerlerle aynı. Katarakt görüntüsünde (`_0_4015166.jpg`) damar yoğunluğu 0.000 ve "disk bulunamadı" uyarısı çıktı, sayfa çökmedi.

---

### Karşılaştığım sorun:
**Geliştirme sırasında çözülenler:**
- Retinanın daire kenarı damar gibi çıktı: görüntü sınırının dışı da "boş" sayılarak çözüldü.
- Katarakt görüntülerinde bulanıklıktan kaynaklı gürültü damar sanıldı: Otsu eşiğine alt sınır konarak çözüldü.
- Glokom örneğindeki kanama lekeleri damar sanıldı: kısa ve dolgun parçalar filtrelenerek azaltıldı, tamamen gitmedi.
- İlk sürümde çukur/disk oranı her görüntüde 0.7-0.9 çıkıyordu (çukur, diskin neredeyse tamamı sayılıyordu): çukur, diskin en parlak kısmı olarak yeniden tanımlandı.
- Damar çıkıntıları diskin yuvarlaklığını düşürüyordu (normal ve diabetic reddediliyordu): yuvarlaklık dış zarf üzerinden ölçülerek çözüldü.

**Başarısız olan görüntüler (hata analizi):**
- Katarakt: 5 görüntüde de disk bulunamadı, damar yoğunluğu 0.000-0.012. Görüntüler sarımsı ve puslu, damar ve disk gözle de seçilmiyor.
- Glokom: `glaucoma_1_4540560.jpg` çok karanlık, parlak bölge görüntü kenarında. Disk büyük olasılıkla aranan alanın dışında kaldı (doğrulanmadı).

**Sınırlılıklar:**
- **İnce damarlar kısmen eksik.** Ön işlenmiş (yeşil kanal + CLAHE) görüntüde ince damar ağı gözle görülüyor, damar haritası ise ağırlıklı olarak kalın ana damarları ve bazı dalları yakalıyor. Gürültü ve lekeler damar sanılmasın diye konan eşik alt sınırı, minimum uzunluk ve leke filtresi ince ve soluk damarları da eliyor. Daha duyarlı bir harita katarakt ve glokomda gürültüyü geri getirebilir, bu takas test edilmedi. Frangi filtresi ve çok ölçekli çekirdek denenmedi.
- Çukur/disk oranı parlaklığa dayalı yaklaşık bir değerdir. Sınıf ortalamaları beklenen sırada (normal < diabetic < glokom), ama sınıf içi dağılım geniş (diabetic: 0.40-0.80) ve elimizde gerçek çukur etiketi yok, doğruluğu ölçülmedi. Glokom ortalaması 4 görüntüden.
- Eşikler ilk 4 görüntüye bakılarak ayarlandı ve bu 4 görüntü 20'lik sette de var. Bu yüzden 20 görüntülük sonuç bağımsız bir doğrulama değil.
- Örnekler elle seçildi, train/test ayrımı kontrol edilmedi. Bazı hastaların iki gözü de dahil (diabetic-100, normal_84, normal_8).
- Optik disk için Hough yerine eşikleme kullanıldı.

**Veri setinde çözünürlük bulgusu (veri_bolme.csv, 4028 görüntü):**
- diabetic_retinopathy ve normal: hepsi 512x512 (1007'şer).
- cataract: 623 adet 256x256, 288 adet 512x512, 96 adet 1848x1224 ile 2592x1728 arası.
- glaucoma: 600 adet 256x256, 306 adet 512x512, 101 adet 1848x1224 ile 2592x1728 arası.
- 256x256 ve çok büyük görüntüler yalnızca cataract ve glaucoma sınıflarında var. Bir model hastalık yerine çözünürlüğe bakarak bu iki grubu diğer ikisinden ayırabilir. Katarakttaki düşük damar yoğunluğunun sebebi (bulanıklık mı, düşük çözünürlük mü) bu testle ayrılamadı.

**Ekip için not:** CNN'in en çok karıştırdığı sınıf çifti glokom ve normal. Çukur/disk oranı bu ikisini ayırması beklenen klinik bir özellik, ama bizim oranımız güvenilirliği ölçülmemiş yaklaşık bir değer. Makale tartışmasında bir yön olarak ele alınabilir, bir sonuç olarak değil.