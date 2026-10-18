# RETINA-AI: Literatur Tablosu

Fundus goruntuleriyle goz hastaligi tespitine yonelik secilmis 12 calisma. Makaledeki Tablo 1 ile aynidir.
Her kunye Crossref/PubMed kaydindan, her sonuc rakami calismanin kendi ozetinden dogrulanmistir.

---

## 1. Tablo

| No | Calisma | Hastalik / siniflar | Veri | Yontem | Bildirilen sonuc | Ilgili kisi |
|---|---|---|---|---|---|---|
| [1] | Walter vd. (2002) | DR (sert eksuda) | Kucuk bir goruntu veri tabani | Morfolojik yeniden yapilandirma | Duyarlilik %92,8; kestirim degeri %92,4 | Kisi 3 |
| [2] | Cheng vd. (2013) | Glokom (CDR) | 650 goruntu (bolutleme); 2 tarama veri seti | Superpiksel siniflandirma ile disk ve cukur bolutleme | AUC 0,800 ve 0,822 | Kisi 2 |
| [3] | Gulshan vd. (2016) | Sevk gerektiren DR | 128 175 egitim goruntusu; EyePACS-1, Messidor-2 | Derin CNN | AUC 0,991 ve 0,990 | Kisi 4 |
| [4] | Pratt vd. (2016) | DR siddeti | Kaggle DR, yaklasik 80 000 goruntu | CNN + veri artirma | Duyarlilik %95; dogruluk %75 | Kisi 4 |
| [5] | Ting vd. (2017) | DR, olasi glokom, YBMD | 494 661 goruntu | Derin ogrenme sistemi | AUC 0,936 / 0,942 / 0,931 | Kisi 4 |
| [6] | Li vd. (2018) | Glokomatoz optik noropati | 48 116 goruntu | Derin ogrenme | AUC 0,986; duyarlilik %95,6; ozgulluk %92,0 | Kisi 4 |
| [7] | Li vd. (2021) | 8 kategori | ODIR: 5000 hasta, 10 000 goruntu | Cesitli derin aglarla kiyaslama | Olcek artisi tek basina yetersiz | Kisi 1 |
| [8] | Junayed vd. (2021) | Katarakt / katarakt disi | 1130 goruntu (artirma ile 4746) | CataractNet (CNN) | Dogruluk %99,13 | Kisi 4 |
| [9] | Babaqi vd. (2023) | Normal, katarakt, DR, glokom | Yaklasik 4200 goruntu (IDRiD, HRF, ODIR vb.) | EfficientNet (transfer ogrenme); CNN | Dogruluk %94; CNN %84 | Kisi 4, 5 |
| [10] | Alsohemi ve Dardouri (2025) | Normal, katarakt, DR, glokom | Kaggle, 4217 goruntu (bizim veri seti) | EfficientNetB3 (transfer ogrenme) | Dogruluk %95,12; F1 %95,00 | Kisi 4, 5 |
| [11] | Cutur ve Inan (2025) | Normal, katarakt, DR, glokom | Kaggle, 4217 goruntu (bizim veri seti) | ViT (transfer ogrenme) | Dogruluk %98,1; ayri test kumesinde %94,21 | Kisi 4, 5 |
| [12] | Acevedo vd. (2025) | 5 sinif (dort sinif + diger retina hastaliklari) | 5 sinifli fundus veri seti | 11 katmanli CNN | Tum olcutlerde %97 | Kisi 4 |

Kisaltmalar: DR diyabetik retinopati; CDR cukur/disk orani; YBMD yasa bagli makula dejenerasyonu;
AUC ROC egrisi altinda kalan alan; ViT goruntu donusturucu.

---

## 2. Ekip icin notlar

- Bizim veri setimizi (Kaggle, 4217 goruntu) kullanan iki calisma var: [10], [11]. [9] ise ayni kaynaklardan (IDRiD, HRF,
  ODIR) derlenmis, yaklasik 4200 goruntuluk benzer bir veri seti kullaniyor. Sonuclarimizi en cok bu ucuyle karsilastiracagiz.
- [9] ve [10] verinin hasta duzeyinde mi goruntu duzeyinde mi bolundugunu yazmiyor. Bizim bolmemiz hasta duzeyinde;
  bu yuzden dogrulugumuz onlardan dusuk cikarsa bu dogal olabilir. Karsilastirirken bunu belirtecegiz.
- 12 calismanin 6'si tek hastaliga bakiyor ve hepsi ya klasik yontem ya derin ogrenme kullaniyor. Bizim projede iki
  yaklasimin ayni test kumesinde karsilastirilmasi makalenin guclu yani.
- Makalede yeni bir kaynak kullanmak isteyen once Kisi 6'ya iletsin: hocanin kurali geregi kaynaklar APA bicimiyle,
  metinde sirali numarayla yazilir; veri seti disinda URL verilmez.

---

## 3. Bolumlere gore yontem kaynaklari

Teslim notunuzda bu yontemleri kullandiysaniz makalede bu kaynaklara atif yapilacak.

| Kisi | Konu | Kaynak |
|---|---|---|
| Kisi 1 | Ayni hastanin goruntuleri egitim ve testte olursa basari yapay olarak yukselir | Yagis vd. (2021); Kapoor ve Narayanan (2023) |
| Kisi 2 | Damar bolutleme (black top-hat, Otsu esigi; alternatif: Frangi filtresi), optik disk ve cukur, CDR | Zana ve Klein (2001); Otsu (1979); Frangi vd. (1998); Cheng vd. (2013); Li vd. (2018) |
| Kisi 3 | Eksuda tespiti, SVM ve Random Forest | Walter vd. (2002); Cortes ve Vapnik (1995); Breiman (2001) |
| Kisi 4 | EfficientNet-B0, ImageNet on egitimi, AdamW, etiket yumusatma, Grad-CAM | Tan ve Le (2019); Deng vd. (2009); Loshchilov ve Hutter (2019); Szegedy vd. (2016); Selvaraju vd. (2017) |
| Kisi 5 | ROC egrisi ve AUC; ayni veri setinde bildirilen sonuclar (karsilastirma icin); modelin kisa yol ogrenmesi | Fawcett (2006); Alsohemi ve Dardouri (2025); Cutur ve Inan (2025); Babaqi vd. (2023); Geirhos vd. (2020) |

---

## 4. Kunyeler (APA)

Tablodaki calismalar:

[1] Walter, T., Klein, J.-C., Massin, P., & Erginay, A. (2002). A contribution of image processing to the diagnosis of diabetic retinopathy—Detection of exudates in color fundus images of the human retina. *IEEE Transactions on Medical Imaging, 21*(10), 1236–1243.

[2] Cheng, J., Liu, J., Xu, Y., Yin, F., Wong, D. W. K., Tan, N.-M., Tao, D., Cheng, C.-Y., Aung, T., & Wong, T. Y. (2013). Superpixel classification based optic disc and optic cup segmentation for glaucoma screening. *IEEE Transactions on Medical Imaging, 32*(6), 1019–1032.

[3] Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., Venugopalan, S., Widner, K., Madams, T., Cuadros, J., Kim, R., Raman, R., Nelson, P. C., Mega, J. L., & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA, 316*(22), 2402–2410.

[4] Pratt, H., Coenen, F., Broadbent, D. M., Harding, S. P., & Zheng, Y. (2016). Convolutional neural networks for diabetic retinopathy. *Procedia Computer Science, 90*, 200–205.

[5] Ting, D. S. W., Cheung, C. Y.-L., Lim, G., Tan, G. S. W., Quang, N. D., Gan, A., Hamzah, H., Garcia-Franco, R., San Yeo, I. Y., Lee, S. Y., Wong, E. Y. M., Sabanayagam, C., Baskaran, M., Ibrahim, F., Tan, N. C., Finkelstein, E. A., Lamoureux, E. L., Wong, I. Y., Bressler, N. M., . . . Wong, T. Y. (2017). Development and validation of a deep learning system for diabetic retinopathy and related eye diseases using retinal images from multiethnic populations with diabetes. *JAMA, 318*(22), 2211–2223.

[6] Li, Z., He, Y., Keel, S., Meng, W., Chang, R. T., & He, M. (2018). Efficacy of a deep learning system for detecting glaucomatous optic neuropathy based on color fundus photographs. *Ophthalmology, 125*(8), 1199–1206.

[7] Li, N., Li, T., Hu, C., Wang, K., & Kang, H. (2021). A benchmark of ocular disease intelligent recognition: One shot for multi-disease detection. In F. Wolf & W. Gao (Eds.), *Benchmarking, measuring, and optimizing* (pp. 177–193). Springer.

[8] Junayed, M. S., Islam, M. B., Sadeghzadeh, A., & Rahman, S. (2021). CataractNet: An automated cataract detection system using deep learning for fundus images. *IEEE Access, 9*, 128799–128808.

[9] Babaqi, T., Jaradat, M., Yildirim, A. E., Al-Nimer, S. H., & Won, D. (2023). *Eye disease classification using deep learning techniques* (arXiv:2307.10501). arXiv.

[10] Alsohemi, R., & Dardouri, S. (2025). Fundus image-based eye disease detection using EfficientNetB3 architecture. *Journal of Imaging, 11*(8), Article 279.

[11] Cutur, E. S., & Inan, N. G. (2025). Multi-class classification of retinal eye diseases from ophthalmoscopy images using transfer learning-based vision transformers. *Journal of Imaging Informatics in Medicine, 38*(5), 3110–3124.

[12] Acevedo, E., Orantes, D., Acevedo, M., & Carreño, R. (2025). Identification of eye diseases through deep learning. *Diagnostics, 15*(7), Article 916.

Yontem kaynaklari:

- Yagis, E., Atnafu, S. W., García Seco de Herrera, A., Marzi, C., Scheda, R., Giannelli, M., Tessa, C., Citi, L., & Diciotti, S. (2021). Effect of data leakage in brain MRI classification using 2D convolutional neural networks. *Scientific Reports, 11*(1), Article 22544.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Article 100804.
- Zana, F., & Klein, J.-C. (2001). Segmentation of vessel-like patterns using mathematical morphology and curvature evaluation. *IEEE Transactions on Image Processing, 10*(7), 1010–1019.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics, 9*(1), 62–66.
- Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998). Multiscale vessel enhancement filtering. In W. M. Wells, A. Colchester, & S. Delp (Eds.), *Medical image computing and computer-assisted intervention — MICCAI’98* (pp. 130–137). Springer.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In K. Chaudhuri & R. Salakhutdinov (Eds.), *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 6105–6114). PMLR.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 248–255). IEEE.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR 2019)*.
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2818–2826). IEEE.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *2017 IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). IEEE.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*(11), 665–673.
