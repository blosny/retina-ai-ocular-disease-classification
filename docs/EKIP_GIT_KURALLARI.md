# RETINA-AI: Ekip Git ve GitHub Calisma Kurallari

Bu dokuman, 6 kisilik proje ekibinin GitHub uzerinde cakisma (merge conflict) yasamadan, duzenli ve profesyonel bir sekilde calisabilmesi icin hazirlanmistir.

---

## 1. Temel Kural: Buyuk Verileri Asla Push Etmeyin
- Gorseller, zip arsivleri ve egitilmis modeller (.pth, .h5) GitHub'a kesinlikle yuklenmemelidir.
- GitHub dosya basina 100 MB limiti uygular ve depo 2 GB'i gecerse kilitlenebilir.
- Tum veri seti Google Drive uzerinde tutulur. GitHub'da sadece kaynak kodlar (.py, .ipynb, .md) yer alir.
- .gitignore dosyasini silmeyiniz veya degistirmeyiniz.

---

## 2. Branch (Dal) Stratejisi

### Neden Branch Acilmali?
1. Guvenlik: Yanlislikla hatali kod veya buyuk dosya eklendiginde ana (main) dal bozulmaz.
2. Cakismasiz Calisma: Iki kisi ayni anda push etmeye calistiginda yasancak cakismalar engellenir.
3. Takip Kolayligi: Herkesin katkisi kendi dalinda net gorulur.

### Branch Isimlendirme Standardi:
- Kisi 1: feature/veri-ve-on-isleme
- Kisi 2: feature/damar-ve-disk
- Kisi 3: feature/lezyon-ve-klasik-model
- Kisi 4: feature/derin-ogrenme-cnn
- Kisi 5: feature/degerlendirme-ve-streamlit
- Kisi 6: docs/makale-ve-raporlama

---

## 3. Conventional Commits (Commit Mesaji Standardi)

Commit mesajlari icin asagidaki format kullanilmalidir:

```text
<tur>(<kapsam>): <kisa ve net aciklama>
```

### Gecerli Turler:
- feat: Yeni bir ozellik veya modul eklendiginde (ornek: feat(preprocessing): add auto-crop pipeline)
- fix: Hata duzeltildiginde (ornek: fix(vessel): correct green channel threshold)
- docs: Dokumantasyon degisikliklerinde (ornek: docs(readme): update setup instructions)
- refactor: Kod yapisi iyilestirildiginde (ornek: refactor(data): optimize loop in split logic)
- chore: Kutuphane veya ayar guncellemelerinde (ornek: chore(git): update gitignore)

---

## 4. Gunluk Calisma Rutini

1. Gune baslarken ana dali guncelleyin:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Kendi daliniza gecip ana dali birlestirin:
   ```bash
   git checkout feature/<dal-adiniz>
   git merge main
   ```
3. Kodunuzu yazip sadece ilgili dosyalari ekleyin:
   ```bash
   git add src/preprocessing.py
   git commit -m "feat(preprocessing): implement on_isle function"
   ```
4. Kendi dalinizi gonderin:
   ```bash
   git push origin feature/<dal-adiniz>
   ```
5. GitHub uzerinden Pull Request (PR) acin.
