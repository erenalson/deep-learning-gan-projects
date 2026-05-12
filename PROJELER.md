# 📁 GAN Projeleri Klasör Yapısı

Bu klasörde 2 farklı GAN projesi var:

---

## 🔢 PROJE 1: MNIST Rakam Üretici (Orijinal GAN)

**Ne Yapıyor:** Rastgele gürültüden rakam görüntüleri üretir (0-9)

**Dosyalar:**
```
mnist_gan/
├── model.py                    # Generator ve Discriminator
├── train.py                    # Eğitim kodu
├── utils.py                    # Yardımcı fonksiyonlar
├── checkpoint_hizli_kullan.py  # Checkpoint yükleme
├── README.md                   # Kullanım rehberi
└── (Eğitim sonrası)
    ├── images/                 # Üretilen görüntüler
    ├── checkpoints/            # Kaydedilen modeller
    └── training_data/          # Eğitim verileri
```

**Hızlı Başlangıç:**
```bash
cd mnist_gan
python train.py
```

---

## 🌅 PROJE 2: CycleGAN Görüntü Dönüştürücü

**Ne Yapıyor:** Görüntüleri başka stile çevirir (Yaz→Kış, At→Zebra, vb.)

**Dosyalar:**
```
cyclegan/
├── cyclegan_model.py           # CycleGAN modeli
├── cyclegan_train.py           # Eğitim kodu
├── cyclegan_test.py            # Test kodu
├── cyclegan_download_data.py   # Veri indirme rehberi
├── create_demo_dataset.py      # Demo veri oluşturucu
├── README.md                   # Kullanım rehberi
└── (Eğitim sonrası)
    ├── cyclegan_results/       # Sonuçlar
    ├── cyclegan_checkpoints/   # Modeller
    └── data/                   # Veri setleri
```

**Hızlı Başlangıç:**
```bash
cd cyclegan
python create_demo_dataset.py 30  # Demo veri
# veya
python cyclegan_download_data.py  # Gerçek veri
python cyclegan_train.py
```

---

## 📚 Ortak Dosyalar (Ana Klasörde)

```
GAN/
├── mnist_gan/                  # 🔢 Rakam üretici
├── cyclegan/                   # 🌅 Görüntü dönüştürücü
├── PROJELER.md                 # Bu dosya
└── requirements.txt            # Gerekli paketler
```

---

## 🚀 Hangi Projeyi Kullanmalıyım?

| Görev | Kullan | Örnek |
|-------|--------|-------|
| Yeni içerik üret | **MNIST GAN** | Rakam, yüz, nesne üret |
| Görüntü dönüştür | **CycleGAN** | Yaz→Kış, At→Zebra |
| GAN öğren | **MNIST GAN** | Basit, hızlı |
| İleri seviye | **CycleGAN** | Gerçek projeler |

---

## 💡 Bağımsız Çalışma

Her proje **tamamen bağımsız**:
- Kendi klasöründe çalışır
- Dosyalar karışmaz
- Ayrı ayrı eğitilebilir

---

## ✅ Yapılacaklar

- [ ] `mnist_gan/` klasörü oluşturuldu
- [ ] `cyclegan/` klasörü oluşturuldu
- [ ] Dosyalar organize edildi
- [ ] Her klasörde README.md var
- [ ] Test edildi

---

**Sonraki adım:** Dosyaları düzenliyorum! 🔄

