# 🔢 MNIST Rakam Üretici (GAN)

Rastgele gürültüden el yazısı rakam görüntüleri üretir (0-9)

---

## 🎯 Ne Yapıyor?

Bu klasik GAN projesi:
- **Girdi:** 100 boyutlu rastgele gürültü vektörü
- **Çıktı:** 28×28 piksel rakam görüntüsü
- **Veri:** MNIST veri seti (otomatik indirilir)

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Bu klasöre gel
cd mnist_gan

# 2. Eğitimi başlat (veri otomatik indirilir)
python train.py

# Sonuç: images/ klasöründe üretilen rakamlar
```

**Süre:** 30-60 dakika (CPU), 5-10 dakika (GPU)

---

## 📁 Dosyalar

- **`model.py`** - Generator ve Discriminator modelleri
- **`train.py`** - Eğitim kodu
- **`checkpoint_hizli_kullan.py`** - Kaydedilmiş modeli yükle
- **`GAN_NASIL_OGRENIR.md`** - Detaylı açıklama
- **`GERCEK_EGITIM_KILAVUZU.md`** - Eğitim rehberi
- **`NASIL_CALISIR.md`** - Teknik detaylar

---

## ⚙️ Ayarlar (train.py içinde)

```python
config = {
    'n_epochs': 200,        # Epoch sayısı
    'batch_size': 64,       # Batch boyutu
    'lr': 0.0002,          # Öğrenme hızı
    'latent_dim': 100,     # Gürültü boyutu
    'sample_interval': 10   # Kaç epoch'ta bir görüntü kaydet
}
```

---

## 📊 Beklenen Sonuçlar

```
Epoch 50:  Rakam şekilleri belli oluyor
Epoch 100: Rakamlar net
Epoch 200: Mükemmel kalite! ✅
```

**Sonuçlar:**
- `images/` - Üretilen görüntüler
- `checkpoints/` - Kaydedilen modeller
- `training_data/` - Eğitim verileri

---

## 🧪 Kaydedilmiş Model ile Test

```python
python checkpoint_hizli_kullan.py
# Checkpoint'ten model yükler ve yeni rakamlar üretir
```

---

## 💡 İpuçları

- **GPU yoksa:** `batch_size: 32` yap (daha hızlı)
- **Daha iyi kalite:** `n_epochs: 300-400`
- **Hızlı test:** `n_epochs: 50`

---

## 📚 Daha Fazla Bilgi

- `GAN_NASIL_OGRENIR.md` - GAN'ların öğrenme süreci
- `GERCEK_EGITIM_KILAVUZU.md` - Detaylı eğitim rehberi
- `NASIL_CALISIR.md` - Teknik açıklamalar

---

**Sorun mu var?** Ana klasördeki `PROJELER.md` dosyasına bak!

