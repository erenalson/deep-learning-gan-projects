# 📚 GAN PROJESİ NASIL ÇALIŞIR?

## 🎯 Genel Bakış

Bu proje **GAN (Generative Adversarial Network)** teknolojisi kullanarak sıfırdan görüntü üretir.

### Basit Açıklama
İki yapay zeka birbirine karşı oynar:
- **Generator (Sanatçı)**: Sahte para üretmeye çalışan sahteci
- **Discriminator (Dedektif)**: Sahte parayı yakalamaya çalışan dedektif

Sahteci gittikçe iyileşir → Dedektif gittikçe iyileşir → En sonunda mükemmel görüntüler üretilir!

---

## 📦 VERİ NEREDEN GELİYOR?

### MNIST Veri Seti

**Veri Kaynağı:** http://yann.lecun.com/exdb/mnist/

**İçeriği:**
- 60,000 eğitim görüntüsü
- 10,000 test görüntüsü
- El yazısı rakamlar (0-9)
- Her görüntü 28x28 piksel, siyah-beyaz

**Otomatik İndirme:**
```python
# train.py dosyasında, 66. satırda:
dataset = datasets.MNIST(
    root='./data',         # 👈 Buraya indirilir!
    train=True,
    download=True,         # 👈 Otomatik indir
    transform=transform
)
```

**İlk çalıştırmada:**
1. Internet'ten MNIST indirilir (~12 MB)
2. `./data` klasörüne kaydedilir
3. Sonraki çalıştırmalarda yerel dosyadan yüklenir

**Veri Formatı:**
- Ham görüntü: 0-255 arası piksel değerleri
- Normalize edilmiş: -1 ile 1 arası (GAN için optimal)

---

## 🏗️ PROJE YAPISI

```
GAN/
├── 📄 model.py          # Generator ve Discriminator tanımları
├── 📄 train.py          # Eğitim scripti (ANA DOSYA)
├── 📄 generate.py       # Yeni görüntü üretme
├── 📄 utils.py          # Yardımcı fonksiyonlar
├── 📄 requirements.txt  # Gerekli Python paketleri
│
├── 📁 data/             # MNIST veri seti (otomatik indirilir)
│   └── MNIST/
│       └── raw/
│           ├── train-images-idx3-ubyte
│           ├── train-labels-idx1-ubyte
│           └── ...
│
├── 📁 images/           # Üretilen görüntüler (eğitim sırasında)
│   ├── epoch_0010.png
│   ├── epoch_0020.png
│   ├── ...
│   ├── loss_plot.png
│   └── training_progress.gif
│
└── 📁 checkpoints/      # Kaydedilen modeller
    ├── gan_epoch_50.pt
    ├── gan_epoch_100.pt
    ├── generator_final.pt
    └── discriminator_final.pt
```

---

## 🔄 ÇALIŞMA AKIŞI

### 1️⃣ BAŞLATMA (Initialization)

```python
# train.py çalıştırıldığında:

1. CUDA kontrolü (GPU var mı?)
2. Generator oluştur
   - 100 sayı → 128 → 256 → 512 → 1024 → 784 (28x28)
3. Discriminator oluştur
   - 784 (28x28) → 512 → 256 → 128 → 1
4. MNIST'i indir/yükle
5. Klasörleri oluştur (images/, checkpoints/)
```

### 2️⃣ EĞİTİM DÖNGÜSÜ (Training Loop)

Her epoch için (200 kez):
```
┌─────────────────────────────────────────────────────┐
│  EPOCH 1                                            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  📦 BATCH 1 (64 görüntü):                          │
│  ┌─────────────────────────────────────┐          │
│  │ 1. Generator'ı Eğit                 │          │
│  │    - Gürültü üret: [100 sayı]      │          │
│  │    - Görüntü üret: [28x28]         │          │
│  │    - Discriminator'ı kandırmaya     │          │
│  │      çalış                           │          │
│  │    - Loss hesapla, ağırlık güncelle│          │
│  │                                     │          │
│  │ 2. Discriminator'ı Eğit            │          │
│  │    - Gerçek görüntülere bak → 1 de │          │
│  │    - Sahte görüntülere bak → 0 de  │          │
│  │    - Loss hesapla, ağırlık güncelle│          │
│  └─────────────────────────────────────┘          │
│                                                     │
│  📦 BATCH 2 (64 görüntü):                          │
│  ... (aynı işlem)                                  │
│                                                     │
│  📦 BATCH 938 (son batch):                         │
│  ... (aynı işlem)                                  │
│                                                     │
│  ✅ Epoch 1 tamamlandı!                            │
│     - Ortalama Generator Loss: 0.8234             │
│     - Ortalama Discriminator Loss: 0.6543         │
│     - 25 örnek görüntü kaydet: epoch_0010.png     │
└─────────────────────────────────────────────────────┘

... 200 epoch daha ...
```

### 3️⃣ KAYDETME (Saving)

**Her 10 Epoch:** Örnek görüntüler üret ve kaydet
```python
# images/epoch_0010.png, epoch_0020.png, ...
# 5x5 grid (25 rakam)
```

**Her 50 Epoch:** Model checkpoint kaydet
```python
# checkpoints/gan_epoch_50.pt
# İçerik: Tüm model ağırlıkları + optimizer durumu
```

**Son Epoch:** Final sonuçlar
```python
# checkpoints/generator_final.pt (görüntü üretmek için)
# images/loss_plot.png (grafik)
# images/training_progress.gif (animasyon)
```

---

## 🧠 KODLAR NE İŞE YARAR?

### 1. `model.py` - Model Tanımları

#### Generator Sınıfı
```python
# 🎨 NE YAPAR: Rastgele gürültüden görüntü üretir

Girdi:  [100 rastgele sayı]
        ↓
Katman 1: 100 → 128 nöron (Linear + BatchNorm + LeakyReLU)
        ↓
Katman 2: 128 → 256 nöron
        ↓
Katman 3: 256 → 512 nöron
        ↓
Katman 4: 512 → 1024 nöron
        ↓
Katman 5: 1024 → 784 (Linear + Tanh)
        ↓
Çıktı:  [28x28 görüntü] ([-1, 1] aralığında)
```

#### Discriminator Sınıfı
```python
# 🔍 NE YAPAR: Görüntünün gerçek mi sahte mi olduğunu tahmin eder

Girdi:  [28x28 görüntü] → Düzleştir → [784 sayı]
        ↓
Katman 1: 784 → 512 nöron (Linear + LeakyReLU + Dropout)
        ↓
Katman 2: 512 → 256 nöron (Linear + LeakyReLU + Dropout)
        ↓
Katman 3: 256 → 128 nöron (Linear + LeakyReLU + Dropout)
        ↓
Katman 4: 128 → 1 (Linear + Sigmoid)
        ↓
Çıktı:  [1 sayı] (0-1 arası: 0=sahte, 1=gerçek)
```

### 2. `train.py` - Eğitim Scripti

**Ana Sınıf: GANTrainer**

```python
__init__():
    # 🔧 BAŞLATMA
    - GPU/CPU seç
    - Generator oluştur
    - Discriminator oluştur
    - Loss fonksiyonu tanımla (BCELoss)
    - Optimizerler oluştur (Adam)
    - MNIST yükle
    - Klasörler oluştur

_get_dataloader():
    # 📦 VERİ YÜKLEME
    - MNIST'i indir (yoksa)
    - Normalize et: [0, 255] → [-1, 1]
    - Batch'lere böl: 60,000 → 938 batch (64'lük)
    - DataLoader döndür

train():
    # 🎯 ANA EĞİTİM
    For her epoch (200 kez):
        For her batch (938 kez):
            # Generator Eğitimi:
            1. Gürültü üret (100 sayı)
            2. Görüntü üret
            3. Discriminator'a göster
            4. Loss hesapla (gerçek etiket kullan)
            5. Backpropagation
            6. Ağırlıkları güncelle
            
            # Discriminator Eğitimi:
            1. Gerçek görüntüleri değerlendir
            2. Sahte görüntüleri değerlendir
            3. Loss hesapla (her ikisi için)
            4. Backpropagation
            5. Ağırlıkları güncelle
        
        Epoch bitti:
        - Loss'ları kaydet
        - Her 10 epoch: Görüntü kaydet
        - Her 50 epoch: Checkpoint kaydet
    
    Eğitim bitti:
    - Final modelleri kaydet
    - Grafik çiz
    - GIF oluştur

save_checkpoint():
    # 💾 CHECKPOINT KAYDET
    - Tüm model durumunu .pt dosyasına kaydet
    - Eğitimi daha sonra devam ettirebilmek için

save_final_results():
    # ✅ SON SONUÇLAR
    - Generator ve Discriminator'ı kaydet
    - Loss grafiği çiz
    - Eğitim animasyonu (GIF) oluştur
```

### 3. `utils.py` - Yardımcı Fonksiyonlar

```python
save_generated_images():
    # 🖼️ GÖRÜNTÜ KAYDETME
    - 25 adet görüntü üret (5x5 grid)
    - PNG olarak kaydet: epoch_0010.png

plot_losses():
    # 📊 GRAFİK ÇİZME
    - Generator ve Discriminator loss'larını çiz
    - 2 panel yan yana grafik
    - Kaydedilir: loss_plot.png

create_gif():
    # 🎬 ANİMASYON OLUŞTURMA
    - Tüm epoch görüntülerini birleştir
    - GIF animasyon oluştur
    - Eğitim ilerlemesini gösterir

generate_samples():
    # 🎨 YENİ GÖRÜNTÜ ÜRETME
    - Kaydedilmiş modelden görüntü üret
    - İstediğiniz sayıda (default: 25)
```

### 4. `generate.py` - Görüntü Üretici

```python
generate_images():
    # 🎨 ANA FONKSİYON
    1. Eğitilmiş modeli yükle (generator_final.pt)
    2. Rastgele gürültü üret
    3. Generator'dan geçir
    4. Görüntüleri grid şeklinde göster
    5. PNG olarak kaydet

# Komut satırı argümanları:
--model: Model dosya yolu
--num: Kaç görüntü üretilecek
--latent_dim: Gürültü boyutu
--output: Çıktı dosya adı
```

---

## 🎮 NASIL KULLANILIR?

### Adım 1: Kurulum
```bash
pip install -r requirements.txt
```

### Adım 2: Eğitim
```bash
python train.py
```

**Ne Olur:**
1. MNIST otomatik indirilir → `data/` klasörü
2. 200 epoch eğitim başlar
3. Her 10 epoch: Görüntü kaydedilir → `images/epoch_XXXX.png`
4. Her 50 epoch: Model kaydedilir → `checkpoints/gan_epoch_XX.pt`
5. Bittikten sonra:
   - `checkpoints/generator_final.pt` (final model)
   - `images/loss_plot.png` (grafikler)
   - `images/training_progress.gif` (animasyon)

**Süre:**
- CPU: ~60-90 dakika
- GPU: ~10-15 dakika

### Adım 3: Yeni Görüntü Üret
```bash
python generate.py --num 25
```

**Ne Olur:**
1. `generator_final.pt` yüklenir
2. 25 adet yeni rakam üretilir
3. `images/generated_images.png` kaydedilir

---

## 📊 SONUÇLARI ANLAMA

### Loss Grafikleri

```
Generator Loss (Mavi):
- Düşüyor → Generator iyileşiyor
- Çok düşük → Discriminator çok zayıf (kötü!)
- Optimal: 0.5-1.5 arası

Discriminator Loss (Kırmızı):
- Düşüyor → Discriminator iyileşiyor
- Çok düşük → Generator çok zayıf (kötü!)
- Optimal: 0.5-0.8 arası

İdeal Durum:
İki loss birbirine yakın ve dengeli
```

### Üretilen Görüntüler

```
Epoch 10:  😕 Gürültülü, belirsiz
Epoch 50:  🤔 Rakam şekilleri belli oluyor
Epoch 100: 🙂 Net rakamlar
Epoch 200: 😃 Çok gerçekçi!
```

---

## 🔬 DETAYLI İÇ ÇALIŞMA

### Generator İleri Geçiş Örneği

```python
# Girdi: 100 rastgele sayı
z = [0.5, -0.3, 0.8, ..., 0.2]  # 100 sayı

# Katman 1: 100 → 128
x1 = Linear(z)              # [128]
x1 = BatchNorm(x1)          # Normalize
x1 = LeakyReLU(x1)          # Aktivasyon

# Katman 2: 128 → 256
x2 = Linear(x1)             # [256]
x2 = BatchNorm(x2)
x2 = LeakyReLU(x2)

# Katman 3: 256 → 512
x3 = Linear(x2)             # [512]
x3 = BatchNorm(x3)
x3 = LeakyReLU(x3)

# Katman 4: 512 → 1024
x4 = Linear(x3)             # [1024]
x4 = BatchNorm(x4)
x4 = LeakyReLU(x4)

# Katman 5: 1024 → 784
x5 = Linear(x4)             # [784]
x5 = Tanh(x5)               # [-1, 1] aralığı

# Yeniden şekillendir: 784 → 28x28
output = x5.reshape(1, 28, 28)  # [1, 28, 28]

# Sonuç: Bir rakam görüntüsü!
```

### Discriminator İleri Geçiş Örneği

```python
# Girdi: 28x28 görüntü
img = [28x28 piksel matris]

# Düzleştir: 28x28 → 784
x = img.flatten()           # [784]

# Katman 1: 784 → 512
x1 = Linear(x)              # [512]
x1 = LeakyReLU(x1)
x1 = Dropout(x1, 0.3)       # %30 nöron kapat

# Katman 2: 512 → 256
x2 = Linear(x1)             # [256]
x2 = LeakyReLU(x2)
x2 = Dropout(x2, 0.3)

# Katman 3: 256 → 128
x3 = Linear(x2)             # [128]
x3 = LeakyReLU(x3)
x3 = Dropout(x3, 0.3)

# Katman 4: 128 → 1
x4 = Linear(x3)             # [1]
output = Sigmoid(x4)        # [0-1 arası]

# Sonuç: 0.85 (büyük ihtimalle gerçek!)
#        0.12 (muhtemelen sahte!)
```

### Loss Hesaplama

```python
# Binary Cross Entropy Loss

# Örnek 1: Discriminator gerçek görüntüye bakıyor
prediction = 0.95       # Discriminator "gerçek" diyor
target = 1.0            # Gerçek etiket: gerçek
loss = -log(0.95) = 0.05  # Düşük loss (iyi!)

# Örnek 2: Discriminator sahte görüntüye bakıyor
prediction = 0.85       # Discriminator "gerçek" diyor
target = 0.0            # Gerçek etiket: sahte
loss = -log(1-0.85) = 1.90  # Yüksek loss (kötü!)

# Örnek 3: Generator eğitimi
prediction = 0.30       # Discriminator "sahte" diyor
target = 1.0            # Generator "gerçek" demesini istiyor
loss = -log(0.30) = 1.20  # Yüksek loss (Generator iyileşmeli)
```

---

## 💡 İPUÇLARI

### Eğitim İyileştirme

1. **Epoch sayısını artır**: 200 → 500
2. **Batch size'ı ayarla**: RAM/GPU'ya göre
3. **Learning rate değiştir**: 0.0002 optimal ama deneyebilirsiniz
4. **Farklı veri setleri**: Fashion-MNIST, CIFAR-10

### Sorun Giderme

**Mode Collapse:**
- Generator hep aynı görüntüyü üretiyor
- Çözüm: Learning rate'i düşür

**Training Collapse:**
- Loss'lardan biri çok düşük, diğeri çok yüksek
- Çözüm: Modeli yeniden eğit

**Kötü Görüntüler:**
- Daha fazla epoch eğit
- Hyperparameter'ları ayarla

---

## 🎓 EĞİTİM KAYNKLARI

- **Original GAN Paper**: https://arxiv.org/abs/1406.2661
- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/

---

**Başarılar! 🚀**

