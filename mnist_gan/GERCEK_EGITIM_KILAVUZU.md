# ============================================================================
# 🎯 GERÇEK EĞİTİM VERİLERİNİ KAYDEDEN KILAVUZ
# ============================================================================

## 📍 GÜRÜLTÜ NEREDE ÜRETİLİYOR VE VERİLİYOR?

### 1. GÜRÜLTÜ ÜRETİMİ (train.py - Satır 249)
```python
z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
```
**Bu satırda:**
- `torch.randn()` fonksiyonu ile RASTGELE gürültü üretilir
- Seed YOK! Her batch'te tamamen farklı gürültü
- Boyut: (64, 100) - 64 örnek, her biri 100 sayı

### 2. GENERATOR'A GÜRÜLTÜ VERİLMESİ (train.py - Satır 254)
```python
gen_imgs = self.generator(z)
```
**Bu satırda:**
- Generator'ın `forward()` metodu çağrılır
- GİRDİ: z → (64, 100) gürültü vektörü
- ÇIKTI: gen_imgs → (64, 1, 28, 28) görüntüler

### 3. DISCRIMINATOR'A GÖRÜNTÜ VERİLMESİ

#### 3a. Generator'dan gelen sahte görüntüler (Satır 262)
```python
g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
```
**Bu satırda:**
- Discriminator'ın `forward()` metodu çağrılır
- GİRDİ: gen_imgs → (64, 1, 28, 28) sahte görüntü
- ÇIKTI: skor → (64, 1) - 0 ile 1 arası değerler

#### 3b. MNIST'ten gelen gerçek görüntüler (Satır 285)
```python
real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
```
**Bu satırda:**
- Discriminator'ın `forward()` metodu çağrılır
- GİRDİ: real_imgs → (64, 1, 28, 28) gerçek MNIST görüntüsü
- ÇIKTI: skor → (64, 1) - 0 ile 1 arası değerler

---

## 🔄 TAM AKIŞ DİYAGRAMI

```
GERÇEK EĞİTİM SÜRECİ:
=====================

1. MNIST VERİ SETI
   ↓
   real_imgs (64, 1, 28, 28) ← Gerçek rakam görüntüleri
   
2. GÜRÜLTÜ ÜRETİMİ
   ↓
   z = torch.randn(64, 100) ← ⭐ BURADA ÜRETİLİYOR!
   Seed YOK, her batch farklı!
   
3. GENERATOR
   ↓
   gen_imgs = generator(z) ← ⭐ BURADA VERİLİYOR!
   GİRDİ: (64, 100) gürültü
   ÇIKTI: (64, 1, 28, 28) sahte görüntü
   
4. DISCRIMINATOR - İki Kez Çağrılır:
   
   4a. Gerçek görüntüler için:
       skor_real = discriminator(real_imgs)
       GİRDİ: (64, 1, 28, 28) MNIST
       ÇIKTI: (64, 1) skor - 1'e yakın olmalı
       
   4b. Sahte görüntüler için:
       skor_fake = discriminator(gen_imgs)
       GİRDİ: (64, 1, 28, 28) Generator'dan
       ÇIKTI: (64, 1) skor - 0'a yakın olmalı
       
5. LOSS HESAPLAMA
   ↓
   g_loss: Generator loss (Discriminator'ı kandırabildi mi?)
   d_loss: Discriminator loss (Gerçek ve sahteyi ayırt edebildi mi?)
   
6. BACKPROPAGATION
   ↓
   Ağırlıklar güncellenir, bir sonraki batch'e geç!
```

---

## 📦 GİRDİ VE ÇIKTILAR ÖZET

### GENERATOR:
```
GİRDİ:  z              → (batch_size, 100)      # Rastgele gürültü
İŞLEM:  forward(z)     → Linear katmanlar
ÇIKTI:  gen_imgs       → (batch_size, 1, 28, 28)  # Sahte görüntü
```

### DISCRIMINATOR:
```
GİRDİ:  img            → (batch_size, 1, 28, 28)  # Gerçek veya sahte
İŞLEM:  forward(img)   → Linear katmanlar  
ÇIKTI:  skor           → (batch_size, 1)          # 0-1 arası (0=sahte, 1=gerçek)
```

---

## 💾 GERÇEK EĞİTİM VERİLERİNİ KAYDETMEK İÇİN

train.py dosyasına şu fonksiyonu ekleyin (GANTrainer sınıfı içine):

```python
def save_training_data(self, noise, generated_imgs, epoch, batch, g_loss, d_loss):
    """Gerçek eğitim verilerini kaydet"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Kaydetme klasörü
    save_dir = f'training_data/epoch_{epoch+1}_batch_{batch+1}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Gürültü vektörünü kaydet
    noise_np = noise[:4].cpu().detach().numpy()  # İlk 4 örnek
    np.save(f'{save_dir}/noise.npy', noise_np)
    
    # 2. Üretilen görüntüleri kaydet
    imgs_np = generated_imgs[:4].cpu().detach().numpy()
    imgs_np = (imgs_np + 1) / 2  # [-1, 1] -> [0, 1]
    
    # 3. Görselleştir: Gürültü + Görüntü
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'Epoch {epoch+1}, Batch {batch+1}\\n'
                 f'G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}',
                 fontsize=14, fontweight='bold')
    
    # Üst satır: Gürültü vektörleri
    for i in range(4):
        axes[0, i].plot(noise_np[i], 'b-', linewidth=0.8)
        axes[0, i].set_title(f'Gürültü #{i+1}', fontsize=10)
        axes[0, i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, i].set_ylim(-3, 3)
        axes[0, i].grid(True, alpha=0.3)
    
    # Alt satır: Üretilen görüntüler
    for i in range(4):
        axes[1, i].imshow(imgs_np[i, 0], cmap='gray')
        axes[1, i].set_title(f'Görüntü #{i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/visualization.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 4. İstatistikleri kaydet
    with open(f'{save_dir}/stats.txt', 'w') as f:
        f.write(f"Epoch: {epoch+1}\\n")
        f.write(f"Batch: {batch+1}\\n")
        f.write(f"Generator Loss: {g_loss.item():.6f}\\n")
        f.write(f"Discriminator Loss: {d_loss.item():.6f}\\n\\n")
        f.write(f"Gürültü İstatistikleri:\\n")
        f.write(f"  Mean: {noise_np.mean():.4f}\\n")
        f.write(f"  Std:  {noise_np.std():.4f}\\n")
        f.write(f"  Min:  {noise_np.min():.4f}\\n")
        f.write(f"  Max:  {noise_np.max():.4f}\\n")
```

---

## 🔧 KULLANIM

### train.py'de şu satırı ekleyin (epoch döngüsü içinde):

```python
# Her 50 batch'te bir kaydet
if (i + 1) % 50 == 0:
    self.save_training_data(z, gen_imgs, epoch, i, g_loss, d_loss)
    print(f"💾 Eğitim verileri kaydedildi: training_data/epoch_{epoch+1}_batch_{i+1}/")
```

---

## 📁 OLUŞACAK YAPILAR

```
training_data/
├── epoch_1_batch_50/
│   ├── noise.npy              ← Gerçek gürültü vektörleri
│   ├── visualization.png      ← Gürültü + Görüntü görseli
│   └── stats.txt              ← İstatistikler
├── epoch_1_batch_100/
│   ├── noise.npy
│   ├── visualization.png
│   └── stats.txt
├── epoch_1_batch_150/
...
```

---

## 🎯 SONUÇ

**Gerçek eğitim verileri:**
- ✅ Her batch'te farklı gürültü üretilir (Satır 249)
- ✅ Generator'a gürültü verilir (Satır 254)  
- ✅ Discriminator'a görüntü verilir (Satır 262, 285, 293)
- ✅ Yukarıdaki kodla kaydedilebilir!

**Simülasyon dosyaları:**
- ❌ Silindi (show_training_process.py, show_each_step.py, visualize_progress.py)
- ✅ Sadece gerçek eğitim kaldı!

