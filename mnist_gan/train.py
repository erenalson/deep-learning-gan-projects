# -*- coding: utf-8 -*-
import sys
import io

# Windows console encoding düzeltmesi
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# 📦 KÜTÜPHANE İÇE AKTARMALARI (IMPORTS)
# ============================================================================

# PyTorch: Derin öğrenme için ana kütüphane
import torch
# Sinir ağı katmanları ve loss fonksiyonları
import torch.nn as nn
# Optimizasyon algoritmaları (Adam, SGD vb.)
import torch.optim as optim
# Veri yükleme için araçlar (batch'lere böler, karıştırır)
from torch.utils.data import DataLoader
# Hazır veri setleri (MNIST) ve görüntü dönüştürme araçları
from torchvision import datasets, transforms
# Dosya ve klasör işlemleri için
import os
# Kendi yazdığımız model dosyasından Generator ve Discriminator'ı içe aktar
from model import Generator, Discriminator
# Yardımcı fonksiyonlar (görüntü kaydetme, grafik çizme, GIF oluşturma)
from utils import save_generated_images, create_gif, plot_losses
# Süre ölçümü için
import time

class GANTrainer:
    """
    🎓 GAN EĞİTİM SINIFI
    
    Bu sınıf GAN'ın tüm eğitim sürecini yönetir.
    Generator ve Discriminator'ı birbirine karşı eğitir.
    
    Nasıl çalışır:
    1. MNIST veri setini yükler (60,000 el yazısı rakam)
    2. Generator'ı başlatır (gürültüden görüntü üretecek)
    3. Discriminator'ı başlatır (gerçek/sahte ayırt edecek)
    4. İki modeli birbirine karşı eğitir (çekişmeli öğrenme)
    5. Periyodik olarak sonuçları kaydeder
    """
    
    def __init__(self, config):
        """
        🔧 BAŞLATMA (Initialization)
        
        GANTrainer'ı kurar ve tüm gerekli bileşenleri hazırlar.
        
        Parametreler:
            config: Eğitim ayarlarını içeren sözlük
                    - n_epochs: Kaç epoch eğitilecek
                    - batch_size: Her adımda kaç görüntü işlenecek
                    - lr: Öğrenme hızı
                    - latent_dim: Gürültü vektörü boyutu
                    vb.
        """
        # Konfigürasyonu sakla
        self.config = config
        
        # 🖥️ CİHAZ SEÇİMİ
        # CUDA (NVIDIA GPU) varsa kullan, yoksa CPU kullan
        # GPU 10-20x daha hızlıdır!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🖥️  Kullanılan cihaz: {self.device}")
        if self.device.type == "cuda":
            # GPU adını göster (örn: RTX 3080)
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # 🎨 MODELLERİ OLUŞTUR
        # Generator: Rastgele gürültüden görüntü üretir
        self.generator = Generator(
            config['latent_dim'],      # 100 boyutlu gürültü vektörü
            config['img_shape']        # (1, 28, 28) çıktı boyutu
        ).to(self.device)              # Modeli GPU/CPU'ya taşı
        
        # Discriminator: Görüntüleri değerlendirir (gerçek/sahte)
        self.discriminator = Discriminator(
            config['img_shape']        # (1, 28, 28) girdi boyutu
        ).to(self.device)              # Modeli GPU/CPU'ya taşı
        
        # 📉 LOSS FONKSİYONU
        # Binary Cross Entropy Loss: İki sınıflı problemler için (gerçek/sahte)
        # Discriminator'ın tahminlerini gerçek etiketlerle karşılaştırır
        self.adversarial_loss = nn.BCELoss()
        
        # 🎯 OPTİMİZERLER (Ağırlık Güncelleyiciler)
        # Adam optimizasyon algoritması: Gradient descent'in gelişmiş versiyonu
        
        # Generator için optimizer
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),    # Generator'ın tüm ağırlıkları
            lr=config['lr'],                # Öğrenme hızı (learning rate)
            betas=(config['b1'], config['b2'])  # Adam'ın momentum parametreleri
        )
        
        # Discriminator için optimizer
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),    # Discriminator'ın tüm ağırlıkları
            lr=config['lr'],                    # Öğrenme hızı
            betas=(config['b1'], config['b2'])  # Adam'ın momentum parametreleri
        )
        
        # 📊 VERİ YÜKLEYİCİ
        # MNIST veri setini yükle ve batch'lere böl
        self.dataloader = self._get_dataloader()
        
        # 📈 KAYITLAR (Eğitim ilerlemesini takip için)
        self.g_losses = []     # Generator loss değerlerini saklar (her epoch için)
        self.d_losses = []     # Discriminator loss değerlerini saklar (her epoch için)
        
        # 📁 KLASÖRLER OLUŞTUR
        os.makedirs("images", exist_ok=True)        # Üretilen görüntüler buraya kaydedilir
        os.makedirs("checkpoints", exist_ok=True)   # Model checkpointleri buraya kaydedilir
        os.makedirs("training_data", exist_ok=True) # ⭐ GERÇEK EĞİTİM VERİLERİ BURAYA!
        os.makedirs("training_data/noises", exist_ok=True)   # Gürültü vektörleri
        os.makedirs("training_data/images", exist_ok=True)   # Üretilen görüntüler
        os.makedirs("training_data/samples", exist_ok=True)  # Örnek görseller
    
    def _get_dataloader(self):
        """
        📦 VERİ SETİNİ YÜKLE
        
        ⭐ ÖNEMLİ: VERİ BURADAN GELİYOR! ⭐
        
        MNIST veri setini otomatik olarak indirir ve hazırlar.
        
        MNIST Nedir?
        - 60,000 eğitim + 10,000 test görüntüsü
        - El yazısı rakamlar (0-9)
        - Her görüntü 28x28 piksel, siyah-beyaz
        - Makine öğrenmede en popüler veri seti
        
        Veri Nereden Gelir?
        - İlk çalıştırmada internet üzerinden indirilir
        - './data' klasörüne kaydedilir
        - Sonraki çalıştırmalarda yerel dosyadan yüklenir"
        
        Döndürür:
            dataloader: Veriyi batch'ler halinde veren PyTorch DataLoader
        """
        # 🔄 VERİ DÖNÜŞTÜRMELERİ
        transform = transforms.Compose([
            # 1. PIL Image veya numpy array'i PyTorch tensor'a çevir
            #    Değer aralığı: [0, 255] -> [0, 1]
            transforms.ToTensor(),
            
            # 2. Normalize et: [0, 1] -> [-1, 1]
            #    Ortalama=0.5, Standart Sapma=0.5
            #    Bu işlem eğitimi stabilize eder
            #    Generator'ın Tanh çıktısıyla uyumludur
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 📥 MNIST VERİ SETİNİ İNDİR VE YÜK
        dataset = datasets.MNIST(
            root='./data',         # Verinin kaydedileceği klasör
            train=True,            # Eğitim setini kullan (60,000 görüntü)
            download=True,         # Yoksa indir, varsa yükle
            transform=transform    # Yukarıdaki dönüşümleri uygula
        )
        
        # 📦 DATALOADER OLUŞTUR
        # DataLoader: Veriyi batch'lere böler, karıştırır ve verir
        dataloader = DataLoader(
            dataset,                                # Yüklediğimiz MNIST veri seti
            batch_size=self.config['batch_size'],   # Her batch'te kaç görüntü (örn: 64)
            shuffle=True,                           # Her epoch'ta veriyi karıştır
            num_workers=0                           # Paralel veri yükleme (Windows için 0)
        )
        
        return dataloader
    
    def train(self):
        """
        🎯 ANA EĞİTİM DÖNGÜSÜ
        
        GAN'ın kalbidir! Burada Generator ve Discriminator birbirine karşı öğrenir.
        
        ⭐ GERÇEK EĞİTİM VERİLERİ KAYDEDILIYOR! ⭐
        
        Eğitim Süreci (Her Batch İçin):
        ================================
        
        1️⃣ DISCRIMINATOR'I EĞİT:
           - Gerçek görüntülere 1 (gerçek) de
           - Sahte görüntülere 0 (sahte) de
           - Amacı: Gerçek ve sahteyi ayırt et
        
        2️⃣ GENERATOR'I EĞİT:
           - Sahte görüntüler üret
           - Discriminator'ı kandırmaya çalış (1 olarak etiketle)
           - Amacı: Discriminator'ı aldatacak kadar gerçekçi görüntü üret
        
        Bu iki ağ birbirine karşı "savaşarak" öğrenir!
        - Discriminator iyileşir -> Daha iyi ayırt eder
        - Generator iyileşir -> Daha gerçekçi görüntü üretir
        """
        print(f"\n🚀 Eğitim başlıyor...")
        print(f"   Epoch sayısı: {self.config['n_epochs']}")
        print(f"   Batch boyutu: {self.config['batch_size']}")
        print(f"   Latent boyutu: {self.config['latent_dim']}")
        print(f"\n⭐ GERÇEK EĞİTİM VERİLERİ KAYDEDILECEK!")
        print(f"   - Her 50 batch'te bir gürültü ve görüntü kaydedilecek")
        print(f"   - Klasör: training_data/")
        print("-" * 60)
        
        # ⏱️ Süre ölçümü başlat
        start_time = time.time()
        
        # 📊 İstatistikler
        total_batches = 0
        saved_samples = 0
        
        # 🔄 EPOCH DÖNGÜSÜ
        # Her epoch: Tüm veri setini bir kez görmek
        for epoch in range(self.config['n_epochs']):
            # Bu epoch için toplam loss değerlerini sakla
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            # 📦 BATCH DÖNGÜSÜ
            # Her batch: 64 (veya batch_size kadar) görüntü
            # enumerate(): Hem index (i) hem de veriyi (imgs, labels) verir
            # _ : Etiketleri (0-9 rakamları) kullanmıyoruz, görmezden gel
            for i, (imgs, _) in enumerate(self.dataloader):
                # Bu batch'teki görüntü sayısı (genelde 64, son batch daha az olabilir)
                batch_size = imgs.shape[0]
                
                # 🏷️ ETİKETLER HAZIRLA
                # Gerçek görüntüler için etiket: 1 (ones)
                # Boyut: (batch_size, 1) örn: [[1], [1], [1], ...]
                valid = torch.ones(batch_size, 1).to(self.device)
                
                # Sahte görüntüler için etiket: 0 (zeros)
                # Boyut: (batch_size, 1) örn: [[0], [0], [0], ...]
                fake = torch.zeros(batch_size, 1).to(self.device)
                
                # Gerçek görüntüleri GPU/CPU'ya taşı
                real_imgs = imgs.to(self.device)                   # GERÇEK GÖRÜNTÜLER
                
                # ============================================================
                # 1️⃣ GENERATOR'I EĞİT
                # ============================================================
                # Amaç: Discriminator'ı kandıracak kadar gerçekçi görüntü üret
                
                # Gradyanları sıfırla (her eğitim adımı öncesi yapılmalı)
                self.optimizer_G.zero_grad()
                
                # 🎲 RASTGELE GÜRÜLTÜ ÜRET
                # Normal dağılımdan (ortalama=0, std=1) rastgele sayılar
                # Boyut: (batch_size, 100) örn: 64 adet 100 boyutlu vektör
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)              # !!! GÜRÜLTÜ ÜRETİR !!!!!!!
                
                # 🎨 SAHTE GÖRÜNTÜLER ÜRET
                # Generator gürültüyü görüntüye çevirir
                # Boyut: (batch_size, 1, 28, 28)
                gen_imgs = self.generator(z)                                 # GÜRÜLTÜYÜ GÖRÜNTÜYE ÇEVİRİR
                
                # 📉 GENERATOR LOSS HESAPLA
                # Generator'ın amacı: Discriminator'ı kandırmak
                # Sahte görüntüleri Discriminator'a göster
                # Ama Generator bunların "gerçek" (valid) olarak etiketlenmesini istiyor!
                # Eğer Discriminator 1 derse -> loss düşük (başarılı!)
                # Eğer Discriminator 0 derse -> loss yüksek (başarısız!)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)           # !!! DISCRIMINATOR'A SAHTE GÖRÜNTÜYÜ GÖSTERİR
                
                # ⬅️ GERIYE YAYILIM (Backpropagation)
                # Loss'tan gradyanları hesapla
                g_loss.backward()
                
                # 📈 AĞIRLIKLARI GÜNCELLE
                # Adam optimizer kullanarak Generator'ın ağırlıklarını güncelle
                self.optimizer_G.step()
                
                # ============================================================
                # 2️⃣ DISCRIMINATOR'I EĞİT
                # ============================================================
                # Amaç: Gerçek ve sahte görüntüleri doğru şekilde ayırt et
                
                # Gradyanları sıfırla
                self.optimizer_D.zero_grad()
                
                # 📉 GERÇEK GÖRÜNTÜLER İÇİN LOSS
                # Gerçek görüntüleri Discriminator'a göster
                # Beklenen çıktı: 1 (gerçek)
                # Eğer Discriminator 1 derse -> loss düşük (başarılı!)
                # Eğer Discriminator 0 derse -> loss yüksek (başarısız!)
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)      # !!! DISCRIMINATOR'A GERÇEK GÖRÜNTÜYÜ GÖSTERİR
                
                # 📉 SAHTE GÖRÜNTÜLER İÇİN LOSS
                # Sahte görüntüleri Discriminator'a göster
                # .detach(): Generator'ın gradyanlarını hesaplama (sadece Discriminator eğitiliyor)
                # Beklenen çıktı: 0 (sahte)
                # Eğer Discriminator 0 derse -> loss düşük (başarılı!)
                # Eğer Discriminator 1 derse -> loss yüksek (başarısız!)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                
                # 📊 TOPLAM DISCRIMINATOR LOSS
                # İki loss'un ortalaması: Hem gerçeği tanımalı hem sahteyi
                d_loss = (real_loss + fake_loss) / 2
                
                # ⬅️ GERIYE YAYILIM
                d_loss.backward()
                
                # 📈 AĞIRLIKLARI GÜNCELLE
                self.optimizer_D.step()
                
                # 📝 LOSS'LARI KAYDET
                # .item(): Tensor'dan Python sayısına çevir
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                # ============================================================
                # 💾 GERÇEK EĞİTİM VERİLERİNİ KAYDET
                # ============================================================
                # Her 50 batch'te bir gerçek eğitim verilerini kaydet
                if (i + 1) % 50 == 0:
                    # Discriminator skorlarını al
                    with torch.no_grad():
                        real_output = self.discriminator(real_imgs)
                        fake_output = self.discriminator(gen_imgs)
                    
                    self.save_training_data(
                        z, gen_imgs, real_imgs, 
                        epoch, i, g_loss, d_loss,
                        real_output, fake_output
                    )
            
            # ============================================================
            # 📊 EPOCH İSTATİSTİKLERİ
            # ============================================================
            
            # Bu epoch'taki ortalama loss'ları hesapla
            # Toplam loss / batch sayısı
            avg_g_loss = epoch_g_loss / len(self.dataloader)
            avg_d_loss = epoch_d_loss / len(self.dataloader)
            
            # Loss'ları listeye ekle (grafik için)
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            # 🖨️ İLERLEME GÖSTER
            # Her 5 epoch'ta bir ekrana yazdır
            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{self.config['n_epochs']}] | "
                      f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | "
                      f"Süre: {elapsed:.1f}s")
            
            # 🖼️ GÖRÜNTÜ KAYDET
            # Her 10 epoch'ta bir (sample_interval) örnek görüntüler üret ve kaydet
            # Bu sayede eğitim ilerlemesini görebiliriz
            if (epoch + 1) % self.config['sample_interval'] == 0:
                save_generated_images(
                    self.generator,                 # Üretici model
                    self.config['latent_dim'],      # Gürültü boyutu
                    epoch + 1,                      # Epoch numarası (dosya adı için)
                    self.device                     # GPU/CPU
                )
            
            # 💾 MODEL CHECKPOINT KAYDET
            # Her 50 epoch'ta bir modeli kaydet (eğitim yarıda kalırsa devam edebilmek için)
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch + 1)
        
        # ============================================================
        # ✅ EĞİTİM TAMAMLANDI
        # ============================================================
        
        # Toplam süreyi hesapla ve göster
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✅ Eğitim tamamlandı! Toplam süre: {total_time/60:.2f} dakika")
        print("=" * 60)
        
        # Son sonuçları kaydet (final modeller, grafikler, GIF)
        self.save_final_results()
    
    
    def save_training_data(self, noise, generated_imgs, real_imgs, epoch, batch, 
                          g_loss, d_loss, real_output, fake_output):
        """
        💾 GERÇEK EĞİTİM VERİLERİNİ KAYDET
        
        Bu fonksiyon eğitim sırasında kullanılan gerçek verileri kaydeder:
        - Gürültü vektörleri
        - Üretilen görüntüler
        - MNIST görüntüleri
        - Loss değerleri
        - Discriminator skorları
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Kaydetme klasörü
        save_dir = f'training_data/epoch_{epoch+1:03d}_batch_{batch+1:04d}'
        os.makedirs(save_dir, exist_ok=True)
        
        # İlk 8 örneği al
        noise_np = noise[:8].cpu().detach().numpy()
        gen_imgs_np = generated_imgs[:8].cpu().detach().numpy()
        real_imgs_np = real_imgs[:8].cpu().detach().numpy()
        
        # Normalize: [-1, 1] -> [0, 1]
        gen_imgs_np = (gen_imgs_np + 1) / 2
        real_imgs_np = (real_imgs_np + 1) / 2
        
        # 1. Gürültü vektörünü kaydet
        np.save(f'{save_dir}/noise.npy', noise_np)
        
        # 2. Detaylı Görselleştirme
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'GERÇEK EĞİTİM VERİLERİ\\n'
                     f'Epoch {epoch+1}/{self.config["n_epochs"]}, '
                     f'Batch {batch+1}/{len(self.dataloader)}\\n'
                     f'G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}',
                     fontsize=16, fontweight='bold')
        
        # Üst kısım: Gürültü vektörleri
        for i in range(8):
            ax = plt.subplot(4, 8, i + 1)
            ax.plot(noise_np[i], 'b-', linewidth=0.6)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title(f'Gürültü #{i+1}', fontsize=9)
            ax.set_ylim(-3, 3)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([-2, 0, 2])
        
        # İkinci kısım: MNIST (Gerçek görüntüler)
        for i in range(8):
            ax = plt.subplot(4, 8, i + 9)
            ax.imshow(real_imgs_np[i, 0], cmap='gray')
            ax.set_title(f'MNIST #{i+1}\\n'
                        f'Score: {real_output[i].item():.2f}',
                        fontsize=9, color='green')
            ax.axis('off')
        
        # Üçüncü kısım: Generator çıktıları
        for i in range(8):
            ax = plt.subplot(4, 8, i + 17)
            ax.imshow(gen_imgs_np[i, 0], cmap='gray')
            ax.set_title(f'Generator #{i+1}\\n'
                        f'Score: {fake_output[i].item():.2f}',
                        fontsize=9, color='blue')
            ax.axis('off')
        
        # Alt kısım: İstatistikler
        ax = plt.subplot(4, 1, 4)
        ax.axis('off')
        
        stats_text = (
            f"═══ GİRDİ VE ÇIKTI BİLGİLERİ ═══\\n\\n"
            f"1️⃣ GÜRÜLTÜ ÜRETİMİ (Satır 249: z = torch.randn(...))\\n"
            f"   • Boyut: {noise.shape}\\n"
            f"   • Mean: {noise_np.mean():.4f}, Std: {noise_np.std():.4f}\\n"
            f"   • Min: {noise_np.min():.4f}, Max: {noise_np.max():.4f}\\n\\n"
            
            f"2️⃣ GENERATOR (Satır 254: gen_imgs = generator(z))\\n"
            f"   • GİRDİ: {noise.shape} (gürültü)\\n"
            f"   • ÇIKTI: {generated_imgs.shape} (sahte görüntü)\\n\\n"
            
            f"3️⃣ DISCRIMINATOR - Gerçek görüntüler (Satır 285)\\n"
            f"   • GİRDİ: {real_imgs.shape} (MNIST)\\n"
            f"   • ÇIKTI: {real_output.shape} (skor)\\n"
            f"   • Ortalama skor: {real_output.mean().item():.4f} (1'e yakın olmalı)\\n\\n"
            
            f"4️⃣ DISCRIMINATOR - Sahte görüntüler (Satır 293)\\n"
            f"   • GİRDİ: {generated_imgs.shape} (Generator'dan)\\n"
            f"   • ÇIKTI: {fake_output.shape} (skor)\\n"
            f"   • Ortalama skor: {fake_output.mean().item():.4f} (0'a yakın olmalı)"
        )
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=10, va='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/full_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Basit Görselleştirme (Gürültü -> Görüntü)
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle(f'Epoch {epoch+1}, Batch {batch+1}: Gürültü -> Görüntü',
                     fontsize=14, fontweight='bold')
        
        for i in range(8):
            # Gürültü
            axes[0, i].plot(noise_np[i], 'b-', linewidth=0.8)
            axes[0, i].set_title(f'#{i+1}', fontsize=9)
            axes[0, i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, i].set_ylim(-3, 3)
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_xticks([])
            
            # Görüntü
            axes[1, i].imshow(gen_imgs_np[i, 0], cmap='gray')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/simple_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. İstatistikleri kaydet
        with open(f'{save_dir}/stats.txt', 'w', encoding='utf-8') as f:
            f.write("═"*70 + "\\n")
            f.write("GERÇEK EĞİTİM VERİLERİ\\n")
            f.write("═"*70 + "\\n\\n")
            
            f.write(f"Epoch: {epoch+1}/{self.config['n_epochs']}\\n")
            f.write(f"Batch: {batch+1}/{len(self.dataloader)}\\n\\n")
            
            f.write("─"*70 + "\\n")
            f.write("LOSS DEĞERLERİ\\n")
            f.write("─"*70 + "\\n")
            f.write(f"Generator Loss:     {g_loss.item():.6f}\\n")
            f.write(f"Discriminator Loss: {d_loss.item():.6f}\\n\\n")
            
            f.write("─"*70 + "\\n")
            f.write("GÜRÜLTÜ İSTATİSTİKLERİ (Satır 249: z = torch.randn(...))\\n")
            f.write("─"*70 + "\\n")
            f.write(f"Boyut: {noise.shape}\\n")
            f.write(f"Mean:  {noise_np.mean():.6f}\\n")
            f.write(f"Std:   {noise_np.std():.6f}\\n")
            f.write(f"Min:   {noise_np.min():.6f}\\n")
            f.write(f"Max:   {noise_np.max():.6f}\\n")
            f.write(f"İlk 10 değer: {noise_np[0, :10].tolist()}\\n\\n")
            
            f.write("─"*70 + "\\n")
            f.write("DISCRIMINATOR SKORLARI\\n")
            f.write("─"*70 + "\\n")
            f.write(f"Gerçek görüntüler (MNIST): {real_output.mean().item():.4f} (1'e yakın olmalı)\\n")
            f.write(f"Sahte görüntüler (Generator): {fake_output.mean().item():.4f} (0'a yakın olmalı)\\n\\n")
            
            f.write("─"*70 + "\\n")
            f.write("GİRDİ VE ÇIKTILAR\\n")
            f.write("─"*70 + "\\n\\n")
            
            f.write("1. GÜRÜLTÜ ÜRETİMİ (train.py - Satır 249)\\n")
            f.write(f"   z = torch.randn({list(noise.shape)})\\n")
            f.write(f"   Seed YOK! Her batch farklı gürültü\\n\\n")
            
            f.write("2. GENERATOR (train.py - Satır 254)\\n")
            f.write(f"   GİRDİ:  z shape = {list(noise.shape)}\\n")
            f.write(f"   ÇIKTI:  gen_imgs shape = {list(generated_imgs.shape)}\\n\\n")
            
            f.write("3. DISCRIMINATOR - Gerçek (train.py - Satır 285)\\n")
            f.write(f"   GİRDİ:  real_imgs shape = {list(real_imgs.shape)}\\n")
            f.write(f"   ÇIKTI:  skor shape = {list(real_output.shape)}\\n\\n")
            
            f.write("4. DISCRIMINATOR - Sahte (train.py - Satır 293)\\n")
            f.write(f"   GİRDİ:  gen_imgs shape = {list(generated_imgs.shape)}\\n")
            f.write(f"   ÇIKTI:  skor shape = {list(fake_output.shape)}\\n\\n")
        
        print(f"\\n{'='*70}")
        print(f"💾 GERÇEK EĞİTİM VERİLERİ KAYDEDİLDİ!")
        print(f"   Klasör: {save_dir}/")
        print(f"   - full_visualization.png (detaylı)")
        print(f"   - simple_visualization.png (basit)")
        print(f"   - noise.npy (gürültü vektörleri)")
        print(f"   - stats.txt (istatistikler)")
        print(f"{'='*70}\\n")
    
    def save_checkpoint(self, epoch):
        """
        💾 MODEL CHECKPOINT KAYDET
        
        Eğitimi daha sonra devam ettirebilmek için model durumunu kaydeder.
        
        Checkpoint İçeriği:
        - Epoch numarası
        - Generator ağırlıkları
        - Discriminator ağırlıkları
        - Optimizer durumları
        - Loss geçmişi
        
        Kullanımı:
        Eğitim yarıda kalırsa bu checkpoint'ten devam edilebilir.
        """
        # Tüm önemli bilgileri bir sözlükte topla
        checkpoint = {
            'epoch': epoch,                                            # Kaçıncı epoch
            'generator_state_dict': self.generator.state_dict(),       # Generator ağırlıkları
            'discriminator_state_dict': self.discriminator.state_dict(),  # Discriminator ağırlıkları
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),   # Generator optimizer durumu
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),   # Discriminator optimizer durumu
            'g_losses': self.g_losses,                                 # Generator loss geçmişi
            'd_losses': self.d_losses,                                 # Discriminator loss geçmişi
        }
        
        # Dosya yolu oluştur: checkpoints/gan_epoch_50.pt
        path = f"checkpoints/gan_epoch_{epoch}.pt"
        # PyTorch formatında (.pt) kaydet
        torch.save(checkpoint, path)
        print(f"   💾 Checkpoint kaydedildi: {path}")
    
    def save_final_results(self):
        """
        💾 SON SONUÇLARI KAYDET
        
        Eğitim tamamlandığında çağrılır.
        Final modelleri, grafikleri ve animasyonu kaydeder.
        
        Kaydedilenler:
        1. Final Generator modeli (görüntü üretmek için kullanılır)
        2. Final Discriminator modeli (opsiyonel, genelde gerek yok)
        3. Loss grafikleri (eğitim ilerlemesini gösterir)
        4. GIF animasyon (eğitim boyunca üretilen görüntüler)
        """
        # 1️⃣ SON MODELLERİ KAYDET
        # Sadece ağırlıkları kaydet (state_dict)
        # Checkpoint'ten farklı: Sadece model ağırlıkları, optimizer yok
        torch.save(self.generator.state_dict(), "checkpoints/generator_final.pt")
        torch.save(self.discriminator.state_dict(), "checkpoints/discriminator_final.pt")
        print("💾 Final modeller kaydedildi: checkpoints/")
        
        # 2️⃣ LOSS GRAFİKLERİNİ KAYDET
        # Generator ve Discriminator loss'larının epoch'lara göre grafiği
        # Bu grafik eğitimin ne kadar iyi gittiğini gösterir
        plot_losses(self.g_losses, self.d_losses)
        print("📊 Loss grafikleri kaydedildi: images/loss_plot.png")
        
        # 3️⃣ GIF OLUŞTUR
        # Eğitim boyunca kaydedilen tüm görüntülerden animasyon oluştur
        # Epoch 10, 20, 30... görüntülerini GIF'e çevir
        # Bu animasyon Generator'ın nasıl geliştiğini gösterir
        create_gif()
        print("🎬 Eğitim animasyonu oluşturuldu: images/training_progress.gif")


# ============================================================================
# 🚀 ANA PROGRAM (Main)
# ============================================================================

if __name__ == "__main__":
    """
    Bu kısım "python train.py" komutu çalıştırıldığında çalışır.
    
    Adımlar:
    1. Eğitim parametrelerini ayarla (config)
    2. GANTrainer oluştur
    3. Eğitimi başlat
    4. Sonuçları otomatik kaydet
    """
    
    # ⚙️ EĞİTİM KONFİGÜRASYONU
    # Bu parametreleri değiştirerek eğitimi özelleştirebilirsiniz
    config = {
        'n_epochs': 200,           # Epoch sayısı (200 önerilen, daha fazla = daha iyi kalite)
        'batch_size': 64,          # Batch boyutu (RAM/GPU yetersizse 32 yapın)
        'lr': 0.0002,              # Öğrenme oranı (learning rate) - 0.0002 GAN için optimal
        'b1': 0.5,                 # Adam optimizer beta1 parametresi (momentum)
        'b2': 0.999,               # Adam optimizer beta2 parametresi (momentum)
        'latent_dim': 100,         # Gürültü vektörü boyutu (100 standart)
        'img_shape': (1, 28, 28),  # Görüntü boyutu: 1 kanal (siyah-beyaz), 28x28 piksel
        'sample_interval': 10      # Kaç epoch'ta bir görüntü kaydet (10 = her 10 epoch)
    }
    
    print("=" * 60)
    print("  [*] GAN GORUNTU URETIMI EGITIMI")
    print("=" * 60)
    print("\n[*] Egitim Bilgileri:")
    print(f"   - MNIST veri seti otomatik indirilecek")
    print(f"   - {config['n_epochs']} epoch boyunca egitim yapilacak")
    print(f"   - Her {config['sample_interval']} epoch'ta ornek goruntu kaydedilecek")
    print(f"   - Sonuclar: images/ ve checkpoints/ klasorlerine kaydedilecek")
    print(f"   - GERCEK EGITIM VERILERI: training_data/ klasorune kaydedilecek (Her 50 batch)")
    print()
    
    # 🎓 TRAINER OLUŞTUR VE EĞİT
    trainer = GANTrainer(config)  # Trainer'ı başlat (modeller burada oluşturulur)
    trainer.train()                # Eğitimi başlat (uzun sürebilir!)

