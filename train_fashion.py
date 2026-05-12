"""
👕 FASHION-MNIST İLE GAN EĞİTİMİ

Rakam yerine kıyafet üret! (Tişört, pantolon, ayakkabı, çanta...)

Fashion-MNIST:
- 60,000 eğitim görüntüsü
- 10 kategori: Tişört, pantolon, kazak, elbise, mont, sandalet, 
                gömlek, spor ayakkabı, çanta, bot
- 28x28 piksel, siyah-beyaz

Kullanım:
    python train_fashion.py
    
Sonuç:
    MNIST ile aynı ama kıyafet görüntüleri üretir!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import Generator, Discriminator
from utils import save_generated_images, create_gif, plot_losses
import time

# ============================================================================
# 🎯 AYARLAR
# ============================================================================

config = {
    'n_epochs': 200,           # Epoch sayısı
    'batch_size': 64,          # Batch boyutu
    'lr': 0.0002,              # Öğrenme oranı
    'b1': 0.5,                 # Adam beta1
    'b2': 0.999,               # Adam beta2
    'latent_dim': 100,         # Gürültü vektörü boyutu
    'img_shape': (1, 28, 28),  # Görüntü boyutu (Fashion-MNIST de 28x28)
    'sample_interval': 10      # Kaç epoch'ta bir görüntü kaydet
}

# ============================================================================
# 📦 VERİ YÜKLEME - FASHION-MNIST
# ============================================================================

def get_fashion_dataloader(batch_size):
    """Fashion-MNIST veri setini yükle"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 👕 FASHION-MNIST YÜKLEME
    # MNIST yerine FashionMNIST kullanıyoruz!
    dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,  # İlk seferde indirilir (~30 MB)
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print("\n📦 Fashion-MNIST Veri Seti:")
    print(f"   - Toplam görüntü: {len(dataset)}")
    print(f"   - Kategoriler: Tişört, Pantolon, Kazak, Elbise, Mont,")
    print(f"                 Sandalet, Gömlek, Ayakkabı, Çanta, Bot")
    print(f"   - Boyut: 28x28 piksel (siyah-beyaz)")
    
    return dataloader

# ============================================================================
# 🎓 EĞİTİM
# ============================================================================

def train_fashion_gan():
    """Fashion-MNIST ile GAN eğit"""
    
    print("=" * 70)
    print("  👕 FASHION-MNIST GAN EĞİTİMİ")
    print("=" * 70)
    print("\n🎯 Hedef: Kıyafet görüntüleri üretmek!")
    print("   (Tişört, pantolon, ayakkabı, çanta, vb.)")
    print()
    
    # Cihaz seç
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Kullanılan cihaz: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Modelleri oluştur
    generator = Generator(config['latent_dim'], config['img_shape']).to(device)
    discriminator = Discriminator(config['img_shape']).to(device)
    
    # Loss ve optimizerler
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config['lr'], 
                              betas=(config['b1'], config['b2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], 
                              betas=(config['b1'], config['b2']))
    
    # Veri yükle
    dataloader = get_fashion_dataloader(config['batch_size'])
    
    # Klasörler
    os.makedirs("images_fashion", exist_ok=True)
    os.makedirs("checkpoints_fashion", exist_ok=True)
    
    # Loss kayıtları
    g_losses = []
    d_losses = []
    
    print(f"\n🚀 Eğitim başlıyor... ({config['n_epochs']} epoch)")
    print("-" * 70)
    
    start_time = time.time()
    
    # Eğitim döngüsü
    for epoch in range(config['n_epochs']):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            # Etiketler
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # Gerçek görüntüler (Fashion-MNIST'ten)
            real_imgs = imgs.to(device)
            
            # Generator'ı eğit
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # Discriminator'ı eğit
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Loss kaydet
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
        
        # Epoch ortalaması
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # İlerleme
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{config['n_epochs']}] | "
                  f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | "
                  f"Süre: {elapsed:.1f}s")
        
        # Görüntü kaydet
        if (epoch + 1) % config['sample_interval'] == 0:
            with torch.no_grad():
                z = torch.randn(25, config['latent_dim']).to(device)
                gen_imgs = generator(z).cpu().numpy()
                gen_imgs = (gen_imgs + 1) / 2
                
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(5, 5, figsize=(10, 10))
                fig.suptitle(f'Fashion-MNIST Epoch {epoch+1}', 
                            fontsize=16, fontweight='bold')
                
                for i in range(5):
                    for j in range(5):
                        idx = i * 5 + j
                        axes[i, j].imshow(gen_imgs[idx, 0], cmap='gray')
                        axes[i, j].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'images_fashion/epoch_{epoch+1:04d}.png', 
                           dpi=100, bbox_inches='tight')
                plt.close()
        
        # Checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
            }
            path = f"checkpoints_fashion/gan_epoch_{epoch+1}.pt"
            torch.save(checkpoint, path)
            print(f"   💾 Checkpoint: {path}")
    
    # Eğitim bitti
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ Eğitim tamamlandı! Toplam süre: {total_time/60:.2f} dakika")
    print("=" * 70)
    
    # Final modeller
    torch.save(generator.state_dict(), "checkpoints_fashion/generator_final.pt")
    torch.save(discriminator.state_dict(), "checkpoints_fashion/discriminator_final.pt")
    print("💾 Final modeller: checkpoints_fashion/")
    
    # Loss grafiği
    plot_losses(g_losses, d_losses)
    import shutil
    shutil.move("images/loss_plot.png", "images_fashion/loss_plot.png")
    print("📊 Loss grafikleri: images_fashion/loss_plot.png")
    
    # GIF
    import glob
    from PIL import Image
    image_files = sorted(glob.glob('images_fashion/epoch_*.png'))
    if image_files:
        images = [Image.open(f) for f in image_files]
        images[0].save(
            'images_fashion/training_progress.gif',
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        print("🎬 Animasyon: images_fashion/training_progress.gif")
    
    print("\n" + "=" * 70)
    print("🎉 Fashion-MNIST GAN eğitimi tamamlandı!")
    print("📁 Sonuçlar:")
    print("   - Görüntüler: images_fashion/")
    print("   - Modeller: checkpoints_fashion/")
    print("\n💡 Yeni görüntü üret:")
    print("   python generate_fashion.py")
    print("=" * 70)


if __name__ == "__main__":
    train_fashion_gan()

