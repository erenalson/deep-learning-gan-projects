"""
💾 CHECKPOINT DOSYALARINI KULLANMA KILAVUZU

.pt dosyaları PyTorch model checkpoint'leridir.
Bu script ile checkpoint'leri yükleyebilir ve kullanabilirsiniz.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Generator, Discriminator
import os

def checkpoint_bilgisi_goster(checkpoint_path):
    """
    📊 CHECKPOINT BİLGİLERİNİ GÖSTER
    
    Checkpoint dosyasının içeriğini inceler ve bilgileri gösterir.
    
    Kullanım:
        python checkpoint_kullan.py
        veya
        checkpoint_bilgisi_goster('checkpoints/gan_epoch_200.pt')
    """
    print("=" * 70)
    print(f"📂 Checkpoint Dosyası: {checkpoint_path}")
    print("=" * 70)
    
    # Checkpoint'i yükle
    if not os.path.exists(checkpoint_path):
        print(f"❌ Dosya bulunamadı: {checkpoint_path}")
        return
    
    # CPU'da yükle (GPU yoksa da çalışır)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n📋 İçerik:")
    print("-" * 70)
    
    # Checkpoint içeriğini göster
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            if key == 'epoch':
                print(f"   🔢 Epoch: {checkpoint[key]}")
            elif key == 'g_losses':
                print(f"   📉 Generator Loss Geçmişi: {len(checkpoint[key])} epoch")
                print(f"      → Son loss: {checkpoint[key][-1]:.4f}")
            elif key == 'd_losses':
                print(f"   📉 Discriminator Loss Geçmişi: {len(checkpoint[key])} epoch")
                print(f"      → Son loss: {checkpoint[key][-1]:.4f}")
            elif 'state_dict' in key:
                print(f"   🧠 {key}: Model ağırlıkları mevcut")
            else:
                print(f"   📦 {key}: {type(checkpoint[key])}")
    
    print("\n" + "=" * 70)
    print("✅ Checkpoint başarıyla yüklendi ve incelendi!")
    print("=" * 70)
    
    return checkpoint


def checkpoint_ten_goruntu_uret(checkpoint_path, num_images=25, save_path='images/checkpoint_output.png'):
    """
    🎨 CHECKPOINT'TEN GÖRÜNTÜ ÜRET
    
    Kaydedilmiş checkpoint'ten Generator'ı yükler ve yeni görüntüler üretir.
    
    Parametreler:
        checkpoint_path: Checkpoint dosya yolu (örn: 'checkpoints/gan_epoch_200.pt')
        num_images: Üretilecek görüntü sayısı (default: 25)
        save_path: Çıktı dosya yolu
    
    Kullanım:
        checkpoint_ten_goruntu_uret('checkpoints/gan_epoch_200.pt', num_images=25)
    """
    print(f"\n🎨 Checkpoint'ten {num_images} görüntü üretiliyor...")
    print(f"📂 Checkpoint: {checkpoint_path}")
    
    # Checkpoint'i yükle
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Generator'ı oluştur
    generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
    
    # Ağırlıkları yükle
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()  # Evaluation moduna geç
    
    # Görüntü üret
    with torch.no_grad():
        # Rastgele gürültü
        z = torch.randn(num_images, 100)
        
        # Görüntüleri üret
        gen_imgs = generator(z).cpu().numpy()
        
        # Normalize: [-1, 1] -> [0, 1]
        gen_imgs = (gen_imgs + 1) / 2
        
        # Grid boyutu
        n_row = int(np.sqrt(num_images))
        n_col = (num_images + n_row - 1) // n_row
        
        # Görselleştir
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*2, n_row*2))
        fig.suptitle(f'Checkpoint Epoch {checkpoint["epoch"]} - Üretilen Görüntüler', 
                     fontsize=16, fontweight='bold')
        
        if num_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i in range(num_images):
            axes[i].imshow(gen_imgs[i, 0], cmap='gray')
            axes[i].axis('off')
        
        # Boş subplotları gizle
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Kaydet
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Görüntüler kaydedildi: {save_path}")
        
        plt.show()


def checkpoint_loss_grafigi(checkpoint_path):
    """
    📊 CHECKPOINT'TEKİ LOSS GRAFİĞİNİ ÇİZ
    
    Checkpoint içindeki loss geçmişini görselleştirir.
    """
    print(f"\n📊 Loss grafikleri çiziliyor...")
    
    # Checkpoint'i yükle
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    g_losses = checkpoint['g_losses']
    d_losses = checkpoint['d_losses']
    
    plt.figure(figsize=(14, 5))
    
    # Generator Loss
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Generator Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Discriminator Loss
    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label='Discriminator Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Discriminator Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    output_path = 'images/checkpoint_loss_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Loss grafikleri kaydedildi: {output_path}")
    
    plt.show()


def checkpointleri_karsilastir(checkpoint_paths):
    """
    🔍 FARKLI CHECKPOINT'LERİ KARŞILAŞTIR
    
    Birden fazla checkpoint'i karşılaştırır ve görüntüler üretir.
    
    Kullanım:
        checkpointleri_karsilastir([
            'checkpoints/gan_epoch_50.pt',
            'checkpoints/gan_epoch_100.pt',
            'checkpoints/gan_epoch_200.pt'
        ])
    """
    print(f"\n🔍 {len(checkpoint_paths)} checkpoint karşılaştırılıyor...")
    
    # Aynı gürültü vektörünü kullan (adil karşılaştırma için)
    z = torch.randn(16, 100)
    
    fig, axes = plt.subplots(len(checkpoint_paths), 4, figsize=(12, len(checkpoint_paths)*3))
    
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Atlanıyor (dosya yok): {checkpoint_path}")
            continue
        
        # Checkpoint'i yükle
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint['epoch']
        
        # Generator'ı yükle
        generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        # Görüntü üret
        with torch.no_grad():
            gen_imgs = generator(z[:4]).cpu().numpy()
            gen_imgs = (gen_imgs + 1) / 2
        
        # Görselleştir
        for i in range(4):
            ax = axes[idx, i] if len(checkpoint_paths) > 1 else axes[i]
            ax.imshow(gen_imgs[i, 0], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Checkpoint Karşılaştırması', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'images/checkpoint_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Karşılaştırma kaydedildi: {output_path}")
    
    plt.show()


def egitim_devam_et(checkpoint_path, ek_epoch=50):
    """
    🔄 EĞİTİME DEVAM ET
    
    Checkpoint'ten eğitime devam eder.
    
    UYARI: Bu fonksiyon örnek amaçlıdır. 
    Gerçek eğitim için train.py'ı checkpoint ile başlatmak daha iyidir.
    """
    print(f"\n🔄 Checkpoint'ten eğitime devam ediliyor...")
    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"➕ Ek epoch: {ek_epoch}")
    print("\n⚠️  NOT: Bu özellik için train.py dosyasını checkpoint yükleme ile")
    print("   özelleştirmeniz gerekir. Şu an sadece bilgi amaçlıdır.")
    
    # Checkpoint'i yükle
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n📊 Mevcut Durum:")
    print(f"   - Mevcut Epoch: {checkpoint['epoch']}")
    print(f"   - Son Generator Loss: {checkpoint['g_losses'][-1]:.4f}")
    print(f"   - Son Discriminator Loss: {checkpoint['d_losses'][-1]:.4f}")
    print(f"\n🎯 Hedef:")
    print(f"   - Yeni Epoch: {checkpoint['epoch'] + ek_epoch}")


# ============================================================================
# 🚀 ANA PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  💾 GAN CHECKPOINT KULLANIM ARACI")
    print("=" * 70)
    
    # En son checkpoint'i bul
    checkpoint_dir = 'checkpoints'
    
    if not os.path.exists(checkpoint_dir):
        print(f"\n❌ Checkpoint klasörü bulunamadı: {checkpoint_dir}")
        print("   Önce train.py ile model eğitmeniz gerekiyor.")
        exit(1)
    
    # Tüm checkpoint'leri listele
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    
    if not checkpoints:
        print(f"\n❌ Checkpoint dosyası bulunamadı!")
        print("   Önce train.py ile model eğitmeniz gerekiyor.")
        exit(1)
    
    print(f"\n📁 Bulunan Checkpoint'ler:")
    print("-" * 70)
    for i, cp in enumerate(checkpoints, 1):
        print(f"   {i}. {cp}")
    print("-" * 70)
    
    # En son checkpoint'i kullan
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    print(f"\n🎯 Kullanılacak checkpoint: {checkpoints[-1]}")
    print("\n" + "=" * 70)
    print("MENÜ - Ne yapmak istersiniz?")
    print("=" * 70)
    print("1. 📊 Checkpoint bilgilerini göster")
    print("2. 🎨 Checkpoint'ten görüntü üret (25 adet)")
    print("3. 📈 Loss grafiklerini çiz")
    print("4. 🔍 Checkpoint'leri karşılaştır")
    print("5. 🎯 Tümünü yap (1+2+3)")
    print("-" * 70)
    
    try:
        secim = input("\nSeçiminiz (1-5): ").strip()
        
        if secim == '1':
            checkpoint_bilgisi_goster(latest_checkpoint)
        
        elif secim == '2':
            checkpoint_ten_goruntu_uret(latest_checkpoint, num_images=25)
        
        elif secim == '3':
            checkpoint_loss_grafigi(latest_checkpoint)
        
        elif secim == '4':
            # Tüm epoch checkpoint'lerini karşılaştır
            epoch_checkpoints = [os.path.join(checkpoint_dir, cp) 
                                for cp in checkpoints if 'epoch' in cp]
            if len(epoch_checkpoints) >= 2:
                # En fazla 4 checkpoint karşılaştır
                checkpointleri_karsilastir(epoch_checkpoints[-4:])
            else:
                print("⚠️  Karşılaştırma için en az 2 epoch checkpoint gerekli!")
        
        elif secim == '5':
            print("\n🎯 Tüm işlemler yapılıyor...\n")
            checkpoint_bilgisi_goster(latest_checkpoint)
            checkpoint_ten_goruntu_uret(latest_checkpoint, num_images=25)
            checkpoint_loss_grafigi(latest_checkpoint)
        
        else:
            print("❌ Geçersiz seçim!")
    
    except KeyboardInterrupt:
        print("\n\n👋 İşlem iptal edildi.")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
    
    print("\n" + "=" * 70)
    print("✅ İşlem tamamlandı!")
    print("=" * 70)

