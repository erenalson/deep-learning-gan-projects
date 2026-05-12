"""
⚡ CHECKPOINT'TEN HIZLI GÖRÜNTÜ ÜRET

Tek satır ile checkpoint'ten görüntü üretmenin en basit yolu!

Kullanım:
    python checkpoint_hizli_kullan.py
    python checkpoint_hizli_kullan.py --checkpoint checkpoints/gan_epoch_100.pt --num 50
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
import argparse
import os

def hizli_uret(checkpoint_path='checkpoints/gan_epoch_200.pt', num_images=25):
    """
    ⚡ EN BASIT YÖNTEM: Checkpoint'ten hızlıca görüntü üret
    """
    print("=" * 70)
    print("  ⚡ CHECKPOINT'TEN HIZLI GÖRÜNTÜ ÜRETİMİ")
    print("=" * 70)
    
    # Checkpoint dosyası var mı kontrol et
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint bulunamadı: {checkpoint_path}")
        print("\nMevcut checkpoint'ler:")
        
        if os.path.exists('checkpoints'):
            checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
            if checkpoints:
                for cp in sorted(checkpoints):
                    print(f"   📁 checkpoints/{cp}")
            else:
                print("   ⚠️  Hiç checkpoint yok! Önce train.py çalıştırın.")
        return
    
    print(f"\n📂 Yüklenen checkpoint: {checkpoint_path}")
    
    # 1️⃣ CHECKPOINT'İ YÜKLE
    # CPU'da yükle (GPU yoksa sorun olmaz)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📊 Epoch: {checkpoint['epoch']}")
    print(f"📉 Generator Loss: {checkpoint['g_losses'][-1]:.4f}")
    print(f"📉 Discriminator Loss: {checkpoint['d_losses'][-1]:.4f}")
    
    # 2️⃣ GENERATOR'I OLUŞTUR VE AĞIRLIKLARI YÜKLE
    generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()  # Evaluation modu (dropout vs. kapalı)
    
    print(f"\n🎨 {num_images} görüntü üretiliyor...")
    
    # 3️⃣ GÖRÜNTÜ ÜRET
    with torch.no_grad():  # Gradyan hesaplama (eğitim yok)
        # Rastgele gürültü üret
        z = torch.randn(num_images, 100)
        
        # Generator'dan geçir
        gen_imgs = generator(z)
        
        # Numpy'a çevir ve normalize et
        gen_imgs = gen_imgs.cpu().numpy()
        gen_imgs = (gen_imgs + 1) / 2  # [-1, 1] -> [0, 1]
    
    # 4️⃣ GÖRSELLEŞTİR
    n_row = int(np.sqrt(num_images))
    n_col = (num_images + n_row - 1) // n_row
    
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*2, n_row*2))
    fig.suptitle(f'Checkpoint Epoch {checkpoint["epoch"]} - Üretilen Görüntüler', 
                 fontsize=16, fontweight='bold')
    
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_images):
        axes[i].imshow(gen_imgs[i, 0], cmap='gray')
        axes[i].axis('off')
    
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 5️⃣ KAYDET
    os.makedirs('images', exist_ok=True)
    output_path = f'images/checkpoint_epoch_{checkpoint["epoch"]}_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"✅ Görüntüler kaydedildi: {output_path}")
    print("\n" + "=" * 70)
    
    plt.show()


if __name__ == "__main__":
    # Komut satırı argümanları
    parser = argparse.ArgumentParser(description='Checkpoint\'ten hızlıca görüntü üret')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/gan_epoch_200.pt',
                        help='Checkpoint dosya yolu')
    parser.add_argument('--num', type=int, default=25,
                        help='Üretilecek görüntü sayısı')
    
    args = parser.parse_args()
    
    hizli_uret(args.checkpoint, args.num)

