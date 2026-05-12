"""
Eğitilmiş GAN modelinden yeni görüntüler üret
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
import argparse
import os

def generate_images(model_path, num_images=25, latent_dim=100, output_file='generated_images.png'):
    """
    Eğitilmiş modelden yeni görüntüler üret
    """
    # Cihaz ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Kullanılan cihaz: {device}")
    
    # Modeli yükle
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        print("\nÖnce modeli eğitmeniz gerekiyor:")
        print("  python train.py")
        return
    
    print(f"📂 Model yükleniyor: {model_path}")
    generator = Generator(latent_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    print(f"🎨 {num_images} adet görüntü üretiliyor...")
    
    with torch.no_grad():
        # Rastgele gürültü üret
        z = torch.randn(num_images, latent_dim).to(device)
        
        # Görüntüleri üret
        gen_imgs = generator(z).cpu().numpy()
        
        # Normalize et: [-1, 1] -> [0, 1]
        gen_imgs = (gen_imgs + 1) / 2
        
        # Grid boyutunu hesapla
        n_row = int(np.sqrt(num_images))
        n_col = (num_images + n_row - 1) // n_row
        
        # Görselleştir
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*2, n_row*2))
        fig.suptitle('GAN ile Üretilmiş Görüntüler', fontsize=16, fontweight='bold')
        
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
        os.makedirs('images', exist_ok=True)
        output_path = os.path.join('images', output_file)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Görüntüler kaydedildi: {output_path}")
        
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAN ile görüntü üret')
    parser.add_argument('--model', type=str, default='checkpoints/generator_final.pt',
                        help='Model dosyası yolu')
    parser.add_argument('--num', type=int, default=25,
                        help='Üretilecek görüntü sayısı')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent vektör boyutu')
    parser.add_argument('--output', type=str, default='generated_images.png',
                        help='Çıktı dosya adı')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🎨 GAN GÖRÜNTÜ ÜRETİCİ")
    print("=" * 60)
    
    generate_images(
        model_path=args.model,
        num_images=args.num,
        latent_dim=args.latent_dim,
        output_file=args.output
    )

