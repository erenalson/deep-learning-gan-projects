"""
👕 FASHION-MNIST GÖRÜNTÜ ÜRETİCİ

Eğitilmiş Fashion-MNIST modelinden yeni kıyafet görüntüleri üret!

Kullanım:
    python generate_fashion.py
    python generate_fashion.py --num 50
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
import argparse
import os

# Fashion-MNIST kategorileri
FASHION_CATEGORIES = {
    0: "Tişört/Üst",
    1: "Pantolon",
    2: "Kazak",
    3: "Elbise",
    4: "Mont",
    5: "Sandalet",
    6: "Gömlek",
    7: "Spor Ayakkabı",
    8: "Çanta",
    9: "Bot"
}

def generate_fashion_images(model_path, num_images=25):
    """Fashion-MNIST modelinden görüntü üret"""
    
    print("=" * 70)
    print("  👕 FASHION-MNIST GÖRÜNTÜ ÜRETİCİ")
    print("=" * 70)
    
    # Model yükle
    if not os.path.exists(model_path):
        print(f"\n❌ Model bulunamadı: {model_path}")
        print("\nÖnce Fashion-MNIST modelini eğitin:")
        print("  python train_fashion.py")
        return
    
    print(f"\n📂 Model yükleniyor: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Kullanılan cihaz: {device}")
    
    # Generator
    generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    print(f"\n🎨 {num_images} kıyafet görüntüsü üretiliyor...")
    print("   (Tişört, pantolon, ayakkabı, çanta, vb.)")
    
    # Görüntü üret
    with torch.no_grad():
        z = torch.randn(num_images, 100).to(device)
        gen_imgs = generator(z).cpu().numpy()
        gen_imgs = (gen_imgs + 1) / 2
    
    # Görselleştir
    n_row = int(np.sqrt(num_images))
    n_col = (num_images + n_row - 1) // n_row
    
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*2.5, n_row*2.5))
    fig.suptitle('Fashion-MNIST ile Üretilmiş Kıyafetler', 
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
    
    # Kaydet
    os.makedirs('images_fashion', exist_ok=True)
    output_path = 'images_fashion/generated_fashion.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Görüntüler kaydedildi: {output_path}")
    
    # Kategori bilgisi
    print("\n📋 Fashion-MNIST Kategorileri:")
    for cat_id, cat_name in FASHION_CATEGORIES.items():
        print(f"   {cat_id}: {cat_name}")
    
    print("\n💡 Not: Model rastgele kıyafet türleri üretir.")
    print("   Belirli bir kategori için Conditional GAN kullanın!")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fashion-MNIST görüntü üret')
    parser.add_argument('--model', type=str, 
                        default='checkpoints_fashion/generator_final.pt',
                        help='Model dosyası')
    parser.add_argument('--num', type=int, default=25,
                        help='Üretilecek görüntü sayısı')
    
    args = parser.parse_args()
    
    generate_fashion_images(args.model, args.num)

