import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob

def save_generated_images(generator, latent_dim, epoch, device, n_row=5):
    """
    Generator'dan örnek görüntüler üret ve kaydet
    """
    generator.eval()
    with torch.no_grad():
        # Rastgele gürültü üret
        z = torch.randn(n_row * n_row, latent_dim).to(device)
        
        # Görüntü üret
        gen_imgs = generator(z)
        
        # CPU'ya taşı ve numpy'a çevir
        gen_imgs = gen_imgs.cpu().numpy()
        
        # [-1, 1] aralığından [0, 1] aralığına dönüştür
        gen_imgs = (gen_imgs + 1) / 2
        
        # Grid oluştur
        fig, axes = plt.subplots(n_row, n_row, figsize=(10, 10))
        fig.suptitle(f'Epoch {epoch}', fontsize=16, fontweight='bold')
        
        for i in range(n_row):
            for j in range(n_row):
                idx = i * n_row + j
                axes[i, j].imshow(gen_imgs[idx, 0], cmap='gray')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'images/epoch_{epoch:04d}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    generator.train()


def plot_losses(g_losses, d_losses):
    """
    Eğitim sırasında loss değerlerini görselleştir
    """
    plt.figure(figsize=(12, 5))
    
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
    plt.savefig('images/loss_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_gif(duration=100):
    """
    Eğitim sırasında kaydedilen görüntülerden animasyon oluştur
    """
    # Tüm epoch görüntülerini bul
    image_files = sorted(glob.glob('images/epoch_*.png'))
    
    if len(image_files) == 0:
        print("⚠️  Animasyon için görüntü bulunamadı!")
        return
    
    # Görüntüleri yükle
    images = []
    for filename in image_files:
        images.append(Image.open(filename))
    
    # GIF oluştur
    if images:
        images[0].save(
            'images/training_progress.gif',
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )


def generate_samples(model_path, latent_dim=100, num_samples=25, device='cpu'):
    """
    Kaydedilmiş bir modelden yeni görüntüler üret
    """
    from model import Generator
    
    # Modeli yükle
    generator = Generator(latent_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    with torch.no_grad():
        # Rastgele gürültü
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Görüntü üret
        gen_imgs = generator(z).cpu().numpy()
        gen_imgs = (gen_imgs + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Görselleştir
        n_row = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(n_row, n_row, figsize=(10, 10))
        
        for i in range(n_row):
            for j in range(n_row):
                idx = i * n_row + j
                axes[i, j].imshow(gen_imgs[idx, 0], cmap='gray')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('images/generated_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"✅ {num_samples} adet görüntü üretildi!")


if __name__ == "__main__":
    print("🛠️  Yardımcı fonksiyonlar hazır!")
    print("\nKullanılabilir fonksiyonlar:")
    print("  - save_generated_images(): Eğitim sırasında görüntü kaydet")
    print("  - plot_losses(): Loss grafiklerini çiz")
    print("  - create_gif(): Eğitim animasyonu oluştur")
    print("  - generate_samples(): Eğitilmiş modelden görüntü üret")

