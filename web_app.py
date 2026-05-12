# -*- coding: utf-8 -*-
"""
🌐 GAN WEB ARAYÜZÜ

Gradio ile etkileşimli arayüz:
- Gürültü vektörü ayarla
- Anlık görüntü üret
- Farklı checkpoint'leri test et
"""

import torch
import gradio as gr
import numpy as np
from model import Generator
import matplotlib.pyplot as plt
from PIL import Image

class GANWebApp:
    """Web arayüzü için GAN sınıfı"""
    
    def __init__(self, checkpoint_path="checkpoints/generator_final.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generator'ı yükle
        self.generator = Generator(latent_dim=100, img_shape=(1, 28, 28)).to(self.device)
        self.generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.generator.eval()
        
        print(f"✓ Generator yüklendi: {checkpoint_path}")
    
    def generate_from_sliders(self, *slider_values):
        """
        Slider'lardan gürültü oluştur ve görüntü üret
        
        Parametreler:
            slider_values: 10 slider değeri (her biri -3 ile 3 arası)
        """
        # 10 slider değerini 100 boyutlu vektöre çevir
        # Strateji: Her slider değerini 10 kez tekrarla
        noise = []
        for val in slider_values:
            noise.extend([val] * 10)
        
        z = torch.tensor([noise], dtype=torch.float32).to(self.device)
        
        # Görüntü üret
        with torch.no_grad():
            gen_img = self.generator(z)
        
        # Numpy'a çevir ve normalize et
        img_np = gen_img[0, 0].cpu().numpy()
        img_np = (img_np + 1) / 2  # [-1, 1] -> [0, 1]
        img_np = (img_np * 255).astype(np.uint8)
        
        # PIL Image'e çevir
        img_pil = Image.fromarray(img_np, mode='L')
        return img_pil
    
    def generate_random(self):
        """Rastgele görüntü üret"""
        z = torch.randn(1, 100).to(self.device)
        
        with torch.no_grad():
            gen_img = self.generator(z)
        
        img_np = gen_img[0, 0].cpu().numpy()
        img_np = (img_np + 1) / 2
        img_np = (img_np * 255).astype(np.uint8)
        
        return Image.fromarray(img_np, mode='L')
    
    def generate_grid(self, n_samples=16):
        """
        Grid halinde görüntü üret
        
        Parametreler:
            n_samples: Kaç görüntü (4x4=16, 5x5=25 gibi)
        """
        grid_size = int(np.sqrt(n_samples))
        z = torch.randn(grid_size ** 2, 100).to(self.device)
        
        with torch.no_grad():
            gen_imgs = self.generator(z)
        
        # Grid oluştur
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                img = gen_imgs[idx, 0].cpu().numpy()
                img = (img + 1) / 2
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('temp_grid.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        return Image.open('temp_grid.png')


def create_interface():
    """Gradio arayüzü oluştur"""
    
    app = GANWebApp()
    
    # Tab 1: Slider ile kontrol
    with gr.Blocks(title="GAN Görüntü Üretici") as demo:
        gr.Markdown("# 🎨 GAN Görüntü Üretici")
        gr.Markdown("Eğitilmiş GAN modelinizle etkileşimli görüntü üretin!")
        
        with gr.Tab("🎛️ Slider Kontrolü"):
            gr.Markdown("### Gürültü Vektörünü Manuel Ayarla")
            gr.Markdown("Her slider gürültü vektörünün bir bölümünü kontrol eder")
            
            # 10 slider oluştur
            sliders = []
            with gr.Row():
                for i in range(5):
                    sliders.append(gr.Slider(-3, 3, value=0, step=0.1, label=f"Gürültü {i+1}"))
            with gr.Row():
                for i in range(5, 10):
                    sliders.append(gr.Slider(-3, 3, value=0, step=0.1, label=f"Gürültü {i+1}"))
            
            generate_btn = gr.Button("🎨 Görüntü Üret", variant="primary")
            output_img = gr.Image(label="Üretilen Görüntü", type="pil")
            
            generate_btn.click(
                fn=app.generate_from_sliders,
                inputs=sliders,
                outputs=output_img
            )
        
        with gr.Tab("🎲 Rastgele Üretim"):
            gr.Markdown("### Rastgele Görüntü Üret")
            random_btn = gr.Button("🎲 Rastgele Üret", variant="primary")
            random_output = gr.Image(label="Rastgele Görüntü", type="pil")
            
            random_btn.click(
                fn=app.generate_random,
                outputs=random_output
            )
        
        with gr.Tab("🎬 Grid Görünümü"):
            gr.Markdown("### Çoklu Görüntü Üret")
            n_samples = gr.Slider(4, 25, value=16, step=1, label="Kaç görüntü?")
            grid_btn = gr.Button("🎬 Grid Üret", variant="primary")
            grid_output = gr.Image(label="Görüntü Grid'i", type="pil")
            
            grid_btn.click(
                fn=app.generate_grid,
                inputs=n_samples,
                outputs=grid_output
            )
        
        gr.Markdown("""
        ### 💡 İpuçları:
        - Slider değerlerini değiştirerek farklı rakamlar üretin
        - Ekstrem değerler (-3, +3) daha değişik sonuçlar verir
        - Grid görünümü ile çeşitliliği görün
        """)
    
    return demo


# ============================================================================
# ÇALIŞTIRMA
# ============================================================================

if __name__ == "__main__":
    print("🌐 Web arayüzü başlatılıyor...")
    print("💡 Önce modeli eğitmeyi unutmayın!")
    print("   python train.py ile eğitin")
    print("   checkpoints/generator_final.pt oluşacak")
    print("\n📦 Gerekli paket: pip install gradio\n")
    
    try:
        demo = create_interface()
        demo.launch(share=False)  # share=True ile herkese açık link
    except FileNotFoundError:
        print("❌ Model bulunamadı!")
        print("   Önce train.py ile modeli eğitin")

