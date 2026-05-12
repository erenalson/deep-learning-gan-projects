# -*- coding: utf-8 -*-
"""
🔬 WGAN (Wasserstein GAN) İYİLEŞTİRMELERİ

Farklar:
1. Sigmoid yerine linear output (Discriminator → Critic)
2. BCE Loss yerine Wasserstein Loss
3. Weight clipping veya gradient penalty
4. Daha stabil eğitim!
"""

import torch
import torch.nn as nn

class WGANLoss:
    """
    📊 WASSERSTEIN LOSS
    
    Normal GAN: BCE Loss (log tabanlı)
    WGAN: Wasserstein Distance (direkt fark)
    
    Avantaj: Daha stabil, anlamlı loss değerleri
    """
    
    @staticmethod
    def discriminator_loss(real_validity, fake_validity):
        """
        Discriminator (Critic) loss
        
        Amaç: Real'i maximize, Fake'i minimize et
        """
        return -torch.mean(real_validity) + torch.mean(fake_validity)
    
    @staticmethod
    def generator_loss(fake_validity):
        """
        Generator loss
        
        Amaç: Fake'in skorunu maximize et
        """
        return -torch.mean(fake_validity)
    
    @staticmethod
    def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
        """
        Gradient Penalty (WGAN-GP)
        
        Weight clipping yerine gradient penalty kullan
        Daha iyi sonuç verir!
        """
        batch_size = real_imgs.size(0)
        
        # Rastgele interpolasyon faktörü
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        
        # Interpolate edilmiş görüntü
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        
        # Discriminator skorları
        d_interpolates = discriminator(interpolates)
        
        # Gradyanları hesapla
        fake = torch.ones(batch_size, 1).to(device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradyan normunu hesapla
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


class WGANCritic(nn.Module):
    """
    🔍 WGAN CRITIC (Discriminator'ın WGAN versiyonu)
    
    Farklar:
    - Sigmoid yok (linear output)
    - "Gerçek mi sahte mi?" yerine "ne kadar gerçek?"
    - Skor aralığı: (-∞, +∞)
    """
    def __init__(self, img_shape=(1, 28, 28)):
        super(WGANCritic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            # Sigmoid YOK! Linear output
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# ============================================================================
# WGAN EĞİTİM DÖNGÜSÜ ÖRNEĞİ
# ============================================================================

def wgan_training_step_example():
    """
    WGAN eğitim adımı örneği
    (train.py'ye entegre edilebilir)
    """
    
    # Pseudo-kod
    print("""
    # ========================================
    # WGAN EĞİTİM DÖNGÜSÜ
    # ========================================
    
    lambda_gp = 10  # Gradient penalty katsayısı
    n_critic = 5    # Her G güncellemesi için 5 D güncellemesi
    
    for epoch in range(n_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            
            # ====================================
            # CRITIC'İ EĞİT (n_critic kez)
            # ====================================
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                
                # Gerçek ve sahte görüntüler
                z = torch.randn(batch_size, 100)
                fake_imgs = generator(z)
                
                # Wasserstein loss
                real_validity = critic(real_imgs)
                fake_validity = critic(fake_imgs.detach())
                
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                
                # Gradient penalty ekle
                gp = gradient_penalty(critic, real_imgs, fake_imgs, device)
                d_loss = d_loss + lambda_gp * gp
                
                d_loss.backward()
                optimizer_D.step()
            
            # ====================================
            # GENERATOR'I EĞİT (1 kez)
            # ====================================
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, 100)
            fake_imgs = generator(z)
            fake_validity = critic(fake_imgs)
            
            g_loss = -torch.mean(fake_validity)
            
            g_loss.backward()
            optimizer_G.step()
            
            print(f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    """)


# ============================================================================
# KULLANIM
# ============================================================================

if __name__ == "__main__":
    print("=== WGAN Improvements ===\n")
    
    print("✓ WGAN Loss fonksiyonları hazır")
    print("✓ Gradient Penalty implementasyonu hazır")
    print("✓ WGAN Critic (Discriminator) hazır")
    
    print("\n💡 WGAN Avantajları:")
    print("   1. Daha stabil eğitim")
    print("   2. Anlamlı loss değerleri (loss azalıyor = kalite artıyor)")
    print("   3. Mode collapse daha az")
    print("   4. Hiperparametre seçimi daha kolay")
    
    print("\n📚 Kullanım:")
    print("   train.py'de BCELoss yerine WGANLoss kullanın")
    print("   Discriminator yerine WGANCritic kullanın")
    print("   n_critic=5 yapın (her G güncellemesi için 5 D)")
    
    wgan_training_step_example()

