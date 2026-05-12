# -*- coding: utf-8 -*-
"""
🎨 DCGAN (Deep Convolutional GAN)

Farkı: Fully Connected yerine Convolutional katmanlar kullanır
Avantajı: Daha iyi görüntü kalitesi, özellikle büyük görüntülerde
"""

import torch
import torch.nn as nn

class DCGenerator(nn.Module):
    """
    🎨 DCGAN GENERATOR
    
    Gürültü → ConvTranspose2d katmanlarıyla büyütülür → Görüntü
    
    Süreç:
    100 → [4x4] → [8x8] → [16x16] → [28x28]
    """
    def __init__(self, latent_dim=100, channels=1):
        super(DCGenerator, self).__init__()
        
        # İlk projeksiyon: 100 → 7x7x128
        self.init_size = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        # Convolutional bloğu
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            # Upsample 1: 7x7 → 14x14
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample 2: 14x14 → 28x28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Son katman: 28x28, kanal sayısını 1'e düşür
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        """
        İleri geçiş
        
        z: [batch, 100] → img: [batch, 1, 28, 28]
        """
        # 100 → 7x7x128
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Convolutional katmanlardan geçir
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    """
    🔍 DCGAN DISCRIMINATOR
    
    Görüntü → Conv2d katmanlarıyla küçültülür → Skor
    
    Süreç:
    [28x28] → [14x14] → [7x7] → [1]
    """
    def __init__(self, channels=1):
        super(DCDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            """Discriminator bloğu"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),  # Boyut yarıya düşer
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            # 28x28 → 14x14
            *discriminator_block(channels, 16, bn=False),
            # 14x14 → 7x7
            *discriminator_block(16, 32),
            # 7x7 → 3x3 (integer division)
            *discriminator_block(32, 64),
            # 3x3 → 1x1
            *discriminator_block(64, 128),
        )
        
        # Son katman: Flatten + Linear → Skor
        # 128 * 1 * 1 (veya 128 * 2 * 2 depending on dimensions)
        ds_size = 2  # Sonuç boyutu (28 -> 14 -> 7 -> 3 -> 1 ama padding nedeniyle 2 olabilir)
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        """
        İleri geçiş
        
        img: [batch, 1, 28, 28] → validity: [batch, 1]
        """
        out = self.model(img)
        out = out.view(out.shape[0], -1)  # Flatten
        validity = self.adv_layer(out)
        return validity


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    print("=== DCGAN Model Test ===\n")
    
    # Modelleri oluştur
    generator = DCGenerator()
    discriminator = DCDiscriminator()
    
    print("✓ DCGAN Generator oluşturuldu")
    print(f"  Parametreler: {sum(p.numel() for p in generator.parameters()):,}")
    
    print("✓ DCGAN Discriminator oluşturuldu")
    print(f"  Parametreler: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test
    z = torch.randn(8, 100)
    gen_imgs = generator(z)
    print(f"\n✓ Üretilen görüntü boyutu: {gen_imgs.shape}")
    
    validity = discriminator(gen_imgs)
    print(f"✓ Discriminator çıktısı: {validity.shape}")
    
    print("\n💡 DCGAN, özellikle büyük görüntülerde (64x64, 128x128) çok daha iyi!")
    print("   MNIST için küçük bir iyileşme, Fashion-MNIST/CIFAR-10 için büyük fark!")

