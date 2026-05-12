# -*- coding: utf-8 -*-
"""
🎨 CONDITIONAL GAN (cGAN) - İstediğin Rakamı Üret!

Bu versiyon Generator'a hangi rakamı üreteceğini söyleyebilirsiniz.
"""

import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    """
    🎨 KOŞULLU GENERATOR
    
    Fark: Gürültüye ek olarak etiket (label) de alır
    Örnek: "5 rakamı üret" komutu verebilirsin
    """
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(1, 28, 28)):
        super(ConditionalGenerator, self).__init__()
        self.img_shape = img_shape
        
        # Etiket gömme (embedding) katmanı
        # Her rakam (0-9) için 50 boyutlu bir vektör öğrenir
        self.label_emb = nn.Embedding(n_classes, 50)
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Model: Gürültü (100) + Etiket (50) = 150 boyutlu girdi
        self.model = nn.Sequential(
            *block(latent_dim + 50, 128, normalize=False),  # 150 -> 128
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        """
        İleri geçiş - Gürültü + Etiket → Görüntü
        
        Parametreler:
            z: Gürültü [batch_size, 100]
            labels: Rakam etiketi [batch_size] - Örnek: [5, 3, 7, 5, ...]
        """
        # Etiketi vektöre çevir: [64] -> [64, 50]
        label_embedding = self.label_emb(labels)
        
        # Gürültü + Etiket birleştir: [64, 100] + [64, 50] = [64, 150]
        gen_input = torch.cat((z, label_embedding), -1)
        
        # Modelden geçir
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class ConditionalDiscriminator(nn.Module):
    """
    🔍 KOŞULLU DISCRIMINATOR
    
    Fark: Görüntüyle birlikte etiket de alır
    "Bu görüntü 5 rakamı mı ve gerçek mi?" diye sorar
    """
    def __init__(self, n_classes=10, img_shape=(1, 28, 28)):
        super(ConditionalDiscriminator, self).__init__()
        
        # Etiket gömme
        self.label_emb = nn.Embedding(n_classes, 50)
        
        # Model: Görüntü (784) + Etiket (50) = 834 boyutlu girdi
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))) + 50, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        """
        İleri geçiş - Görüntü + Etiket → Skor
        """
        # Görüntüyü düzleştir
        img_flat = img.view(img.size(0), -1)  # [64, 784]
        
        # Etiketi vektöre çevir
        label_embedding = self.label_emb(labels)  # [64, 50]
        
        # Görüntü + Etiket birleştir
        d_input = torch.cat((img_flat, label_embedding), -1)  # [64, 834]
        
        # Skor ver
        validity = self.model(d_input)
        return validity


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    print("=== Conditional GAN Test ===\n")
    
    # Modelleri oluştur
    generator = ConditionalGenerator()
    discriminator = ConditionalDiscriminator()
    
    # Test verisi
    batch_size = 8
    z = torch.randn(batch_size, 100)  # Gürültü
    
    # İstediğimiz rakamlar
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # Her rakamdan 1'er tane
    
    # Görüntü üret
    gen_imgs = generator(z, labels)
    print(f"✓ Üretilen görüntüler: {gen_imgs.shape}")
    print(f"  İstenen rakamlar: {labels.tolist()}")
    
    # Discriminator'a göster
    validity = discriminator(gen_imgs, labels)
    print(f"✓ Discriminator skorları: {validity.shape}")
    print(f"  Skorlar: {validity.squeeze().tolist()}")
    
    print("\n💡 Artık 'Sadece 5 rakamı üret' diyebilirsiniz!")
    labels_all_5 = torch.tensor([5, 5, 5, 5, 5, 5, 5, 5])
    imgs_of_5 = generator(z, labels_all_5)
    print(f"✓ 8 adet 5 rakamı üretildi: {imgs_of_5.shape}")

