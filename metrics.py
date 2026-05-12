# -*- coding: utf-8 -*-
"""
📊 GAN EVALUATION METRICS

GAN kalitesini ölçmek için metrikler:
- Inception Score (IS): Görüntü kalitesi ve çeşitliliği
- Frechet Inception Distance (FID): Gerçek ve sahte arasındaki mesafe
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3

class InceptionScore:
    """
    📈 INCEPTION SCORE (IS)
    
    Ne ölçer:
    - Yüksek IS = İyi kalite + Yüksek çeşitlilik
    - Düşük IS = Kötü kalite veya düşük çeşitlilik
    
    Aralık: 1-10 (10 en iyi)
    """
    def __init__(self, device='cuda'):
        self.device = device
        # Inception v3 modelini yükle
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
    
    def calculate(self, images, batch_size=32, splits=10):
        """
        IS hesapla
        
        Parametreler:
            images: Üretilen görüntüler [N, 3, 299, 299]
            batch_size: İşlem batch boyutu
            splits: Kaça bölünerek hesaplansın
        
        Döndürür:
            mean, std: IS skoru (ortalama, standart sapma)
        """
        N = len(images)
        
        # Tüm görüntüleri işle
        preds = []
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size].to(self.device)
            with torch.no_grad():
                pred = self.inception(batch)
                pred = torch.nn.functional.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # IS hesapla
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py)))
            split_scores.append(np.exp(np.mean(scores)))
        
        return np.mean(split_scores), np.std(split_scores)


class FrechetInceptionDistance:
    """
    📏 FRECHET INCEPTION DISTANCE (FID)
    
    Ne ölçer:
    - Gerçek ve sahte görüntülerin dağılım farkı
    - Düşük FID = Daha iyi (gerçeğe yakın)
    
    Aralık: 0-∞ (0 en iyi)
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
    
    def get_features(self, images, batch_size=32):
        """Görüntülerden özellik çıkar"""
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            with torch.no_grad():
                feat = self.inception(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)
    
    def calculate(self, real_images, fake_images):
        """
        FID hesapla
        
        Parametreler:
            real_images: Gerçek görüntüler
            fake_images: Üretilen görüntüler
        
        Döndürür:
            fid: FID skoru (düşük = iyi)
        """
        # Özellik çıkar
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # İstatistikleri hesapla
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # FID hesapla
        diff = mu_real - mu_fake
        covmean = sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return fid


class SimpleQualityMetrics:
    """
    🔍 BASİT KALİTE METRİKLERİ (MNIST için)
    
    Inception Score ve FID büyük modeller gerektirir.
    MNIST için daha basit metrikler:
    """
    
    @staticmethod
    def pixel_diversity(images):
        """
        Piksel çeşitliliği
        
        Yüksek = Her görüntü farklı (iyi!)
        Düşük = Hep aynı görüntü (mode collapse!)
        """
        images_np = images.cpu().numpy()
        # Her pikselin standart sapması
        diversity = np.std(images_np, axis=0).mean()
        return diversity
    
    @staticmethod
    def mean_brightness(images):
        """Ortalama parlaklık"""
        return images.mean().item()
    
    @staticmethod
    def contrast(images):
        """Kontrast (max - min)"""
        return (images.max() - images.min()).item()
    
    @staticmethod
    def sharpness(images):
        """
        Keskinlik - Laplacian variance
        
        Yüksek = Keskin kenarlar (iyi!)
        Düşük = Bulanık (kötü!)
        """
        # Basit Laplacian filtre
        kernel = torch.tensor([[[[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]]]], dtype=torch.float32)
        
        images = images.cpu()
        laplacian = torch.nn.functional.conv2d(images, kernel, padding=1)
        sharpness = laplacian.var().item()
        return sharpness


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    print("=== GAN Metrics Test ===\n")
    
    # Test görüntüleri (random)
    fake_images = torch.randn(100, 1, 28, 28)
    
    # Basit metrikler
    metrics = SimpleQualityMetrics()
    
    print("📊 Basit Metrikler:")
    print(f"  Çeşitlilik: {metrics.pixel_diversity(fake_images):.4f}")
    print(f"  Parlaklık: {metrics.mean_brightness(fake_images):.4f}")
    print(f"  Kontrast: {metrics.contrast(fake_images):.4f}")
    print(f"  Keskinlik: {metrics.sharpness(fake_images):.4f}")
    
    print("\n💡 Bu metrikleri train.py'de her epoch'ta hesaplayıp kaydedin!")
    print("   Eğitim ilerledikçe çeşitlilik ve keskinlik artmalı!")

