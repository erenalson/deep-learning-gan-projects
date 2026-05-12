# PyTorch kütüphanesini içe aktar (derin öğrenme için temel kütüphane)
import torch
# Sinir ağı modülleri (katmanlar, aktivasyon fonksiyonları vb.)
import torch.nn as nn

class Generator(nn.Module):
    """
    🎨 GENERATOR (ÜRETİCİ) AĞI
    
    Bu ağın görevi: Rastgele gürültüden (100 sayı) gerçekçi görüntü üretmek
    
    Nasıl çalışır:
    1. 100 rastgele sayı alır (latent vector/gizli vektör)
    2. Bu sayıları katmanlardan geçirir
    3. Sonunda 28x28 piksel görüntü üretir (MNIST rakam boyutu)
    
    """
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        # Üst sınıfın (nn.Module) yapıcı metodunu çağır
        super(Generator, self).__init__()
        # Üretilecek görüntünün boyutunu sakla: (1, 28, 28) = 1 kanal, 28x28 piksel
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            """
            🧱 YAPI BLOĞU
            Generator'ın temel yapı taşı. Her blok şunları içerir:
            1. Linear (Tam Bağlı) Katman: Girdiyi dönüştürür
            2. Batch Normalization: Eğitimi hızlandırır ve stabilize eder
            3. LeakyReLU Aktivasyon: Negatif değerleri de korur (normal ReLU'dan daha iyi)
            """
            # Boş bir liste oluştur, katmanları buraya ekleyeceğiz
            layers = [nn.Linear(in_feat, out_feat)]  # Tam bağlı katman: girdi -> çıktı
            
            if normalize:
                # Batch normalization: Verileri normalize eder, eğitimi hızlandırır
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            
            # LeakyReLU: Negatif değerler için küçük bir eğim (0.2) kullanır
            # inplace=True: Bellek tasarrufu için değeri yerinde değiştirir
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # 🏗️ GENERATOR MİMARİSİ (Katmanlar zincirleme)
        # İlerleme: 100 -> 128 -> 256 -> 512 -> 1024 -> 784 (28x28)
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),  # İlk blok: 100 -> 128 nöron
            *block(128, 256),                          # 128 -> 256 nöron
            *block(256, 512),                          # 256 -> 512 nöron
            *block(512, 1024),                         # 512 -> 1024 nöron
            # Son katman: 1024 -> 784 (28*28=784 piksel)
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            # Tanh aktivasyon: Çıktıyı [-1, 1] aralığına sıkıştırır (MNIST veri formatıyla uyumlu)
            nn.Tanh()
        )
    
    def forward(self, z):
        """
        ⚡ İLERİ GEÇİŞ (Forward Pass)
        
        Generator'ın ana işlevi. Gürültüyü görüntüye çevirir.
        
        Parametreler:
            z: Rastgele gürültü vektörü, boyut: (batch_size, 100)
               Örnek: 64 örnek için (64, 100)
        
        Döndürür:
            img: Üretilmiş görüntü, boyut: (batch_size, 1, 28, 28)
                 Örnek: 64 örnek için (64, 1, 28, 28)
        """
        # Gürültüyü tüm katmanlardan geçir
        img = self.model(z)
        # Düz vektörü (batch_size, 784) görüntü formatına (batch_size, 1, 28, 28) dönüştür
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """
    🔍 DISCRIMINATOR (AYIRT EDİCİ) AĞI
    
    Bu ağın görevi: Bir görüntüye bakıp gerçek mi sahte mi olduğunu anlamak
    
    Nasıl çalışır:
    1. Görüntüyü alır (28x28 piksel)
    2. Katmanlardan geçirir
    3. Tek bir sayı verir: 0 (sahte) ile 1 (gerçek) arası
    
    Örnek: [28x28 görüntü] -> 0.85 (büyük ihtimalle gerçek!)
           [28x28 görüntü] -> 0.12 (muhtemelen sahte!)
    
    Polis gibidir: Generator'ın ürettiği sahte görüntüleri yakalamaya çalışır!
    """
    def __init__(self, img_shape=(1, 28, 28)):
        # Üst sınıfın yapıcı metodunu çağır
        super(Discriminator, self).__init__()
        
        # 🏗️ DISCRIMINATOR MİMARİSİ
        # İlerleme: 784 (28x28) -> 512 -> 256 -> 128 -> 1 (gerçek/sahte skoru)
        self.model = nn.Sequential(
            # İlk katman: 784 piksel (28*28) -> 512 nöron
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            # LeakyReLU: Negatif değerler için 0.2 eğim
            nn.LeakyReLU(0.2, inplace=True),
            # Dropout (0.3): Rastgele %30 nöronu kapat -> Overfitting'i önler
            nn.Dropout(0.3),
            
            # İkinci katman: 512 -> 256 nöron
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Yine %30 dropout
            
            # Üçüncü katman: 256 -> 128 nöron
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Yine %30 dropout
            
            # Son katman: 128 -> 1 (tek bir skor değeri)
            nn.Linear(128, 1),
            # Sigmoid: Çıktıyı 0 ile 1 arasına sıkıştırır (olasılık değeri)
            # 0 = %100 sahte, 1 = %100 gerçek, 0.5 = emin değil
            nn.Sigmoid()
        )
    
    def forward(self, img):
        """
        ⚡ İLERİ GEÇİŞ (Forward Pass)
        
        Görüntünün gerçek mi sahte mi olduğunu değerlendirir.
        
        Parametreler:
            img: Girdi görüntüsü, boyut: (batch_size, 1, 28, 28)
                 Örnek: 64 görüntü için (64, 1, 28, 28)
        
        Döndürür:
            validity: Gerçeklik skoru, boyut: (batch_size, 1)
                     Her görüntü için 0-1 arası değer
                     Örnek: [[0.85], [0.12], [0.67], ...] 64 görüntü için
        """
        # Görüntüyü düzleştir: (batch_size, 1, 28, 28) -> (batch_size, 784)
        # -1: PyTorch otomatik olarak boyutu hesaplar
        img_flat = img.view(img.size(0), -1)
        # Düzleştirilmiş görüntüyü katmanlardan geçir
        validity = self.model(img_flat)
        return validity


if __name__ == "__main__":
    """
    🧪 TEST BÖLÜMÜ
    
    Bu kısım sadece "python model.py" komutunu çalıştırdığınızda çalışır.
    Modellerin doğru kurulup kurulmadığını test eder.
    
    Ne yapar:
    1. Generator ve Discriminator oluşturur
    2. Rastgele gürültüden görüntü üretir
    3. Discriminator'a gösterir ve değerlendirir
    4. Boyutların doğru olup olmadığını kontrol eder
    """
    print("=== GAN Model Test ===")
    
    # Test için gerekli parametreler
    batch_size = 64              # Aynı anda 64 görüntü test edeceğiz
    latent_dim = 100             # 100 boyutlu gürültü vektörü
    img_shape = (1, 28, 28)      # 1 kanal (siyah-beyaz), 28x28 piksel
    
    # Modelleri oluştur (GPU yoksa CPU'da çalışır)
    generator = Generator(latent_dim, img_shape)
    discriminator = Discriminator(img_shape)
    
    # Generator'daki toplam parametre sayısını say
    print(f"\n✓ Generator oluşturuldu")
    print(f"  Parametreler: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Discriminator'daki toplam parametre sayısını say
    print(f"\n✓ Discriminator oluşturuldu")
    print(f"  Parametreler: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 64 adet 100 boyutlu rastgele gürültü üret
    # Boyut: (64, 100) - 64 örnek, her biri 100 sayı
    z = torch.randn(batch_size, latent_dim)
    
    # Generator ile gürültüden görüntü üret
    fake_imgs = generator(z)
    # Beklenen boyut: (64, 1, 28, 28) - 64 adet 28x28 görüntü
    print(f"\n✓ Üretilen görüntü boyutu: {fake_imgs.shape}")
    
    # Discriminator ile üretilen görüntüleri değerlendir
    validity = discriminator(fake_imgs)
    # Beklenen boyut: (64, 1) - Her görüntü için 1 skor (0-1 arası)
    print(f"✓ Discriminator çıktısı boyutu: {validity.shape}")
    
    print("\n✅ Tüm testler başarılı!")
    print("\n💡 İpucu: Şimdi 'python train.py' ile eğitime başlayabilirsiniz!")

