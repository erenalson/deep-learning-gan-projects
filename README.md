# PyTorch GAN Koleksiyonu

Yapay zeka ile **yeni görüntü üretimi** ve **görüntü stil dönüşümü** (ör. **gündüz → gece**, **gece → gündüz**) için PyTorch ile geliştirilmiş bir proje koleksiyonu. CycleGAN ile **eşleştirilmiş (paired) girdi–çıktı çiftlerine ihtiyaç duymadan** iki görüntü kümesi arasında dönüşüm mümkündür.

### CycleGAN karşılaştırma şeritleri (`assets/`)

Her görsel yatay **3 panel**: **orijinal** → **hedef domaine çeviri** → **cycle ile geri oluşturma** (dashcam / sokak senaryosu, gündüz↔gece).

| Örnek | Örnek | Örnek |
|:---:|:---:|:---:|

| ![comparison_0004](assets/comparison_0004.jpg) | ![comparison_0011](assets/comparison_0011.jpg) |
| ![comparison_0014](assets/comparison_0014.jpg) | ![comparison_0074](assets/comparison_0074.jpg) | ![comparison_0103](assets/comparison_0103.jpg) |


---

## Bu repo ne işe yarar?

- **Görüntü üretimi:** Gürültüden öğrenilmiş dağılıma uygun yeni görüntüler (ör. el yazısı rakamlar).
- **Gündüz / gece dönüşümü:** CycleGAN ile sokak veya sahne görüntülerini gündüzden geceye veya tersine çevirme.
- **Diğer stil örnekleri (aynı çerçeve):** Yaz↔kış, at↔zebra vb. — veri kümeleri uygun olduğunda aynı mimariyle eğitilebilir.

**“Paired veri yok” ne demek?** Aynı kadrajın gündüz ve gece fotoğrafı çiftlerini vermek zorunda değilsin; yeter ki bir klasörde gündüz, başkada gece örnekleri olsun. Model iki “domain” arasındaki ilişkiyi Cycle tutarlılığı ve adversarial öğrenme ile öğrenir.

---

## Özellikler

| Bileşen | Açıklama |
|--------|----------|
| **CycleGAN** | 2 generator + 2 discriminator, PatchGAN, cycle / identity / LSGAN kayıpları, residual bloklar |
| **MNIST GAN** | Klasik taban çizgisi, `mnist_gan/` altında |
| **DCGAN** | Konvolüsyon tabanlı üretici ayrımı |
| **Conditional GAN** | Etiket ile yönlendirilmiş üretim |
| **WGAN** | Wasserstein / daha stabil eğitim fikirleri |
| **Metrikler** | FID, Inception Score (`metrics.py`) |
| **Demo arayüz** | Gradio (`web_app.py` — ayrıca `gradio` kurulumu gerekir) |

---

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADIN/REPO_ADIN.git
cd REPO_ADIN
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Gradio arayüzü için:

```bash
pip install gradio
```

CycleGAN için veri indirme / Kaggle yardımcıları: `cyclegan/` içindeki rehber dosyalara bakın; isteğe bağlı: `pip install kaggle`.

---

## Hızlı kullanım

### CycleGAN (gündüz–gece vb.)

```bash
cd cyclegan
python create_demo_dataset.py 30
python cyclegan_train.py
python cyclegan_test.py
```

### MNIST GAN

```bash
cd mnist_gan
python train.py
```

### Ana klasördeki ek scriptler

Fashion-MNIST eğitimi (`train_fashion.py`), üretim (`generate.py`, `generate_fashion.py`) ve diğer modüller kök dizinde; ayrıntılar için ilgili `.py` dosyalarının başındaki açıklamalara bakın.

---

## Proje yapısı (özet)

```
.
├── assets/                 # comparison_*.jpg — 3 panelli README örnekleri
├── cyclegan/               # CycleGAN model, eğitim, veri hazırlığı
├── mnist_gan/              # MNIST üretim pipeline’ı
├── model_dcgan.py
├── model_conditional.py
├── wgan_improvements.py
├── metrics.py
├── web_app.py
├── requirements.txt
└── README.md
```

---




