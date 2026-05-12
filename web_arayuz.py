"""
🌐 GAN WEB ARAYÜZÜ (Streamlit)

En etkileyici geliştirme! Tarayıcıdan görüntü üret!

Kurulum:
    pip install streamlit

Çalıştırma:
    streamlit run web_arayuz.py

Tarayıcıda otomatik açılır!
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
import os
from PIL import Image
import io

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🎨 GAN Görüntü Üretici",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık
st.title("🎨 GAN Görüntü Üretici")
st.markdown("Yapay zeka ile yeni rakam görüntüleri üretin!")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")

# Model seçimi
checkpoint_dir = "checkpoints"
if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if checkpoints:
        selected_model = st.sidebar.selectbox(
            "Model Seç:",
            checkpoints,
            index=len(checkpoints)-1  # En son model varsayılan
        )
        model_path = os.path.join(checkpoint_dir, selected_model)
    else:
        st.error("❌ Checkpoint bulunamadı! Önce train.py çalıştırın.")
        st.stop()
else:
    st.error("❌ checkpoints/ klasörü bulunamadı!")
    st.stop()

# Görüntü sayısı
num_images = st.sidebar.slider(
    "Kaç görüntü üretilsin?",
    min_value=1,
    max_value=100,
    value=16,
    step=1
)

# Grid boyutu hesapla
n_cols = st.sidebar.slider(
    "Sütun sayısı:",
    min_value=1,
    max_value=10,
    value=4,
    step=1
)

# Seed (tekrarlanabilirlik için)
use_seed = st.sidebar.checkbox("Sabit seed kullan (aynı sonuçlar için)", value=False)
if use_seed:
    seed = st.sidebar.number_input("Seed değeri:", min_value=0, max_value=9999, value=42)
else:
    seed = None

# Model bilgileri göster
with st.sidebar.expander("📊 Model Bilgileri"):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        st.write(f"**Epoch:** {checkpoint['epoch']}")
        st.write(f"**Generator Loss:** {checkpoint['g_losses'][-1]:.4f}")
        st.write(f"**Discriminator Loss:** {checkpoint['d_losses'][-1]:.4f}")
    except Exception as e:
        st.write(f"Bilgi yüklenemedi: {e}")

# Yardım
with st.sidebar.expander("❓ Nasıl Kullanılır?"):
    st.markdown("""
    1. Soldaki ayarlardan model seçin
    2. Kaç görüntü üretmek istediğinizi seçin
    3. **"Görüntü Üret"** butonuna tıklayın
    4. İsterseniz görüntüleri indirin!
    
    **İpucu:** Sabit seed kullanarak aynı görüntüleri tekrar üretebilirsiniz.
    """)

# Ana içerik
st.markdown("---")

# Görüntü üretme butonu
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("🎨 Görüntü Üret", use_container_width=True, type="primary")

if generate_button or 'generated_images' not in st.session_state:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Model yükle
        status_text.text("🔄 Model yükleniyor...")
        progress_bar.progress(20)
        
        generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
        checkpoint = torch.load(model_path, map_location='cpu')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        # Görüntü üret
        status_text.text("🎨 Görüntüler üretiliyor...")
        progress_bar.progress(50)
        
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            z = torch.randn(num_images, 100)
            gen_imgs = generator(z).cpu().numpy()
            gen_imgs = (gen_imgs + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Görselleştir
        status_text.text("🖼️ Görselleştiriliyor...")
        progress_bar.progress(80)
        
        n_rows = (num_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        fig.patch.set_facecolor('white')
        
        if num_images == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for i in range(num_images):
            axes[i].imshow(gen_imgs[i, 0], cmap='gray')
            axes[i].axis('off')
        
        # Boş subplotları gizle
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Session state'e kaydet
        st.session_state['generated_images'] = gen_imgs
        st.session_state['fig'] = fig
        
        progress_bar.progress(100)
        status_text.text("✅ Tamamlandı!")
        
    except Exception as e:
        st.error(f"❌ Hata: {e}")
        st.stop()

# Görüntüleri göster
if 'fig' in st.session_state:
    st.pyplot(st.session_state['fig'])
    
    # İndirme butonları
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # PNG olarak indir
        buf = io.BytesIO()
        st.session_state['fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="📥 PNG İndir",
            data=buf,
            file_name="gan_generated_images.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col2:
        # Tek tek indir butonu (ilk 10 görüntü için)
        if st.button("💾 Tek Tek Kaydet", use_container_width=True):
            os.makedirs("downloads", exist_ok=True)
            for i, img in enumerate(st.session_state['generated_images'][:10]):
                img_pil = Image.fromarray((img[0] * 255).astype(np.uint8))
                img_pil.save(f"downloads/image_{i+1}.png")
            st.success(f"✅ İlk {min(10, num_images)} görüntü downloads/ klasörüne kaydedildi!")
    
    with col3:
        # Yeni görüntü üret
        if st.button("🔄 Yeniden Üret", use_container_width=True):
            del st.session_state['generated_images']
            del st.session_state['fig']
            st.rerun()

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎨 GAN Görüntü Üretici | Yapay Zeka ile Üretildi</p>
    <p>Model: Generative Adversarial Network (GAN)</p>
</div>
""", unsafe_allow_html=True)

# Sağ sidebar - İstatistikler
with st.sidebar:
    st.markdown("---")
    st.subheader("📈 İstatistikler")
    if 'generated_images' in st.session_state:
        st.metric("Üretilen Görüntü", len(st.session_state['generated_images']))
        st.metric("Görüntü Boyutu", "28×28 piksel")
        st.metric("Model Tipi", "Vanilla GAN")

