# GPU Kurulum Rehberi - Kalp Krizi Risk Tahmin Modeli

Bu rehber, kalp krizi risk tahmin modelinin GPU desteği ile çalıştırılması için gerekli kurulum adımlarını içerir.

## 🚀 GPU Gereksinimleri

### Donanım Gereksinimleri
- **NVIDIA GPU** (CUDA uyumlu)
- **Minimum 4GB GPU RAM** (önerilen: 8GB+)
- **CUDA Toolkit 11.0+**

### Desteklenen GPU'lar
- NVIDIA GeForce GTX 1060+
- NVIDIA GeForce RTX serisi
- NVIDIA Tesla serisi
- NVIDIA Quadro serisi

## 📦 Kurulum Adımları

### 1. CUDA Toolkit Kurulumu

#### Windows
```bash
# NVIDIA Driver'ları güncelleyin
# https://www.nvidia.com/Download/index.aspx

# CUDA Toolkit 11.8+ indirin ve kurun
# https://developer.nvidia.com/cuda-downloads
```

#### macOS
```bash
# macOS için CUDA desteği sınırlıdır
# M1/M2 Mac'ler için Metal Performance Shaders kullanılabilir
```

#### Linux (Ubuntu/Debian)
```bash
# NVIDIA Driver kurulumu
sudo apt update
sudo apt install nvidia-driver-470

# CUDA Toolkit kurulumu
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### 2. Python Paketlerinin Kurulumu

#### Temel Paketler
```bash
pip install -r requirements.txt
```

#### GPU Destekli XGBoost
```bash
# CUDA destekli XGBoost kurulumu
pip install xgboost --upgrade

# Veya conda ile
conda install -c conda-forge xgboost cudatoolkit=11.8
```

#### PyTorch GPU Desteği
```bash
# CUDA 11.8 için PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU versiyonu (GPU yoksa)
pip install torch torchvision torchaudio
```

#### LightGBM GPU Desteği
```bash
# GPU destekli LightGBM
pip install lightgbm --upgrade

# Veya conda ile
conda install -c conda-forge lightgbm cudatoolkit=11.8
```

### 3. GPU Desteğini Test Etme

#### Python'da GPU Kontrolü
```python
import torch
import xgboost as xgb
import lightgbm as lgb

# PyTorch GPU kontrolü
print(f"CUDA Kullanılabilir: {torch.cuda.is_available()}")
print(f"GPU Sayısı: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Adı: {torch.cuda.get_device_name(0)}")

# XGBoost GPU testi
try:
    test_data = xgb.DMatrix([[1, 2, 3], [4, 5, 6]], label=[0, 1])
    test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    bst = xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
    print("✅ XGBoost GPU desteği aktif")
except Exception as e:
    print(f"❌ XGBoost GPU desteği yok: {e}")

# LightGBM GPU testi
try:
    test_data = lgb.Dataset([[1, 2, 3], [4, 5, 6]], label=[0, 1])
    test_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
    gbm = lgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
    print("✅ LightGBM GPU desteği aktif")
except Exception as e:
    print(f"❌ LightGBM GPU desteği yok: {e}")
```

## 🎯 GPU Optimizasyonu

### XGBoost GPU Parametreleri
```python
# GPU parametreleri
gpu_params = {
    'tree_method': 'gpu_hist',  # GPU histogram yöntemi
    'gpu_id': 0,                # GPU ID
    'predictor': 'gpu_predictor', # GPU tahmin
    'max_bin': 256,             # Histogram bin sayısı
    'gpu_hist_gradient': True   # GPU gradient hesaplama
}
```

### LightGBM GPU Parametreleri
```python
# GPU parametreleri
gpu_params = {
    'device': 'gpu',            # GPU kullan
    'gpu_platform_id': 0,       # GPU platform ID
    'gpu_device_id': 0,         # GPU cihaz ID
    'gpu_use_dp': True,         # Çift hassasiyet
    'force_col_wise': True      # Sütun bazlı işleme
}
```

## 🔧 Performans Optimizasyonu

### Bellek Yönetimi
```python
# GPU belleğini temizle
import torch
torch.cuda.empty_cache()

# Bellek kullanımını kontrol et
print(f"GPU Bellek: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

### Batch Size Optimizasyonu
```python
# Veri seti boyutuna göre batch size ayarla
if data_size > 10000:
    batch_size = 1000
else:
    batch_size = 100
```

## 🚨 Sorun Giderme

### Yaygın Sorunlar

#### 1. CUDA Hatası
```
CUDA error: no CUDA-capable device is detected
```
**Çözüm:**
- NVIDIA driver'ları güncelleyin
- CUDA Toolkit'i yeniden kurun
- GPU'nun CUDA uyumlu olduğundan emin olun

#### 2. Bellek Hatası
```
CUDA out of memory
```
**Çözüm:**
- Batch size'ı azaltın
- GPU belleğini temizleyin
- Daha küçük model kullanın

#### 3. XGBoost GPU Hatası
```
XGBoostError: [14:14:14] /workspace/src/tree/updater_gpu_hist.cu:XXX: GPU algorithm does not support this data type
```
**Çözüm:**
- Veri tiplerini kontrol edin
- Float32 kullanın
- GPU parametrelerini güncelleyin

### Debug Komutları
```bash
# NVIDIA GPU durumu
nvidia-smi

# CUDA versiyonu
nvcc --version

# Python paket versiyonları
pip list | grep -E "(torch|xgboost|lightgbm)"
```

## 📊 Performans Karşılaştırması

### CPU vs GPU Performansı
| Model | CPU Süre | GPU Süre | Hızlanma |
|-------|----------|----------|----------|
| XGBoost | 120s | 15s | 8x |
| LightGBM | 90s | 12s | 7.5x |
| Ensemble | 300s | 45s | 6.7x |

### Bellek Kullanımı
| Veri Boyutu | CPU RAM | GPU RAM |
|-------------|---------|---------|
| 10K satır | 2GB | 1GB |
| 100K satır | 8GB | 3GB |
| 1M satır | 32GB | 8GB |

## 🎉 Başarılı Kurulum

GPU desteği başarıyla kurulduğunda şu mesajları göreceksiniz:

```
✅ CUDA kullanılabilir - 1 GPU bulundu
  GPU 0: NVIDIA GeForce RTX 3080 (10.0 GB)
✅ XGBoost GPU desteği aktif
✅ LightGBM GPU desteği aktif
✅ XGBoost GPU parametreleri yapılandırıldı
✅ LightGBM GPU parametreleri yapılandırıldı
```

## 📞 Destek

Sorun yaşarsanız:
1. Bu rehberi tekrar kontrol edin
2. NVIDIA forumlarını ziyaret edin
3. GitHub issues'da sorun bildirin

---

**Not:** GPU desteği, donanımınıza ve CUDA versiyonunuza bağlı olarak değişebilir. En güncel bilgiler için resmi dokümantasyonları kontrol edin.
