"""
GPU Test Script - Kalp Krizi Risk Tahmin Modeli
GPU desteğinin doğru çalışıp çalışmadığını test eder.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_gpu_availability():
    """GPU kullanılabilirliğini test et."""
    print("="*60)
    print("GPU KULLANILABİLİRLİK TESTİ")
    print("="*60)
    
    gpu_status = {
        'cuda_available': False,
        'xgboost_gpu': False,
        'lightgbm_gpu': False,
        'gpu_count': 0,
        'gpu_names': []
    }
    
    # 1. PyTorch CUDA Testi
    print("\n1. PyTorch CUDA Testi:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_status['cuda_available'] = True
            gpu_status['gpu_count'] = torch.cuda.device_count()
            
            print(f"✅ CUDA kullanılabilir")
            print(f"   GPU Sayısı: {gpu_status['gpu_count']}")
            
            for i in range(gpu_status['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_status['gpu_names'].append(gpu_name)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA kullanılamıyor")
    except ImportError:
        print("❌ PyTorch yüklü değil")
    except Exception as e:
        print(f"❌ PyTorch CUDA hatası: {e}")
    
    # 2. XGBoost GPU Testi
    print("\n2. XGBoost GPU Testi:")
    try:
        import xgboost as xgb
        
        # Test verisi oluştur
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 2, 1000)
        
        # GPU parametreleri ile test
        try:
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'max_depth': 3,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            
            start_time = time.time()
            bst = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
            gpu_time = time.time() - start_time
            
            gpu_status['xgboost_gpu'] = True
            print(f"✅ XGBoost GPU desteği aktif")
            print(f"   Eğitim süresi: {gpu_time:.2f} saniye")
            
        except Exception as e:
            print(f"❌ XGBoost GPU hatası: {e}")
            
    except ImportError:
        print("❌ XGBoost yüklü değil")
    
    # 3. LightGBM GPU Testi
    print("\n3. LightGBM GPU Testi:")
    try:
        import lightgbm as lgb
        
        # Test verisi oluştur
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 2, 1000)
        
        # GPU parametreleri ile test
        try:
            train_data = lgb.Dataset(X, label=y)
            params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9
            }
            
            start_time = time.time()
            gbm = lgb.train(params, train_data, num_boost_round=10, verbose_eval=False)
            gpu_time = time.time() - start_time
            
            gpu_status['lightgbm_gpu'] = True
            print(f"✅ LightGBM GPU desteği aktif")
            print(f"   Eğitim süresi: {gpu_time:.2f} saniye")
            
        except Exception as e:
            print(f"❌ LightGBM GPU hatası: {e}")
            
    except ImportError:
        print("❌ LightGBM yüklü değil")
    
    return gpu_status

def test_model_performance():
    """Model performansını test et."""
    print("\n" + "="*60)
    print("MODEL PERFORMANS TESTİ")
    print("="*60)
    
    # Test verisi oluştur
    print("Test verisi oluşturuluyor...")
    np.random.seed(42)
    n_samples = 5000
    n_features = 20
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    print(f"Veri boyutu: {X.shape}")
    
    # XGBoost CPU vs GPU karşılaştırması
    print("\nXGBoost CPU vs GPU Karşılaştırması:")
    
    try:
        import xgboost as xgb
        
        # CPU testi
        dtrain = xgb.DMatrix(X, label=y)
        cpu_params = {
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        start_time = time.time()
        bst_cpu = xgb.train(cpu_params, dtrain, num_boost_round=50, verbose_eval=False)
        cpu_time = time.time() - start_time
        
        print(f"CPU Eğitim süresi: {cpu_time:.2f} saniye")
        
        # GPU testi (eğer mevcutsa)
        try:
            gpu_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            
            start_time = time.time()
            bst_gpu = xgb.train(gpu_params, dtrain, num_boost_round=50, verbose_eval=False)
            gpu_time = time.time() - start_time
            
            print(f"GPU Eğitim süresi: {gpu_time:.2f} saniye")
            speedup = cpu_time / gpu_time
            print(f"Hızlanma oranı: {speedup:.2f}x")
            
        except Exception as e:
            print(f"GPU testi başarısız: {e}")
            
    except ImportError:
        print("XGBoost yüklü değil")

def test_memory_usage():
    """GPU bellek kullanımını test et."""
    print("\n" + "="*60)
    print("GPU BELLEK KULLANIMI TESTİ")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Bellek durumunu kontrol et
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"GPU {i} Bellek Durumu:")
                print(f"  Toplam: {total:.2f} GB")
                print(f"  Ayrılan: {allocated:.2f} GB")
                print(f"  Rezerve: {reserved:.2f} GB")
                print(f"  Boş: {total - reserved:.2f} GB")
                
            # Bellek temizleme testi
            print("\nBellek temizleme testi...")
            torch.cuda.empty_cache()
            print("✅ GPU belleği temizlendi")
            
    except ImportError:
        print("PyTorch yüklü değil")

def main():
    """Ana test fonksiyonu."""
    print("🚀 GPU TEST SCRIPTİ BAŞLATILIYOR")
    print("="*80)
    
    # 1. GPU kullanılabilirlik testi
    gpu_status = test_gpu_availability()
    
    # 2. Model performans testi
    test_model_performance()
    
    # 3. Bellek kullanımı testi
    test_memory_usage()
    
    # 4. Özet rapor
    print("\n" + "="*80)
    print("TEST ÖZET RAPORU")
    print("="*80)
    
    print(f"CUDA Kullanılabilir: {'✅' if gpu_status['cuda_available'] else '❌'}")
    print(f"GPU Sayısı: {gpu_status['gpu_count']}")
    print(f"XGBoost GPU: {'✅' if gpu_status['xgboost_gpu'] else '❌'}")
    print(f"LightGBM GPU: {'✅' if gpu_status['lightgbm_gpu'] else '❌'}")
    
    if gpu_status['gpu_names']:
        print(f"GPU'lar: {', '.join(gpu_status['gpu_names'])}")
    
    # Öneriler
    print("\n📋 ÖNERİLER:")
    
    if not gpu_status['cuda_available']:
        print("❌ CUDA kurulumu gerekli")
        print("   - NVIDIA driver'ları güncelleyin")
        print("   - CUDA Toolkit kurun")
    
    if not gpu_status['xgboost_gpu']:
        print("❌ XGBoost GPU desteği eksik")
        print("   - GPU destekli XGBoost kurun")
    
    if not gpu_status['lightgbm_gpu']:
        print("❌ LightGBM GPU desteği eksik")
        print("   - GPU destekli LightGBM kurun")
    
    if gpu_status['cuda_available'] and gpu_status['xgboost_gpu']:
        print("✅ GPU desteği hazır! main_optimized.py çalıştırabilirsiniz.")
    
    print("\n🎉 Test tamamlandı!")

if __name__ == "__main__":
    main()
