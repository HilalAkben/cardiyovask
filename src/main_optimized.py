"""
Kalp Krizi Risk Tahmin Modeli - Optimizasyon Ana Çalıştırma Dosyası
Sadece Gradient Boosting ve XGBoost için - Outlier'lı Verilerle
GPU Desteği ile
"""

import sys
import pickle
import os
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.advanced_feature_engineering import AdvancedFeatureEngineer
from analysis.hyperparameter_tuning import HyperparameterTuner
from analysis.ensemble_methods import EnsembleMethods
from utils.data_loader import DataLoader

def check_gpu_availability():
    """GPU kullanılabilirliğini kontrol et ve yapılandır."""
    gpu_config = {
        'xgboost_gpu': False,
        'lightgbm_gpu': False,
        'cuda_available': False,
        'gpu_count': 0
    }
    
    print("\n" + "="*60)
    print("GPU KULLANILABİLİRLİK KONTROLÜ")
    print("="*60)
    
    # CUDA kontrolü
    try:
        import torch
        if torch.cuda.is_available():
            gpu_config['cuda_available'] = True
            gpu_config['gpu_count'] = torch.cuda.device_count()
            print(f"✅ CUDA kullanılabilir - {gpu_config['gpu_count']} GPU bulundu")
            
            for i in range(gpu_config['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA kullanılamıyor")
    except ImportError:
        print("❌ PyTorch yüklü değil - CUDA kontrolü yapılamıyor")
    
    # XGBoost GPU kontrolü
    try:
        import xgboost as xgb
        # XGBoost'un GPU desteğini kontrol et
        try:
            # GPU parametresi ile test et
            test_data = xgb.DMatrix([[1, 2, 3], [4, 5, 6]], label=[0, 1])
            test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            bst = xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
            gpu_config['xgboost_gpu'] = True
            print("✅ XGBoost GPU desteği aktif")
        except Exception as e:
            print(f"❌ XGBoost GPU desteği yok: {str(e)}")
    except ImportError:
        print("❌ XGBoost yüklü değil")
    
    # LightGBM GPU kontrolü
    try:
        import lightgbm as lgb
        # LightGBM'in GPU desteğini kontrol et
        try:
            test_data = lgb.Dataset([[1, 2, 3], [4, 5, 6]], label=[0, 1])
            test_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
            gbm = lgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
            gpu_config['lightgbm_gpu'] = True
            print("✅ LightGBM GPU desteği aktif")
        except Exception as e:
            print(f"❌ LightGBM GPU desteği yok: {str(e)}")
    except ImportError:
        print("❌ LightGBM yüklü değil")
    
    return gpu_config

def configure_gpu_parameters(gpu_config):
    """GPU parametrelerini yapılandır."""
    gpu_params = {}
    
    if gpu_config['xgboost_gpu']:
        gpu_params['xgboost'] = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'predictor': 'gpu_predictor'
        }
        print("✅ XGBoost GPU parametreleri yapılandırıldı")
    
    if gpu_config['lightgbm_gpu']:
        gpu_params['lightgbm'] = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'gpu_use_dp': True
        }
        print("✅ LightGBM GPU parametreleri yapılandırıldı")
    
    return gpu_params

def save_models_to_pickle(tuner, ensemble, data, tuning_results, ensemble_results):
    """Eğitilmiş modelleri pickle formatında kaydet."""
    
    # Models klasörü oluştur
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("MODELLERİ PICKLE FORMATINDA KAYDETME")
    print("="*60)
    
    # 1. Hyperparameter Tuning Modelleri
    print("\n--- HYPERPARAMETER TUNING MODELLERİ KAYDEDİLİYOR ---")
    
    for name, model in tuner.best_models.items():
        filename = f"tuning_{name.lower().replace(' ', '_')}.pkl"
        filepath = models_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✅ {name} kaydedildi: {filename}")
    
    # 2. Ensemble Modelleri
    print("\n--- ENSEMBLE MODELLERİ KAYDEDİLİYOR ---")
    
    for name, model in ensemble.ensemble_models.items():
        filename = f"ensemble_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        filepath = models_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✅ {name} kaydedildi: {filename}")
    
    # 3. En İyi Modelleri Ayrıca Kaydet
    print("\n--- EN İYİ MODELLERİ KAYDETME ---")
    
    # En iyi tuning modeli
    best_tuning = max(tuning_results.keys(), 
                     key=lambda x: tuning_results[x]['accuracy'])
    
    # En iyi ensemble modeli
    best_ensemble = ensemble_results['best_model_name']
    
    # En iyi modelleri kaydet
    best_models = {
        'best_tuning': tuner.best_models[best_tuning],
        'best_ensemble': ensemble.ensemble_models[best_ensemble]
    }
    
    for name, model in best_models.items():
        filename = f"{name}.pkl"
        filepath = models_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✅ {name} kaydedildi: {filename}")
    
    # 4. Feature Engineering Bilgilerini Kaydet
    print("\n--- FEATURE ENGINEERING BİLGİLERİ KAYDEDİLİYOR ---")
    
    fe_info = {
        'feature_names': data['feature_names'],
        'scaler': data.get('scaler', None),
        'label_encoders': data.get('label_encoders', {}),
        'data_type': 'with_outliers'
    }
    
    with open(models_dir / "feature_engineering.pkl", 'wb') as f:
        pickle.dump(fe_info, f)
    print(f"  ✅ Feature engineering bilgileri kaydedildi")
    
    # 5. Model Performans Sonuçlarını Kaydet
    print("\n--- MODEL PERFORMANS SONUÇLARI KAYDEDİLİYOR ---")
    
    performance_results = {
        'tuning_results': tuning_results,
        'ensemble_results': ensemble_results,
        'best_tuning_model': best_tuning,
        'best_ensemble_model': best_ensemble,
        'best_tuning_accuracy': tuning_results[best_tuning]['accuracy'],
        'best_ensemble_accuracy': ensemble_results['best_accuracy']
    }
    
    with open(models_dir / "model_performance_results.pkl", 'wb') as f:
        pickle.dump(performance_results, f)
    print(f"  ✅ Model performans sonuçları kaydedildi")
    
    print(f"\n📁 Tüm modeller '{models_dir}' klasörüne kaydedildi!")
    print(f"📊 Toplam kaydedilen dosya sayısı: {len(list(models_dir.glob('*.pkl')))}")
    
    return models_dir

def main():
    """Ana fonksiyon - Gradient Boosting ve XGBoost için optimizasyon (outlier'lı verilerle) - GPU Desteği ile."""
    print("="*80)
    print("KALP KRİZİ RİSK TAHMİN MODELİ - GB & XGBOOST OPTİMİZASYON")
    print("OUTLIER'LI VERİLERLE - GPU DESTEĞİ İLE")
    print("="*80)
    
    # GPU kontrolü ve yapılandırması
    gpu_config = check_gpu_availability()
    gpu_params = configure_gpu_parameters(gpu_config)
    
    # 1. Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        print("Lütfen veri dosyasının doğru konumda olduğundan emin olun.")
        return
    
    print(f"Veri dosyası: {data_path}")
    
    # 2. GELİŞMİŞ FEATURE ENGINEERING
    print("\n" + "="*60)
    print("1. GELİŞMİŞ FEATURE ENGINEERING")
    print("="*60)
    
    advanced_fe = AdvancedFeatureEngineer()
    
    # Outlier'lı verilerle işleme
    print("\n--- GELİŞMİŞ FEATURE ENGINEERING ---")
    data = advanced_fe.advanced_pipeline_with_outliers(str(data_path))
    if data is None:
        print("Veri işleme başarısız!")
        return
    
    print(f"\nFeature Engineering Tamamlandı!")
    print(f"Feature sayısı: {data['X_train'].shape[1]}")
    print(f"Eğitim seti: {data['X_train'].shape}")
    print(f"Test seti: {data['X_test'].shape}")
    
    # 3. HYPERPARAMETER TUNING - GPU DESTEĞİ İLE
    print("\n" + "="*60)
    print("2. HYPERPARAMETER TUNING - GRADIENT BOOSTING & XGBOOST (GPU)")
    print("="*60)
    
    tuner = HyperparameterTuner()
    
    print("\n--- HYPERPARAMETER TUNING (GPU DESTEĞİ İLE) ---")
    
    # GPU parametrelerini hyperparameter tuning'e geçir
    best_models = tuner.tune_all_models(
        data['X_train'],
        data['y_train'],
        cv=3,
        n_jobs=-1,
        gpu_params=gpu_params  # GPU parametrelerini geçir
    )
    
    # 4. TUNE EDİLMİŞ MODELLERİ DEĞERLENDİR
    print("\n" + "="*60)
    print("3. TUNE EDİLMİŞ MODELLERİN DEĞERLENDİRİLMESİ")
    print("="*60)
    
    print("\n--- TUNE EDİLMİŞ MODELLERİN DEĞERLENDİRİLMESİ ---")
    tuning_results = tuner.evaluate_tuned_models(
        data['X_test'],
        data['y_test']
    )
    
    # 5. ENSEMBLE METHODS - GPU DESTEĞİ İLE
    print("\n" + "="*60)
    print("4. ENSEMBLE METHODS - GRADIENT BOOSTING & XGBOOST (GPU)")
    print("="*60)
    
    ensemble = EnsembleMethods()
    
    print("\n--- ENSEMBLE METHODS (GPU DESTEĞİ İLE) ---")
    ensemble_results = ensemble.run_ensemble_analysis(
        data['X_train'],
        data['X_test'],
        data['y_train'],
        data['y_test'],
        gpu_params=gpu_params  # GPU parametrelerini geçir
    )
    
    # 6. FİNAL SONUÇLAR
    print("\n" + "="*80)
    print("FİNAL OPTİMİZASYON SONUÇLARI (GPU DESTEĞİ İLE)")
    print("="*80)
    
    # GPU kullanım bilgileri
    print(f"\nGPU KULLANIM BİLGİLERİ:")
    print(f"CUDA Kullanılabilir: {'✅' if gpu_config['cuda_available'] else '❌'}")
    print(f"GPU Sayısı: {gpu_config['gpu_count']}")
    print(f"XGBoost GPU: {'✅' if gpu_config['xgboost_gpu'] else '❌'}")
    print(f"LightGBM GPU: {'✅' if gpu_config['lightgbm_gpu'] else '❌'}")
    
    # Veri seti bilgileri
    print(f"\nVERİ SETİ BİLGİLERİ:")
    print(f"Eğitim seti: {data['X_train'].shape}")
    print(f"Test seti: {data['X_test'].shape}")
    print(f"Feature sayısı: {data['X_train'].shape[1]}")
    
    # Hyperparameter Tuning sonuçları
    print(f"\nHYPERPARAMETER TUNING SONUÇLARI:")
    print("-" * 60)
    
    # En iyi tuning modeli
    best_tuning = max(tuning_results.keys(), 
                     key=lambda x: tuning_results[x]['accuracy'])
    best_tuning_acc = tuning_results[best_tuning]['accuracy']
    
    print(f"En iyi tuning modeli: {best_tuning}")
    print(f"En iyi accuracy: {best_tuning_acc:.4f}")
    
    # Ensemble Methods sonuçları
    print(f"\nENSEMBLE METHODS SONUÇLARI:")
    print("-" * 60)
    
    best_ensemble = ensemble_results['best_model_name']
    best_ensemble_acc = ensemble_results['best_accuracy']
    
    print(f"En iyi ensemble modeli: {best_ensemble}")
    print(f"En iyi accuracy: {best_ensemble_acc:.4f}")
    
    # 7. KARŞILAŞTIRMALI ÖZET
    print(f"\n" + "="*60)
    print("KARŞILAŞTIRMALI ÖZET")
    print("="*60)
    
    print(f"Hyperparameter Tuning (En iyi): {best_tuning_acc:.4f}")
    print(f"Ensemble Methods (En iyi): {best_ensemble_acc:.4f}")
    
    if best_ensemble_acc > best_tuning_acc:
        print(f"🏆 En iyi sonuç: {best_ensemble} ({best_ensemble_acc:.4f})")
        print(f"Ensemble methods, hyperparameter tuning'den {(best_ensemble_acc - best_tuning_acc)*100:.2f}% daha iyi!")
    else:
        print(f"🏆 En iyi sonuç: {best_tuning} ({best_tuning_acc:.4f})")
        print(f"Hyperparameter tuning, ensemble methods'dan {(best_tuning_acc - best_ensemble_acc)*100:.2f}% daha iyi!")
    
    # 8. DETAYLI SONUÇ TABLOLARI
    print(f"\n" + "="*80)
    print("DETAYLI SONUÇ TABLOLARI")
    print("="*80)
    
    import pandas as pd
    
    # Hyperparameter Tuning sonuçları tablosu
    print(f"\nHYPERPARAMETER TUNING SONUÇLARI:")
    print("-" * 60)
    
    tuning_df = pd.DataFrame({
        'Model': list(tuning_results.keys()),
        'Accuracy': [tuning_results[name]['accuracy'] for name in tuning_results.keys()],
        'Precision': [tuning_results[name]['precision'] for name in tuning_results.keys()],
        'Recall': [tuning_results[name]['recall'] for name in tuning_results.keys()],
        'F1-Score': [tuning_results[name]['f1'] for name in tuning_results.keys()],
        'ROC AUC': [tuning_results[name]['roc_auc'] for name in tuning_results.keys()]
    })
    print(tuning_df.to_string(index=False))
    
    # Ensemble sonuçları tablosu
    print(f"\nENSEMBLE METHODS SONUÇLARI:")
    print("-" * 60)
    
    ensemble_df = ensemble_results['results_df']
    print(ensemble_df.to_string(index=False))
    
    # 9. MODELLERİ PICKLE FORMATINDA KAYDET
    models_dir = save_models_to_pickle(
        tuner, ensemble, data, tuning_results, ensemble_results
    )
    
    # 10. GPU PERFORMANS BİLGİLERİ
    print("\n" + "="*80)
    print("GPU PERFORMANS BİLGİLERİ")
    print("="*80)
    
    if gpu_config['cuda_available']:
        try:
            import torch
            for i in range(gpu_config['gpu_count']):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i} Bellek Kullanımı:")
                print(f"  Ayrılan: {memory_allocated:.2f} GB")
                print(f"  Rezerve: {memory_reserved:.2f} GB")
        except:
            pass
    
    print("\n" + "="*80)
    print("OPTİMİZASYON PIPELINE TAMAMLANDI! (GPU DESTEĞİ İLE)")
    print("="*80)
    print("Oluşturulan dosyalar:")
    print("📊 Grafikler:")
    print("  - hyperparameter_tuning_results.png")
    print("  - ensemble_comparison.png")
    print("  - cv_comparison.png")
    print(f"🤖 Modeller ({models_dir} klasöründe):")
    print("  - tuning_gradient_boosting.pkl")
    print("  - tuning_xgboost.pkl")
    print("  - ensemble_voting_soft.pkl")
    print("  - ensemble_voting_hard.pkl")
    print("  - ensemble_weighted_average.pkl")
    print("  - best_tuning.pkl")
    print("  - best_ensemble.pkl")
    print("  - feature_engineering.pkl")
    print("  - model_performance_results.pkl")
    print("\n🎯 En iyi modeli kullanarak tahmin yapabilirsin!")
    print("🚀 GPU desteği ile hızlandırılmış eğitim tamamlandı!")

if __name__ == "__main__":
    main()