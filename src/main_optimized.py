"""
Kalp Krizi Risk Tahmin Modeli - Optimizasyon Ana Çalıştırma Dosyası
Sadece Gradient Boosting ve XGBoost için - Outlier'lı ve Outlier'sız Karşılaştırması
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.advanced_feature_engineering import AdvancedFeatureEngineer
from analysis.hyperparameter_tuning import HyperparameterTuner
from analysis.ensemble_methods import EnsembleMethods
from utils.data_loader import DataLoader

def main():
    """Ana fonksiyon - Gradient Boosting ve XGBoost için outlier karşılaştırmalı optimizasyon."""
    print("="*80)
    print("KALP KRİZİ RİSK TAHMİN MODELİ - GB & XGBOOST OPTİMİZASYON")
    print("OUTLIER'LI vs OUTLIER'SIZ KARŞILAŞTIRMASI")
    print("="*80)
    
    # 1. Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        print("Lütfen veri dosyasının doğru konumda olduğundan emin olun.")
        return
    
    print(f"Veri dosyası: {data_path}")
    
    # 2. GELİŞMİŞ FEATURE ENGINEERING - İKİ FARKLI PIPELINE
    print("\n" + "="*60)
    print("1. GELİŞMİŞ FEATURE ENGINEERING")
    print("="*60)
    
    advanced_fe = AdvancedFeatureEngineer()
    
    # Outlier'lı verilerle işleme
    print("\n--- OUTLIER'LAR İLE İŞLEME ---")
    data_with_outliers = advanced_fe.advanced_pipeline_with_outliers(str(data_path))
    if data_with_outliers is None:
        print("Outlier'lı veri işleme başarısız!")
        return
    
    # Outlier'lar çıkarılarak işleme
    print("\n--- OUTLIER'LAR ÇIKARILARAK İŞLEME ---")
    data_without_outliers = advanced_fe.advanced_pipeline_without_outliers(str(data_path))
    if data_without_outliers is None:
        print("Outlier'lar çıkarılarak veri işleme başarısız!")
        return
    
    print(f"\nFeature Engineering Tamamlandı!")
    print(f"Outlier'lı veri - Feature sayısı: {data_with_outliers['X_train'].shape[1]}")
    print(f"Outlier'sız veri - Feature sayısı: {data_without_outliers['X_train'].shape[1]}")
    
    # 3. HYPERPARAMETER TUNING - İKİ VERİ SETİ İÇİN
    print("\n" + "="*60)
    print("2. HYPERPARAMETER TUNING - GRADIENT BOOSTING & XGBOOST")
    print("="*60)
    
    tuner = HyperparameterTuner()
    
    # Outlier'lı veriler için tuning
    print("\n--- OUTLIER'LI VERİLER İÇİN TUNING ---")
    best_models_with_outliers = tuner.tune_all_models(
        data_with_outliers['X_train'],
        data_with_outliers['y_train'],
        cv=5,
        n_jobs=-1
    )
    
    # Outlier'sız veriler için tuning
    print("\n--- OUTLIER'SIZ VERİLER İÇİN TUNING ---")
    best_models_without_outliers = tuner.tune_all_models(
        data_without_outliers['X_train'],
        data_without_outliers['y_train'],
        cv=5,
        n_jobs=-1
    )
    
    # 4. TUNE EDİLMİŞ MODELLERİ DEĞERLENDİR
    print("\n" + "="*60)
    print("3. TUNE EDİLMİŞ MODELLERİN DEĞERLENDİRİLMESİ")
    print("="*60)
    
    # Outlier'lı veriler için değerlendirme
    print("\n--- OUTLIER'LI VERİLER İÇİN DEĞERLENDİRME ---")
    tuning_results_with_outliers = tuner.evaluate_tuned_models(
        data_with_outliers['X_test'],
        data_with_outliers['y_test']
    )
    
    # Outlier'sız veriler için değerlendirme
    print("\n--- OUTLIER'SIZ VERİLER İÇİN DEĞERLENDİRME ---")
    tuning_results_without_outliers = tuner.evaluate_tuned_models(
        data_without_outliers['X_test'],
        data_without_outliers['y_test']
    )
    
    # 5. ENSEMBLE METHODS - İKİ VERİ SETİ İÇİN
    print("\n" + "="*60)
    print("4. ENSEMBLE METHODS - GRADIENT BOOSTING & XGBOOST")
    print("="*60)
    
    ensemble = EnsembleMethods()
    
    # Outlier'lı veriler için ensemble
    print("\n--- OUTLIER'LI VERİLER İÇİN ENSEMBLE ---")
    ensemble_results_with_outliers = ensemble.run_ensemble_analysis(
        data_with_outliers['X_train'],
        data_with_outliers['X_test'],
        data_with_outliers['y_train'],
        data_with_outliers['y_test']
    )
    
    # Outlier'sız veriler için ensemble
    print("\n--- OUTLIER'SIZ VERİLER İÇİN ENSEMBLE ---")
    ensemble_results_without_outliers = ensemble.run_ensemble_analysis(
        data_without_outliers['X_train'],
        data_without_outliers['X_test'],
        data_without_outliers['y_train'],
        data_without_outliers['y_test']
    )
    
    # 6. FİNAL KARŞILAŞTIRMALI SONUÇLAR
    print("\n" + "="*80)
    print("FİNAL KARŞILAŞTIRMALI OPTİMİZASYON SONUÇLARI")
    print("="*80)
    
    # Veri seti bilgileri
    print(f"\nVERİ SETİ BİLGİLERİ:")
    print(f"Outlier'lı veri - Eğitim: {data_with_outliers['X_train'].shape}, Test: {data_with_outliers['X_test'].shape}")
    print(f"Outlier'sız veri - Eğitim: {data_without_outliers['X_train'].shape}, Test: {data_without_outliers['X_test'].shape}")
    
    # Hyperparameter Tuning sonuçları
    print(f"\nHYPERPARAMETER TUNING SONUÇLARI:")
    print("-" * 60)
    
    # Outlier'lı veriler için en iyi model
    best_tuning_with_outliers = max(tuning_results_with_outliers.keys(), 
                                   key=lambda x: tuning_results_with_outliers[x]['accuracy'])
    best_tuning_acc_with_outliers = tuning_results_with_outliers[best_tuning_with_outliers]['accuracy']
    
    # Outlier'sız veriler için en iyi model
    best_tuning_without_outliers = max(tuning_results_without_outliers.keys(), 
                                      key=lambda x: tuning_results_without_outliers[x]['accuracy'])
    best_tuning_acc_without_outliers = tuning_results_without_outliers[best_tuning_without_outliers]['accuracy']
    
    print(f"Outlier'lı veri - En iyi: {best_tuning_with_outliers} ({best_tuning_acc_with_outliers:.4f})")
    print(f"Outlier'sız veri - En iyi: {best_tuning_without_outliers} ({best_tuning_acc_without_outliers:.4f})")
    
    # Ensemble Methods sonuçları
    print(f"\nENSEMBLE METHODS SONUÇLARI:")
    print("-" * 60)
    
    best_ensemble_with_outliers = ensemble_results_with_outliers['best_model_name']
    best_ensemble_acc_with_outliers = ensemble_results_with_outliers['best_accuracy']
    
    best_ensemble_without_outliers = ensemble_results_without_outliers['best_model_name']
    best_ensemble_acc_without_outliers = ensemble_results_without_outliers['best_accuracy']
    
    print(f"Outlier'lı veri - En iyi: {best_ensemble_with_outliers} ({best_ensemble_acc_with_outliers:.4f})")
    print(f"Outlier'sız veri - En iyi: {best_ensemble_without_outliers} ({best_ensemble_acc_without_outliers:.4f})")
    
    # 7. KARŞILAŞTIRMALI ÖZET
    print(f"\n" + "="*60)
    print("KARŞILAŞTIRMALI ÖZET")
    print("="*60)
    
    print(f"\nOUTLIER'LI VERİLER:")
    print(f"  Hyperparameter Tuning: {best_tuning_acc_with_outliers:.4f}")
    print(f"  Ensemble Methods: {best_ensemble_acc_with_outliers:.4f}")
    
    print(f"\nOUTLIER'SIZ VERİLER:")
    print(f"  Hyperparameter Tuning: {best_tuning_acc_without_outliers:.4f}")
    print(f"  Ensemble Methods: {best_ensemble_acc_without_outliers:.4f}")
    
    # En iyi genel sonuç
    all_accuracies = {
        'Outlier\'lı - Tuning': best_tuning_acc_with_outliers,
        'Outlier\'lı - Ensemble': best_ensemble_acc_with_outliers,
        'Outlier\'sız - Tuning': best_tuning_acc_without_outliers,
        'Outlier\'sız - Ensemble': best_ensemble_acc_without_outliers
    }
    
    best_overall = max(all_accuracies.keys(), key=lambda x: all_accuracies[x])
    best_overall_acc = all_accuracies[best_overall]
    
    print(f"\n EN İYİ GENEL SONUÇ:")
    print(f"  {best_overall}: {best_overall_acc:.4f}")
    
    # Outlier etkisi analizi
    print(f"\nOUTLIER ETKİSİ ANALİZİ:")
    tuning_diff = best_tuning_acc_without_outliers - best_tuning_acc_with_outliers
    ensemble_diff = best_ensemble_acc_without_outliers - best_ensemble_acc_with_outliers
    
    print(f"  Hyperparameter Tuning: Outlier'sız veri {tuning_diff:+.4f} farkla {'daha iyi' if tuning_diff > 0 else 'daha kötü'}")
    print(f"  Ensemble Methods: Outlier'sız veri {ensemble_diff:+.4f} farkla {'daha iyi' if ensemble_diff > 0 else 'daha kötü'}")
    
    # 8. DETAYLI SONUÇ TABLOLARI
    print(f"\n" + "="*80)
    print("DETAYLI SONUÇ TABLOLARI")
    print("="*80)
    
    import pandas as pd
    
    # Hyperparameter Tuning sonuçları tablosu
    print(f"\nHYPERPARAMETER TUNING SONUÇLARI:")
    print("-" * 60)
    
    tuning_comparison = pd.DataFrame({
        'Model': list(tuning_results_with_outliers.keys()),
        'Outlier\'lı Accuracy': [tuning_results_with_outliers[name]['accuracy'] for name in tuning_results_with_outliers.keys()],
        'Outlier\'sız Accuracy': [tuning_results_without_outliers[name]['accuracy'] for name in tuning_results_without_outliers.keys()],
        'Outlier\'lı F1': [tuning_results_with_outliers[name]['f1'] for name in tuning_results_with_outliers.keys()],
        'Outlier\'sız F1': [tuning_results_without_outliers[name]['f1'] for name in tuning_results_without_outliers.keys()],
        'Outlier\'lı ROC AUC': [tuning_results_with_outliers[name]['roc_auc'] for name in tuning_results_with_outliers.keys()],
        'Outlier\'sız ROC AUC': [tuning_results_without_outliers[name]['roc_auc'] for name in tuning_results_without_outliers.keys()]
    })
    print(tuning_comparison.to_string(index=False))
    
    # Ensemble sonuçları tablosu
    print(f"\nENSEMBLE METHODS SONUÇLARI:")
    print("-" * 60)
    
    ensemble_comparison = pd.DataFrame({
        'Model': list(ensemble_results_with_outliers['results_df']['Model']),
        'Outlier\'lı Accuracy': ensemble_results_with_outliers['results_df']['Accuracy'],
        'Outlier\'sız Accuracy': ensemble_results_without_outliers['results_df']['Accuracy'],
        'Outlier\'lı F1': ensemble_results_with_outliers['results_df']['F1-Score'],
        'Outlier\'sız F1': ensemble_results_without_outliers['results_df']['F1-Score'],
        'Outlier\'lı ROC AUC': ensemble_results_with_outliers['results_df']['ROC AUC'],
        'Outlier\'sız ROC AUC': ensemble_results_without_outliers['results_df']['ROC AUC']
    })
    print(ensemble_comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("OPTİMİZASYON PIPELINE TAMAMLANDI!")
    print("="*80)
    print("Oluşturulan grafikler:")
    print("- hyperparameter_tuning_results.png")
    print("- ensemble_comparison.png")
    print("- cv_comparison.png")
    print("\nEn iyi modeli kullanarak tahmin yapabilirsin!")

if __name__ == "__main__":
    main() 