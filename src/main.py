"""
Kalp Krizi Risk Tahmin Modeli - Ana Çalıştırma Dosyası
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.feature_engineering import FeatureEngineer
from analysis.data_analysis import ComparativeAnalyzer
from utils.data_loader import DataLoader

def main():
    """Ana fonksiyon - Tüm pipeline'ı çalıştır."""
    print("="*80)
    print("KALP KRİZİ RİSK TAHMİN MODELİ - KARŞILAŞTIRMALI ANALİZ PIPELINE")
    print("="*80)
    
    # 1. Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        print("Lütfen veri dosyasının doğru konumda olduğundan emin olun.")
        return
    
    print(f"Veri dosyası: {data_path}")
    
    # 2. Feature Engineering - İki farklı pipeline
    print("\n" + "="*50)
    print("1. VERİ İŞLEME VE FEATURE ENGINEERING")
    print("="*50)
    
    fe = FeatureEngineer()
    
    # Outlier'lı verilerle işleme
    print("\n--- Outlier'lı Verilerle İşleme ---")
    data_with_outliers = fe.process_pipeline_with_outliers(str(data_path))
    if data_with_outliers is None:
        print("Outlier'lı veri işleme başarısız!")
        return
    
    # Outlier'lar çıkarılarak işleme
    print("\n--- Outlier'lar Çıkarılarak İşleme ---")
    data_without_outliers = fe.process_pipeline_without_outliers(str(data_path))
    if data_without_outliers is None:
        print("Outlier'lar çıkarılarak veri işleme başarısız!")
        return
    
    # 3. Karşılaştırmalı Model Analizi
    print("\n" + "="*50)
    print("2. KARŞILAŞTIRMALI MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("="*50)
    
    comparative_analyzer = ComparativeAnalyzer()
    outlier_results, no_outlier_results = comparative_analyzer.run_comparative_analysis(
        data_with_outliers, data_without_outliers
    )
    
    # 4. Final Sonuçlar
    print("\n" + "="*80)
    print("FİNAL KARŞILAŞTIRMALI SONUÇLAR")
    print("="*80)
    
    print(f"\nOUTLIER'LAR İLE:")
    print(f"X_train.shape: {data_with_outliers['X_train'].shape}")
    print(f"X_test.shape: {data_with_outliers['X_test'].shape}")
    print(f"En iyi model: {outlier_results['best_model_name']}")
    print(f"En iyi F1-Score: {outlier_results['best_score']:.4f}")
    
    # Outlier sayıları
    print(f"\nOutlier Analizi (çıkarılmadan):")
    for col, count in data_with_outliers['outlier_counts'].items():
        print(f"  {col}: {count} outlier")
    
    print(f"\nOUTLIER'LAR ÇIKARILARAK:")
    print(f"X_train.shape: {data_without_outliers['X_train'].shape}")
    print(f"X_test.shape: {data_without_outliers['X_test'].shape}")
    print(f"En iyi model: {no_outlier_results['best_model_name']}")
    print(f"En iyi F1-Score: {no_outlier_results['best_score']:.4f}")
    
    # Outlier sayıları (çıkarıldıktan sonra)
    print(f"\nOutlier Analizi (çıkarıldıktan sonra):")
    for col, count in data_without_outliers['outlier_counts'].items():
        print(f"  {col}: {count} outlier (çıkarıldı)")
    
    # Model performans karşılaştırması
    print(f"\nModel Performans Karşılaştırması (Outlier'lı):")
    print(outlier_results['metrics_df'].to_string(index=False))
    
    print(f"\nModel Performans Karşılaştırması (Outlier'lar çıkarılmış):")
    print(no_outlier_results['metrics_df'].to_string(index=False))
    
    # Feature importance (Random Forest ve XGBoost için)
    if outlier_results['importance_dfs']:
        print(f"\nEn Önemli 10 Feature - Outlier'lı Veriler (Random Forest):")
        if 'Random Forest' in outlier_results['importance_dfs']:
            top_features = outlier_results['importance_dfs']['Random Forest'].head(10)
            for idx, row in top_features.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    if no_outlier_results['importance_dfs']:
        print(f"\nEn Önemli 10 Feature - Outlier'lar çıkarılmış (Random Forest):")
        if 'Random Forest' in no_outlier_results['importance_dfs']:
            top_features = no_outlier_results['importance_dfs']['Random Forest'].head(10)
            for idx, row in top_features.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n" + "="*80)
    print("KARŞILAŞTIRMALI ANALİZ PIPELINE TAMAMLANDI!")
    print("="*80)
    print("Grafikler proje dizinine kaydedildi:")
    print("- confusion_matrices.png (her iki veri seti için)")
    print("- roc_curves.png (her iki veri seti için)")
    print("- metrics_comparison.png (her iki veri seti için)")
    print("- rf_feature_importance.png (her iki veri seti için)")
    print("- xgb_feature_importance.png (her iki veri seti için)")
    print("- comparative_analysis.png (karşılaştırmalı sonuçlar)")

if __name__ == "__main__":
    main() 