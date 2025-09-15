"""
Kalp Krizi Risk Tahmin Modeli - Ana Çalıştırma Dosyası
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.preprocessor import DataPreprocessor
from analysis.data_analysis import ComparativeAnalyzer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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
    
    # 2. Veri Ön İşleme - İki farklı pipeline
    print("\n" + "="*50)
    print("1. VERİ ÖN İŞLEME VE FEATURE ENGINEERING")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    
    # Outlier'lı verilerle işleme
    print("\n--- Outlier'lı Verilerle İşleme ---")
    data_with_outliers = preprocessor.complete_preprocessing_pipeline(
        str(data_path), remove_outliers=False
    )
    if data_with_outliers is None:
        print("Outlier'lı veri işleme başarısız!")
        return
    
    # Outlier'lar çıkarılarak işleme
    print("\n--- Outlier'lar Çıkarılarak İşleme ---")
    data_without_outliers = preprocessor.complete_preprocessing_pipeline(
        str(data_path), remove_outliers=True
    )
    if data_without_outliers is None:
        print("Outlier'lar çıkarılarak veri işleme başarısız!")
        return
    
    # Veri hazırlama (encoding ve scaling)
    def prepare_ml_data(df):
        """Veriyi makine öğrenmesi için hazırla."""
        df_ml = df.copy()
        
        # Kategorik değişkenleri encode et
        categorical_columns = ['gender', 'smoke', 'alco', 'active']
        for col in categorical_columns:
            if col in df_ml.columns:
                le = LabelEncoder()
                df_ml[col] = le.fit_transform(df_ml[col])
        
        # Hedef değişkeni ayır
        y = df_ml['cardio']
        X = df_ml.drop(['id', 'age', 'cardio'], axis=1, errors='ignore')
        
        # Sürekli değişkenleri ölçekle
        continuous_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        available_continuous = [col for col in continuous_columns if col in X.columns]
        
        if available_continuous:
            scaler = StandardScaler()
            X[available_continuous] = scaler.fit_transform(X[available_continuous])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, list(X.columns)
    
    # Veri hazırlama
    X_train_with, X_test_with, y_train_with, y_test_with, feature_names_with = prepare_ml_data(data_with_outliers['data'])
    X_train_without, X_test_without, y_train_without, y_test_without, feature_names_without = prepare_ml_data(data_without_outliers['data'])
    
    # Model eğitimi için veri yapısını hazırla
    data_with_outliers_ml = {
        'X_train': X_train_with,
        'X_test': X_test_with,
        'y_train': y_train_with,
        'y_test': y_test_with,
        'feature_names': feature_names_with
    }
    
    data_without_outliers_ml = {
        'X_train': X_train_without,
        'X_test': X_test_without,
        'y_train': y_train_without,
        'y_test': y_test_without,
        'feature_names': feature_names_without
    }
    
    # 3. Karşılaştırmalı Model Analizi
    print("\n" + "="*50)
    print("2. KARŞILAŞTIRMALI MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("="*50)
    
    comparative_analyzer = ComparativeAnalyzer()
    outlier_results, no_outlier_results = comparative_analyzer.run_comparative_analysis(
        data_with_outliers_ml, data_without_outliers_ml
    )
    
    # 4. Final Sonuçlar
    print("\n" + "="*80)
    print("FİNAL KARŞILAŞTIRMALI SONUÇLAR")
    print("="*80)
    
    print(f"\nOUTLIER'LAR İLE:")
    print(f"X_train.shape: {data_with_outliers_ml['X_train'].shape}")
    print(f"X_test.shape: {data_with_outliers_ml['X_test'].shape}")
    print(f"En iyi model: {outlier_results['best_model_name']}")
    print(f"En iyi F1-Score: {outlier_results['best_score']:.4f}")
    
    # Outlier sayıları
    print(f"\nOutlier Analizi (çıkarılmadan):")
    for col, info in data_with_outliers['outlier_summary'].items():
        if isinstance(info, dict) and 'outlier_count' in info:
            print(f"  {col}: {info['outlier_count']} outlier")
    
    print(f"\nOUTLIER'LAR ÇIKARILARAK:")
    print(f"X_train.shape: {data_without_outliers_ml['X_train'].shape}")
    print(f"X_test.shape: {data_without_outliers_ml['X_test'].shape}")
    print(f"En iyi model: {no_outlier_results['best_model_name']}")
    print(f"En iyi F1-Score: {no_outlier_results['best_score']:.4f}")
    
    # Outlier sayıları (çıkarıldıktan sonra)
    print(f"\nOutlier Analizi (çıkarıldıktan sonra):")
    for col, info in data_without_outliers['outlier_summary'].items():
        if isinstance(info, dict) and 'outlier_count' in info:
            print(f"  {col}: {info['outlier_count']} outlier (çıkarıldı)")
    
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