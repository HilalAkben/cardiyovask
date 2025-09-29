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
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

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
    
    # 3.1 Olasılık Temelli Metrikler ve Kalibrasyon (Brier, Log Loss, Calibration Curve)
    def compute_probability_metrics(results_dict, y_test, tag):
        print("\n" + "="*50)
        print(f"OLASILIK TEMELLİ METRİKLER VE KALİBRASYON ({tag})")
        print("="*50)

        # Kalibrasyon eğrileri için figür
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Mükemmel Kalibrasyon')

        for model_name, splits in results_dict.items():
            y_proba = splits['test'].get('y_proba', None)
            if y_proba is None:
                continue

            try:
                brier = brier_score_loss(y_test, y_proba)
                ll = log_loss(y_test, y_proba)
                print(f"{model_name}: Brier={brier:.4f}, LogLoss={ll:.4f}")

                prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='quantile')
                plt.plot(prob_pred, prob_true, marker='o', linewidth=1.5, label=model_name)
            except Exception as e:
                print(f"{model_name}: Olasılık metrikleri hesaplanamadı -> {e}")

        plt.xlabel('Tahmin Olasılığı (Mean Predicted)')
        plt.ylabel('Gerçek Pozitif Oranı (Fraction of Positives)')
        plt.title(f'Kalibrasyon Eğrileri - {tag}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = f'calibration_{tag}.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Kalibrasyon grafiği kaydedildi: {out_path}")

    # Olasılık metriklerini hesapla (with/without outliers)
    compute_probability_metrics(outlier_results['results'], y_test_with, 'with_outliers')
    compute_probability_metrics(no_outlier_results['results'], y_test_without, 'without_outliers')

    # 4. Feature Importance'ları yazdır (tüm modeller için uygun olanlarda)
    def print_top_features(importance_dfs, tag):
        print("\n" + "="*50)
        print(f"FEATURE IMPORTANCE - {tag}")
        print("="*50)
        for model_name, imp_df in importance_dfs.items():
            print(f"\n{model_name} - Top 10 Features:")
            top10 = imp_df.head(10)
            for _, row in top10.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")

    print_top_features(outlier_results['importance_dfs'], 'with_outliers')
    print_top_features(no_outlier_results['importance_dfs'], 'without_outliers')

    # 5. Final Sonuçlar
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