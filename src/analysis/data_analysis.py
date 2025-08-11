"""
Kalp Krizi Risk Tahmin Modeli - Veri Analizi ve Modelleme Modülü
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost bulunamadı. XGBoost modelleri çalıştırılmayacak.")
    XGBOOST_AVAILABLE = False

# Türkçe karakter desteği için
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelAnalyzer:
    """Model eğitimi ve değerlendirme sınıfı."""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # XGBoost ekle
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                random_state=42, 
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                eval_metric='logloss'
            )
        
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_models(self, X_train, y_train):
        """Tüm modelleri eğit."""
        print("=== MODEL EĞİTİMİ ===")
        
        for name, model in self.models.items():
            print(f"\n{name} eğitiliyor...")
            model.fit(X_train, y_train)
            print(f"{name} eğitimi tamamlandı.")
        
        print("\nTüm modeller eğitildi!")
    
    def evaluate_models(self, X_test, y_test):
        """Modelleri değerlendir ve metrikleri hesapla."""
        print("\n=== MODEL DEĞERLENDİRMESİ ===")
        
        for name, model in self.models.items():
            print(f"\n{name} değerlendiriliyor...")
            
            # Tahminler
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrikler
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            
            # Sonuçları sakla
            self.results[name] = {
                'y_pred': y_pred,
                'y_proba': y_proba,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if roc_auc:
                print(f"ROC AUC: {roc_auc:.4f}")
    
    def find_best_model(self):
        """En iyi modeli belirle."""
        print("\n=== EN İYİ MODEL SEÇİMİ ===")
        
        best_score = 0
        best_name = None
        
        for name, metrics in self.results.items():
            # F1-score'u ana kriter olarak kullan
            score = metrics['f1']
            print(f"{name}: F1-Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"\nEn iyi model: {best_name} (F1-Score: {best_score:.4f})")
        
        return best_name, best_score
    
    def plot_confusion_matrices(self):
        """Confusion matrix'leri görselleştir."""
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
            axes[row, col].set_title(f'{name} - Confusion Matrix')
            axes[row, col].set_xlabel('Tahmin Edilen')
            axes[row, col].set_ylabel('Gerçek')
        
        # Boş subplot'ları gizle
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_test):
        """ROC eğrilerini görselleştir."""
        plt.figure(figsize=(10, 8))
        
        for name, metrics in self.results.items():
            if metrics['y_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_test, metrics['y_proba'])
                auc = metrics['roc_auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Eğrileri')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self):
        """Model metriklerini karşılaştır."""
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[name]['accuracy'] for name in self.results.keys()],
            'Precision': [self.results[name]['precision'] for name in self.results.keys()],
            'Recall': [self.results[name]['recall'] for name in self.results.keys()],
            'F1-Score': [self.results[name]['f1'] for name in self.results.keys()],
            'ROC AUC': [self.results[name]['roc_auc'] for name in self.results.keys()]
        })
        
        # Metrikleri görselleştir
        n_metrics = 5
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        
        for idx, metric in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            
            axes[row, col].bar(metrics_df['Model'], metrics_df[metric])
            axes[row, col].set_title(f'{metric} Karşılaştırması')
            axes[row, col].set_ylabel(metric)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Değerleri çubukların üzerine yaz
            for i, v in enumerate(metrics_df[metric]):
                if not pd.isna(v):
                    axes[row, col].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Boş subplot'ları gizle
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def plot_feature_importance(self, feature_names):
        """Feature importance'ları görselleştir (Random Forest ve XGBoost için)."""
        importance_dfs = {}
        
        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            importance_dfs['Random Forest'] = importance_df
            
            # İlk 15 feature'ı göster
            top_features = importance_df.head(15)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_features, x='Importance', y='Feature')
            plt.title('Random Forest - Feature Importance (İlk 15)')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # XGBoost feature importance
        if XGBOOST_AVAILABLE and 'XGBoost' in self.models:
            xgb_model = self.models['XGBoost']
            feature_importance = xgb_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            importance_dfs['XGBoost'] = importance_df
            
            # İlk 15 feature'ı göster
            top_features = importance_df.head(15)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_features, x='Importance', y='Feature')
            plt.title('XGBoost - Feature Importance (İlk 15)')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig('xgb_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return importance_dfs
    
    def detailed_classification_report(self, model_name, y_test):
        """Detaylı sınıflandırma raporu."""
        if model_name in self.results:
            y_pred = self.results[model_name]['y_pred']
            print(f"\n=== {model_name} - DETAYLI SINIFLANDIRMA RAPORU ===")
            print(classification_report(y_test, y_pred, target_names=['Kardiyovasküler Yok', 'Kardiyovasküler Var']))
    
    def run_complete_analysis(self, X_train, X_test, y_train, y_test, feature_names, data_type=""):
        """Tam analiz pipeline'ını çalıştır."""
        print(f"=== KALP KRİZİ RİSK TAHMİN MODELİ - TAM ANALİZ ({data_type.upper()}) ===\n")
        
        # 1. Modelleri eğit
        self.train_models(X_train, y_train)
        
        # 2. Modelleri değerlendir
        self.evaluate_models(X_test, y_test)
        
        # 3. En iyi modeli bul
        best_name, best_score = self.find_best_model()
        
        # 4. Detaylı rapor
        self.detailed_classification_report(best_name, y_test)
        
        # 5. Görselleştirmeler
        print("\nGörselleştirmeler oluşturuluyor...")
        
        # Confusion matrices
        self.plot_confusion_matrices()
        
        # ROC curves
        self.plot_roc_curves(y_test)
        
        # Metrics comparison
        metrics_df = self.plot_metrics_comparison()
        
        # Feature importance
        importance_dfs = self.plot_feature_importance(feature_names)
        
        print(f"\n=== ANALİZ TAMAMLANDI ({data_type.upper()}) ===")
        print(f"En iyi model: {best_name}")
        print(f"En iyi F1-Score: {best_score:.4f}")
        
        return {
            'best_model': self.best_model,
            'best_model_name': best_name,
            'best_score': best_score,
            'metrics_df': metrics_df,
            'importance_dfs': importance_dfs,
            'results': self.results,
            'data_type': data_type
        }

class ComparativeAnalyzer:
    """Outlier'lı ve outliersız veriler için karşılaştırmalı analiz."""
    
    def __init__(self):
        self.outlier_results = None
        self.no_outlier_results = None
        
    def run_comparative_analysis(self, data_with_outliers, data_without_outliers):
        """Karşılaştırmalı analiz çalıştır."""
        print("="*80)
        print("KALP KRİZİ RİSK TAHMİN MODELİ - KARŞILAŞTIRMALI ANALİZ")
        print("="*80)
        
        # 1. Outlier'lı verilerle analiz
        print("\n" + "="*40)
        print("1. OUTLIER'LAR İLE ANALİZ")
        print("="*40)
        
        analyzer_with_outliers = ModelAnalyzer()
        self.outlier_results = analyzer_with_outliers.run_complete_analysis(
            data_with_outliers['X_train'],
            data_with_outliers['X_test'],
            data_with_outliers['y_train'],
            data_with_outliers['y_test'],
            data_with_outliers['feature_names'],
            "with_outliers"
        )
        
        # 2. Outlier'lar çıkarılarak analiz
        print("\n" + "="*40)
        print("2. OUTLIER'LAR ÇIKARILARAK ANALİZ")
        print("="*40)
        
        analyzer_without_outliers = ModelAnalyzer()
        self.no_outlier_results = analyzer_without_outliers.run_complete_analysis(
            data_without_outliers['X_train'],
            data_without_outliers['X_test'],
            data_without_outliers['y_train'],
            data_without_outliers['y_test'],
            data_without_outliers['feature_names'],
            "without_outliers"
        )
        
        # 3. Karşılaştırmalı sonuçlar
        self.plot_comparative_results()
        
        return self.outlier_results, self.no_outlier_results
    
    def plot_comparative_results(self):
        """Karşılaştırmalı sonuçları görselleştir."""
        print("\n" + "="*40)
        print("3. KARŞILAŞTIRMALI SONUÇLAR")
        print("="*40)
        
        # Metrikleri karşılaştır
        metrics_comparison = pd.DataFrame()
        
        # Outlier'lı veriler
        outlier_metrics = self.outlier_results['metrics_df'].copy()
        outlier_metrics['Data_Type'] = 'With Outliers'
        
        # Outlier'lar çıkarılmış veriler
        no_outlier_metrics = self.no_outlier_results['metrics_df'].copy()
        no_outlier_metrics['Data_Type'] = 'Without Outliers'
        
        # Birleştir
        metrics_comparison = pd.concat([outlier_metrics, no_outlier_metrics], ignore_index=True)
        
        # Karşılaştırmalı grafikler
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            if idx < len(axes):
                # Pivot table oluştur
                pivot_data = metrics_comparison.pivot(index='Model', columns='Data_Type', values=metric)
                
                pivot_data.plot(kind='bar', ax=axes[idx], width=0.8)
                axes[idx].set_title(f'{metric} Karşılaştırması')
                axes[idx].set_ylabel(metric)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
                
                # Değerleri çubukların üzerine yaz
                for i, (model, row) in enumerate(pivot_data.iterrows()):
                    for j, (data_type, value) in enumerate(row.items()):
                        if not pd.isna(value):
                            axes[idx].text(i + j*0.4 - 0.2, value + 0.01, f'{value:.3f}', 
                                         ha='center', va='bottom', fontsize=8)
        
        # Son subplot'u gizle
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Sonuçları yazdır
        print("\n=== KARŞILAŞTIRMALI SONUÇLAR TABLOSU ===")
        print(metrics_comparison.to_string(index=False))
        
        # En iyi modelleri karşılaştır
        print(f"\n=== EN İYİ MODEL KARŞILAŞTIRMASI ===")
        print(f"Outlier'lı veriler: {self.outlier_results['best_model_name']} (F1: {self.outlier_results['best_score']:.4f})")
        print(f"Outlier'lar çıkarılmış: {self.no_outlier_results['best_model_name']} (F1: {self.no_outlier_results['best_score']:.4f})")
        
        # Performans artışı/azalışı
        f1_diff = self.no_outlier_results['best_score'] - self.outlier_results['best_score']
        if f1_diff > 0:
            print(f"Outlier temizleme ile F1-Score artışı: +{f1_diff:.4f}")
        else:
            print(f"Outlier temizleme ile F1-Score azalışı: {f1_diff:.4f}")
        
        return metrics_comparison

def main():
    """Ana fonksiyon."""
    # Feature engineering modülünü import et
    import sys
    from pathlib import Path
    
    # Proje kök dizinini Python path'ine ekle
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from data.feature_engineering import FeatureEngineer
    
    # Feature engineering
    fe = FeatureEngineer()
    
    # Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    # Karşılaştırmalı analiz
    comparative_analyzer = ComparativeAnalyzer()
    
    # Outlier'lı verilerle analiz
    data_with_outliers = fe.process_pipeline_with_outliers(str(data_path))
    if data_with_outliers is None:
        print("Outlier'lı veri işleme başarısız!")
        return
    
    # Outlier'lar çıkarılarak analiz
    data_without_outliers = fe.process_pipeline_without_outliers(str(data_path))
    if data_without_outliers is None:
        print("Outlier'lar çıkarılarak veri işleme başarısız!")
        return
    
    # Karşılaştırmalı analiz çalıştır
    outlier_results, no_outlier_results = comparative_analyzer.run_comparative_analysis(
        data_with_outliers, data_without_outliers
    )
    
    # Final sonuçlar
    print("\n" + "="*80)
    print("FİNAL KARŞILAŞTIRMALI SONUÇLAR")
    print("="*80)
    
    print(f"\nOUTLIER'LAR İLE:")
    print(f"X_train.shape: {data_with_outliers['X_train'].shape}")
    print(f"X_test.shape: {data_with_outliers['X_test'].shape}")
    print(f"En iyi model: {outlier_results['best_model_name']}")
    print(f"En iyi F1-Score: {outlier_results['best_score']:.4f}")
    
    print(f"\nOUTLIER'LAR ÇIKARILARAK:")
    print(f"X_train.shape: {data_without_outliers['X_train'].shape}")
    print(f"X_test.shape: {data_without_outliers['X_test'].shape}")
    print(f"En iyi model: {no_outlier_results['best_model_name']}")
    print(f"En iyi F1-Score: {no_outlier_results['best_score']:.4f}")

if __name__ == "__main__":
    main() 