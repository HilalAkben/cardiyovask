"""
Kalp Krizi Risk Tahmin Modeli - Ensemble Methods Modülü
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost bulunamadı. XGBoost modelleri çalıştırılmayacak.")
    XGBOOST_AVAILABLE = False

class EnsembleMethods:
    """Ensemble metodları sınıfı."""
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        self.results = {}
        
    def create_base_models(self):
        """Temel modelleri oluştur."""
        self.base_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5, 
                min_samples_leaf=2, random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, 
                min_samples_split=5, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', max_iter=1000, random_state=42
            ),
            'SVM': SVC(
                C=10, kernel='rbf', gamma='scale', probability=True, random_state=42
            )
        }
        
        # XGBoost ekle
        if XGBOOST_AVAILABLE:
            self.base_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            )
        
        print(f"{len(self.base_models)} temel model oluşturuldu.")
    
    def create_voting_classifier(self, voting='soft'):
        """Voting Classifier oluştur."""
        print(f"=== VOTING CLASSIFIER ({voting.upper()}) ===")
        
        # Estimator listesi
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        self.ensemble_models[f'Voting ({voting})'] = voting_clf
        return voting_clf
    
    def create_stacking_classifier(self):
        """Stacking Classifier oluştur."""
        print("=== STACKING CLASSIFIER ===")
        
        # Estimator listesi
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42)
        
        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        self.ensemble_models['Stacking'] = stacking_clf
        return stacking_clf
    
    def create_weighted_average(self, weights=None):
        """Ağırlıklı ortalama ensemble oluştur."""
        print("=== WEIGHTED AVERAGE ENSEMBLE ===")
        
        if weights is None:
            # Eşit ağırlıklar
            weights = [1.0] * len(self.base_models)
        
        # Ağırlıkları normalize et
        weights = np.array(weights) / sum(weights)
        
        class WeightedAverageEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def fit(self, X, y):
                for model in self.models.values():
                    model.fit(X, y)
                return self
            
            def predict_proba(self, X):
                probas = []
                for model in self.models.values():
                    proba = model.predict_proba(X)[:, 1]
                    probas.append(proba)
                
                # Ağırlıklı ortalama
                weighted_proba = np.zeros(len(X))
                for i, weight in enumerate(self.weights):
                    weighted_proba += weight * probas[i]
                
                # 2 sınıf için proba formatına çevir
                return np.column_stack([1 - weighted_proba, weighted_proba])
            
            def predict(self, X):
                proba = self.predict_proba(X)[:, 1]
                return (proba > 0.5).astype(int)
        
        weighted_ensemble = WeightedAverageEnsemble(self.base_models, weights)
        self.ensemble_models['Weighted Average'] = weighted_ensemble
        return weighted_ensemble
    
    def train_all_models(self, X_train, y_train):
        """Tüm modelleri eğit."""
        print("=== TÜM MODELLERİN EĞİTİMİ ===")
        
        # Temel modelleri eğit
        for name, model in self.base_models.items():
            print(f"{name} eğitiliyor...")
            model.fit(X_train, y_train)
            print(f"{name} eğitimi tamamlandı.")
        
        # Ensemble modelleri eğit
        for name, model in self.ensemble_models.items():
            print(f"{name} eğitiliyor...")
            model.fit(X_train, y_train)
            print(f"{name} eğitimi tamamlandı.")
        
        print("Tüm modeller eğitildi!")
    
    def evaluate_all_models(self, X_test, y_test):
        """Tüm modelleri değerlendir."""
        print("=== TÜM MODELLERİN DEĞERLENDİRMESİ ===")
        
        all_models = {**self.base_models, **self.ensemble_models}
        
        for name, model in all_models.items():
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
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            if roc_auc:
                print(f"  ROC AUC: {roc_auc:.4f}")
    
    def find_best_model(self):
        """En iyi modeli bul."""
        print("\n=== EN İYİ MODEL SEÇİMİ ===")
        
        best_accuracy = 0
        best_model_name = None
        
        for name, metrics in self.results.items():
            accuracy = metrics['accuracy']
            print(f"{name}: Accuracy = {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
        
        print(f"\nEn iyi model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return best_model_name, best_accuracy
    
    def plot_ensemble_comparison(self):
        """Ensemble modellerinin karşılaştırmasını görselleştir."""
        # Sonuçları DataFrame'e çevir
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[name]['accuracy'] for name in self.results.keys()],
            'Precision': [self.results[name]['precision'] for name in self.results.keys()],
            'Recall': [self.results[name]['recall'] for name in self.results.keys()],
            'F1-Score': [self.results[name]['f1'] for name in self.results.keys()],
            'ROC AUC': [self.results[name]['roc_auc'] for name in self.results.keys()]
        })
        
        # Model türlerini belirle
        base_models = list(self.base_models.keys())
        ensemble_models = list(self.ensemble_models.keys())
        
        results_df['Type'] = results_df['Model'].apply(
            lambda x: 'Ensemble' if x in ensemble_models else 'Base'
        )
        
        # Görselleştirme
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Accuracy karşılaştırması
        ax1 = axes[0, 0]
        colors = ['lightblue' if x == 'Base' else 'lightgreen' for x in results_df['Type']]
        bars = ax1.bar(results_df['Model'], results_df['Accuracy'], color=colors)
        ax1.set_title('Accuracy Karşılaştırması')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Değerleri çubukların üzerine yaz
        for bar, acc in zip(bars, results_df['Accuracy']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Precision karşılaştırması
        ax2 = axes[0, 1]
        bars = ax2.bar(results_df['Model'], results_df['Precision'], color=colors)
        ax2.set_title('Precision Karşılaştırması')
        ax2.set_ylabel('Precision')
        ax2.tick_params(axis='x', rotation=45)
        
        # Recall karşılaştırması
        ax3 = axes[0, 2]
        bars = ax3.bar(results_df['Model'], results_df['Recall'], color=colors)
        ax3.set_title('Recall Karşılaştırması')
        ax3.set_ylabel('Recall')
        ax3.tick_params(axis='x', rotation=45)
        
        # F1-Score karşılaştırması
        ax4 = axes[1, 0]
        bars = ax4.bar(results_df['Model'], results_df['F1-Score'], color=colors)
        ax4.set_title('F1-Score Karşılaştırması')
        ax4.set_ylabel('F1-Score')
        ax4.tick_params(axis='x', rotation=45)
        
        # ROC AUC karşılaştırması
        ax5 = axes[1, 1]
        bars = ax5.bar(results_df['Model'], results_df['ROC AUC'], color=colors)
        ax5.set_title('ROC AUC Karşılaştırması')
        ax5.set_ylabel('ROC AUC')
        ax5.tick_params(axis='x', rotation=45)
        
        # Model türü dağılımı
        ax6 = axes[1, 2]
        type_counts = results_df['Type'].value_counts()
        ax6.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax6.set_title('Model Türü Dağılımı')
        
        plt.tight_layout()
        plt.savefig('ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df
    
    def cross_validation_comparison(self, X, y, cv=5):
        """Cross-validation ile model karşılaştırması."""
        print("=== CROSS-VALIDATION KARŞILAŞTIRMASI ===")
        
        all_models = {**self.base_models, **self.ensemble_models}
        cv_results = {}
        
        for name, model in all_models.items():
            print(f"{name} cross-validation...")
            
            # Accuracy için CV
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_results[name] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # CV sonuçlarını görselleştir
        cv_df = pd.DataFrame({
            'Model': list(cv_results.keys()),
            'Mean CV Accuracy': [cv_results[name]['mean_accuracy'] for name in cv_results.keys()],
            'Std CV Accuracy': [cv_results[name]['std_accuracy'] for name in cv_results.keys()]
        })
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(cv_df['Model'], cv_df['Mean CV Accuracy'], 
                      yerr=cv_df['Std CV Accuracy'], capsize=5, color='lightcoral')
        plt.title('Cross-Validation Accuracy Karşılaştırması')
        plt.ylabel('Mean CV Accuracy')
        plt.xticks(rotation=45)
        
        # Değerleri çubukların üzerine yaz
        for bar, mean_acc in zip(bars, cv_df['Mean CV Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cv_results
    
    def run_ensemble_analysis(self, X_train, X_test, y_train, y_test):
        """Tam ensemble analizi çalıştır."""
        print("="*60)
        print("ENSEMBLE METHODS ANALİZİ")
        print("="*60)
        
        # 1. Temel modelleri oluştur
        self.create_base_models()
        
        # 2. Ensemble modelleri oluştur
        self.create_voting_classifier(voting='soft')
        self.create_voting_classifier(voting='hard')
        self.create_stacking_classifier()
        self.create_weighted_average()
        
        # 3. Tüm modelleri eğit
        self.train_all_models(X_train, y_train)
        
        # 4. Test setinde değerlendir
        self.evaluate_all_models(X_test, y_test)
        
        # 5. En iyi modeli bul
        best_model_name, best_accuracy = self.find_best_model()
        
        # 6. Karşılaştırmalı görselleştirme
        results_df = self.plot_ensemble_comparison()
        
        # 7. Cross-validation karşılaştırması
        cv_results = self.cross_validation_comparison(X_train, y_train)
        
        print("\n" + "="*60)
        print("ENSEMBLE ANALİZİ TAMAMLANDI!")
        print("="*60)
        print(f"En iyi model: {best_model_name}")
        print(f"En iyi accuracy: {best_accuracy:.4f}")
        
        return {
            'best_model_name': best_model_name,
            'best_accuracy': best_accuracy,
            'results_df': results_df,
            'cv_results': cv_results,
            'all_results': self.results
        }

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
    
    # Veri işleme (outlier'lı verilerle)
    data_dict = fe.process_pipeline_with_outliers(str(data_path))
    if data_dict is None:
        print("Veri işleme başarısız!")
        return
    
    # Ensemble analizi
    ensemble = EnsembleMethods()
    results = ensemble.run_ensemble_analysis(
        data_dict['X_train'],
        data_dict['X_test'],
        data_dict['y_train'],
        data_dict['y_test']
    )
    
    print("\n" + "="*60)
    print("FİNAL ENSEMBLE SONUÇLARI")
    print("="*60)
    print(f"En iyi model: {results['best_model_name']}")
    print(f"En iyi accuracy: {results['best_accuracy']:.4f}")

if __name__ == "__main__":
    main() 