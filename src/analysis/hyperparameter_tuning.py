"""
Kalp Krizi Risk Tahmin Modeli - Hyperparameter Tuning Modülü
Sadece Gradient Boosting ve XGBoost için
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings 
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """Hyperparameter tuning sınıfı - Sadece GB ve XGBoost için."""
    
    def __init__(self):
        self.best_models = {}
        self.tuning_results = {}
        
    def tune_gradient_boosting(self, X_train, y_train, cv=5, n_jobs=-1):
        """Gradient Boosting için hyperparameter tuning."""
        print("Gradient Boosting hyperparameter tuning başlatılıyor...")
        
        # Gradient Boosting parametre grid'i
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'random_state': [42]
        }
        
        # Grid search
        gb = GradientBoostingClassifier()
        grid_search = GridSearchCV(
            estimator=gb,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi cross-validation skoru: {grid_search.best_score_:.4f}")
        
        self.best_models['Gradient Boosting'] = grid_search.best_estimator_
        self.tuning_results['Gradient Boosting'] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        return grid_search.best_estimator_
    
    def tune_xgboost(self, X_train, y_train, cv=5, n_jobs=-1, gpu_params=None):
        """XGBoost için hyperparameter tuning."""
        print("XGBoost hyperparameter tuning başlatılıyor...")
        
        # XGBoost parametre grid'i
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'random_state': [42]
        }
        
        # GPU parametrelerini ekle
        if gpu_params and 'xgboost' in gpu_params:
            for key, value in gpu_params['xgboost'].items():
                param_grid[key] = [value]
            print("✅ GPU parametreleri XGBoost'a eklendi")
        
        # Grid search
        xgb_model = xgb.XGBClassifier()
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi cross-validation skoru: {grid_search.best_score_:.4f}")
        
        self.best_models['XGBoost'] = grid_search.best_estimator_
        self.tuning_results['XGBoost'] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        return grid_search.best_estimator_
    
    def nested_cross_validation(self, X, y, model_name, param_grid, cv_outer=5, cv_inner=5, n_jobs=-1, gpu_params=None):
        """Nested cross-validation implementation for robust model evaluation."""
        print(f"\n=== NESTED CROSS-VALIDATION: {model_name} ===")
        print(f"Outer CV folds: {cv_outer}, Inner CV folds: {cv_inner}")
        
        # Model seçimi
        if model_name.lower() == 'gradient boosting':
            base_model = GradientBoostingClassifier()
        elif model_name.lower() == 'xgboost':
            base_model = xgb.XGBClassifier()
            # GPU parametrelerini ekle
            if gpu_params and 'xgboost' in gpu_params:
                for key, value in gpu_params['xgboost'].items():
                    param_grid[key] = [value]
                print("✅ GPU parametreleri XGBoost'a eklendi")
        else:
            raise ValueError(f"Desteklenmeyen model: {model_name}")
        
        # Nested CV için scoring metrikleri
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Nested cross-validation
        nested_scores = cross_validate(
            GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_inner,
                scoring='accuracy',
                n_jobs=n_jobs,
                verbose=0
            ),
            X=X,
            y=y,
            cv=cv_outer,
            scoring=scoring,
            return_train_score=True,
            n_jobs=n_jobs
        )
        
        # Sonuçları hesapla
        results = {}
        for metric in scoring.keys():
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            if test_key in nested_scores:
                results[metric] = {
                    'test_mean': nested_scores[test_key].mean(),
                    'test_std': nested_scores[test_key].std(),
                    'test_scores': nested_scores[test_key]
                }
                
                if train_key in nested_scores:
                    results[metric]['train_mean'] = nested_scores[train_key].mean()
                    results[metric]['train_std'] = nested_scores[train_key].std()
                    results[metric]['train_scores'] = nested_scores[train_key]
        
        # Sonuçları yazdır
        print(f"\n{model_name} Nested CV Sonuçları:")
        print("-" * 50)
        for metric, scores in results.items():
            print(f"{metric.upper()}:")
            print(f"  Test: {scores['test_mean']:.4f} (+/- {scores['test_std'] * 2:.4f})")
            if 'train_mean' in scores:
                print(f"  Train: {scores['train_mean']:.4f} (+/- {scores['train_std'] * 2:.4f})")
                # Overfitting kontrolü
                overfitting = scores['train_mean'] - scores['test_mean']
                print(f"  Overfitting: {overfitting:.4f}")
        
        return results
    
    def nested_cv_all_models(self, X, y, cv_outer=5, cv_inner=5, n_jobs=-1, gpu_params=None):
        """Tüm modeller için nested cross-validation."""
        print("="*80)
        print("NESTED CROSS-VALIDATION - TÜM MODELLER")
        print("="*80)
        
        all_results = {}
        
        # Gradient Boosting parametre grid'i
        gb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'random_state': [42]
        }
        
        # XGBoost parametre grid'i
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'random_state': [42]
        }
        
        # Gradient Boosting nested CV
        print("\n1. GRADIENT BOOSTING NESTED CV")
        gb_results = self.nested_cross_validation(
            X, y, 'Gradient Boosting', gb_param_grid, 
            cv_outer, cv_inner, n_jobs
        )
        all_results['Gradient Boosting'] = gb_results
        
        # XGBoost nested CV
        print("\n2. XGBOOST NESTED CV")
        xgb_results = self.nested_cross_validation(
            X, y, 'XGBoost', xgb_param_grid, 
            cv_outer, cv_inner, n_jobs, gpu_params
        )
        all_results['XGBoost'] = xgb_results
        
        # Sonuçları görselleştir
        self.plot_nested_cv_results(all_results)
        
        print("\n" + "="*80)
        print("NESTED CROSS-VALIDATION TAMAMLANDI")
        print("="*80)
        
        return all_results
    
    def plot_nested_cv_results(self, results):
        """Nested CV sonuçlarını görselleştir."""
        if not results:
            print("Nested CV sonuçları bulunamadı!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Nested Cross-Validation Sonuçları', fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Her metrik için ayrı subplot
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            test_means = []
            test_stds = []
            train_means = []
            train_stds = []
            
            for model in models:
                if metric in results[model]:
                    test_means.append(results[model][metric]['test_mean'])
                    test_stds.append(results[model][metric]['test_std'])
                    
                    if 'train_mean' in results[model][metric]:
                        train_means.append(results[model][metric]['train_mean'])
                        train_stds.append(results[model][metric]['train_std'])
                    else:
                        train_means.append(0)
                        train_stds.append(0)
            
            x = np.arange(len(models))
            width = 0.35
            
            # Test ve train skorları
            bars1 = ax.bar(x - width/2, test_means, width, yerr=test_stds, 
                          label='Test', capsize=5, alpha=0.8, color='lightcoral')
            bars2 = ax.bar(x + width/2, train_means, width, yerr=train_stds, 
                          label='Train', capsize=5, alpha=0.8, color='lightblue')
            
            ax.set_xlabel('Modeller')
            ax.set_ylabel(f'{metric.upper()} Score')
            ax.set_title(f'{metric.upper()} Karşılaştırması')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Değerleri bar'ların üzerine yaz
            for bar, mean, std in zip(bars1, test_means, test_stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Overfitting analizi
        ax = axes[1, 2]
        overfitting_data = []
        for model in models:
            if 'accuracy' in results[model] and 'train_mean' in results[model]['accuracy']:
                overfitting = results[model]['accuracy']['train_mean'] - results[model]['accuracy']['test_mean']
                overfitting_data.append(overfitting)
            else:
                overfitting_data.append(0)
        
        bars = ax.bar(models, overfitting_data, color=['red' if x > 0.05 else 'orange' if x > 0.02 else 'green' for x in overfitting_data])
        ax.set_xlabel('Modeller')
        ax.set_ylabel('Overfitting (Train - Test)')
        ax.set_title('Overfitting Analizi')
        ax.set_xticklabels(models, rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Overfitting değerlerini yaz
        for bar, value in zip(bars, overfitting_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('nested_cv_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Nested CV sonuçları 'nested_cv_results.png' dosyasına kaydedildi.")
    
    def tune_all_models(self, X_train, y_train, cv=5, n_jobs=-1, gpu_params=None):
        """Tüm modeller için hyperparameter tuning (sadece GB ve XGBoost)."""
        print("="*60)
        print("HYPERPARAMETER TUNING - GRADIENT BOOSTING VE XGBOOST")
        print("="*60)
        
        # Gradient Boosting tuning
        self.tune_gradient_boosting(X_train, y_train, cv, n_jobs)
        
        print("\n" + "-"*40)
        
        # XGBoost tuning
        self.tune_xgboost(X_train, y_train, cv, n_jobs, gpu_params)
        
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING TAMAMLANDI")
        print("="*60)
        
        return self.best_models
    
    def evaluate_tuned_models(self, X_test, y_test):
        """Tune edilmiş modelleri test setinde değerlendir."""
        print("\nTune edilmiş modeller test setinde değerlendiriliyor...")
        
        results = {}
        
        for name, model in self.best_models.items():
            # Tahminler
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrikler
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'model': model
            }
            
            print(f"\n{name} Test Sonuçları:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def plot_tuning_results(self):
        """Tuning sonuçlarını görselleştir."""
        if not self.tuning_results:
            print("Tuning sonuçları bulunamadı!")
            return
        
        # Cross-validation skorları
        models = list(self.tuning_results.keys())
        cv_scores = [self.tuning_results[model]['best_cv_score'] for model in models]
        
        plt.figure(figsize=(12, 8))
        
        # Cross-validation skorları
        plt.subplot(2, 2, 1)
        bars = plt.bar(models, cv_scores, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Cross-Validation Skorları', fontsize=14, fontweight='bold')
        plt.ylabel('CV Score')
        plt.ylim(0, 1)
        
        # Skorları bar'ların üzerine yaz
        for bar, score in zip(bars, cv_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # En iyi parametreler
        plt.subplot(2, 2, 2)
        plt.text(0.1, 0.9, 'En İyi Parametreler:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
        
        y_pos = 0.8
        for model in models:
            params = self.tuning_results[model]['best_params']
            plt.text(0.1, y_pos, f'{model}:', fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
            y_pos -= 0.05
            
            for param, value in params.items():
                if param != 'random_state':
                    plt.text(0.15, y_pos, f'  {param}: {value}', fontsize=9, transform=plt.gca().transAxes)
                    y_pos -= 0.03
            y_pos -= 0.02
        
        plt.axis('off')
        
        # Feature importance (eğer varsa)
        if self.best_models:
            plt.subplot(2, 2, 3)
            for i, (name, model) in enumerate(self.best_models.items()):
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    top_indices = np.argsort(importances)[-10:]  # En önemli 10 feature
                    
                    plt.barh(range(len(top_indices)), importances[top_indices], 
                           label=name, alpha=0.7)
                    plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
            
            plt.title('En Önemli 10 Feature', fontsize=12, fontweight='bold')
            plt.xlabel('Importance')
            plt.legend()
        
        # Model karşılaştırması
        plt.subplot(2, 2, 4)
        if hasattr(self, 'test_results') and self.test_results:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            x = np.arange(len(metrics))
            width = 0.35
            
            for i, (name, results) in enumerate(self.test_results.items()):
                values = [results[metric] for metric in metrics]
                plt.bar(x + i*width, values, width, label=name, alpha=0.8)
            
            plt.xlabel('Metrikler')
            plt.ylabel('Skor')
            plt.title('Model Performans Karşılaştırması', fontsize=12, fontweight='bold')
            plt.xticks(x + width/2, metrics, rotation=45)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Tuning sonuçları 'hyperparameter_tuning_results.png' dosyasına kaydedildi.")

def main():
    """Ana fonksiyon."""
    # Hyperparameter tuning
    tuner = HyperparameterTuner()
    
    # Örnek veri (gerçek kullanımda advanced feature engineering'den gelecek)
    print("Bu modül main_optimized.py üzerinden çalıştırılmalıdır.")

if __name__ == "__main__":
    main() 