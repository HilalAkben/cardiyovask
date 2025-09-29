"""
Model Karşılaştırma Servisi
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.analysis.data_analysis import ComparativeAnalyzer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

class ModelComparisonService:
    """Model karşılaştırma servisi."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.comparative_analyzer = ComparativeAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def run_model_comparison(self, data_path: str) -> Dict[str, Any]:
        """
        Tüm modelleri karşılaştır ve sonuçları döndür.
        
        Args:
            data_path: Veri dosyası yolu
            
        Returns:
            Model karşılaştırma sonuçları
        """
        try:
            # Asenkron olarak çalıştır
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._run_comparison_sync, 
                data_path
            )
            return result
        except Exception as e:
            raise Exception(f"Model karşılaştırma hatası: {str(e)}")
    
    def _run_comparison_sync(self, data_path: str) -> Dict[str, Any]:
        """Senkron model karşılaştırma."""
        print("🔄 Model karşılaştırması başlatılıyor...")
        
        # 1. Veri Ön İşleme
        print("📊 Veri ön işleme...")
        
        # Outlier'lı verilerle işleme
        data_with_outliers = self.preprocessor.complete_preprocessing_pipeline(
            data_path, remove_outliers=False
        )
        if data_with_outliers is None:
            raise Exception("Outlier'lı veri işleme başarısız!")
        
        # Outlier'lar çıkarılarak işleme
        data_without_outliers = self.preprocessor.complete_preprocessing_pipeline(
            data_path, remove_outliers=True
        )
        if data_without_outliers is None:
            raise Exception("Outlier'lar çıkarılarak veri işleme başarısız!")
        
        # 2. Veri Hazırlama
        print("🔧 Veri hazırlama...")
        
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
        print("🤖 Model analizi...")
        outlier_results, no_outlier_results = self.comparative_analyzer.run_comparative_analysis(
            data_with_outliers_ml, data_without_outliers_ml
        )
        
        # 4. Olasılık Temelli Metrikler
        print("📈 Olasılık metrikleri hesaplanıyor...")
        
        def compute_probability_metrics(results_dict, y_test, tag):
            """Olasılık temelli metrikleri hesapla."""
            metrics = {}
            
            for model_name, splits in results_dict.items():
                y_proba = splits['test'].get('y_proba', None)
                if y_proba is None:
                    continue
                
                try:
                    brier = brier_score_loss(y_test, y_proba)
                    ll = log_loss(y_test, y_proba)
                    metrics[model_name] = {
                        'brier_score': float(brier),
                        'log_loss': float(ll)
                    }
                except Exception as e:
                    print(f"{model_name}: Olasılık metrikleri hesaplanamadı -> {e}")
                    metrics[model_name] = {
                        'brier_score': None,
                        'log_loss': None
                    }
            
            return metrics
        
        # Olasılık metriklerini hesapla
        prob_metrics_with = compute_probability_metrics(outlier_results['results'], y_test_with, 'with_outliers')
        prob_metrics_without = compute_probability_metrics(no_outlier_results['results'], y_test_without, 'without_outliers')
        
        # 5. Sonuçları Hazırla
        print("📋 Sonuçlar hazırlanıyor...")
        
        # Model performanslarını sırala (F1-Score'a göre)
        def sort_models_by_performance(results_dict):
            """Modelleri performansa göre sırala."""
            model_scores = []
            for model_name, splits in results_dict.items():
                test_metrics = splits['test']
                f1_score = test_metrics.get('f1', 0)
                model_scores.append({
                    'model_name': model_name,
                    'f1_score': f1_score,
                    'accuracy': test_metrics.get('accuracy', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0),
                    'roc_auc': test_metrics.get('roc_auc', 0)
                })
            
            # F1-Score'a göre sırala (yüksekten düşüğe)
            model_scores.sort(key=lambda x: x['f1_score'], reverse=True)
            return model_scores
        
        # Outlier'lı ve outlier'sız modelleri sırala
        sorted_models_with = sort_models_by_performance(outlier_results['results'])
        sorted_models_without = sort_models_by_performance(no_outlier_results['results'])
        
        # Risk skorları hesapla (F1-Score'a göre normalize edilmiş)
        def calculate_risk_scores(sorted_models):
            """Risk skorlarını hesapla."""
            if not sorted_models:
                return []
            
            max_f1 = sorted_models[0]['f1_score']
            min_f1 = sorted_models[-1]['f1_score']
            f1_range = max_f1 - min_f1 if max_f1 != min_f1 else 1
            
            risk_scores = []
            for i, model in enumerate(sorted_models):
                # F1-Score'a göre risk skoru (yüksek F1 = düşük risk)
                normalized_f1 = (model['f1_score'] - min_f1) / f1_range
                risk_score = 1 - normalized_f1  # Ters çevir (yüksek performans = düşük risk)
                
                risk_scores.append({
                    'rank': i + 1,
                    'model_name': model['model_name'],
                    'f1_score': model['f1_score'],
                    'accuracy': model['accuracy'],
                    'precision': model['precision'],
                    'recall': model['recall'],
                    'roc_auc': model['roc_auc'],
                    'risk_score': risk_score,
                    'risk_level': self._get_risk_level(risk_score)
                })
            
            return risk_scores
        
        risk_scores_with = calculate_risk_scores(sorted_models_with)
        risk_scores_without = calculate_risk_scores(sorted_models_without)
        
        # Final sonuçlar
        result = {
            'comparison_summary': {
                'total_models': len(outlier_results['results']),
                'data_with_outliers': {
                    'train_shape': data_with_outliers_ml['X_train'].shape,
                    'test_shape': data_with_outliers_ml['X_test'].shape,
                    'best_model': outlier_results['best_model_name'],
                    'best_f1_score': outlier_results['best_score']
                },
                'data_without_outliers': {
                    'train_shape': data_without_outliers_ml['X_train'].shape,
                    'test_shape': data_without_outliers_ml['X_test'].shape,
                    'best_model': no_outlier_results['best_model_name'],
                    'best_f1_score': no_outlier_results['best_score']
                }
            },
            'model_rankings': {
                'with_outliers': risk_scores_with,
                'without_outliers': risk_scores_without
            },
            'probability_metrics': {
                'with_outliers': prob_metrics_with,
                'without_outliers': prob_metrics_without
            },
            'feature_importance': {
                'with_outliers': self._extract_feature_importance(outlier_results['importance_dfs']),
                'without_outliers': self._extract_feature_importance(no_outlier_results['importance_dfs'])
            },
            'outlier_analysis': {
                'with_outliers': data_with_outliers['outlier_summary'],
                'without_outliers': data_without_outliers['outlier_summary']
            }
        }
        
        print("✅ Model karşılaştırması tamamlandı!")
        return result
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Risk skoruna göre risk seviyesini belirle."""
        if risk_score <= 0.2:
            return "Çok Düşük"
        elif risk_score <= 0.4:
            return "Düşük"
        elif risk_score <= 0.6:
            return "Orta"
        elif risk_score <= 0.8:
            return "Yüksek"
        else:
            return "Çok Yüksek"
    
    def _extract_feature_importance(self, importance_dfs: Dict) -> Dict[str, List[Dict]]:
        """Feature importance'ları çıkar."""
        feature_importance = {}
        
        for model_name, imp_df in importance_dfs.items():
            if imp_df is not None and not imp_df.empty:
                top_features = imp_df.head(10).to_dict('records')
                feature_importance[model_name] = top_features
        
        return feature_importance

# Global service instance
model_comparison_service = ModelComparisonService()
