"""
Model Eğitimi Servisi
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.analysis.hyperparameter_tuning import HyperparameterTuner
from src.analysis.ensemble_methods import EnsembleMethods
from src.data.advanced_feature_engineering import AdvancedFeatureEngineer
from api.core.database import ModelRecord, SessionLocal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainingService:
    """Model eğitimi servisi."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.tuner = HyperparameterTuner()
        self.ensemble = EnsembleMethods()
        self.advanced_fe = AdvancedFeatureEngineer()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def train_selected_model(
        self, 
        model_name: str, 
        data_path: str,
        use_advanced_features: bool = True,
        use_gpu: bool = True
    ) -> Dict[str, Any]:
        """
        Seçilen modeli eğit ve detaylı analiz sağla.
        
        Args:
            model_name: Eğitilecek model adı
            data_path: Veri dosyası yolu
            use_advanced_features: Gelişmiş feature engineering kullan
            use_gpu: GPU kullan
            
        Returns:
            Eğitim sonuçları ve analiz
        """
        try:
            # Asenkron olarak çalıştır
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._train_model_sync, 
                model_name, data_path, use_advanced_features, use_gpu
            )
            return result
        except Exception as e:
            raise Exception(f"Model eğitimi hatası: {str(e)}")
    
    def _train_model_sync(
        self, 
        model_name: str, 
        data_path: str,
        use_advanced_features: bool,
        use_gpu: bool
    ) -> Dict[str, Any]:
        """Senkron model eğitimi."""
        print(f"🔄 {model_name} modeli eğitiliyor...")
        
        # 1. Veri Hazırlama
        print("📊 Veri hazırlama...")
        
        if use_advanced_features:
            # Gelişmiş feature engineering kullan
            data = self.advanced_fe.advanced_pipeline_with_outliers(data_path)
            if data is None:
                raise Exception("Gelişmiş feature engineering başarısız!")
            
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']
            feature_names = data['feature_names']
            
        else:
            # Basit preprocessing kullan
            preprocessed_data = self.preprocessor.complete_preprocessing_pipeline(
                data_path, remove_outliers=True
            )
            if preprocessed_data is None:
                raise Exception("Veri ön işleme başarısız!")
            
            # Veriyi ML için hazırla
            X_train, X_test, y_train, y_test, feature_names = self._prepare_ml_data(
                preprocessed_data['data']
            )
        
        print(f"✅ Veri hazırlandı: Train {X_train.shape}, Test {X_test.shape}")
        
        # 2. Model Eğitimi
        print(f"🤖 {model_name} eğitimi...")
        
        trained_model, training_params = self._train_specific_model(
            model_name, X_train, y_train, use_gpu
        )
        
        # 3. Model Değerlendirmesi
        print("📈 Model değerlendirmesi...")
        
        evaluation_results = self._evaluate_model(
            trained_model, X_test, y_test
        )
        
        # 4. Feature Importance Analizi
        print("🔍 Feature importance analizi...")
        
        feature_importance = self._analyze_feature_importance(
            trained_model, feature_names, model_name
        )
        
        # 5. Model Kaydetme
        print("💾 Model kaydediliyor...")
        
        model_path = self._save_model(
            trained_model, model_name, evaluation_results, feature_names, training_params
        )
        
        # 6. Veritabanına Kaydetme
        print("🗄️ Veritabanına kaydediliyor...")
        
        model_record_id = self._save_to_database(
            model_name, evaluation_results, model_path, feature_names, training_params
        )
        
        # 7. Detaylı Analiz
        print("📊 Detaylı analiz...")
        
        detailed_analysis = self._generate_detailed_analysis(
            trained_model, X_test, y_test, evaluation_results, feature_importance
        )
        
        # Final sonuçlar
        result = {
            'training_summary': {
                'model_name': model_name,
                'model_type': self._get_model_type(model_name),
                'training_params': training_params,
                'data_shape': {
                    'train': X_train.shape,
                    'test': X_test.shape,
                    'features': len(feature_names)
                },
                'use_advanced_features': use_advanced_features,
                'use_gpu': use_gpu
            },
            'evaluation_results': evaluation_results,
            'feature_importance': feature_importance,
            'model_path': model_path,
            'model_record_id': model_record_id,
            'detailed_analysis': detailed_analysis,
            'training_timestamp': datetime.utcnow().isoformat()
        }
        
        print(f"✅ {model_name} eğitimi tamamlandı!")
        return result
    
    def _prepare_ml_data(self, df):
        """Veriyi makine öğrenmesi için hazırla."""
        df_ml = df.copy()
        
        # Kategorik değişkenleri encode et
        categorical_columns = ['gender', 'smoke', 'alco', 'active']
        label_encoders = {}
        
        for col in categorical_columns:
            if col in df_ml.columns:
                le = LabelEncoder()
                df_ml[col] = le.fit_transform(df_ml[col])
                label_encoders[col] = le
        
        # Hedef değişkeni ayır
        y = df_ml['cardio']
        X = df_ml.drop(['id', 'age', 'cardio'], axis=1, errors='ignore')
        
        # Sürekli değişkenleri ölçekle
        continuous_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        available_continuous = [col for col in continuous_columns if col in X.columns]
        
        scaler = None
        if available_continuous:
            scaler = StandardScaler()
            X[available_continuous] = scaler.fit_transform(X[available_continuous])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, list(X.columns)
    
    def _train_specific_model(self, model_name: str, X_train, y_train, use_gpu: bool):
        """Belirli bir modeli eğit."""
        # GPU parametrelerini hazırla
        gpu_params = {}
        if use_gpu:
            gpu_params = self._get_gpu_params()
        
        # Model parametrelerini belirle
        if model_name.lower() in ['random forest', 'randomforest']:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5, 
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            training_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            
        elif model_name.lower() in ['gradient boosting', 'gradientboosting']:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, 
                min_samples_split=5, random_state=42
            )
            training_params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'random_state': 42
            }
            
        elif model_name.lower() == 'xgboost':
            try:
                import xgboost as xgb
                xgb_params = {
                    'n_estimators': 200, 
                    'learning_rate': 0.1, 
                    'max_depth': 6,
                    'subsample': 0.9, 
                    'colsample_bytree': 0.9, 
                    'random_state': 42
                }
                
                # GPU parametrelerini ekle
                if gpu_params and 'xgboost' in gpu_params:
                    xgb_params.update(gpu_params['xgboost'])
                
                model = xgb.XGBClassifier(**xgb_params)
                training_params = xgb_params
                
            except ImportError:
                raise Exception("XGBoost yüklü değil!")
                
        elif model_name.lower() == 'lightgbm':
            try:
                import lightgbm as lgb
                lgb_params = {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': 42,
                    'verbose': -1
                }
                
                # GPU parametrelerini ekle
                if gpu_params and 'lightgbm' in gpu_params:
                    lgb_params.update(gpu_params['lightgbm'])
                
                model = lgb.LGBMClassifier(**lgb_params)
                training_params = lgb_params
                
            except ImportError:
                raise Exception("LightGBM yüklü değil!")
                
        elif model_name.lower() in ['logistic regression', 'logisticregression']:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', max_iter=1000, random_state=42
            )
            training_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'max_iter': 1000,
                'random_state': 42
            }
            
        elif model_name.lower() == 'svm':
            from sklearn.svm import SVC
            model = SVC(
                C=10, kernel='rbf', gamma='scale', probability=True, random_state=42
            )
            training_params = {
                'C': 10,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            }
            
        else:
            raise Exception(f"Desteklenmeyen model: {model_name}")
        
        # Modeli eğit
        model.fit(X_train, y_train)
        
        return model, training_params
    
    def _get_gpu_params(self):
        """GPU parametrelerini al."""
        gpu_params = {}
        
        # XGBoost GPU kontrolü
        try:
            import xgboost as xgb
            gpu_params['xgboost'] = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            }
        except:
            pass
        
        # LightGBM GPU kontrolü
        try:
            import lightgbm as lgb
            gpu_params['lightgbm'] = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'gpu_use_dp': True
            }
        except:
            pass
        
        return gpu_params
    
    def _evaluate_model(self, model, X_test, y_test):
        """Modeli değerlendir."""
        # Tahminler
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist() if y_proba is not None else None
        }
    
    def _analyze_feature_importance(self, model, feature_names, model_name):
        """Feature importance analizi."""
        feature_importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance[model_name] = [
                    {'feature': name, 'importance': float(imp)}
                    for name, imp in zip(feature_names, importances)
                ]
                # Önem sırasına göre sırala
                feature_importance[model_name].sort(key=lambda x: x['importance'], reverse=True)
                
        except Exception as e:
            print(f"Feature importance analizi başarısız: {e}")
            feature_importance[model_name] = []
        
        return feature_importance
    
    def _save_model(self, model, model_name, evaluation_results, feature_names, training_params):
        """Modeli kaydet."""
        # Models klasörü oluştur
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Dosya adı oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
        filepath = models_dir / filename
        
        # Model bilgilerini hazırla
        model_data = {
            'model': model,
            'model_name': model_name,
            'evaluation_results': evaluation_results,
            'feature_names': feature_names,
            'training_params': training_params,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Modeli kaydet
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        return str(filepath)
    
    def _save_to_database(self, model_name, evaluation_results, model_path, feature_names, training_params):
        """Modeli veritabanına kaydet."""
        db = SessionLocal()
        try:
            model_record = ModelRecord(
                model_name=model_name,
                model_type=self._get_model_type(model_name),
                accuracy=evaluation_results['accuracy'],
                precision=evaluation_results['precision'],
                recall=evaluation_results['recall'],
                f1_score=evaluation_results['f1_score'],
                roc_auc=evaluation_results['roc_auc'],
                model_path=model_path,
                feature_names=json.dumps(feature_names),
                training_params=json.dumps(training_params),
                is_active=True
            )
            
            db.add(model_record)
            db.commit()
            db.refresh(model_record)
            
            return model_record.id
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Veritabanı kayıt hatası: {str(e)}")
        finally:
            db.close()
    
    def _get_model_type(self, model_name):
        """Model tipini belirle."""
        if model_name.lower() in ['random forest', 'gradient boosting', 'xgboost', 'lightgbm']:
            return 'tree_based'
        elif model_name.lower() in ['logistic regression', 'svm']:
            return 'linear'
        else:
            return 'other'
    
    def _generate_detailed_analysis(self, model, X_test, y_test, evaluation_results, feature_importance):
        """Detaylı analiz oluştur."""
        analysis = {
            'performance_summary': {
                'overall_performance': self._get_performance_level(evaluation_results['f1_score']),
                'strengths': self._identify_strengths(evaluation_results),
                'weaknesses': self._identify_weaknesses(evaluation_results)
            },
            'feature_analysis': {
                'top_features': feature_importance.get(list(feature_importance.keys())[0], [])[:10],
                'feature_count': len(feature_importance.get(list(feature_importance.keys())[0], []))
            },
            'model_characteristics': {
                'complexity': self._assess_model_complexity(model),
                'interpretability': self._assess_interpretability(model),
                'scalability': self._assess_scalability(model)
            }
        }
        
        return analysis
    
    def _get_performance_level(self, f1_score):
        """Performans seviyesini belirle."""
        if f1_score >= 0.9:
            return "Mükemmel"
        elif f1_score >= 0.8:
            return "Çok İyi"
        elif f1_score >= 0.7:
            return "İyi"
        elif f1_score >= 0.6:
            return "Orta"
        else:
            return "Düşük"
    
    def _identify_strengths(self, evaluation_results):
        """Güçlü yanları belirle."""
        strengths = []
        
        if evaluation_results['accuracy'] >= 0.8:
            strengths.append("Yüksek doğruluk")
        if evaluation_results['precision'] >= 0.8:
            strengths.append("Yüksek hassasiyet")
        if evaluation_results['recall'] >= 0.8:
            strengths.append("Yüksek duyarlılık")
        if evaluation_results['roc_auc'] and evaluation_results['roc_auc'] >= 0.8:
            strengths.append("Güçlü sınıflandırma yeteneği")
        
        return strengths if strengths else ["Orta seviye performans"]
    
    def _identify_weaknesses(self, evaluation_results):
        """Zayıf yanları belirle."""
        weaknesses = []
        
        if evaluation_results['accuracy'] < 0.7:
            weaknesses.append("Düşük doğruluk")
        if evaluation_results['precision'] < 0.7:
            weaknesses.append("Düşük hassasiyet")
        if evaluation_results['recall'] < 0.7:
            weaknesses.append("Düşük duyarlılık")
        if evaluation_results['roc_auc'] and evaluation_results['roc_auc'] < 0.7:
            weaknesses.append("Zayıf sınıflandırma yeteneği")
        
        return weaknesses if weaknesses else ["Önemli zayıflık bulunamadı"]
    
    def _assess_model_complexity(self, model):
        """Model karmaşıklığını değerlendir."""
        if hasattr(model, 'n_estimators'):
            return "Yüksek" if model.n_estimators > 100 else "Orta"
        elif hasattr(model, 'max_depth'):
            return "Yüksek" if model.max_depth > 10 else "Orta"
        else:
            return "Düşük"
    
    def _assess_interpretability(self, model):
        """Model yorumlanabilirliğini değerlendir."""
        model_name = type(model).__name__.lower()
        
        if 'logistic' in model_name:
            return "Yüksek"
        elif 'tree' in model_name or 'forest' in model_name:
            return "Orta"
        else:
            return "Düşük"
    
    def _assess_scalability(self, model):
        """Model ölçeklenebilirliğini değerlendir."""
        model_name = type(model).__name__.lower()
        
        if 'logistic' in model_name or 'svm' in model_name:
            return "Yüksek"
        elif 'tree' in model_name or 'forest' in model_name:
            return "Orta"
        else:
            return "Düşük"

# Global service instance
model_training_service = ModelTrainingService()
