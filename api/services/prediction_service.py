"""
Risk Tahmin Servisi
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.core.database import ModelRecord, PredictionRecord, SessionLocal
from sklearn.preprocessing import StandardScaler, LabelEncoder

class PredictionService:
    """Risk tahmin servisi."""
    
    def __init__(self):
        self.models_dir = project_root / "models"
    
    async def predict_cardio_risk(
        self, 
        model_id: int,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hastanın kardiyovasküler riskini tahmin et.
        
        Args:
            model_id: Kullanılacak model ID'si
            patient_data: Hasta verileri
            
        Returns:
            Risk tahmini sonuçları
        """
        try:
            # 1. Modeli veritabanından al
            model_record = self._get_model_from_database(model_id)
            if not model_record:
                raise Exception(f"Model ID {model_id} bulunamadı!")
            
            # 2. Modeli yükle
            trained_model = self._load_model(model_record.model_path)
            if not trained_model:
                raise Exception("Model yüklenemedi!")
            
            # 3. Hasta verilerini hazırla
            prepared_data = self._prepare_patient_data(
                patient_data, 
                model_record.feature_names,
                trained_model.get('feature_names', [])
            )
            
            # 4. Tahmin yap
            prediction_result = self._make_prediction(trained_model['model'], prepared_data)
            
            # 5. Risk skorunu hesapla
            risk_score = self._calculate_risk_score(
                prediction_result['probability'],
                model_record.f1_score
            )
            
            # 6. Risk seviyesini belirle
            risk_level = self._determine_risk_level(risk_score)
            
            # 7. Önerileri oluştur
            recommendations = self._generate_recommendations(
                patient_data, risk_score, risk_level
            )
            
            # 8. Tahmini veritabanına kaydet
            prediction_record_id = self._save_prediction_to_database(
                model_id, patient_data, prediction_result, risk_score
            )
            
            # Final sonuçlar
            result = {
                'prediction_summary': {
                    'model_name': model_record.model_name,
                    'model_accuracy': model_record.accuracy,
                    'prediction': int(prediction_result['prediction']),
                    'probability': float(prediction_result['probability']),
                    'risk_score': float(risk_score),
                    'risk_level': risk_level,
                    'prediction_id': prediction_record_id
                },
                'patient_analysis': {
                    'input_data': patient_data,
                    'risk_factors': self._identify_risk_factors(patient_data),
                    'protective_factors': self._identify_protective_factors(patient_data)
                },
                'recommendations': recommendations,
                'model_info': {
                    'model_type': model_record.model_type,
                    'training_date': model_record.created_at.isoformat(),
                    'feature_count': len(model_record.feature_names) if model_record.feature_names else 0
                },
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Tahmin hatası: {str(e)}")
    
    def _get_model_from_database(self, model_id: int) -> Optional[ModelRecord]:
        """Modeli veritabanından al."""
        db = SessionLocal()
        try:
            model_record = db.query(ModelRecord).filter(
                ModelRecord.id == model_id,
                ModelRecord.is_active == True
            ).first()
            return model_record
        except Exception as e:
            print(f"Veritabanı hatası: {e}")
            return None
        finally:
            db.close()
    
    def _load_model(self, model_path: str):
        """Modeli dosyadan yükle."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return None
    
    def _prepare_patient_data(self, patient_data: Dict, model_feature_names: List, trained_feature_names: List):
        """Hasta verilerini model için hazırla."""
        # Önce model kaydındaki feature names'i kullan
        feature_names = model_feature_names if model_feature_names else trained_feature_names
        
        if not feature_names:
            raise Exception("Model feature names bulunamadı!")
        
        # Veri doğrulama
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        missing_fields = [field for field in required_fields if field not in patient_data]
        
        if missing_fields:
            raise Exception(f"Eksik alanlar: {', '.join(missing_fields)}")
        
        # Veri tiplerini kontrol et ve dönüştür
        processed_data = {}
        
        # Yaş dönüşümü (gün -> yıl)
        if 'age' in patient_data:
            age_days = patient_data['age']
            if age_days > 1000:  # Gün formatında ise
                processed_data['age_years'] = int(age_days / 365)
            else:  # Yıl formatında ise
                processed_data['age_years'] = int(age_days)
        
        # Diğer sayısal alanlar
        numeric_fields = ['height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        for field in numeric_fields:
            if field in patient_data:
                processed_data[field] = float(patient_data[field])
        
        # Kategorik alanlar
        categorical_fields = ['gender', 'smoke', 'alco', 'active']
        for field in categorical_fields:
            if field in patient_data:
                processed_data[field] = int(patient_data[field])
        
        # DataFrame oluştur
        df = pd.DataFrame([processed_data])
        
        # Feature names'e göre sırala ve eksikleri doldur
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Eksik feature'ları 0 ile doldur
        
        # Sadece model feature'larını al
        df = df[feature_names]
        
        return df
    
    def _make_prediction(self, model, prepared_data):
        """Model ile tahmin yap."""
        try:
            # Tahmin
            prediction = model.predict(prepared_data)[0]
            
            # Olasılık (eğer model destekliyorsa)
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(prepared_data)[0][1]  # Pozitif sınıf olasılığı
            else:
                probability = float(prediction)  # Binary prediction için
            
            return {
                'prediction': prediction,
                'probability': probability
            }
            
        except Exception as e:
            raise Exception(f"Tahmin hatası: {str(e)}")
    
    def _calculate_risk_score(self, probability: float, model_f1_score: float) -> float:
        """
        Risk skorunu hesapla.
        
        Args:
            probability: Model tahmin olasılığı
            model_f1_score: Model F1 skoru
            
        Returns:
            Risk skoru (0-1 arası)
        """
        # Temel risk skoru = tahmin olasılığı
        base_risk = probability
        
        # Model güvenilirliği faktörü (F1 skoruna göre)
        model_reliability = model_f1_score
        
        # Risk skorunu model güvenilirliği ile ayarla
        # Yüksek güvenilirlik = daha güvenilir risk skoru
        adjusted_risk = base_risk * model_reliability + base_risk * (1 - model_reliability) * 0.5
        
        # 0-1 arasında sınırla
        risk_score = max(0.0, min(1.0, adjusted_risk))
        
        return risk_score
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Risk seviyesini belirle."""
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
    
    def _identify_risk_factors(self, patient_data: Dict) -> List[str]:
        """Risk faktörlerini belirle."""
        risk_factors = []
        
        # Yaş faktörü
        age = patient_data.get('age', 0)
        if age > 1000:  # Gün formatında
            age_years = age / 365
        else:
            age_years = age
            
        if age_years > 65:
            risk_factors.append("İleri yaş")
        elif age_years > 50:
            risk_factors.append("Orta yaş")
        
        # Cinsiyet faktörü
        gender = patient_data.get('gender', 1)
        if gender == 1:  # Erkek
            risk_factors.append("Erkek cinsiyet")
        
        # Tansiyon faktörü
        ap_hi = patient_data.get('ap_hi', 0)
        ap_lo = patient_data.get('ap_lo', 0)
        
        if ap_hi >= 140 or ap_lo >= 90:
            risk_factors.append("Yüksek tansiyon")
        elif ap_hi >= 130 or ap_lo >= 80:
            risk_factors.append("Pre-hipertansiyon")
        
        # Kolesterol faktörü
        cholesterol = patient_data.get('cholesterol', 1)
        if cholesterol >= 3:
            risk_factors.append("Yüksek kolesterol")
        
        # Glukoz faktörü
        gluc = patient_data.get('gluc', 1)
        if gluc >= 3:
            risk_factors.append("Yüksek glukoz")
        
        # Sigara faktörü
        smoke = patient_data.get('smoke', 0)
        if smoke == 1:
            risk_factors.append("Sigara kullanımı")
        
        # Alkol faktörü
        alco = patient_data.get('alco', 0)
        if alco == 1:
            risk_factors.append("Alkol kullanımı")
        
        # Hareketsizlik faktörü
        active = patient_data.get('active', 0)
        if active == 0:
            risk_factors.append("Hareketsiz yaşam")
        
        # BMI faktörü
        height = patient_data.get('height', 0)
        weight = patient_data.get('weight', 0)
        if height > 0 and weight > 0:
            bmi = weight / ((height / 100) ** 2)
            if bmi >= 30:
                risk_factors.append("Obezite")
            elif bmi >= 25:
                risk_factors.append("Fazla kilolu")
        
        return risk_factors
    
    def _identify_protective_factors(self, patient_data: Dict) -> List[str]:
        """Koruyucu faktörleri belirle."""
        protective_factors = []
        
        # Aktif yaşam
        active = patient_data.get('active', 0)
        if active == 1:
            protective_factors.append("Aktif yaşam tarzı")
        
        # Sigara kullanmama
        smoke = patient_data.get('smoke', 0)
        if smoke == 0:
            protective_factors.append("Sigara kullanmama")
        
        # Alkol kullanmama
        alco = patient_data.get('alco', 0)
        if alco == 0:
            protective_factors.append("Alkol kullanmama")
        
        # Normal tansiyon
        ap_hi = patient_data.get('ap_hi', 0)
        ap_lo = patient_data.get('ap_lo', 0)
        if ap_hi < 120 and ap_lo < 80:
            protective_factors.append("Normal tansiyon")
        
        # Normal kolesterol
        cholesterol = patient_data.get('cholesterol', 1)
        if cholesterol == 1:
            protective_factors.append("Normal kolesterol")
        
        # Normal glukoz
        gluc = patient_data.get('gluc', 1)
        if gluc == 1:
            protective_factors.append("Normal glukoz")
        
        # Normal BMI
        height = patient_data.get('height', 0)
        weight = patient_data.get('weight', 0)
        if height > 0 and weight > 0:
            bmi = weight / ((height / 100) ** 2)
            if 18.5 <= bmi < 25:
                protective_factors.append("Normal kilo")
        
        return protective_factors
    
    def _generate_recommendations(self, patient_data: Dict, risk_score: float, risk_level: str) -> Dict[str, List[str]]:
        """Öneriler oluştur."""
        recommendations = {
            'lifestyle': [],
            'medical': [],
            'monitoring': []
        }
        
        # Risk seviyesine göre genel öneriler
        if risk_level in ["Yüksek", "Çok Yüksek"]:
            recommendations['lifestyle'].extend([
                "Düzenli egzersiz yapın (haftada en az 150 dakika)",
                "Sağlıklı beslenme programı uygulayın",
                "Stres yönetimi teknikleri öğrenin",
                "Yeterli uyku alın (7-8 saat)"
            ])
            
            recommendations['medical'].extend([
                "Kardiyoloji uzmanına başvurun",
                "Düzenli kan basıncı takibi yapın",
                "Lipid profili kontrolü yaptırın",
                "Diyabet taraması yaptırın"
            ])
            
            recommendations['monitoring'].extend([
                "Günlük kan basıncı ölçümü",
                "Aylık kilo takibi",
                "3 ayda bir kan testleri",
                "Yıllık kardiyak değerlendirme"
            ])
            
        elif risk_level == "Orta":
            recommendations['lifestyle'].extend([
                "Haftada en az 3 gün egzersiz yapın",
                "Meyve ve sebze tüketimini artırın",
                "Tuz tüketimini azaltın",
                "Sigara ve alkol kullanımından kaçının"
            ])
            
            recommendations['medical'].extend([
                "Aile hekiminize düzenli kontrole gidin",
                "Kan basıncınızı takip edin",
                "Kolesterol seviyelerinizi kontrol ettirin"
            ])
            
            recommendations['monitoring'].extend([
                "Haftalık kan basıncı ölçümü",
                "Aylık kilo takibi",
                "6 ayda bir kan testleri"
            ])
            
        else:  # Düşük risk
            recommendations['lifestyle'].extend([
                "Mevcut sağlıklı yaşam tarzınızı sürdürün",
                "Düzenli fiziksel aktivite yapın",
                "Sağlıklı beslenme alışkanlıklarınızı koruyun"
            ])
            
            recommendations['medical'].extend([
                "Yıllık sağlık kontrolü yaptırın",
                "Aile geçmişinizi takip edin"
            ])
            
            recommendations['monitoring'].extend([
                "Yıllık kan basıncı kontrolü",
                "Yıllık kan testleri"
            ])
        
        # Spesifik risk faktörlerine göre öneriler
        risk_factors = self._identify_risk_factors(patient_data)
        
        if "Yüksek tansiyon" in risk_factors:
            recommendations['medical'].append("Hipertansiyon tedavisi için doktora başvurun")
            recommendations['lifestyle'].append("Düşük sodyum diyeti uygulayın")
        
        if "Yüksek kolesterol" in risk_factors:
            recommendations['medical'].append("Kolesterol düşürücü tedavi için doktora başvurun")
            recommendations['lifestyle'].append("Düşük kolesterol diyeti uygulayın")
        
        if "Sigara kullanımı" in risk_factors:
            recommendations['lifestyle'].append("Sigara bırakma programına katılın")
            recommendations['medical'].append("Sigara bırakma danışmanlığı alın")
        
        if "Obezite" in risk_factors:
            recommendations['lifestyle'].append("Kilo verme programına katılın")
            recommendations['medical'].append("Beslenme uzmanından yardım alın")
        
        return recommendations
    
    def _save_prediction_to_database(self, model_id: int, patient_data: Dict, prediction_result: Dict, risk_score: float) -> int:
        """Tahmini veritabanına kaydet."""
        db = SessionLocal()
        try:
            prediction_record = PredictionRecord(
                model_id=model_id,
                patient_data=json.dumps(patient_data),
                prediction=int(prediction_result['prediction']),
                probability=float(prediction_result['probability']),
                risk_score=float(risk_score)
            )
            
            db.add(prediction_record)
            db.commit()
            db.refresh(prediction_record)
            
            return prediction_record.id
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Tahmin kayıt hatası: {str(e)}")
        finally:
            db.close()
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Kullanılabilir modelleri listele."""
        db = SessionLocal()
        try:
            models = db.query(ModelRecord).filter(ModelRecord.is_active == True).all()
            
            model_list = []
            for model in models:
                model_list.append({
                    'id': model.id,
                    'model_name': model.model_name,
                    'model_type': model.model_type,
                    'accuracy': model.accuracy,
                    'f1_score': model.f1_score,
                    'created_at': model.created_at.isoformat(),
                    'feature_count': len(model.feature_names) if model.feature_names else 0
                })
            
            return model_list
            
        except Exception as e:
            print(f"Model listesi hatası: {e}")
            return []
        finally:
            db.close()

# Global service instance
prediction_service = PredictionService()
