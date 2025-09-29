"""
Risk Tahmin Router
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.services.prediction_service import prediction_service

router = APIRouter()

class PatientData(BaseModel):
    """Hasta verileri."""
    id: Optional[int] = None
    age: int = Field(..., description="Yaş (gün veya yıl)", example=45)
    gender: int = Field(..., description="Cinsiyet (1: Erkek, 2: Kadın)", example=1)
    height: float = Field(..., description="Boy (cm)", example=175.0)
    weight: float = Field(..., description="Kilo (kg)", example=75.0)
    ap_hi: int = Field(..., description="Sistolik kan basıncı (mmHg)", example=120)
    ap_lo: int = Field(..., description="Diyastolik kan basıncı (mmHg)", example=80)
    cholesterol: int = Field(..., description="Kolesterol (1: Normal, 2: Yüksek, 3: Çok yüksek)", example=1)
    gluc: int = Field(..., description="Glukoz (1: Normal, 2: Yüksek, 3: Çok yüksek)", example=1)
    smoke: int = Field(..., description="Sigara (0: Hayır, 1: Evet)", example=0)
    alco: int = Field(..., description="Alkol (0: Hayır, 1: Evet)", example=0)
    active: int = Field(..., description="Fiziksel aktivite (0: Hayır, 1: Evet)", example=1)
    
    class Config:
        schema_extra = {
            "example": {
                "id": 12345,
                "age": 45,
                "gender": 1,
                "height": 175.0,
                "weight": 75.0,
                "ap_hi": 120,
                "ap_lo": 80,
                "cholesterol": 1,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        }

class PredictionRequest(BaseModel):
    """Risk tahmin isteği."""
    model_id: int = Field(..., description="Kullanılacak model ID'si", example=1)
    patient_data: PatientData = Field(..., description="Hasta verileri")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": 1,
                "patient_data": {
                    "id": 12345,
                    "age": 45,
                    "gender": 1,
                    "height": 175.0,
                    "weight": 75.0,
                    "ap_hi": 120,
                    "ap_lo": 80,
                    "cholesterol": 1,
                    "gluc": 1,
                    "smoke": 0,
                    "alco": 0,
                    "active": 1
                }
            }
        }

class PredictionResponse(BaseModel):
    """Risk tahmin yanıtı."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Risk tahmini başarıyla tamamlandı",
                "data": {
                    "prediction_summary": {
                        "model_name": "Random Forest",
                        "model_accuracy": 0.8156,
                        "prediction": 0,
                        "probability": 0.2345,
                        "risk_score": 0.1844,
                        "risk_level": "Düşük",
                        "prediction_id": 1
                    },
                    "patient_analysis": {
                        "risk_factors": ["Erkek cinsiyet"],
                        "protective_factors": ["Aktif yaşam tarzı", "Sigara kullanmama", "Alkol kullanmama"]
                    },
                    "recommendations": {
                        "lifestyle": ["Mevcut sağlıklı yaşam tarzınızı sürdürün"],
                        "medical": ["Yıllık sağlık kontrolü yaptırın"],
                        "monitoring": ["Yıllık kan basıncı kontrolü"]
                    }
                }
            }
        }

class AvailableModelsResponse(BaseModel):
    """Kullanılabilir modeller yanıtı."""
    success: bool
    message: str
    models: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Kullanılabilir modeller listelendi",
                "models": [
                    {
                        "id": 1,
                        "model_name": "Random Forest",
                        "model_type": "tree_based",
                        "accuracy": 0.8156,
                        "f1_score": 0.8234,
                        "created_at": "2024-12-01T14:30:22",
                        "feature_count": 12
                    }
                ]
            }
        }

@router.post("/predict", response_model=PredictionResponse)
async def predict_cardio_risk(request: PredictionRequest):
    """
    Hastanın kardiyovasküler riskini tahmin et.
    
    Bu endpoint:
    - Seçilen modeli kullanarak risk tahmini yapar
    - Risk skorunu ve seviyesini hesaplar
    - Risk faktörlerini analiz eder
    - Kişiselleştirilmiş öneriler sunar
    
    **Girdi Parametreleri:**
    - model_id: Kullanılacak model ID'si
    - patient_data: Hasta demografik ve klinik verileri
    
    **Çıktılar:**
    - Risk tahmini (0: Düşük risk, 1: Yüksek risk)
    - Risk olasılığı (0-1 arası)
    - Risk skoru (0-1 arası)
    - Risk seviyesi (Çok Düşük, Düşük, Orta, Yüksek, Çok Yüksek)
    - Risk faktörleri analizi
    - Koruyucu faktörler
    - Kişiselleştirilmiş öneriler
    """
    try:
        # Hasta verilerini dictionary'e çevir
        patient_data_dict = request.patient_data.dict()
        
        # Risk tahminini çalıştır
        result = await prediction_service.predict_cardio_risk(
            model_id=request.model_id,
            patient_data=patient_data_dict
        )
        
        return PredictionResponse(
            success=True,
            message="Risk tahmini başarıyla tamamlandı",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk tahmin hatası: {str(e)}"
        )

@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """
    Risk tahmini için kullanılabilir modelleri listele.
    
    Bu endpoint, risk tahmini yapmak için kullanılabilecek tüm aktif modelleri listeler.
    """
    try:
        models = await prediction_service.get_available_models()
        
        return AvailableModelsResponse(
            success=True,
            message=f"{len(models)} kullanılabilir model bulundu",
            models=models
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model listesi hatası: {str(e)}"
        )

@router.post("/predict/sample")
async def predict_sample():
    """
    Örnek risk tahmini döndür.
    
    Bu endpoint, gerçek model kullanmadan örnek tahmin sonucu döndürür.
    Test ve dokümantasyon amaçlı kullanılır.
    """
    sample_result = {
        "prediction_summary": {
            "model_name": "Random Forest",
            "model_accuracy": 0.8156,
            "prediction": 0,
            "probability": 0.2345,
            "risk_score": 0.1844,
            "risk_level": "Düşük",
            "prediction_id": 1
        },
        "patient_analysis": {
            "input_data": {
                "id": 12345,
                "age": 45,
                "gender": 1,
                "height": 175.0,
                "weight": 75.0,
                "ap_hi": 120,
                "ap_lo": 80,
                "cholesterol": 1,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            },
            "risk_factors": [
                "Erkek cinsiyet"
            ],
            "protective_factors": [
                "Aktif yaşam tarzı",
                "Sigara kullanmama",
                "Alkol kullanmama",
                "Normal tansiyon",
                "Normal kolesterol",
                "Normal glukoz",
                "Normal kilo"
            ]
        },
        "recommendations": {
            "lifestyle": [
                "Mevcut sağlıklı yaşam tarzınızı sürdürün",
                "Düzenli fiziksel aktivite yapın",
                "Sağlıklı beslenme alışkanlıklarınızı koruyun"
            ],
            "medical": [
                "Yıllık sağlık kontrolü yaptırın",
                "Aile geçmişinizi takip edin"
            ],
            "monitoring": [
                "Yıllık kan basıncı kontrolü",
                "Yıllık kan testleri"
            ]
        },
        "model_info": {
            "model_type": "tree_based",
            "training_date": "2024-12-01T14:30:22",
            "feature_count": 12
        },
        "prediction_timestamp": "2024-12-01T15:45:30"
    }
    
    return PredictionResponse(
        success=True,
        message="Örnek risk tahmini",
        data=sample_result
    )

@router.get("/risk-levels")
async def get_risk_levels():
    """
    Risk seviyeleri hakkında bilgi ver.
    
    Bu endpoint, risk seviyelerinin açıklamalarını ve önerilerini döndürür.
    """
    risk_levels = {
        "Çok Düşük": {
            "score_range": "0.0 - 0.2",
            "description": "Kardiyovasküler risk çok düşük",
            "recommendations": [
                "Mevcut sağlıklı yaşam tarzınızı sürdürün",
                "Yıllık sağlık kontrolü yaptırın",
                "Aile geçmişinizi takip edin"
            ],
            "monitoring": "Yıllık kontroller yeterli"
        },
        "Düşük": {
            "score_range": "0.2 - 0.4",
            "description": "Kardiyovasküler risk düşük",
            "recommendations": [
                "Sağlıklı beslenme alışkanlıklarınızı koruyun",
                "Düzenli fiziksel aktivite yapın",
                "Stres yönetimi teknikleri öğrenin"
            ],
            "monitoring": "6 ayda bir kontroller"
        },
        "Orta": {
            "score_range": "0.4 - 0.6",
            "description": "Kardiyovasküler risk orta seviyede",
            "recommendations": [
                "Haftada en az 3 gün egzersiz yapın",
                "Meyve ve sebze tüketimini artırın",
                "Tuz tüketimini azaltın",
                "Sigara ve alkol kullanımından kaçının"
            ],
            "monitoring": "3 ayda bir kontroller"
        },
        "Yüksek": {
            "score_range": "0.6 - 0.8",
            "description": "Kardiyovasküler risk yüksek",
            "recommendations": [
                "Düzenli egzersiz yapın (haftada en az 150 dakika)",
                "Sağlıklı beslenme programı uygulayın",
                "Stres yönetimi teknikleri öğrenin",
                "Yeterli uyku alın (7-8 saat)"
            ],
            "monitoring": "Aylık kontroller"
        },
        "Çok Yüksek": {
            "score_range": "0.8 - 1.0",
            "description": "Kardiyovasküler risk çok yüksek",
            "recommendations": [
                "Acil olarak kardiyoloji uzmanına başvurun",
                "Düzenli egzersiz programına katılın",
                "Beslenme uzmanından yardım alın",
                "Sigara bırakma programına katılın"
            ],
            "monitoring": "Haftalık kontroller"
        }
    }
    
    return {
        "success": True,
        "message": "Risk seviyeleri bilgisi",
        "risk_levels": risk_levels,
        "note": "Risk skorları 0-1 arasında normalize edilmiştir. Yüksek skor = yüksek risk"
    }

@router.get("/risk-factors")
async def get_risk_factors():
    """
    Kardiyovasküler risk faktörleri hakkında bilgi ver.
    
    Bu endpoint, kardiyovasküler hastalık risk faktörlerini ve koruyucu faktörleri açıklar.
    """
    risk_factors_info = {
        "major_risk_factors": {
            "age": {
                "description": "İleri yaş",
                "threshold": "65 yaş üstü",
                "impact": "Yüksek",
                "modifiable": False
            },
            "gender": {
                "description": "Erkek cinsiyet",
                "threshold": "Erkek",
                "impact": "Orta",
                "modifiable": False
            },
            "hypertension": {
                "description": "Yüksek tansiyon",
                "threshold": "140/90 mmHg üstü",
                "impact": "Yüksek",
                "modifiable": True
            },
            "cholesterol": {
                "description": "Yüksek kolesterol",
                "threshold": "200 mg/dL üstü",
                "impact": "Yüksek",
                "modifiable": True
            },
            "diabetes": {
                "description": "Diyabet",
                "threshold": "Glukoz yüksekliği",
                "impact": "Yüksek",
                "modifiable": True
            },
            "smoking": {
                "description": "Sigara kullanımı",
                "threshold": "Aktif sigara içimi",
                "impact": "Yüksek",
                "modifiable": True
            },
            "obesity": {
                "description": "Obezite",
                "threshold": "BMI 30 üstü",
                "impact": "Orta",
                "modifiable": True
            },
            "physical_inactivity": {
                "description": "Hareketsizlik",
                "threshold": "Düzenli egzersiz yapmama",
                "impact": "Orta",
                "modifiable": True
            }
        },
        "protective_factors": {
            "regular_exercise": {
                "description": "Düzenli egzersiz",
                "recommendation": "Haftada en az 150 dakika",
                "benefit": "Kalp sağlığını korur"
            },
            "healthy_diet": {
                "description": "Sağlıklı beslenme",
                "recommendation": "Meyve, sebze, tam tahıl",
                "benefit": "Kolesterol ve tansiyonu düşürür"
            },
            "no_smoking": {
                "description": "Sigara kullanmama",
                "recommendation": "Sigara bırakma",
                "benefit": "Kalp hastalığı riskini azaltır"
            },
            "moderate_alcohol": {
                "description": "Alkol kullanmama/az kullanma",
                "recommendation": "Günde 1-2 kadeh",
                "benefit": "Kalp sağlığını korur"
            },
            "stress_management": {
                "description": "Stres yönetimi",
                "recommendation": "Meditasyon, yoga",
                "benefit": "Tansiyonu düşürür"
            },
            "adequate_sleep": {
                "description": "Yeterli uyku",
                "recommendation": "7-8 saat",
                "benefit": "Genel sağlığı korur"
            }
        },
        "risk_calculation": {
            "method": "Machine Learning Model",
            "factors_considered": [
                "Yaş", "Cinsiyet", "Boy", "Kilo", "Kan basıncı",
                "Kolesterol", "Glukoz", "Sigara", "Alkol", "Fiziksel aktivite"
            ],
            "output": "Risk skoru (0-1), Risk seviyesi, Kişiselleştirilmiş öneriler"
        }
    }
    
    return {
        "success": True,
        "message": "Risk faktörleri bilgisi",
        "risk_factors_info": risk_factors_info
    }

@router.get("/prediction/status")
async def get_prediction_status():
    """
    Risk tahmin servisi durumunu kontrol et.
    
    Bu endpoint, risk tahmin servisinin durumunu döndürür.
    """
    return {
        "status": "ready",
        "message": "Risk tahmin servisi hazır",
        "features": {
            "model_selection": True,
            "risk_scoring": True,
            "factor_analysis": True,
            "personalized_recommendations": True,
            "database_integration": True
        },
        "supported_models": "Tüm eğitilmiş modeller"
    }
