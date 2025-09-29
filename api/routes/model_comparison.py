"""
Model Karşılaştırma Router
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.services.model_comparison_service import model_comparison_service
from api.core.config import settings

router = APIRouter()

class ModelComparisonRequest(BaseModel):
    """Model karşılaştırma isteği."""
    data_path: Optional[str] = None
    use_gpu: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "data_path": "src/data/cardiokaggle.csv",
                "use_gpu": True
            }
        }

class ModelComparisonResponse(BaseModel):
    """Model karşılaştırma yanıtı."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Model karşılaştırması başarıyla tamamlandı",
                "data": {
                    "comparison_summary": {
                        "total_models": 6,
                        "data_with_outliers": {
                            "best_model": "Random Forest",
                            "best_f1_score": 0.8234
                        },
                        "data_without_outliers": {
                            "best_model": "XGBoost",
                            "best_f1_score": 0.8456
                        }
                    },
                    "model_rankings": {
                        "with_outliers": [
                            {
                                "rank": 1,
                                "model_name": "Random Forest",
                                "f1_score": 0.8234,
                                "risk_score": 0.1766,
                                "risk_level": "Düşük"
                            }
                        ]
                    }
                },
                "execution_time": 45.67
            }
        }

@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(request: ModelComparisonRequest):
    """
    Tüm modelleri karşılaştır ve performanslarını analiz et.
    
    Bu endpoint:
    - Tüm makine öğrenmesi modellerini eğitir
    - Performanslarını karşılaştırır
    - Risk skorlarına göre sıralar
    - Detaylı analiz sonuçları döndürür
    
    **Desteklenen Modeller:**
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Logistic Regression
    - SVM
    - Ensemble Methods
    
    **Çıktılar:**
    - Model performans karşılaştırması
    - Risk skorları (F1-Score'a göre)
    - Feature importance analizi
    - Outlier analizi
    - Olasılık metrikleri
    """
    try:
        import time
        start_time = time.time()
        
        # Veri dosyası yolunu belirle
        if request.data_path:
            # Kullanıcı tarafından verilen yol
            if not Path(request.data_path).is_absolute():
                # Relatif yol ise proje kökünden başlat
                data_path = str(settings.project_root / request.data_path)
            else:
                data_path = request.data_path
        else:
            # Varsayılan yol
            data_path = str(settings.project_root / settings.data_file)
        
        # Dosya varlığını kontrol et
        if not Path(data_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Veri dosyası bulunamadı: {data_path}"
            )
        
        # Model karşılaştırmasını çalıştır
        result = await model_comparison_service.run_model_comparison(data_path)
        
        execution_time = time.time() - start_time
        
        return ModelComparisonResponse(
            success=True,
            message="Model karşılaştırması başarıyla tamamlandı",
            data=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model karşılaştırma hatası: {str(e)}"
        )

@router.get("/compare/status")
async def get_comparison_status():
    """
    Model karşılaştırma durumunu kontrol et.
    
    Bu endpoint, mevcut model karşılaştırma işleminin durumunu döndürür.
    """
    return {
        "status": "ready",
        "message": "Model karşılaştırma servisi hazır",
        "supported_models": [
            "Random Forest",
            "Gradient Boosting", 
            "XGBoost",
            "LightGBM",
            "Logistic Regression",
            "SVM",
            "Ensemble Methods"
        ],
        "data_file": str(settings.project_root / settings.data_file)
    }

@router.get("/compare/sample")
async def get_sample_comparison():
    """
    Örnek model karşılaştırma sonucu döndür.
    
    Bu endpoint, gerçek veri işleme yapmadan örnek sonuç döndürür.
    Test ve dokümantasyon amaçlı kullanılır.
    """
    sample_result = {
        "comparison_summary": {
            "total_models": 6,
            "data_with_outliers": {
                "train_shape": [56042, 12],
                "test_shape": [14011, 12],
                "best_model": "Random Forest",
                "best_f1_score": 0.8234
            },
            "data_without_outliers": {
                "train_shape": [52000, 12],
                "test_shape": [13000, 12],
                "best_model": "XGBoost",
                "best_f1_score": 0.8456
            }
        },
        "model_rankings": {
            "with_outliers": [
                {
                    "rank": 1,
                    "model_name": "Random Forest",
                    "f1_score": 0.8234,
                    "accuracy": 0.8156,
                    "precision": 0.8123,
                    "recall": 0.8345,
                    "roc_auc": 0.8456,
                    "risk_score": 0.1766,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 2,
                    "model_name": "XGBoost",
                    "f1_score": 0.8156,
                    "accuracy": 0.8089,
                    "precision": 0.8034,
                    "recall": 0.8278,
                    "roc_auc": 0.8389,
                    "risk_score": 0.1844,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 3,
                    "model_name": "Gradient Boosting",
                    "f1_score": 0.8089,
                    "accuracy": 0.8023,
                    "precision": 0.7989,
                    "recall": 0.8189,
                    "roc_auc": 0.8323,
                    "risk_score": 0.1911,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 4,
                    "model_name": "LightGBM",
                    "f1_score": 0.8023,
                    "accuracy": 0.7956,
                    "precision": 0.7923,
                    "recall": 0.8123,
                    "roc_auc": 0.8256,
                    "risk_score": 0.1977,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 5,
                    "model_name": "Logistic Regression",
                    "f1_score": 0.7654,
                    "accuracy": 0.7589,
                    "precision": 0.7543,
                    "recall": 0.7765,
                    "roc_auc": 0.7898,
                    "risk_score": 0.2346,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 6,
                    "model_name": "SVM",
                    "f1_score": 0.7234,
                    "accuracy": 0.7167,
                    "precision": 0.7123,
                    "recall": 0.7345,
                    "roc_auc": 0.7478,
                    "risk_score": 0.2766,
                    "risk_level": "Düşük"
                }
            ],
            "without_outliers": [
                {
                    "rank": 1,
                    "model_name": "XGBoost",
                    "f1_score": 0.8456,
                    "accuracy": 0.8389,
                    "precision": 0.8345,
                    "recall": 0.8567,
                    "roc_auc": 0.8678,
                    "risk_score": 0.1544,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 2,
                    "model_name": "Random Forest",
                    "f1_score": 0.8389,
                    "accuracy": 0.8323,
                    "precision": 0.8278,
                    "recall": 0.8500,
                    "roc_auc": 0.8600,
                    "risk_score": 0.1611,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 3,
                    "model_name": "Gradient Boosting",
                    "f1_score": 0.8323,
                    "accuracy": 0.8256,
                    "precision": 0.8211,
                    "recall": 0.8434,
                    "roc_auc": 0.8523,
                    "risk_score": 0.1677,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 4,
                    "model_name": "LightGBM",
                    "f1_score": 0.8256,
                    "accuracy": 0.8189,
                    "precision": 0.8145,
                    "recall": 0.8367,
                    "roc_auc": 0.8445,
                    "risk_score": 0.1744,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 5,
                    "model_name": "Logistic Regression",
                    "f1_score": 0.7898,
                    "accuracy": 0.7834,
                    "precision": 0.7789,
                    "recall": 0.8007,
                    "roc_auc": 0.8123,
                    "risk_score": 0.2102,
                    "risk_level": "Düşük"
                },
                {
                    "rank": 6,
                    "model_name": "SVM",
                    "f1_score": 0.7478,
                    "accuracy": 0.7412,
                    "precision": 0.7367,
                    "recall": 0.7589,
                    "roc_auc": 0.7701,
                    "risk_score": 0.2522,
                    "risk_level": "Düşük"
                }
            ]
        },
        "probability_metrics": {
            "with_outliers": {
                "Random Forest": {
                    "brier_score": 0.1567,
                    "log_loss": 0.4234
                },
                "XGBoost": {
                    "brier_score": 0.1634,
                    "log_loss": 0.4456
                }
            },
            "without_outliers": {
                "XGBoost": {
                    "brier_score": 0.1345,
                    "log_loss": 0.3789
                },
                "Random Forest": {
                    "brier_score": 0.1412,
                    "log_loss": 0.4012
                }
            }
        },
        "feature_importance": {
            "with_outliers": {
                "Random Forest": [
                    {"feature": "age_years", "importance": 0.2345},
                    {"feature": "ap_hi", "importance": 0.1987},
                    {"feature": "cholesterol", "importance": 0.1765},
                    {"feature": "weight", "importance": 0.1543},
                    {"feature": "ap_lo", "importance": 0.1321}
                ]
            },
            "without_outliers": {
                "XGBoost": [
                    {"feature": "age_years", "importance": 0.2456},
                    {"feature": "ap_hi", "importance": 0.2034},
                    {"feature": "cholesterol", "importance": 0.1823},
                    {"feature": "weight", "importance": 0.1612},
                    {"feature": "ap_lo", "importance": 0.1398}
                ]
            }
        },
        "outlier_analysis": {
            "with_outliers": {
                "age": {"outlier_count": 1234, "outlier_percentage": 1.76},
                "height": {"outlier_count": 567, "outlier_percentage": 0.81},
                "weight": {"outlier_count": 890, "outlier_percentage": 1.27}
            },
            "without_outliers": {
                "age": {"outlier_count": 0, "outlier_percentage": 0.0},
                "height": {"outlier_count": 0, "outlier_percentage": 0.0},
                "weight": {"outlier_count": 0, "outlier_percentage": 0.0}
            }
        }
    }
    
    return ModelComparisonResponse(
        success=True,
        message="Örnek model karşılaştırma sonucu",
        data=sample_result,
        execution_time=0.0
    )
