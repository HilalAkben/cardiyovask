"""
Model Eğitimi Router
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.services.model_training_service import model_training_service
from api.core.config import settings

router = APIRouter()

class ModelTrainingRequest(BaseModel):
    """Model eğitimi isteği."""
    model_name: str
    data_path: Optional[str] = None
    use_advanced_features: bool = True
    use_gpu: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "Random Forest",
                "data_path": "src/data/cardiokaggle.csv",
                "use_advanced_features": True,
                "use_gpu": True
            }
        }

class ModelTrainingResponse(BaseModel):
    """Model eğitimi yanıtı."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Model eğitimi başarıyla tamamlandı",
                "data": {
                    "training_summary": {
                        "model_name": "Random Forest",
                        "model_type": "tree_based",
                        "data_shape": {
                            "train": [56042, 12],
                            "test": [14011, 12],
                            "features": 12
                        }
                    },
                    "evaluation_results": {
                        "accuracy": 0.8156,
                        "precision": 0.8123,
                        "recall": 0.8345,
                        "f1_score": 0.8234,
                        "roc_auc": 0.8456
                    },
                    "model_path": "models/random_forest_20241201_143022.pkl",
                    "model_record_id": 1
                },
                "execution_time": 23.45
            }
        }

class ModelListResponse(BaseModel):
    """Model listesi yanıtı."""
    success: bool
    message: str
    models: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Modeller başarıyla listelendi",
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

@router.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    """
    Seçilen modeli eğit ve detaylı analiz sağla.
    
    Bu endpoint:
    - Belirtilen modeli eğitir
    - Detaylı performans analizi yapar
    - Feature importance analizi sağlar
    - Modeli kaydeder ve veritabanına kayıt eder
    
    **Desteklenen Modeller:**
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Logistic Regression
    - SVM
    
    **Özellikler:**
    - Gelişmiş feature engineering
    - GPU desteği
    - Detaylı analiz raporu
    - Model kaydetme
    """
    try:
        import time
        start_time = time.time()
        
        # Model adını doğrula
        supported_models = [
            "Random Forest", "random forest", "randomforest",
            "Gradient Boosting", "gradient boosting", "gradientboosting",
            "XGBoost", "xgboost",
            "LightGBM", "lightgbm",
            "Logistic Regression", "logistic regression", "logisticregression",
            "SVM", "svm"
        ]
        
        if request.model_name not in supported_models:
            raise HTTPException(
                status_code=400,
                detail=f"Desteklenmeyen model: {request.model_name}. Desteklenen modeller: {supported_models}"
            )
        
        # Veri dosyası yolunu belirle
        data_path = request.data_path or str(settings.project_root / settings.data_file)
        
        # Dosya varlığını kontrol et
        if not Path(data_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Veri dosyası bulunamadı: {data_path}"
            )
        
        # Model eğitimini çalıştır
        result = await model_training_service.train_selected_model(
            model_name=request.model_name,
            data_path=data_path,
            use_advanced_features=request.use_advanced_features,
            use_gpu=request.use_gpu
        )
        
        execution_time = time.time() - start_time
        
        return ModelTrainingResponse(
            success=True,
            message=f"{request.model_name} modeli başarıyla eğitildi",
            data=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model eğitimi hatası: {str(e)}"
        )

@router.get("/models", response_model=ModelListResponse)
async def get_trained_models():
    """
    Eğitilmiş modelleri listele.
    
    Bu endpoint, veritabanında kayıtlı tüm aktif modelleri listeler.
    """
    try:
        from api.core.database import SessionLocal, ModelRecord
        
        db = SessionLocal()
        try:
            models = db.query(ModelRecord).filter(ModelRecord.is_active == True).all()
            
            model_list = []
            for model in models:
                model_list.append({
                    "id": model.id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "accuracy": model.accuracy,
                    "precision": model.precision,
                    "recall": model.recall,
                    "f1_score": model.f1_score,
                    "roc_auc": model.roc_auc,
                    "created_at": model.created_at.isoformat(),
                    "feature_count": len(model.feature_names) if model.feature_names else 0,
                    "model_path": model.model_path
                })
            
            return ModelListResponse(
                success=True,
                message=f"{len(model_list)} model bulundu",
                models=model_list
            )
            
        finally:
            db.close()
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model listesi hatası: {str(e)}"
        )

@router.get("/models/{model_id}")
async def get_model_details(model_id: int):
    """
    Belirli bir modelin detaylarını getir.
    
    Bu endpoint, belirtilen model ID'sine sahip modelin detaylı bilgilerini döndürür.
    """
    try:
        from api.core.database import SessionLocal, ModelRecord
        
        db = SessionLocal()
        try:
            model = db.query(ModelRecord).filter(
                ModelRecord.id == model_id,
                ModelRecord.is_active == True
            ).first()
            
            if not model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model ID {model_id} bulunamadı"
                )
            
            return {
                "success": True,
                "message": "Model detayları başarıyla getirildi",
                "model": {
                    "id": model.id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "accuracy": model.accuracy,
                    "precision": model.precision,
                    "recall": model.recall,
                    "f1_score": model.f1_score,
                    "roc_auc": model.roc_auc,
                    "model_path": model.model_path,
                    "feature_names": model.feature_names,
                    "training_params": model.training_params,
                    "created_at": model.created_at.isoformat(),
                    "is_active": model.is_active
                }
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model detay hatası: {str(e)}"
        )

@router.delete("/models/{model_id}")
async def delete_model(model_id: int):
    """
    Belirli bir modeli sil (soft delete).
    
    Bu endpoint, belirtilen model ID'sine sahip modeli pasif hale getirir.
    """
    try:
        from api.core.database import SessionLocal, ModelRecord
        
        db = SessionLocal()
        try:
            model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
            
            if not model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model ID {model_id} bulunamadı"
                )
            
            # Soft delete - is_active'i False yap
            model.is_active = False
            db.commit()
            
            return {
                "success": True,
                "message": f"Model {model.model_name} başarıyla silindi"
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model silme hatası: {str(e)}"
        )

@router.get("/supported-models")
async def get_supported_models():
    """
    Desteklenen modelleri listele.
    
    Bu endpoint, eğitilebilecek tüm model türlerini ve özelliklerini döndürür.
    """
    supported_models = {
        "tree_based": [
            {
                "name": "Random Forest",
                "description": "Çoklu karar ağacı ensemble metodu",
                "strengths": ["Yüksek doğruluk", "Feature importance", "Outlier'a dayanıklı"],
                "weaknesses": ["Overfitting riski", "Yavaş tahmin"],
                "best_for": "Genel amaçlı sınıflandırma"
            },
            {
                "name": "Gradient Boosting",
                "description": "Gradient boosting ensemble metodu",
                "strengths": ["Yüksek performans", "Feature importance", "Non-linear ilişkiler"],
                "weaknesses": ["Overfitting riski", "Yavaş eğitim"],
                "best_for": "Yüksek doğruluk gereken durumlar"
            },
            {
                "name": "XGBoost",
                "description": "Extreme Gradient Boosting",
                "strengths": ["En yüksek performans", "GPU desteği", "Hızlı eğitim"],
                "weaknesses": ["Karmaşık parametreler", "Overfitting riski"],
                "best_for": "Competition ve production ortamları"
            },
            {
                "name": "LightGBM",
                "description": "Light Gradient Boosting Machine",
                "strengths": ["Hızlı eğitim", "Düşük bellek kullanımı", "GPU desteği"],
                "weaknesses": ["Küçük veri setlerinde overfitting"],
                "best_for": "Büyük veri setleri"
            }
        ],
        "linear": [
            {
                "name": "Logistic Regression",
                "description": "Doğrusal sınıflandırma modeli",
                "strengths": ["Hızlı", "Yorumlanabilir", "Overfitting riski düşük"],
                "weaknesses": ["Doğrusal ilişkiler varsayımı", "Düşük performans"],
                "best_for": "Baseline model ve yorumlanabilirlik"
            },
            {
                "name": "SVM",
                "description": "Support Vector Machine",
                "strengths": ["Küçük veri setlerinde iyi", "Non-linear kernel desteği"],
                "weaknesses": ["Büyük veri setlerinde yavaş", "Yorumlanabilir değil"],
                "best_for": "Küçük-orta boyutlu veri setleri"
            }
        ]
    }
    
    return {
        "success": True,
        "message": "Desteklenen modeller listelendi",
        "supported_models": supported_models,
        "total_count": sum(len(models) for models in supported_models.values())
    }

@router.get("/training/status")
async def get_training_status():
    """
    Model eğitimi durumunu kontrol et.
    
    Bu endpoint, model eğitimi servisinin durumunu döndürür.
    """
    return {
        "status": "ready",
        "message": "Model eğitimi servisi hazır",
        "features": {
            "advanced_feature_engineering": True,
            "gpu_support": True,
            "model_persistence": True,
            "database_integration": True
        },
        "data_file": str(settings.project_root / settings.data_file)
    }
