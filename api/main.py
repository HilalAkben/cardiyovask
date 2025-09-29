"""
Kalp Krizi Risk Tahmin API - FastAPI Ana Uygulama
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.routes import model_comparison, model_training, prediction
from api.core.config import settings
from api.core.database import init_database

# FastAPI uygulaması oluştur
app = FastAPI(
    title="Kalp Krizi Risk Tahmin API",
    description="""
    ## Kalp Krizi Risk Tahmin Sistemi
    
    Bu API, kardiyovasküler hastalık riskini tahmin etmek için geliştirilmiş makine öğrenmesi modellerini kullanır.
    
    ### Özellikler:
    - **Model Karşılaştırması**: Farklı ML algoritmalarının performansını karşılaştırır
    - **Model Eğitimi**: Seçilen modeli eğitir ve detaylı analiz sağlar
    - **Risk Tahmini**: Hastanın kardiyovasküler risk skorunu hesaplar
    
    ### Desteklenen Modeller:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Logistic Regression
    - SVM
    - Ensemble Methods
    """,
    version="1.0.0",
    contact={
        "name": "Kalp Krizi Risk Tahmin API",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# CORS middleware ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domainler belirtilmeli
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router'ları ekle
app.include_router(
    model_comparison.router,
    prefix="/api/v1/models",
    tags=["Model Karşılaştırması"]
)

app.include_router(
    model_training.router,
    prefix="/api/v1/training",
    tags=["Model Eğitimi"]
)

app.include_router(
    prediction.router,
    prefix="/api/v1/prediction",
    tags=["Risk Tahmini"]
)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında çalışacak fonksiyonlar."""
    print("🚀 Kalp Krizi Risk Tahmin API başlatılıyor...")
    
    # Veritabanını başlat
    await init_database()
    
    # Model klasörünü oluştur
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("✅ API başarıyla başlatıldı!")
    print(f"📊 Swagger UI: http://localhost:8000/docs")
    print(f"📋 ReDoc: http://localhost:8000/redoc")

@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapatıldığında çalışacak fonksiyonlar."""
    print("🛑 API kapatılıyor...")

@app.get("/", tags=["Ana Sayfa"])
async def root():
    """Ana sayfa - API hakkında bilgi."""
    return {
        "message": "Kalp Krizi Risk Tahmin API'ye hoş geldiniz!",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "model_comparison": "/api/v1/models/compare",
            "model_training": "/api/v1/training/train",
            "prediction": "/api/v1/prediction/predict"
        }
    }

@app.get("/health", tags=["Sistem Durumu"])
async def health_check():
    """Sistem sağlık kontrolü."""
    return {
        "status": "healthy",
        "message": "API çalışıyor",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
