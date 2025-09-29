"""
API Konfigürasyon Modülü
"""

from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    """Uygulama ayarları."""
    
    # API Ayarları
    app_name: str = "Kalp Krizi Risk Tahmin API"
    debug: bool = True
    version: str = "1.0.0"
    
    # Veritabanı Ayarları
    database_url: str = "sqlite:///./cardio_models.db"
    
    # Model Ayarları
    models_dir: str = "models"
    data_file: str = "src/data/cardiokaggle.csv"
    
    # GPU Ayarları
    use_gpu: bool = True
    gpu_device_id: int = 0
    
    # Model Parametreleri
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Dosya Yolları
    project_root: Path = Path(__file__).parent.parent.parent
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
