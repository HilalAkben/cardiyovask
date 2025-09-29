"""
Veritabanı Modülü
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from api.core.config import settings

# Veritabanı engine oluştur
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()

class ModelRecord(Base):
    """Eğitilmiş modellerin kayıtları."""
    __tablename__ = "model_records"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'base', 'ensemble', 'tuned'
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=True)
    model_path = Column(String(255), nullable=False)
    feature_names = Column(Text, nullable=True)  # JSON string
    training_params = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self):
        """Model kaydını dictionary'e çevir."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "model_path": self.model_path,
            "feature_names": json.loads(self.feature_names) if self.feature_names else None,
            "training_params": json.loads(self.training_params) if self.training_params else None,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }

class PredictionRecord(Base):
    """Tahmin kayıtları."""
    __tablename__ = "prediction_records"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False, index=True)
    patient_data = Column(Text, nullable=False)  # JSON string
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

async def init_database():
    """Veritabanını başlat."""
    # Tabloları oluştur
    Base.metadata.create_all(bind=engine)
    print("✅ Veritabanı başlatıldı")

def get_db():
    """Veritabanı session'ı al."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
