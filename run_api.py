"""
FastAPI Uygulaması Çalıştırma Scripti
"""

import uvicorn
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    print("🚀 Kalp Krizi Risk Tahmin API başlatılıyor...")
    print("📊 Swagger UI: http://localhost:8000/docs")
    print("📋 ReDoc: http://localhost:8000/redoc")
    print("🔗 API Base URL: http://localhost:8000")
    print("="*60)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
