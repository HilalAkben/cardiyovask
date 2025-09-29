# Kalp Krizi Risk Tahmin API - FastAPI

Bu proje, kardiyovasküler hastalık riskini tahmin etmek için geliştirilmiş makine öğrenmesi modellerini FastAPI ile web servisi haline getirir.

## 🚀 Özellikler

### 📊 Model Karşılaştırması
- **Endpoint**: `/api/v1/models/compare`
- Tüm ML algoritmalarının performansını karşılaştırır
- Risk skorlarına göre sıralar (F1-Score bazlı)
- Desteklenen modeller:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Logistic Regression
  - SVM
  - Ensemble Methods

### 🤖 Model Eğitimi
- **Endpoint**: `/api/v1/training/train`
- Seçilen modeli eğitir ve detaylı analiz sağlar
- Gelişmiş feature engineering
- GPU desteği
- Model kaydetme ve veritabanı entegrasyonu

### 🔮 Risk Tahmini
- **Endpoint**: `/api/v1/prediction/predict`
- Hastanın kardiyovasküler riskini hesaplar
- Risk skoru ve seviyesi belirleme
- Kişiselleştirilmiş öneriler
- Risk faktörleri analizi

## 📋 API Endpoints

### Ana Endpoints
- `GET /` - Ana sayfa ve API bilgileri
- `GET /health` - Sistem sağlık kontrolü
- `GET /docs` - Swagger UI dokümantasyonu
- `GET /redoc` - ReDoc dokümantasyonu

### Model Karşılaştırması
- `POST /api/v1/models/compare` - Model karşılaştırması
- `GET /api/v1/models/compare/status` - Karşılaştırma durumu
- `GET /api/v1/models/compare/sample` - Örnek sonuç

### Model Eğitimi
- `POST /api/v1/training/train` - Model eğitimi
- `GET /api/v1/training/models` - Eğitilmiş modeller
- `GET /api/v1/training/models/{id}` - Model detayları
- `DELETE /api/v1/training/models/{id}` - Model silme
- `GET /api/v1/training/supported-models` - Desteklenen modeller
- `GET /api/v1/training/status` - Eğitim durumu

### Risk Tahmini
- `POST /api/v1/prediction/predict` - Risk tahmini
- `GET /api/v1/prediction/models` - Kullanılabilir modeller
- `POST /api/v1/prediction/sample` - Örnek tahmin
- `GET /api/v1/prediction/risk-levels` - Risk seviyeleri
- `GET /api/v1/prediction/risk-factors` - Risk faktörleri
- `GET /api/v1/prediction/status` - Tahmin durumu

## 🛠️ Kurulum

### 1. Gereksinimler
```bash
pip install -r api_requirements.txt
```

### 2. Veri Dosyası
Veri dosyasının `src/data/cardiokaggle.csv` konumunda olduğundan emin olun.

### 3. API'yi Çalıştırma
```bash
python run_api.py
```

Veya doğrudan:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 Kullanım Örnekleri

### Model Karşılaştırması
```python
import requests

# Model karşılaştırması
response = requests.post("http://localhost:8000/api/v1/models/compare", json={
    "data_path": "src/data/cardiokaggle.csv",
    "use_gpu": True
})

result = response.json()
print(f"En iyi model: {result['data']['comparison_summary']['data_with_outliers']['best_model']}")
```

### Model Eğitimi
```python
# Random Forest modeli eğitimi
response = requests.post("http://localhost:8000/api/v1/training/train", json={
    "model_name": "Random Forest",
    "use_advanced_features": True,
    "use_gpu": True
})

result = response.json()
print(f"Model ID: {result['data']['model_record_id']}")
```

### Risk Tahmini
```python
# Hasta risk tahmini
response = requests.post("http://localhost:8000/api/v1/prediction/predict", json={
    "model_id": 1,
    "patient_data": {
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
})

result = response.json()
print(f"Risk seviyesi: {result['data']['prediction_summary']['risk_level']}")
```

## 🔧 Konfigürasyon

### Environment Variables
`.env` dosyası oluşturun:
```env
# API Ayarları
DEBUG=True
VERSION=1.0.0

# Veritabanı
DATABASE_URL=sqlite:///./cardio_models.db

# Model Ayarları
MODELS_DIR=models
DATA_FILE=src/data/cardiokaggle.csv

# GPU Ayarları
USE_GPU=True
GPU_DEVICE_ID=0
```

## 📊 Veritabanı

API, SQLite veritabanı kullanır:
- **model_records**: Eğitilmiş modeller
- **prediction_records**: Tahmin kayıtları

## 🎯 API Özellikleri

### Swagger UI
- **URL**: http://localhost:8000/docs
- Interaktif API dokümantasyonu
- Test endpoint'leri
- Request/Response örnekleri

### ReDoc
- **URL**: http://localhost:8000/redoc
- Detaylı API dokümantasyonu
- Schema açıklamaları

### CORS Desteği
- Tüm origin'lere izin verilir
- Production'da spesifik domainler belirtilmeli

### Error Handling
- Detaylı hata mesajları
- HTTP status kodları
- Validation hataları

## 🔍 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs
API çalışırken detaylı loglar görüntülenir:
- Request/Response logları
- Hata logları
- Performance metrikleri

## 🚀 Production Deployment

### Docker (Önerilen)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r api_requirements.txt

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Gunicorn
```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📝 Notlar

- API, mevcut ML kodlarınızı bozmadan çalışır
- Profesyonel dosya yapısı korunmuştur
- Clean code standartlarına uygun
- Comprehensive error handling
- Detailed documentation
- Swagger UI ile kolay test

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun
3. Commit yapın
4. Push yapın
5. Pull Request oluşturun

## 📄 Lisans

MIT License
