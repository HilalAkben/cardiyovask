"""
FastAPI Demo Script - Kalp Krizi Risk Tahmin API
"""

import requests
import json
import time
from typing import Dict, Any

def demo_api():
    """API demo fonksiyonu."""
    base_url = "http://localhost:8000"
    
    print("🚀 Kalp Krizi Risk Tahmin API - Demo")
    print("="*60)
    
    # 1. Health Check
    print("\n1️⃣ Sistem Sağlık Kontrolü")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API çalışıyor!")
        else:
            print("❌ API çalışmıyor!")
            return
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}")
        print("💡 Lütfen API'yi başlatın: python run_api.py")
        return
    
    # 2. Ana Sayfa
    print("\n2️⃣ Ana Sayfa Bilgileri")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"📱 API: {data['message']}")
            print(f"📊 Swagger UI: {data['docs']}")
            print(f"📋 ReDoc: {data['redoc']}")
    except Exception as e:
        print(f"❌ Ana sayfa hatası: {e}")
    
    # 3. Desteklenen Modeller
    print("\n3️⃣ Desteklenen Modeller")
    try:
        response = requests.get(f"{base_url}/api/v1/training/supported-models")
        if response.status_code == 200:
            data = response.json()
            print("🤖 Desteklenen Modeller:")
            for category, models in data['supported_models'].items():
                print(f"  {category.upper()}:")
                for model in models:
                    print(f"    - {model['name']}: {model['description']}")
    except Exception as e:
        print(f"❌ Model listesi hatası: {e}")
    
    # 4. Risk Seviyeleri
    print("\n4️⃣ Risk Seviyeleri")
    try:
        response = requests.get(f"{base_url}/api/v1/prediction/risk-levels")
        if response.status_code == 200:
            data = response.json()
            print("⚠️ Risk Seviyeleri:")
            for level, info in data['risk_levels'].items():
                print(f"  {level}: {info['score_range']} - {info['description']}")
    except Exception as e:
        print(f"❌ Risk seviyeleri hatası: {e}")
    
    # 5. Risk Faktörleri
    print("\n5️⃣ Risk Faktörleri")
    try:
        response = requests.get(f"{base_url}/api/v1/prediction/risk-factors")
        if response.status_code == 200:
            data = response.json()
            print("🔍 Ana Risk Faktörleri:")
            for factor, info in data['risk_factors_info']['major_risk_factors'].items():
                print(f"  - {factor}: {info['description']} ({info['impact']} etki)")
    except Exception as e:
        print(f"❌ Risk faktörleri hatası: {e}")
    
    # 6. Örnek Risk Tahmini
    print("\n6️⃣ Örnek Risk Tahmini")
    try:
        response = requests.post(f"{base_url}/api/v1/prediction/sample")
        if response.status_code == 200:
            data = response.json()
            summary = data['data']['prediction_summary']
            analysis = data['data']['patient_analysis']
            recommendations = data['data']['recommendations']
            
            print("👤 Örnek Hasta:")
            print(f"  Yaş: 45, Cinsiyet: Erkek, Boy: 175cm, Kilo: 75kg")
            print(f"  Tansiyon: 120/80, Kolesterol: Normal, Glukoz: Normal")
            print(f"  Sigara: Hayır, Alkol: Hayır, Aktif: Evet")
            
            print(f"\n🎯 Tahmin Sonuçları:")
            print(f"  Model: {summary['model_name']}")
            print(f"  Tahmin: {'Yüksek Risk' if summary['prediction'] == 1 else 'Düşük Risk'}")
            print(f"  Olasılık: {summary['probability']:.2%}")
            print(f"  Risk Skoru: {summary['risk_score']:.2%}")
            print(f"  Risk Seviyesi: {summary['risk_level']}")
            
            print(f"\n⚠️ Risk Faktörleri:")
            for factor in analysis['risk_factors']:
                print(f"  - {factor}")
            
            print(f"\n✅ Koruyucu Faktörler:")
            for factor in analysis['protective_factors']:
                print(f"  - {factor}")
            
            print(f"\n💡 Öneriler:")
            print(f"  Yaşam Tarzı:")
            for rec in recommendations['lifestyle'][:2]:
                print(f"    - {rec}")
            print(f"  Tıbbi:")
            for rec in recommendations['medical'][:2]:
                print(f"    - {rec}")
                
    except Exception as e:
        print(f"❌ Örnek tahmin hatası: {e}")
    
    # 7. Model Durumu
    print("\n7️⃣ Model Durumu")
    try:
        response = requests.get(f"{base_url}/api/v1/training/models")
        if response.status_code == 200:
            data = response.json()
            if data['models']:
                print(f"📊 Eğitilmiş Modeller ({len(data['models'])} adet):")
                for model in data['models']:
                    print(f"  - {model['model_name']} (ID: {model['id']}, Accuracy: {model['accuracy']:.2%})")
            else:
                print("📊 Henüz eğitilmiş model bulunmuyor.")
                print("💡 Model eğitmek için: POST /api/v1/training/train")
    except Exception as e:
        print(f"❌ Model durumu hatası: {e}")
    
    print("\n" + "="*60)
    print("🎉 Demo tamamlandı!")
    print("📊 Swagger UI: http://localhost:8000/docs")
    print("📋 ReDoc: http://localhost:8000/redoc")
    print("🔗 API Base URL: http://localhost:8000")
    print("\n💡 İpuçları:")
    print("  - Model karşılaştırması: POST /api/v1/models/compare")
    print("  - Model eğitimi: POST /api/v1/training/train")
    print("  - Risk tahmini: POST /api/v1/prediction/predict")
    print("  - Detaylı test: python test_api.py")

if __name__ == "__main__":
    demo_api()
