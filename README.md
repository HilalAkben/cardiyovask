# Kalp Krizi Risk Tahmin Modeli

Bu proje, kalp krizi riskini tahmin etmek için makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. **Outlier'lı ve outliersız veriler için karşılaştırmalı analiz** yaparak en optimal modeli belirler. **GPU desteği** ile hızlandırılmış eğitim imkanı sunar.

## 🚀 GPU Desteği

Bu proje artık **GPU desteği** ile gelmektedir! XGBoost ve LightGBM modelleri GPU üzerinde çalıştırılarak eğitim süreleri önemli ölçüde azaltılabilir.

### GPU Gereksinimleri
- NVIDIA GPU (CUDA uyumlu)
- CUDA Toolkit 11.0+
- GPU destekli Python paketleri

### GPU Kurulumu
Detaylı kurulum rehberi için [GPU_SETUP.md](GPU_SETUP.md) dosyasını inceleyin.

### GPU Testi
GPU desteğini test etmek için:
```bash
cd src
python test_gpu.py
```

## Proje Yapısı

```
cardiyovask/
├── data/                   # Veri dosyaları
│   └── cardiokaggle.csv   # Ham veri
├── src/                   # Kaynak kodlar
│   ├── data/             # Veri işleme modülleri
│   │   ├── cardiokaggle.csv
│   │   ├── feature_engineering.py
│   │   └── preprocessor.py
│   ├── analysis/         # Veri analizi modülleri
│   │   └── data_analysis.py
│   ├── utils/            # Yardımcı fonksiyonlar
│   │   └── data_loader.py
│   ├── main.py           # Ana çalıştırma dosyası
│   ├── clean_data.py     # Veri temizleme scripti
│   ├── clean_bp_outliers.py
│   ├── test_analysis.py
│   └── test_cleaned_data.py
├── requirements.txt      # Python bağımlılıkları
├── GPU_SETUP.md         # GPU kurulum rehberi
└── README.md            # Proje dokümantasyonu
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Karşılaştırmalı Analiz Pipeline

Outlier'lı ve outliersız veriler için karşılaştırmalı analiz çalıştırmak için:

```bash
cd src
python main.py
```

### GPU Destekli Optimizasyon Pipeline

GPU desteği ile hızlandırılmış optimizasyon için:

```bash
cd src
python main_optimized.py
```

### Ayrı Modüller

**Karşılaştırmalı Analiz için:**
```bash
python src/analysis/data_analysis.py
```

**Veri Temizleme için:**
```bash
python src/clean_data.py
```

**Test Analizi için:**
```bash
python src/test_analysis.py
```

## Özellikler

### Veri İşleme
- ✅ Yaş kolonunu gün formatından yıl formatına çevirme
- ✅ Eksik değer kontrolü ve temizleme
- ✅ Outlier sayısı hesaplama ve raporlama
- ✅ **Outlier temizleme seçeneği** (IQR metoduna göre)
- ✅ Kategorik değişkenlerin encode edilmesi (gender, smoke, alco, active)
- ✅ Sürekli değişkenlerin ölçeklenmesi (StandardScaler)
- ✅ Eğitim/test setlerine %80-%20 ayrımı

### Modelleme
- ✅ Random Forest Classifier
- ✅ Gradient Boosting Classifier
- ✅ Logistic Regression
- ✅ Support Vector Machine (SVM)
- ✅ **XGBoost Classifier** (GPU desteği ile!)
- ✅ **LightGBM** (GPU desteği ile!)

### Değerlendirme Metrikleri
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ ROC AUC Score
- ✅ Confusion Matrix
- ✅ Feature Importance (Random Forest ve XGBoost)

### Karşılaştırmalı Analiz
- ✅ **Outlier'lı vs Outliersız veri karşılaştırması**
- ✅ Her iki veri seti için ayrı model eğitimi
- ✅ Performans metriklerinin karşılaştırmalı analizi
- ✅ En iyi model seçimi (her veri seti için)
- ✅ Outlier temizlemenin etkisinin ölçülmesi

### Görselleştirme
- ✅ Confusion Matrix grafikleri (her veri seti için)
- ✅ ROC eğrileri (her veri seti için)
- ✅ Model performans karşılaştırması (her veri seti için)
- ✅ Feature importance grafiği (Random Forest ve XGBoost)
- ✅ **Karşılaştırmalı analiz grafikleri** (Yeni!)

## Çıktılar

Pipeline çalıştırıldığında aşağıdaki çıktılar oluşturulur:

### 1. Konsol Çıktıları
- Veri işleme adımları (her iki veri seti için)
- Outlier sayıları (çıkarılmadan önce ve sonra)
- Model performans metrikleri (her iki veri seti için)
- X_train.shape, X_test.shape (her iki veri seti için)
- **Karşılaştırmalı sonuçlar tablosu**
- **En iyi model karşılaştırması**
- **Outlier temizlemenin performans etkisi**

### 2. Grafik Dosyaları
- `confusion_matrices.png` - Tüm modellerin confusion matrix'leri (her veri seti için)
- `roc_curves.png` - ROC eğrileri (her veri seti için)
- `metrics_comparison.png` - Model performans karşılaştırması (her veri seti için)
- `rf_feature_importance.png` - Random Forest feature importance (her veri seti için)
- `xgb_feature_importance.png` - XGBoost feature importance (her veri seti için)
- `comparative_analysis.png` - **Karşılaştırmalı analiz grafikleri** (Yeni!)

## Veri Seti

Kullanılan veri seti aşağıdaki kolonları içerir:
- `id`: Hasta ID
- `age`: Yaş (gün cinsinden)
- `gender`: Cinsiyet
- `height`: Boy
- `weight`: Kilo
- `ap_hi`: Sistolik tansiyon
- `ap_lo`: Diastolik tansiyon
- `cholesterol`: Kolesterol seviyesi
- `gluc`: Glikoz seviyesi
- `smoke`: Sigara kullanımı
- `alco`: Alkol kullanımı
- `active`: Fiziksel aktivite
- `cardio`: Kardiyovasküler hastalık (hedef değişken)

## Karşılaştırmalı Analiz Özellikleri

### Outlier Temizleme
- **IQR Metodu**: Q1 - 1.5*IQR ve Q3 + 1.5*IQR aralığı dışındaki değerler outlier olarak kabul edilir
- **Temizlenen Kolonlar**: height, weight, ap_hi, ap_lo, cholesterol, gluc
- **Karşılaştırma**: Outlier'lı ve outliersız veriler için ayrı model eğitimi

### Model Karşılaştırması
- Her veri seti için 5 farklı model eğitilir
- F1-Score'a göre en iyi model seçilir
- Outlier temizlemenin performans etkisi ölçülür

### Görselleştirme
- Karşılaştırmalı bar grafikleri
- Her metrik için ayrı karşılaştırma
- Outlier'lı vs outliersız performans farkı

## Gereksinimler

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- **xgboost >= 1.6.0** (Yeni!) 
#############################################################
================================================================================
KALP KRİZİ RİSK TAHMİN MODELİ - KARŞILAŞTIRMALI ANALİZ PIPELINE
================================================================================
Veri dosyası: /Users/aybukealtuntas/Desktop/cardiovasktrain/cardiyovask/src/data/cardiokaggle.csv

==================================================
1. VERİ İŞLEME VE FEATURE ENGINEERING
==================================================

--- Outlier'lı Verilerle İşleme ---
=== KALP KRİZİ RİSK TAHMİN MODELİ - OUTLIER'LAR İLE VERİ İŞLEME ===

Veri başarıyla yüklendi! Boyut: (70000, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
Yaş dönüşümü tamamlandı. Örnek değerler:
Gün: [18393, 20228, 18857] -> Yıl: [50, 55, 51]

=== EKSİK DEĞER ANALİZİ ===
Toplam eksik değer: 0
Eksik değer bulunmamaktadır.
Eksik değerler silindi. Yeni boyut: (70000, 14)

=== OUTLIER ANALİZİ ===
height: 519 outlier (0.74%)
weight: 1819 outlier (2.60%)
ap_hi: 1435 outlier (2.05%)
ap_lo: 4632 outlier (6.62%)
cholesterol: 0 outlier (0.00%)
gluc: 10521 outlier (15.03%)

Toplam outlier sayısı: 18926

=== KATEGORİK DEĞİŞKEN ENCODING ===
gender: 2 benzersiz değer -> [1, 2] -> [0, 1]
smoke: 2 benzersiz değer -> [0, 1] -> [0, 1]
alco: 2 benzersiz değer -> [0, 1] -> [0, 1]
active: 2 benzersiz değer -> [0, 1] -> [0, 1]

=== SÜREKLİ DEĞİŞKEN ÖLÇEKLEME ===
Ölçeklenen kolonlar: ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
StandardScaler uygulandı (ortalama=0, standart sapma=1)

Feature matrix boyutu: (70000, 11)
Hedef değişken boyutu: (70000,)
Feature kolonları: ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years']

=== VERİ AYRIMI ===
Eğitim seti: (56000, 11)
Test seti: (14000, 11)
Eğitim hedef: (56000,)
Test hedef: (14000,)

=== OUTLIER'LAR İLE VERİ İŞLEME TAMAMLANDI ===

--- Outlier'lar Çıkarılarak İşleme ---
=== KALP KRİZİ RİSK TAHMİN MODELİ - OUTLIER'LAR ÇIKARILARAK VERİ İŞLEME ===

Veri başarıyla yüklendi! Boyut: (70000, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
Yaş dönüşümü tamamlandı. Örnek değerler:
Gün: [18393, 20228, 18857] -> Yıl: [50, 55, 51]

=== EKSİK DEĞER ANALİZİ ===
Toplam eksik değer: 0
Eksik değer bulunmamaktadır.
Eksik değerler silindi. Yeni boyut: (70000, 14)

=== OUTLIER ANALİZİ ===
height: 519 outlier (0.74%)
weight: 1819 outlier (2.60%)
ap_hi: 1435 outlier (2.05%)
ap_lo: 4632 outlier (6.62%)
cholesterol: 0 outlier (0.00%)
gluc: 10521 outlier (15.03%)

Toplam outlier sayısı: 18926

=== OUTLIER TEMİZLEME ===
height: 519 outlier çıkarıldı
weight: 1819 outlier çıkarıldı
ap_hi: 1435 outlier çıkarıldı
ap_lo: 4632 outlier çıkarıldı
cholesterol: 0 outlier çıkarıldı
gluc: 10521 outlier çıkarıldı
Toplam 18926 outlier çıkarıldı
Orijinal boyut: (70000, 14) -> Temizlenmiş boyut: (53408, 14)

=== KATEGORİK DEĞİŞKEN ENCODING ===
gender: 2 benzersiz değer -> [1, 2] -> [0, 1]
smoke: 2 benzersiz değer -> [0, 1] -> [0, 1]
alco: 2 benzersiz değer -> [0, 1] -> [0, 1]
active: 2 benzersiz değer -> [0, 1] -> [0, 1]

=== SÜREKLİ DEĞİŞKEN ÖLÇEKLEME ===
Ölçeklenen kolonlar: ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
StandardScaler uygulandı (ortalama=0, standart sapma=1)

Feature matrix boyutu: (53408, 11)
Hedef değişken boyutu: (53408,)
Feature kolonları: ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years']

=== VERİ AYRIMI ===
Eğitim seti: (42726, 11)
Test seti: (10682, 11)
Eğitim hedef: (42726,)
Test hedef: (10682,)

=== OUTLIER'LAR ÇIKARILARAK VERİ İŞLEME TAMAMLANDI ===

==================================================
2. KARŞILAŞTIRMALI MODEL EĞİTİMİ VE DEĞERLENDİRME
==================================================
================================================================================
KALP KRİZİ RİSK TAHMİN MODELİ - KARŞILAŞTIRMALI ANALİZ
================================================================================

========================================
1. OUTLIER'LAR İLE ANALİZ
========================================
=== KALP KRİZİ RİSK TAHMİN MODELİ - TAM ANALİZ (WITH_OUTLIERS) ===

=== MODEL EĞİTİMİ ===

Random Forest eğitiliyor...
Random Forest eğitimi tamamlandı.

Gradient Boosting eğitiliyor...
Gradient Boosting eğitimi tamamlandı.

Logistic Regression eğitiliyor...
Logistic Regression eğitimi tamamlandı.

SVM eğitiliyor...
SVM eğitimi tamamlandı.

XGBoost eğitiliyor...
XGBoost eğitimi tamamlandı.

Tüm modeller eğitildi!

=== MODEL DEĞERLENDİRMESİ ===

Random Forest değerlendiriliyor...
Accuracy: 0.7077
Precision: 0.7093
Recall: 0.7033
F1-Score: 0.7063
ROC AUC: 0.7635

Gradient Boosting değerlendiriliyor...
Accuracy: 0.7336
Precision: 0.7507
Recall: 0.6990
F1-Score: 0.7239
ROC AUC: 0.7997

Logistic Regression değerlendiriliyor...
Accuracy: 0.7141
Precision: 0.7318
Recall: 0.6754
F1-Score: 0.7024
ROC AUC: 0.7783

SVM değerlendiriliyor...
Accuracy: 0.7294
Precision: 0.7440
Recall: 0.6990
F1-Score: 0.7208
ROC AUC: 0.7859

XGBoost değerlendiriliyor...
Accuracy: 0.7339
Precision: 0.7530
Recall: 0.6958
F1-Score: 0.7233
ROC AUC: 0.8000

=== EN İYİ MODEL SEÇİMİ ===
Random Forest: F1-Score = 0.7063
Gradient Boosting: F1-Score = 0.7239
Logistic Regression: F1-Score = 0.7024
SVM: F1-Score = 0.7208
XGBoost: F1-Score = 0.7233

En iyi model: Gradient Boosting (F1-Score: 0.7239)

=== Gradient Boosting - DETAYLI SINIFLANDIRMA RAPORU ===
                     precision    recall  f1-score   support

Kardiyovasküler Yok       0.72      0.77      0.74      7004
Kardiyovasküler Var       0.75      0.70      0.72      6996

           accuracy                           0.73     14000
          macro avg       0.73      0.73      0.73     14000
       weighted avg       0.73      0.73      0.73     14000


Görselleştirmeler oluşturuluyor...
2025-08-11 21:20:37.709 python[80049:13502621] +[IMKClient subclass]: chose IMKClient_Legacy
2025-08-11 21:20:37.709 python[80049:13502621] +[IMKInputSession subclass]: chose IMKInputSession_Legacy

=== ANALİZ TAMAMLANDI (WITH_OUTLIERS) ===
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7239

========================================
2. OUTLIER'LAR ÇIKARILARAK ANALİZ
========================================
=== KALP KRİZİ RİSK TAHMİN MODELİ - TAM ANALİZ (WITHOUT_OUTLIERS) ===

=== MODEL EĞİTİMİ ===

Random Forest eğitiliyor...
Random Forest eğitimi tamamlandı.

Gradient Boosting eğitiliyor...
Gradient Boosting eğitimi tamamlandı.

Logistic Regression eğitiliyor...
Logistic Regression eğitimi tamamlandı.

SVM eğitiliyor...
SVM eğitimi tamamlandı.

XGBoost eğitiliyor...
XGBoost eğitimi tamamlandı.

Tüm modeller eğitildi!

=== MODEL DEĞERLENDİRMESİ ===

Random Forest değerlendiriliyor...
Accuracy: 0.7070
Precision: 0.7006
Recall: 0.6714
F1-Score: 0.6857
ROC AUC: 0.7634

Gradient Boosting değerlendiriliyor...
Accuracy: 0.7395
Precision: 0.7570
Recall: 0.6667
F1-Score: 0.7090
ROC AUC: 0.8056

Logistic Regression değerlendiriliyor...
Accuracy: 0.7344
Precision: 0.7653
Recall: 0.6376
F1-Score: 0.6956
ROC AUC: 0.7984

SVM değerlendiriliyor...
Accuracy: 0.7368
Precision: 0.7798
Recall: 0.6232
F1-Score: 0.6928
ROC AUC: 0.7870

XGBoost değerlendiriliyor...
Accuracy: 0.7416
Precision: 0.7658
Recall: 0.6586
F1-Score: 0.7082
ROC AUC: 0.8046

=== EN İYİ MODEL SEÇİMİ ===
Random Forest: F1-Score = 0.6857
Gradient Boosting: F1-Score = 0.7090
Logistic Regression: F1-Score = 0.6956
SVM: F1-Score = 0.6928
XGBoost: F1-Score = 0.7082

En iyi model: Gradient Boosting (F1-Score: 0.7090)

=== Gradient Boosting - DETAYLI SINIFLANDIRMA RAPORU ===
                     precision    recall  f1-score   support

Kardiyovasküler Yok       0.73      0.81      0.76      5597
Kardiyovasküler Var       0.76      0.67      0.71      5085

           accuracy                           0.74     10682
          macro avg       0.74      0.74      0.74     10682
       weighted avg       0.74      0.74      0.74     10682


Görselleştirmeler oluşturuluyor...
2025-08-11 21:52:41.536 python[80049:13502621] _TIPropertyValueIsValid called with 16 on nil context!
2025-08-11 21:52:41.536 python[80049:13502621] imkxpc_getApplicationProperty:reply: called with incorrect property value 16, bailing.
2025-08-11 21:52:41.536 python[80049:13502621] Text input context does not respond to _valueForTIProperty:
2025-08-11 21:54:29.329 python[80049:13502621] _TIPropertyValueIsValid called with 16 on nil context!
2025-08-11 21:54:29.329 python[80049:13502621] imkxpc_getApplicationProperty:reply: called with incorrect property value 16, bailing.
2025-08-11 21:54:29.329 python[80049:13502621] Text input context does not respond to _valueForTIProperty:

=== ANALİZ TAMAMLANDI (WITHOUT_OUTLIERS) ===
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7090

========================================
3. KARŞILAŞTIRMALI SONUÇLAR
========================================

=== KARŞILAŞTIRMALI SONUÇLAR TABLOSU ===
              Model  Accuracy  Precision   Recall  F1-Score  ROC AUC        Data_Type
      Random Forest  0.707714   0.709343 0.703259  0.706288 0.763495    With Outliers
  Gradient Boosting  0.733571   0.750691 0.698971  0.723908 0.799737    With Outliers
Logistic Regression  0.714071   0.731764 0.675386  0.702446 0.778264    With Outliers
                SVM  0.729357   0.743953 0.698971  0.720761 0.785948    With Outliers
            XGBoost  0.733929   0.752978 0.695826  0.723275 0.799985    With Outliers
      Random Forest  0.706984   0.700595 0.671386  0.685680 0.763354 Without Outliers
  Gradient Boosting  0.739468   0.757034 0.666667  0.708983 0.805611 Without Outliers
Logistic Regression  0.734413   0.765345 0.637561  0.695634 0.798412 Without Outliers
                SVM  0.736847   0.779774 0.623206  0.692753 0.787034 Without Outliers
            XGBoost  0.741621   0.765836 0.658604  0.708184 0.804620 Without Outliers

=== EN İYİ MODEL KARŞILAŞTIRMASI ===
Outlier'lı veriler: Gradient Boosting (F1: 0.7239)
Outlier'lar çıkarılmış: Gradient Boosting (F1: 0.7090)
Outlier temizleme ile F1-Score azalışı: -0.0149

================================================================================
FİNAL KARŞILAŞTIRMALI SONUÇLAR
================================================================================

OUTLIER'LAR İLE:
X_train.shape: (56000, 11)
X_test.shape: (14000, 11)
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7239

Outlier Analizi (çıkarılmadan):
  height: 519 outlier
  weight: 1819 outlier
  ap_hi: 1435 outlier
  ap_lo: 4632 outlier
  cholesterol: 0 outlier
  gluc: 10521 outlier

OUTLIER'LAR ÇIKARILARAK:
X_train.shape: (42726, 11)
X_test.shape: (10682, 11)
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7090

Outlier Analizi (çıkarıldıktan sonra):
  height: 519 outlier (çıkarıldı)
  weight: 1819 outlier (çıkarıldı)
  ap_hi: 1435 outlier (çıkarıldı)
  ap_lo: 4632 outlier (çıkarıldı)
  cholesterol: 0 outlier (çıkarıldı)
  gluc: 10521 outlier (çıkarıldı)

Model Performans Karşılaştırması (Outlier'lı):
              Model  Accuracy  Precision   Recall  F1-Score  ROC AUC
      Random Forest  0.707714   0.709343 0.703259  0.706288 0.763495
  Gradient Boosting  0.733571   0.750691 0.698971  0.723908 0.799737
Logistic Regression  0.714071   0.731764 0.675386  0.702446 0.778264
                SVM  0.729357   0.743953 0.698971  0.720761 0.785948
            XGBoost  0.733929   0.752978 0.695826  0.723275 0.799985

Model Performans Karşılaştırması (Outlier'lar çıkarılmış):
              Model  Accuracy  Precision   Recall  F1-Score  ROC AUC
      Random Forest  0.706984   0.700595 0.671386  0.685680 0.763354
  Gradient Boosting  0.739468   0.757034 0.666667  0.708983 0.805611
Logistic Regression  0.734413   0.765345 0.637561  0.695634 0.798412
                SVM  0.736847   0.779774 0.623206  0.692753 0.787034
            XGBoost  0.741621   0.765836 0.658604  0.708184 0.804620

En Önemli 10 Feature - Outlier'lı Veriler (Random Forest):
  weight: 0.2308
  height: 0.2101
  ap_hi: 0.1931
  age_years: 0.1635
  ap_lo: 0.0912
  cholesterol: 0.0392
  gluc: 0.0195
  gender: 0.0185
  active: 0.0161
  smoke: 0.0097

En Önemli 10 Feature - Outlier'lar çıkarılmış (Random Forest):
  weight: 0.2491
  height: 0.2220
  ap_hi: 0.1939
  age_years: 0.1714
  ap_lo: 0.0776
  cholesterol: 0.0367
  gender: 0.0175
  active: 0.0148
  smoke: 0.0093
  alco: 0.0078

================================================================================
KARŞILAŞTIRMALI ANALİZ PIPELINE TAMAMLANDI!
================================================================================
Grafikler proje dizinine kaydedildi:
- confusion_matrices.png (her iki veri seti için)
- roc_curves.png (her iki veri seti için)
- metrics_comparison.png (her iki veri seti için)
- rf_feature_importance.png (her iki veri seti için)
- xgb_feature_importance.png (her iki veri seti için)
- comparative_analysis.png (karşılaştırmalı sonuçlar)
"""
Kalp Krizi Risk Tahmin Modeli - Optimizasyon Ana Çalıştırma Dosyası
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.advanced_feature_engineering import AdvancedFeatureEngineer
from analysis.hyperparameter_tuning import HyperparameterTuner
from analysis.ensemble_methods import EnsembleMethods
from utils.data_loader import DataLoader

def main():
    """Ana fonksiyon - Tüm optimizasyon pipeline'ını çalıştır."""
    print("="*80)
    print("KALP KRİZİ RİSK TAHMİN MODELİ - TAM OPTİMİZASYON PIPELINE")
    print("="*80)
    
    # 1. Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        return
    
    # 2. GELİŞMİŞ FEATURE ENGINEERING
    print("\n" + "="*50)
    print("1. GELİŞMİŞ FEATURE ENGINEERING")
    print("="*50)
    
    advanced_fe = AdvancedFeatureEngineer()
    advanced_data = advanced_fe.advanced_pipeline(str(data_path))
    
    if advanced_data is None:
        print("Gelişmiş feature engineering başarısız!")
        return
    
    print(f"Feature sayısı: {advanced_data['X_train'].shape[1]}")
    
    # 3. HYPERPARAMETER TUNING
    print("\n" + "="*50)
    print("2. HYPERPARAMETER TUNING")
    print("="*50)
    
    tuner = HyperparameterTuner()
    best_models = tuner.tune_all_models(
        advanced_data['X_train'],
        advanced_data['y_train'],
        cv=5,
        n_jobs=-1
    )
    
    # Tune edilmiş modelleri test setinde değerlendir
    tuning_results = tuner.evaluate_tuned_models(
        advanced_data['X_test'],
        advanced_data['y_test']
    )
    
    # 4. ENSEMBLE METHODS
    print("\n" + "="*50)
    print("3. ENSEMBLE METHODS")
    print("="*50)
    
    ensemble = EnsembleMethods()
    ensemble_results = ensemble.run_ensemble_analysis(
        advanced_data['X_train'],
        advanced_data['X_test'],
        advanced_data['y_train'],
        advanced_data['y_test']
    )
    
    # 5. FİNAL SONUÇLAR
    print("\n" + "="*80)
    print("FİNAL OPTİMİZASYON SONUÇLARI")
    print("="*80)
    
    best_tuning_model = max(tuning_results.keys(), key=lambda x: tuning_results[x]['accuracy'])
    best_tuning_accuracy = tuning_results[best_tuning_model]['accuracy']
    
    best_ensemble_model = ensemble_results['best_model_name']
    best_ensemble_accuracy = ensemble_results['best_accuracy']
    
    print(f"Hyperparameter Tuning (En iyi): {best_tuning_accuracy:.4f}")
    print(f"Ensemble Methods (En iyi): {best_ensemble_accuracy:.4f}")
    
    if best_ensemble_accuracy > best_tuning_accuracy:
        print(f"🏆 En iyi sonuç: {best_ensemble_model} ({best_ensemble_accuracy:.4f})")
    else:
        print(f"🏆 En iyi sonuç: {best_tuning_model} ({best_tuning_accuracy:.4f})")

if __name__ == "__main__":
    main()
"""
Kalp Krizi Risk Tahmin Modeli - Optimizasyon Ana Çalıştırma Dosyası
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.advanced_feature_engineering import AdvancedFeatureEngineer
from analysis.hyperparameter_tuning import HyperparameterTuner
from analysis.ensemble_methods import EnsembleMethods
from utils.data_loader import DataLoader

def main():
    """Ana fonksiyon - Tüm optimizasyon pipeline'ını çalıştır."""
    print("="*80)
    print("KALP KRİZİ RİSK TAHMİN MODELİ - TAM OPTİMİZASYON PIPELINE")
    print("="*80)
    
    # 1. Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        return
    
    # 2. GELİŞMİŞ FEATURE ENGINEERING
    print("\n" + "="*50)
    print("1. GELİŞMİŞ FEATURE ENGINEERING")
    print("="*50)
    
    advanced_fe = AdvancedFeatureEngineer()
    advanced_data = advanced_fe.advanced_pipeline(str(data_path))
    
    if advanced_data is None:
        print("Gelişmiş feature engineering başarısız!")
        return
    
    print(f"Feature sayısı: {advanced_data['X_train'].shape[1]}")
    
    # 3. HYPERPARAMETER TUNING
    print("\n" + "="*50)
    print("2. HYPERPARAMETER TUNING")
    print("="*50)
    
    tuner = HyperparameterTuner()
    best_models = tuner.tune_all_models(
        advanced_data['X_train'],
        advanced_data['y_train'],
        cv=5,
        n_jobs=-1
    )
    
    # Tune edilmiş modelleri test setinde değerlendir
    tuning_results = tuner.evaluate_tuned_models(
        advanced_data['X_test'],
        advanced_data['y_test']
    )
    
    # 4. ENSEMBLE METHODS
    print("\n" + "="*50)
    print("3. ENSEMBLE METHODS")
    print("="*50)
    
    ensemble = EnsembleMethods()
    ensemble_results = ensemble.run_ensemble_analysis(
        advanced_data['X_train'],
        advanced_data['X_test'],
        advanced_data['y_train'],
        advanced_data['y_test']
    )
    
    # 5. FİNAL SONUÇLAR
    print("\n" + "="*80)
    print("FİNAL OPTİMİZASYON SONUÇLARI")
    print("="*80)
    
    best_tuning_model = max(tuning_results.keys(), key=lambda x: tuning_results[x]['accuracy'])
    best_tuning_accuracy = tuning_results[best_tuning_model]['accuracy']
    
    best_ensemble_model = ensemble_results['best_model_name']
    best_ensemble_accuracy = ensemble_results['best_accuracy']
    
    print(f"Hyperparameter Tuning (En iyi): {best_tuning_accuracy:.4f}")
    print(f"Ensemble Methods (En iyi): {best_ensemble_accuracy:.4f}")
    
    if best_ensemble_accuracy > best_tuning_accuracy:
        print(f"🏆 En iyi sonuç: {best_ensemble_model} ({best_ensemble_accuracy:.4f})")
    else:
        print(f"🏆 En iyi sonuç: {best_tuning_model} ({best_tuning_accuracy:.4f})")

if __name__ == "__main__":
    main()