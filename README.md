# Kalp Krizi Risk Tahmin Modeli

Bu proje, kalp krizi riskini tahmin etmek için makine öğrenmesi modeli geliştirmeyi amaçlamaktadır.

## Proje Yapısı

```
cardiyovask/
├── data/                   # Veri dosyaları
│   ├── raw/               # Ham veriler
│   └── processed/         # İşlenmiş veriler
├── src/                   # Kaynak kodlar
│   ├── data/             # Veri işleme modülleri
│   ├── analysis/         # Veri analizi modülleri
│   └── utils/            # Yardımcı fonksiyonlar
├── notebooks/            # Jupyter notebook'ları
├── reports/              # Analiz raporları
├── requirements.txt      # Python bağımlılıkları
└── README.md            # Proje dokümantasyonu
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

Veri analizi için:
```bash
python src/analysis/data_analysis.py
``` 