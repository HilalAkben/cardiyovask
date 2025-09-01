"""
Kalp Krizi Risk Tahmin Modeli - Test Scripti
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_data_loading():
    """Veri yükleme testi."""
    print("Veri yükleme testi başlatılıyor...")
    
    # Veri dosyasının yolunu kontrol et
    data_path = Path("data/raw/cardio_train.csv")
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        return None
    
    try:
        # Veriyi yükle
        df = pd.read_csv(data_path)
        print(f"Veri başarıyla yüklendi!")
        print(f"Veri boyutu: {df.shape}")
        print(f"Kolonlar: {list(df.columns)}")
        
        # Veri türlerini göster
        print("\nVeri Türleri:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Eksik veri kontrolü
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        print(f"\nEksik Veri Analizi:")
        print(f"Toplam eksik değer: {total_missing}")
        
        if total_missing > 0:
            print("Eksik veri olan kolonlar:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count}")
        else:
            print("Eksik veri bulunmamaktadır.")
        
        # Kategorik ve nümerik özellikleri ayır
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\nKategorik özellikler ({len(categorical_cols)}): {categorical_cols}")
        print(f"Nümerik özellikler ({len(numerical_cols)}): {numerical_cols}")
        
        # Ayrı DataFrame'ler oluştur
        categorical_features = df[categorical_cols].copy() if categorical_cols else pd.DataFrame()
        numerical_features = df[numerical_cols].copy() if numerical_cols else pd.DataFrame()
        
        print(f"\nKategorik özellikler DataFrame boyutu: {categorical_features.shape}")
        print(f"Nümerik özellikler DataFrame boyutu: {numerical_features.shape}")
        
        # Nümerik özelliklerin istatistiksel özeti
        if not numerical_features.empty:
            print("\nNümerik Özellikler İstatistiksel Özeti:")
            print(numerical_features.describe())
        
        # Kategorik özelliklerin analizi
        if not categorical_features.empty:
            print("\nKategorik Özellikler Analizi:")
            for col in categorical_features.columns:
                value_counts = categorical_features[col].value_counts()
                print(f"\n{col}:")
                print(f"  Benzersiz değer sayısı: {categorical_features[col].nunique()}")
                print(f"  En yaygın değer: {value_counts.index[0]} ({value_counts.iloc[0]} kez)")
        
        return df, numerical_features, categorical_features
        
    except Exception as e:
        print(f"HATA: Veri yükleme sırasında hata oluştu: {str(e)}")
        return None

if __name__ == "__main__":
    print("=== KALP KRİZİ RİSK TAHMİN MODELİ - VERİ ANALİZ TESTİ ===\n")
    
    result = test_data_loading()
    
    if result:
        df, numerical_features, categorical_features = result
        print("\n=== TEST BAŞARILI ===")
        print("Veri analizi tamamlandı!")
    else:
        print("\n=== TEST BAŞARISIZ ===")
        print("Veri analizi sırasında hata oluştu!") 