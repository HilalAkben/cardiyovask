"""
Kalp Krizi Risk Tahmin Modeli - Sadece Tansiyon Aykırı Değer Temizleme
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor


def main():
    """Sadece tansiyon aykırı değerlerini temizle."""
    print("=== KALP KRİZİ RİSK TAHMİN MODELİ - TANSİYON AYKIRI DEĞER TEMİZLEME ===\n")
    
    # DataLoader ve Preprocessor oluştur
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Ham veriyi yükle
    print("Ham veri yükleniyor...")
    data_path = project_root / "data" / "raw" / "cardio_train.csv"
    df_original = data_loader.load_csv(data_path)
    
    print(f"Orijinal veri boyutu: {df_original.shape}")
    
    # Sadece tansiyon değerlerini doğrula
    print("\n1. Tansiyon değerleri doğrulama...")
    df_validated = preprocessor.validate_blood_pressure(df_original)
    
    # Sadece ap_hi ve ap_lo kolonlarındaki aykırı değerleri temizle
    print("\n2. Tansiyon aykırı değer temizleme...")
    columns_to_clean = ['ap_hi', 'ap_lo']
    df_cleaned = preprocessor.clean_outliers(df_validated, columns_to_clean, method='iqr')
    
    # Eksik değerleri doldur
    print("\n3. Eksik değer doldurma...")
    df_filled = preprocessor.fill_missing_values(df_cleaned, strategy='median')
    
    # Temizleme özeti
    print("\n4. Temizleme özeti...")
    summary = preprocessor.get_cleaning_summary(df_original, df_filled)
    preprocessor.print_cleaning_summary(summary)
    
    # Temizlenmiş veriyi kaydet
    print("\n5. Temizlenmiş veriyi kaydetme...")
    output_path = project_root / "data" / "processed" / "cardio_bp_cleaned.csv"
    output_path.parent.mkdir(exist_ok=True)
    
    data_loader.save_csv(df_filled, output_path)
    
    # Temizleme sonrası analiz
    print("\n6. Temizleme sonrası analiz...")
    
    # Tansiyon değerleri analizi
    if 'ap_hi' in df_filled.columns and 'ap_lo' in df_filled.columns:
        print(f"\nTemizleme sonrası tansiyon değerleri:")
        print(f"ap_hi - Ortalama: {df_filled['ap_hi'].mean():.2f}, Medyan: {df_filled['ap_hi'].median():.2f}")
        print(f"ap_lo - Ortalama: {df_filled['ap_lo'].mean():.2f}, Medyan: {df_filled['ap_lo'].median():.2f}")
        
        # Mantıksız değer kontrolü
        invalid_ap_hi = (df_filled['ap_hi'] < 0) | (df_filled['ap_hi'] > 300)
        invalid_ap_lo = (df_filled['ap_lo'] < 0) | (df_filled['ap_lo'] > 200)
        invalid_bp_ratio = df_filled['ap_hi'] < df_filled['ap_lo']
        
        print(f"Kalan geçersiz ap_hi: {invalid_ap_hi.sum()}")
        print(f"Kalan geçersiz ap_lo: {invalid_ap_lo.sum()}")
        print(f"Kalan Sistolik < Diastolik: {invalid_bp_ratio.sum()}")
    
    print("\n=== TANSİYON AYKIRI DEĞER TEMİZLEME TAMAMLANDI ===")
    print(f"Temizlenmiş veri kaydedildi: {output_path}")


if __name__ == "__main__":
    main() 