"""
Kalp Krizi Risk Tahmin Modeli - Veri Ön İşleme Modülü
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Veri ön işleme ve temizleme sınıfı."""
    
    def __init__(self):
        self.cleaning_summary = {}
    
    def validate_blood_pressure(self, df):
        """Tansiyon değerlerini doğrula ve düzelt."""
        print("Tansiyon değerleri doğrulanıyor...")
        
        df_validated = df.copy()
        
        # Geçersiz tansiyon değerlerini temizle
        if 'ap_hi' in df_validated.columns and 'ap_lo' in df_validated.columns:
            # Negatif değerleri temizle
            invalid_ap_hi = (df_validated['ap_hi'] < 0) | (df_validated['ap_hi'] > 300)
            invalid_ap_lo = (df_validated['ap_lo'] < 0) | (df_validated['ap_lo'] > 200)
            
            # Sistolik < Diastolik olan değerleri temizle
            invalid_bp_ratio = df_validated['ap_hi'] < df_validated['ap_lo']
            
            # Tüm geçersiz değerleri NaN yap
            df_validated.loc[invalid_ap_hi, 'ap_hi'] = np.nan
            df_validated.loc[invalid_ap_lo, 'ap_lo'] = np.nan
            df_validated.loc[invalid_bp_ratio, ['ap_hi', 'ap_lo']] = np.nan
            
            print(f"Geçersiz ap_hi değerleri: {invalid_ap_hi.sum()}")
            print(f"Geçersiz ap_lo değerleri: {invalid_ap_lo.sum()}")
            print(f"Sistolik < Diastolik: {invalid_bp_ratio.sum()}")
        
        return df_validated
    
    def clean_outliers(self, df, columns, method='iqr'):
        """Aykırı değerleri temizle."""
        print(f"Aykırı değerler temizleniyor (metod: {method})...")
        
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df_cleaned.columns:
                if method == 'iqr':
                    # IQR metodunu kullan
                    q1 = df_cleaned[col].quantile(0.25)
                    q3 = df_cleaned[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                    df_cleaned.loc[outliers, col] = np.nan
                    
                    print(f"{col}: {outliers.sum()} outlier temizlendi")
                
                elif method == 'zscore':
                    # Z-score metodunu kullan
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                    outliers = z_scores > 3
                    df_cleaned.loc[outliers, col] = np.nan
                    
                    print(f"{col}: {outliers.sum()} outlier temizlendi")
        
        return df_cleaned
    
    def fill_missing_values(self, df, strategy='median'):
        """Eksik değerleri doldur."""
        print(f"Eksik değerler dolduruluyor (strateji: {strategy})...")
        
        df_filled = df.copy()
        
        if strategy == 'median':
            df_filled = df_filled.fillna(df_filled.median())
        elif strategy == 'mean':
            df_filled = df_filled.fillna(df_filled.mean())
        elif strategy == 'mode':
            df_filled = df_filled.fillna(df_filled.mode().iloc[0])
        elif strategy == 'drop':
            df_filled = df_filled.dropna()
        
        return df_filled
    
    def get_cleaning_summary(self, df_original, df_cleaned):
        """Temizleme özeti oluştur."""
        summary = {
            'original_shape': df_original.shape,
            'cleaned_shape': df_cleaned.shape,
            'rows_removed': df_original.shape[0] - df_cleaned.shape[0],
            'columns_removed': df_original.shape[1] - df_cleaned.shape[1],
            'missing_values_original': df_original.isnull().sum().sum(),
            'missing_values_cleaned': df_cleaned.isnull().sum().sum()
        }
        
        self.cleaning_summary = summary
        return summary
    
    def print_cleaning_summary(self, summary):
        """Temizleme özetini yazdır."""
        print("\n=== TEMİZLEME ÖZETİ ===")
        print(f"Orijinal veri boyutu: {summary['original_shape']}")
        print(f"Temizlenmiş veri boyutu: {summary['cleaned_shape']}")
        print(f"Silinen satır sayısı: {summary['rows_removed']}")
        print(f"Silinen kolon sayısı: {summary['columns_removed']}")
        print(f"Orijinal eksik değer sayısı: {summary['missing_values_original']}")
        print(f"Temizleme sonrası eksik değer sayısı: {summary['missing_values_cleaned']}")
        
        if summary['rows_removed'] > 0:
            removal_percentage = (summary['rows_removed'] / summary['original_shape'][0]) * 100
            print(f"Satır silme oranı: {removal_percentage:.2f}%")
    
    def validate_data_types(self, df):
        """Veri türlerini doğrula ve düzelt."""
        print("Veri türleri doğrulanıyor...")
        
        df_validated = df.copy()
        
        # Sayısal kolonları kontrol et
        numeric_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        
        for col in numeric_columns:
            if col in df_validated.columns:
                # Sayısal olmayan değerleri temizle
                df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
        
        # Kategorik kolonları kontrol et
        categorical_columns = ['gender', 'smoke', 'alco', 'active', 'cardio']
        
        for col in categorical_columns:
            if col in df_validated.columns:
                # Kategorik değerleri düzelt
                df_validated[col] = df_validated[col].astype('category')
        
        return df_validated
    
    def remove_duplicates(self, df):
        """Tekrarlanan satırları kaldır."""
        print("Tekrarlanan satırlar kontrol ediliyor...")
        
        original_count = len(df)
        df_no_duplicates = df.drop_duplicates()
        removed_count = original_count - len(df_no_duplicates)
        
        print(f"Tekrarlanan satır sayısı: {removed_count}")
        
        return df_no_duplicates
    
    def check_data_quality(self, df):
        """Veri kalitesini kontrol et."""
        print("Veri kalitesi kontrol ediliyor...")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
        }
        
        print("=== VERİ KALİTE RAPORU ===")
        print(f"Toplam satır: {quality_report['total_rows']}")
        print(f"Toplam kolon: {quality_report['total_columns']}")
        print(f"Eksik değer: {quality_report['missing_values']}")
        print(f"Tekrarlanan satır: {quality_report['duplicate_rows']}")
        print(f"Bellek kullanımı: {quality_report['memory_usage'] / 1024 / 1024:.2f} MB")
        print(f"Nümerik kolon: {quality_report['numeric_columns']}")
        print(f"Kategorik kolon: {quality_report['categorical_columns']}")
        
        return quality_report
    
    def complete_preprocessing_pipeline(self, df):
        """Tam ön işleme pipeline'ını çalıştır."""
        print("=== TAM ÖN İŞLEME PIPELINE ===\n")
        
        # 1. Veri türlerini doğrula
        df = self.validate_data_types(df)
        
        # 2. Tekrarlanan satırları kaldır
        df = self.remove_duplicates(df)
        
        # 3. Tansiyon değerlerini doğrula
        df = self.validate_blood_pressure(df)
        
        # 4. Aykırı değerleri temizle
        columns_to_clean = ['ap_hi', 'ap_lo', 'height', 'weight']
        df = self.clean_outliers(df, columns_to_clean, method='iqr')
        
        # 5. Eksik değerleri doldur
        df = self.fill_missing_values(df, strategy='median')
        
        # 6. Veri kalitesini kontrol et
        quality_report = self.check_data_quality(df)
        
        print("\n=== ÖN İŞLEME TAMAMLANDI ===")
        
        return df, quality_report 