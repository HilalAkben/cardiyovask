"""
Kalp Krizi Risk Tahmin Modeli - Veri Yükleme Yardımcı Modülü
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Veri yükleme ve kaydetme işlemleri için yardımcı sınıf."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.parquet']
    
    def load_csv(self, file_path, sep=';', encoding='utf-8'):
        """CSV dosyasını yükle."""
        try:
            df = pd.read_csv(file_path, sep=sep, encoding=encoding)
            print(f"CSV dosyası başarıyla yüklendi: {file_path}")
            print(f"Veri boyutu: {df.shape}")
            return df
        except Exception as e:
            print(f"CSV yükleme hatası: {e}")
            return None
    
    def save_csv(self, df, file_path, sep=';', index=False):
        """DataFrame'i CSV olarak kaydet."""
        try:
            # Dizini oluştur
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(file_path, sep=sep, index=index)
            print(f"Veri başarıyla kaydedildi: {file_path}")
        except Exception as e:
            print(f"CSV kaydetme hatası: {e}")
    
    def load_excel(self, file_path, sheet_name=0):
        """Excel dosyasını yükle."""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"Excel dosyası başarıyla yüklendi: {file_path}")
            print(f"Veri boyutu: {df.shape}")
            return df
        except Exception as e:
            print(f"Excel yükleme hatası: {e}")
            return None
    
    def save_excel(self, df, file_path, sheet_name='Sheet1', index=False):
        """DataFrame'i Excel olarak kaydet."""
        try:
            # Dizini oluştur
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_excel(file_path, sheet_name=sheet_name, index=index)
            print(f"Veri başarıyla kaydedildi: {file_path}")
        except Exception as e:
            print(f"Excel kaydetme hatası: {e}")
    
    def load_parquet(self, file_path):
        """Parquet dosyasını yükle."""
        try:
            df = pd.read_parquet(file_path)
            print(f"Parquet dosyası başarıyla yüklendi: {file_path}")
            print(f"Veri boyutu: {df.shape}")
            return df
        except Exception as e:
            print(f"Parquet yükleme hatası: {e}")
            return None
    
    def save_parquet(self, df, file_path):
        """DataFrame'i Parquet olarak kaydet."""
        try:
            # Dizini oluştur
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(file_path)
            print(f"Veri başarıyla kaydedildi: {file_path}")
        except Exception as e:
            print(f"Parquet kaydetme hatası: {e}")
    
    def auto_load(self, file_path):
        """Dosya uzantısına göre otomatik yükleme."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Dosya bulunamadı: {file_path}")
            return None
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            return self.load_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return self.load_excel(file_path)
        elif file_extension == '.parquet':
            return self.load_parquet(file_path)
        else:
            print(f"Desteklenmeyen dosya formatı: {file_extension}")
            print(f"Desteklenen formatlar: {self.supported_formats}")
            return None
    
    def get_data_info(self, df):
        """Veri hakkında genel bilgi ver."""
        if df is None:
            return None
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        print("=== VERİ BİLGİLERİ ===")
        print(f"Boyut: {info['shape']}")
        print(f"Kolon sayısı: {len(info['columns'])}")
        print(f"Satır sayısı: {info['shape'][0]}")
        print(f"Bellek kullanımı: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        print(f"Nümerik kolonlar: {len(info['numeric_columns'])}")
        print(f"Kategorik kolonlar: {len(info['categorical_columns'])}")
        
        if info['missing_values']:
            total_missing = sum(info['missing_values'].values())
            print(f"Toplam eksik değer: {total_missing}")
        
        return info
    
    def validate_data_path(self, file_path):
        """Dosya yolunu doğrula."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Dosya bulunamadı: {file_path}")
            return False
        
        if not file_path.is_file():
            print(f"Geçerli bir dosya değil: {file_path}")
            return False
        
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_formats:
            print(f"Desteklenmeyen dosya formatı: {file_extension}")
            return False
        
        return True 