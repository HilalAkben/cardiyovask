"""
Kalp Krizi Risk Tahmin Modeli - Feature Engineering Modülü
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Veri işleme ve feature engineering sınıfı."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = ['gender', 'smoke', 'alco', 'active']
        self.continuous_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        
    def load_data(self, file_path, sep=';'):
        """Veri dosyasını yükle."""
        try:
            df = pd.read_csv(file_path, sep=sep)
            print(f"Veri başarıyla yüklendi! Boyut: {df.shape}")
            print(f"Kolonlar: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return None
    
    def convert_age_to_years(self, df):
        """Age kolonunu gün formatından yıl formatına çevir."""
        if 'age' in df.columns:
            df['age_years'] = (df['age'] / 365).astype(int)
            print(f"Yaş dönüşümü tamamlandı. Örnek değerler:")
            print(f"Gün: {df['age'].head(3).tolist()} -> Yıl: {df['age_years'].head(3).tolist()}")
            return df
        else:
            print("'age' kolonu bulunamadı!")
            return df
    
    def check_missing_values(self, df):
        """Eksik değerleri kontrol et ve raporla."""
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        print("\n=== EKSİK DEĞER ANALİZİ ===")
        print(f"Toplam eksik değer: {total_missing}")
        
        if total_missing > 0:
            print("Eksik değer olan kolonlar:")
            for col, count in missing_counts.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    print(f"  {col}: {count} ({percentage:.2f}%)")
        else:
            print("Eksik değer bulunmamaktadır.")
        
        return missing_counts
    
    def clean_missing_values(self, df, strategy='drop'):
        """Eksik değerleri temizle."""
        if strategy == 'drop':
            df_cleaned = df.dropna()
            print(f"Eksik değerler silindi. Yeni boyut: {df_cleaned.shape}")
        elif strategy == 'median':
            df_cleaned = df.fillna(df.median())
            print("Eksik değerler medyan ile dolduruldu.")
        elif strategy == 'mean':
            df_cleaned = df.fillna(df.mean())
            print("Eksik değerler ortalama ile dolduruldu.")
        
        return df_cleaned
    
    def count_outliers(self, df, columns=None):
        """Outlier sayılarını hesapla ve raporla."""
        if columns is None:
            columns = ['height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        
        print("\n=== OUTLIER ANALİZİ ===")
        outlier_counts = {}
        
        for col in columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
                
                print(f"{col}: {outliers} outlier ({outliers/len(df)*100:.2f}%)")
        
        total_outliers = sum(outlier_counts.values())
        print(f"\nToplam outlier sayısı: {total_outliers}")
        
        return outlier_counts
    
    def remove_outliers(self, df, columns=None):
        """Outlier'ları çıkar."""
        if columns is None:
            columns = ['height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        
        print("\n=== OUTLIER TEMİZLEME ===")
        df_cleaned = df.copy()
        total_removed = 0
        
        for col in columns:
            if col in df_cleaned.columns:
                q1 = df_cleaned[col].quantile(0.25)
                q3 = df_cleaned[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Outlier'ları işaretle
                outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                removed_count = outliers.sum()
                total_removed += removed_count
                
                print(f"{col}: {removed_count} outlier çıkarıldı")
        
        # Tüm outlier'ları tek seferde çıkar
        outlier_mask = pd.Series([False] * len(df_cleaned), index=df_cleaned.index)
        for col in columns:
            if col in df_cleaned.columns:
                q1 = df_cleaned[col].quantile(0.25)
                q3 = df_cleaned[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                col_outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
        
        # Outlier'ları çıkar
        df_no_outliers = df_cleaned[~outlier_mask].reset_index(drop=True)
        
        print(f"Toplam {total_removed} outlier çıkarıldı")
        print(f"Orijinal boyut: {df.shape} -> Temizlenmiş boyut: {df_no_outliers.shape}")
        
        return df_no_outliers
    
    def encode_categorical_variables(self, df):
        """Kategorik değişkenleri encode et."""
        print("\n=== KATEGORİK DEĞİŞKEN ENCODING ===")
        
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                # Label encoding uygula
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                
                print(f"{col}: {len(le.classes_)} benzersiz değer -> {list(le.classes_)} -> {list(range(len(le.classes_)))}")
        
        return df_encoded
    
    def scale_continuous_variables(self, df):
        """Sürekli değişkenleri ölçekle."""
        print("\n=== SÜREKLİ DEĞİŞKEN ÖLÇEKLEME ===")
        
        df_scaled = df.copy()
        
        # Mevcut sürekli kolonları kontrol et
        available_continuous = [col for col in self.continuous_columns if col in df_scaled.columns]
        
        if available_continuous:
            df_scaled[available_continuous] = self.scaler.fit_transform(df_scaled[available_continuous])
            print(f"Ölçeklenen kolonlar: {available_continuous}")
            print("StandardScaler uygulandı (ortalama=0, standart sapma=1)")
        else:
            print("Ölçeklenecek sürekli değişken bulunamadı!")
        
        return df_scaled
    
    def prepare_features(self, df, target_column='cardio'):
        """Hedef değişkeni ayır ve feature matrix'i hazırla."""
        if target_column not in df.columns:
            print(f"Hedef değişken '{target_column}' bulunamadı!")
            return None, None
        
        # Hedef değişkeni ayır
        y = df[target_column]
        
        # Feature matrix'i hazırla (id ve age kolonlarını çıkar)
        exclude_columns = ['id', 'age', target_column]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns]
        
        print(f"\nFeature matrix boyutu: {X.shape}")
        print(f"Hedef değişken boyutu: {y.shape}")
        print(f"Feature kolonları: {list(X.columns)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Veriyi eğitim ve test setlerine ayır."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n=== VERİ AYRIMI ===")
        print(f"Eğitim seti: {X_train.shape}")
        print(f"Test seti: {X_test.shape}")
        print(f"Eğitim hedef: {y_train.shape}")
        print(f"Test hedef: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def process_pipeline_with_outliers(self, file_path, sep=';'):
        """Outlier'ları çıkarmadan veri işleme pipeline'ını çalıştır."""
        print("=== KALP KRİZİ RİSK TAHMİN MODELİ - OUTLIER'LAR İLE VERİ İŞLEME ===\n")
        
        # 1. Veri yükleme
        df = self.load_data(file_path, sep)
        if df is None:
            return None
        
        # 2. Yaş dönüşümü
        df = self.convert_age_to_years(df)
        
        # 3. Eksik değer kontrolü ve temizleme
        missing_counts = self.check_missing_values(df)
        df = self.clean_missing_values(df, strategy='drop')
        
        # 4. Outlier sayısı (çıkarmadan)
        outlier_counts = self.count_outliers(df)
        
        # 5. Kategorik değişken encoding
        df = self.encode_categorical_variables(df)
        
        # 6. Sürekli değişken ölçekleme
        df = self.scale_continuous_variables(df)
        
        # 7. Feature hazırlama
        X, y = self.prepare_features(df)
        if X is None:
            return None
        
        # 8. Veri ayrımı
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\n=== OUTLIER'LAR İLE VERİ İŞLEME TAMAMLANDI ===")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'missing_counts': missing_counts,
            'outlier_counts': outlier_counts,
            'data_type': 'with_outliers'
        }
    
    def process_pipeline_without_outliers(self, file_path, sep=';'):
        """Outlier'ları çıkararak veri işleme pipeline'ını çalıştır."""
        print("=== KALP KRİZİ RİSK TAHMİN MODELİ - OUTLIER'LAR ÇIKARILARAK VERİ İŞLEME ===\n")
        
        # 1. Veri yükleme
        df = self.load_data(file_path, sep)
        if df is None:
            return None
        
        # 2. Yaş dönüşümü
        df = self.convert_age_to_years(df)
        
        # 3. Eksik değer kontrolü ve temizleme
        missing_counts = self.check_missing_values(df)
        df = self.clean_missing_values(df, strategy='drop')
        
        # 4. Outlier sayısı ve temizleme
        outlier_counts = self.count_outliers(df)
        df = self.remove_outliers(df)
        
        # 5. Kategorik değişken encoding
        df = self.encode_categorical_variables(df)
        
        # 6. Sürekli değişken ölçekleme
        df = self.scale_continuous_variables(df)
        
        # 7. Feature hazırlama
        X, y = self.prepare_features(df)
        if X is None:
            return None
        
        # 8. Veri ayrımı
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\n=== OUTLIER'LAR ÇIKARILARAK VERİ İŞLEME TAMAMLANDI ===")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'missing_counts': missing_counts,
            'outlier_counts': outlier_counts,
            'data_type': 'without_outliers'
        }
    
    def process_pipeline(self, file_path, sep=';'):
        """Tüm veri işleme pipeline'ını çalıştır."""
        print("=== KALP KRİZİ RİSK TAHMİN MODELİ - VERİ İŞLEME PIPELINE ===\n")
        
        # 1. Veri yükleme
        df = self.load_data(file_path, sep)
        if df is None:
            return None
        
        # 2. Yaş dönüşümü
        df = self.convert_age_to_years(df)
        
        # 3. Eksik değer kontrolü ve temizleme
        missing_counts = self.check_missing_values(df)
        df = self.clean_missing_values(df, strategy='drop')
        
        # 4. Outlier sayısı
        outlier_counts = self.count_outliers(df)
        
        # 5. Kategorik değişken encoding
        df = self.encode_categorical_variables(df)
        
        # 6. Sürekli değişken ölçekleme
        df = self.scale_continuous_variables(df)
        
        # 7. Feature hazırlama
        X, y = self.prepare_features(df)
        if X is None:
            return None
        
        # 8. Veri ayrımı
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\n=== VERİ İŞLEME TAMAMLANDI ===")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'missing_counts': missing_counts,
            'outlier_counts': outlier_counts
        } 