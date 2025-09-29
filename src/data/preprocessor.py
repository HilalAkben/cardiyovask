"""
Kalp Krizi Risk Tahmin Modeli - Veri Ön İşleme Modülü
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataPreprocessor:
    """Veri ön işleme ve temizleme sınıfı."""
    
    def __init__(self):
        self.cleaning_summary = {}
        self.outlier_rules = {
            'age': {'min': 6570, 'max': 36500, 'extreme_max': 43800},  # 18-100 yaş, 120 yaş üstü kesin hatalı
            'height': {'min': 120, 'max': 220},  # cm
            'weight': {'min': 30, 'max': 200, 'extreme_max': 300},  # kg
            'ap_hi': {'min': 80, 'max': 240},  # sistolik
            'ap_lo': {'min': 40, 'max': 150},  # diyastolik
            'gender': {'valid_values': [1, 2]},
            'cholesterol': {'valid_values': [1, 2, 3]},
            'gluc': {'valid_values': [1, 2, 3]},
            'smoke': {'valid_values': [0, 1]},
            'alco': {'valid_values': [0, 1]},
            'active': {'valid_values': [0, 1]},
            'cardio': {'valid_values': [0, 1]}
        }
    
    def convert_age_to_years(self, df):
        """Age kolonunu gün formatından yıl formatına çevir."""
        if 'age' in df.columns:
            # NaN değerleri kontrol et ve temizle
            age_clean = df['age'].dropna()
            if len(age_clean) > 0:
                df['age_years'] = (df['age'] / 365).round().astype('Int64')  # Nullable integer
                print(f"Yaş dönüşümü tamamlandı. Örnek değerler:")
                print(f"Gün: {df['age'].head(3).tolist()} -> Yıl: {df['age_years'].head(3).tolist()}")
            else:
                print("'age' kolonunda geçerli değer bulunamadı!")
                df['age_years'] = pd.Series(dtype='Int64')
            return df
        else:
            print("'age' kolonu bulunamadı!")
            return df
    
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
    
    def check_duplicates(self, df):
        """Duplike kayıtları kontrol et."""
        print("\n=== DUPLİKE KAYIT KONTROLÜ ===")
        
        duplicate_count = df.duplicated().sum()
        print(f"Toplam duplike kayıt sayısı: {duplicate_count}")
        
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            print(f"Duplike kayıt oranı: {duplicate_percentage:.2f}%")
            
            # Duplike kayıtları göster
            duplicate_rows = df[df.duplicated(keep=False)]
            print(f"Duplike kayıt örnekleri (ilk 5):")
            print(duplicate_rows.head())
        else:
            print("Duplike kayıt bulunmamaktadır.")
        
        return duplicate_count
    
    def analyze_missing_values(self, df):
        """Eksik değerleri detaylı analiz et."""
        print("\n=== EKSİK DEĞER ANALİZİ ===")
        
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        total_cells = df.shape[0] * df.shape[1]
        
        print(f"Toplam eksik değer: {total_missing}")
        print(f"Toplam hücre sayısı: {total_cells}")
        print(f"Eksik değer oranı: {(total_missing / total_cells) * 100:.2f}%")
        
        if total_missing > 0:
            print("\nKolon bazında eksik değer analizi:")
            missing_df = pd.DataFrame({
                'Kolon': missing_counts.index,
                'Eksik_Sayı': missing_counts.values,
                'Eksik_Oranı': (missing_counts.values / len(df)) * 100
            }).sort_values('Eksik_Oranı', ascending=False)
            
            print(missing_df.to_string(index=False))
            
            # Eksik değer görselleştirmesi
            self.plot_missing_values(df)
        else:
            print("Eksik değer bulunmamaktadır.")
        
        return missing_counts
    
    def plot_missing_values(self, df):
        """Eksik değerleri görselleştir."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Sadece eksik değeri olan kolonları göster
        missing_data = pd.DataFrame({
            'Kolon': missing_counts.index,
            'Eksik_Sayı': missing_counts.values,
            'Eksik_Oranı': missing_percentages.values
        })
        missing_data = missing_data[missing_data['Eksik_Sayı'] > 0].sort_values('Eksik_Oranı', ascending=True)
        
        if len(missing_data) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot - eksik değer sayıları
            ax1.barh(missing_data['Kolon'], missing_data['Eksik_Sayı'])
            ax1.set_xlabel('Eksik Değer Sayısı')
            ax1.set_title('Kolon Bazında Eksik Değer Sayıları')
            ax1.grid(True, alpha=0.3)
            
            # Bar plot - eksik değer oranları
            ax2.barh(missing_data['Kolon'], missing_data['Eksik_Oranı'])
            ax2.set_xlabel('Eksik Değer Oranı (%)')
            ax2.set_title('Kolon Bazında Eksik Değer Oranları')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def detect_domain_outliers(self, df):
        """Domain-specific outlier'ları tespit et."""
        print("\n=== DOMAIN-SPECIFIC OUTLIER TESPİTİ ===")
        
        outlier_summary = {}
        total_outliers = 0
        
        for column, rules in self.outlier_rules.items():
            if column in df.columns:
                if 'valid_values' in rules:
                    # Kategorik değişkenler için
                    invalid_mask = ~df[column].isin(rules['valid_values'])
                    outlier_count = invalid_mask.sum()
                    outlier_summary[column] = {
                        'outlier_count': outlier_count,
                        'outlier_percentage': (outlier_count / len(df)) * 100,
                        'type': 'categorical',
                        'invalid_values': df[column][invalid_mask].unique().tolist() if outlier_count > 0 else []
                    }
                else:
                    # Sayısal değişkenler için
                    min_val = rules['min']
                    max_val = rules['max']
                    
                    # Normal outlier'lar
                    normal_outliers = (df[column] < min_val) | (df[column] > max_val)
                    normal_outlier_count = normal_outliers.sum()
                    
                    # Ek güvenlik kontrolü (varsa)
                    extreme_outliers = 0
                    if 'extreme_max' in rules:
                        extreme_outliers = (df[column] > rules['extreme_max']).sum()
                    
                    total_column_outliers = normal_outlier_count
                    outlier_summary[column] = {
                        'outlier_count': total_column_outliers,
                        'outlier_percentage': (total_column_outliers / len(df)) * 100,
                        'type': 'numerical',
                        'normal_outliers': normal_outlier_count,
                        'extreme_outliers': extreme_outliers,
                        'min_valid': min_val,
                        'max_valid': max_val
                    }
                
                total_outliers += outlier_summary[column]['outlier_count']
                print(f"{column}: {outlier_summary[column]['outlier_count']} outlier ({outlier_summary[column]['outlier_percentage']:.2f}%)")
        
        # Özel tansiyon kuralı kontrolü
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            bp_invalid = df['ap_hi'] < df['ap_lo']
            bp_invalid_count = bp_invalid.sum()
            if bp_invalid_count > 0:
                print(f"Tansiyon kuralı ihlali (ap_hi < ap_lo): {bp_invalid_count} kayıt")
                outlier_summary['bp_rule_violation'] = {
                    'outlier_count': bp_invalid_count,
                    'outlier_percentage': (bp_invalid_count / len(df)) * 100,
                    'type': 'rule_violation'
                }
                total_outliers += bp_invalid_count
        
        print(f"\nToplam domain-specific outlier: {total_outliers}")
        
        # Outlier görselleştirmesi
        self.plot_domain_outliers(df, outlier_summary)
        
        return outlier_summary
    
    def plot_domain_outliers(self, df, outlier_summary):
        """Domain-specific outlier'ları görselleştir."""
        # Sayısal değişkenler için box plot
        numerical_columns = [col for col, info in outlier_summary.items() 
                           if info['type'] == 'numerical' and col in df.columns]
        
        if numerical_columns:
            n_cols = 3
            n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, col in enumerate(numerical_columns):
                row = idx // n_cols
                col_idx = idx % n_cols
                
                # Box plot
                axes[row, col_idx].boxplot(df[col].dropna())
                axes[row, col_idx].set_title(f'{col} - Box Plot')
                axes[row, col_idx].set_ylabel('Değer')
                
                # Outlier kurallarını çiz
                rules = self.outlier_rules[col]
                axes[row, col_idx].axhline(y=rules['min'], color='r', linestyle='--', alpha=0.7, label=f'Min: {rules["min"]}')
                axes[row, col_idx].axhline(y=rules['max'], color='r', linestyle='--', alpha=0.7, label=f'Max: {rules["max"]}')
                
                if 'extreme_max' in rules:
                    axes[row, col_idx].axhline(y=rules['extreme_max'], color='darkred', linestyle='-', alpha=0.7, label=f'Extreme: {rules["extreme_max"]}')
                
                axes[row, col_idx].legend()
                axes[row, col_idx].grid(True, alpha=0.3)
            
            # Boş subplot'ları gizle
            for idx in range(len(numerical_columns), n_rows * n_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('domain_outliers_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Kategorik değişkenler için bar plot
        categorical_columns = [col for col, info in outlier_summary.items() 
                             if info['type'] == 'categorical' and col in df.columns]
        
        if categorical_columns:
            n_cols = 2
            n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, col in enumerate(categorical_columns):
                row = idx // n_cols
                col_idx = idx % n_cols
                
                # Value counts
                value_counts = df[col].value_counts().sort_index()
                axes[row, col_idx].bar(value_counts.index, value_counts.values)
                axes[row, col_idx].set_title(f'{col} - Değer Dağılımı')
                axes[row, col_idx].set_xlabel('Değer')
                axes[row, col_idx].set_ylabel('Frekans')
                axes[row, col_idx].grid(True, alpha=0.3)
                
                # Geçerli değerleri vurgula
                valid_values = self.outlier_rules[col]['valid_values']
                for i, val in enumerate(value_counts.index):
                    if val in valid_values:
                        axes[row, col_idx].bar(i, value_counts.iloc[i], color='green', alpha=0.7)
                    else:
                        axes[row, col_idx].bar(i, value_counts.iloc[i], color='red', alpha=0.7)
            
            # Boş subplot'ları gizle
            for idx in range(len(categorical_columns), n_rows * n_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('categorical_outliers_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
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
    
    def clean_domain_outliers(self, df, remove_outliers=True):
        """Domain-specific outlier'ları temizle."""
        print(f"\n=== DOMAIN-SPECIFIC OUTLIER TEMİZLEME ===")
        
        df_cleaned = df.copy()
        total_removed = 0
        
        for column, rules in self.outlier_rules.items():
            if column in df_cleaned.columns:
                if 'valid_values' in rules:
                    # Kategorik değişkenler için
                    invalid_mask = ~df_cleaned[column].isin(rules['valid_values'])
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        if remove_outliers:
                            df_cleaned.loc[invalid_mask, column] = np.nan
                            print(f"{column}: {invalid_count} geçersiz değer NaN yapıldı")
                        else:
                            print(f"{column}: {invalid_count} geçersiz değer tespit edildi (temizlenmedi)")
                        total_removed += invalid_count
                else:
                    # Sayısal değişkenler için
                    min_val = rules['min']
                    max_val = rules['max']
                    
                    # Normal outlier'lar
                    normal_outliers = (df_cleaned[column] < min_val) | (df_cleaned[column] > max_val)
                    normal_outlier_count = normal_outliers.sum()
                    
                    if normal_outlier_count > 0:
                        if remove_outliers:
                            df_cleaned.loc[normal_outliers, column] = np.nan
                            print(f"{column}: {normal_outlier_count} outlier NaN yapıldı")
                        else:
                            print(f"{column}: {normal_outlier_count} outlier tespit edildi (temizlenmedi)")
                        total_removed += normal_outlier_count
                    
                    # Ek güvenlik kontrolü (varsa)
                    if 'extreme_max' in rules:
                        extreme_outliers = df_cleaned[column] > rules['extreme_max']
                        extreme_outlier_count = extreme_outliers.sum()
                        if extreme_outlier_count > 0:
                            if remove_outliers:
                                df_cleaned.loc[extreme_outliers, column] = np.nan
                                print(f"{column}: {extreme_outlier_count} extreme outlier NaN yapıldı")
                            else:
                                print(f"{column}: {extreme_outlier_count} extreme outlier tespit edildi (temizlenmedi)")
                            total_removed += extreme_outlier_count
        
        # Özel tansiyon kuralı kontrolü
        if 'ap_hi' in df_cleaned.columns and 'ap_lo' in df_cleaned.columns:
            bp_invalid = df_cleaned['ap_hi'] < df_cleaned['ap_lo']
            bp_invalid_count = bp_invalid.sum()
            if bp_invalid_count > 0:
                if remove_outliers:
                    df_cleaned.loc[bp_invalid, ['ap_hi', 'ap_lo']] = np.nan
                    print(f"Tansiyon kuralı ihlali: {bp_invalid_count} kayıt NaN yapıldı")
                else:
                    print(f"Tansiyon kuralı ihlali: {bp_invalid_count} kayıt tespit edildi (temizlenmedi)")
                total_removed += bp_invalid_count
        
        print(f"Toplam temizlenen outlier: {total_removed}")
        return df_cleaned
    
    def clean_outliers(self, df, columns, method='iqr'):
        """Aykırı değerleri temizle (eski IQR metodu)."""
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
            # Sadece sayısal kolonlar için median hesapla
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_filled[col].isnull().any():
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            
            # Kategorik kolonlar için mode kullan
            categorical_columns = df_filled.select_dtypes(include=['category', 'object']).columns
            for col in categorical_columns:
                if df_filled[col].isnull().any():
                    mode_value = df_filled[col].mode()
                    if len(mode_value) > 0:
                        df_filled[col] = df_filled[col].fillna(mode_value.iloc[0])
                    else:
                        # Eğer mode yoksa, en sık kullanılan değeri kullan
                        df_filled[col] = df_filled[col].fillna(df_filled[col].value_counts().index[0])
                        
        elif strategy == 'mean':
            # Sadece sayısal kolonlar için mean hesapla
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_filled[col].isnull().any():
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            
            # Kategorik kolonlar için mode kullan
            categorical_columns = df_filled.select_dtypes(include=['category', 'object']).columns
            for col in categorical_columns:
                if df_filled[col].isnull().any():
                    mode_value = df_filled[col].mode()
                    if len(mode_value) > 0:
                        df_filled[col] = df_filled[col].fillna(mode_value.iloc[0])
                        
        elif strategy == 'mode':
            for col in df_filled.columns:
                if df_filled[col].isnull().any():
                    mode_value = df_filled[col].mode()
                    if len(mode_value) > 0:
                        df_filled[col] = df_filled[col].fillna(mode_value.iloc[0])
                        
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
    
    def apply_explicit_outlier_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kurala dayalı outlier düzenleme: sayısalları kırp, kategorik geçersizleri sil, bp kuralını uygula, NaN'ları düş."""
        df_adj = df.copy()

        print("\n=== KURAL BAZLI OUTLIER DÜZENLEME ===")

        # 1) age: 6570 <= age <= 43800, aksi satırı sil
        if 'age' in df_adj.columns:
            before = len(df_adj)
            df_adj = df_adj[(df_adj['age'] >= 6570) & (df_adj['age'] <= 43800)]
            dropped_age = before - len(df_adj)
            if dropped_age > 0:
                print(f"[DROP] age sınırı nedeniyle silinen satır: {dropped_age}")

        # Yardımcı: sayısal bir kolonu kırpıp kaç değer kırpıldığını döndür
        def clip_and_count(series: pd.Series, low: float, high: float, name: str) -> int:
            if series is None:
                return 0
            below = (series < low).sum()
            above = (series > high).sum()
            clipped = below + above
            if clipped > 0:
                print(f"[CLIP] {name}: {clipped} değer aralığa kırpıldı (low<{below}, high>{above})")
            return clipped

        # 2) height: [120, 220]
        if 'height' in df_adj.columns:
            clip_and_count(df_adj['height'], 120, 220, 'height')
            df_adj['height'] = df_adj['height'].clip(lower=120, upper=220)

        # 3) weight: [30, 200]
        if 'weight' in df_adj.columns:
            clip_and_count(df_adj['weight'], 30, 200, 'weight')
            df_adj['weight'] = df_adj['weight'].clip(lower=30, upper=200)

        # 4) ap_hi: [80, 240]
        if 'ap_hi' in df_adj.columns:
            clip_and_count(df_adj['ap_hi'], 80, 240, 'ap_hi')
            df_adj['ap_hi'] = df_adj['ap_hi'].clip(lower=80, upper=240)

        # 5) ap_lo: [40, 150]
        if 'ap_lo' in df_adj.columns:
            clip_and_count(df_adj['ap_lo'], 40, 150, 'ap_lo')
            df_adj['ap_lo'] = df_adj['ap_lo'].clip(lower=40, upper=150)

        # 6-12) Kategorikler için geçersizleri sil
        categorical_rules = {
            'gender': [1, 2],
            'cholesterol': [1, 2, 3],
            'gluc': [1, 2, 3],
            'smoke': [0, 1],
            'alco': [0, 1],
            'active': [0, 1],
            'cardio': [0, 1],
        }
        for col, valid in categorical_rules.items():
            if col in df_adj.columns:
                before = len(df_adj)
                df_adj = df_adj[df_adj[col].isin(valid)]
                dropped_cat = before - len(df_adj)
                if dropped_cat > 0:
                    print(f"[DROP] {col} geçersiz değerleri nedeniyle silinen satır: {dropped_cat} (geçerli: {valid})")

        # 13) ap_hi < ap_lo olan satırları sil
        if 'ap_hi' in df_adj.columns and 'ap_lo' in df_adj.columns:
            before = len(df_adj)
            df_adj = df_adj[df_adj['ap_hi'] >= df_adj['ap_lo']]
            dropped_bp = before - len(df_adj)
            if dropped_bp > 0:
                print(f"[DROP] bp kuralı (ap_hi < ap_lo) nedeniyle silinen satır: {dropped_bp}")

        # Oluşan tüm NaN değerleri sil
        before_rows = len(df_adj)
        df_adj = df_adj.dropna()
        removed_by_na = before_rows - len(df_adj)
        if removed_by_na > 0:
            print(f"[DROPNA] NaN nedeniyle silinen satır: {removed_by_na}")

        return df_adj

    def complete_preprocessing_pipeline(self, df):
        """Tam ön işleme pipeline'ını çalıştır."""
        print("=== TAM ÖN İŞLEME PIPELINE ===\n")
        
        # 1. Veri türlerini doğrula
        df = self.validate_data_types(df)
        
        # 2. Tekrarlanan satırları kaldır
        df = self.remove_duplicates(df)
        
        # 3. Tansiyon değerlerini doğrula
        df = self.validate_blood_pressure(df)
        
        # 4-5. Kural bazlı outlier düzenleme (kırp/sil) ve NaN düş
        df = self.apply_explicit_outlier_rules(df)
        
        # 6. Veri kalitesini kontrol et
        quality_report = self.check_data_quality(df)
        
        print("\n=== ÖN İŞLEME TAMAMLANDI ===")
        
        return df, quality_report
    
    def complete_preprocessing_pipeline(self, file_path, sep=';', remove_outliers=True):
        """Tam ön işleme pipeline'ını çalıştır."""
        print("="*80)
        print("KALP KRİZİ RİSK TAHMİN MODELİ - TAM ÖN İŞLEME PIPELINE")
        print("="*80)
        
        # 1. Veri yükleme
        df = self.load_data(file_path, sep)
        if df is None:
            return None
        
        # 2. Yaş dönüşümü (gün → yıl)
        df = self.convert_age_to_years(df)
        
        # 3. Duplike kayıt kontrolü
        duplicate_count = self.check_duplicates(df)
        
        # 4. Eksik değer analizi
        missing_counts = self.analyze_missing_values(df)
        
        # 5. Domain-specific outlier tespiti
        outlier_summary = self.detect_domain_outliers(df)
        
        # 6. Veri türlerini doğrula
        df = self.validate_data_types(df)
        
        # 7. Tekrarlanan satırları kaldır
        df = self.remove_duplicates(df)
        
        # 8-9. Kural bazlı outlier düzenleme (kırp/sil) ve NaN düş (median yok)
        df = self.apply_explicit_outlier_rules(df)
        
        # 10. Veri kalitesini kontrol et
        quality_report = self.check_data_quality(df)
        
        # 11. Temizleme özeti
        summary = self.get_cleaning_summary(self.load_data(file_path, sep), df)
        self.print_cleaning_summary(summary)
        
        print("\n" + "="*80)
        print("TAM ÖN İŞLEME PIPELINE TAMAMLANDI!")
        print("="*80)
        
        return {
            'data': df,
            'quality_report': quality_report,
            'missing_counts': missing_counts,
            'outlier_summary': outlier_summary,
            'duplicate_count': duplicate_count,
            'cleaning_summary': summary
        }

def main():
    """Test fonksiyonu - preprocessor.py'yi test etmek için."""
    import sys
    from pathlib import Path
    
    # Proje kök dizinini Python path'ine ekle
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    # Veri dosyası yolu
    data_path = project_root / "src" / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        return
    
    # Preprocessor test
    preprocessor = DataPreprocessor()
    
    print("="*80)
    print("PREPROCESSOR TEST - OUTLIER'LAR İLE")
    print("="*80)
    
    result_with_outliers = preprocessor.complete_preprocessing_pipeline(
        str(data_path), remove_outliers=False
    )
    
    if result_with_outliers:
        print(f"\nOutlier'lı veri işleme başarılı!")
        print(f"Final veri boyutu: {result_with_outliers['data'].shape}")
    
    print("\n" + "="*80)
    print("PREPROCESSOR TEST - OUTLIER'LAR ÇIKARILARAK")
    print("="*80)
    
    result_without_outliers = preprocessor.complete_preprocessing_pipeline(
        str(data_path), remove_outliers=True
    )
    
    if result_without_outliers:
        print(f"\nOutlier'lar çıkarılarak veri işleme başarılı!")
        print(f"Final veri boyutu: {result_without_outliers['data'].shape}")

if __name__ == "__main__":
    main() 