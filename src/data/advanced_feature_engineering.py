"""
Kalp Krizi Risk Tahmin Modeli - Gelişmiş Feature Engineering Modülü
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Gelişmiş feature engineering sınıfı."""
    
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
            print(f"Yaş dönüşümü tamamlandı.")
            return df
        else:
            print("'age' kolonu bulunamadı!")
            return df
    
    def check_missing_values(self, df):
        """Eksik değerleri kontrol et."""
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Eksik değerler bulundu:")
            print(missing_counts[missing_counts > 0])
        else:
            print("Eksik değer bulunamadı.")
        return missing_counts
    
    def clean_missing_values(self, df, strategy='drop'):
        """Eksik değerleri temizle."""
        if strategy == 'drop':
            df_cleaned = df.dropna()
            print(f"Eksik değerler silindi. Yeni boyut: {df_cleaned.shape}")
        return df_cleaned
    
    def count_outliers(self, df, columns=None):
        """Outlier sayılarını hesapla."""
        if columns is None:
            columns = self.continuous_columns
        
        outlier_counts = {}
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_counts[col] = len(outliers)
        
        print("Outlier sayıları:")
        for col, count in outlier_counts.items():
            print(f"  {col}: {count} outlier")
        
        return outlier_counts
    
    def remove_outliers(self, df, columns=None):
        """Outlier'ları çıkar."""
        if columns is None:
            columns = self.continuous_columns
        
        df_no_outliers = df.copy()
        total_removed = 0
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Outlier'ları çıkar
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df_no_outliers = df_no_outliers[mask]
                removed = len(df) - len(df_no_outliers)
                total_removed += removed
        
        print(f"Toplam {total_removed} outlier çıkarıldı. Yeni boyut: {df_no_outliers.shape}")
        return df_no_outliers
    
    def create_age_groups(self, df):
        """Yaş grupları oluştur."""
        if 'age_years' in df.columns:
            # Yaş grupları
            df['age_group'] = pd.cut(df['age_years'], 
                                   bins=[0, 30, 40, 50, 60, 70, 100], 
                                   labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
            
            # Yaş grubu encoding
            le = LabelEncoder()
            df['age_group_encoded'] = le.fit_transform(df['age_group'])
            self.label_encoders['age_group'] = le
            
            print("Yaş grupları oluşturuldu.")
            return df
        return df
    
    def create_bmi(self, df):
        """BMI (Body Mass Index) hesapla."""
        if 'height' in df.columns and 'weight' in df.columns:
            # BMI = weight (kg) / (height (m))^2
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            
            # BMI kategorileri
            df['bmi_category'] = pd.cut(df['bmi'], 
                                      bins=[0, 18.5, 25, 30, 100], 
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            
            # BMI kategori encoding
            le = LabelEncoder()
            df['bmi_category_encoded'] = le.fit_transform(df['bmi_category'])
            self.label_encoders['bmi_category'] = le
            
            print("BMI ve BMI kategorileri oluşturuldu.")
            return df
        return df
    
    def create_blood_pressure_features(self, df):
        """Tansiyon ile ilgili özellikler oluştur."""
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            # Tansiyon farkı
            df['bp_diff'] = df['ap_hi'] - df['ap_lo']
            
            # Ortalama tansiyon
            df['bp_mean'] = (df['ap_hi'] + df['ap_lo']) / 2
            
            # Tansiyon kategorileri (WHO sınıflandırması)
            def categorize_bp(systolic, diastolic):
                if systolic < 120 and diastolic < 80:
                    return 'Normal'
                elif systolic < 130 and diastolic < 80:
                    return 'Elevated'
                elif systolic < 140 and diastolic < 90:
                    return 'High Normal'
                elif systolic < 160 and diastolic < 100:
                    return 'Stage 1'
                elif systolic < 180 and diastolic < 110:
                    return 'Stage 2'
                else:
                    return 'Stage 3'
            
            df['bp_category'] = df.apply(lambda x: categorize_bp(x['ap_hi'], x['ap_lo']), axis=1)
            
            # Tansiyon kategori encoding
            le = LabelEncoder()
            df['bp_category_encoded'] = le.fit_transform(df['bp_category'])
            self.label_encoders['bp_category'] = le
            
            print("Tansiyon özellikleri oluşturuldu.")
            return df
        return df
    
    def create_health_risk_score(self, df):
        """Sağlık risk skoru oluştur."""
        risk_score = 0
        
        # Yaş riski
        if 'age_years' in df.columns:
            age_risk = np.where(df['age_years'] > 50, 2, 
                              np.where(df['age_years'] > 40, 1, 0))
            risk_score += age_risk
        
        # BMI riski
        if 'bmi' in df.columns:
            bmi_risk = np.where(df['bmi'] > 30, 2,
                              np.where(df['bmi'] > 25, 1, 0))
            risk_score += bmi_risk
        
        # Tansiyon riski
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            bp_risk = np.where((df['ap_hi'] > 140) | (df['ap_lo'] > 90), 2,
                             np.where((df['ap_hi'] > 130) | (df['ap_lo'] > 80), 1, 0))
            risk_score += bp_risk
        
        # Kolesterol riski
        if 'cholesterol' in df.columns:
            chol_risk = np.where(df['cholesterol'] > 2, 1, 0)
            risk_score += chol_risk
        
        # Şeker riski
        if 'gluc' in df.columns:
            gluc_risk = np.where(df['gluc'] > 2, 1, 0)
            risk_score += gluc_risk
        
        # Sigara ve alkol riski
        if 'smoke' in df.columns:
            smoke_risk = df['smoke'] * 1
            risk_score += smoke_risk
        
        if 'alco' in df.columns:
            alco_risk = df['alco'] * 1
            risk_score += alco_risk
        
        df['health_risk_score'] = risk_score
        
        # Risk kategorileri
        df['risk_category'] = pd.cut(df['health_risk_score'], 
                                   bins=[0, 3, 6, 9, 15], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Risk kategori encoding
        le = LabelEncoder()
        df['risk_category_encoded'] = le.fit_transform(df['risk_category'])
        self.label_encoders['risk_category'] = le
        
        print("Sağlık risk skoru oluşturuldu.")
        return df
    
    def create_interaction_features(self, df):
        """Etkileşim özellikleri oluştur."""
        # Yaş x BMI
        if 'age_years' in df.columns and 'bmi' in df.columns:
            df['age_bmi_interaction'] = df['age_years'] * df['bmi']
        
        # Yaş x Tansiyon
        if 'age_years' in df.columns and 'ap_hi' in df.columns:
            df['age_systolic_interaction'] = df['age_years'] * df['ap_hi']
        
        # Cinsiyet x Yaş
        if 'gender' in df.columns and 'age_years' in df.columns:
            df['gender_age_interaction'] = df['gender'] * df['age_years']
        
        # BMI x Tansiyon
        if 'bmi' in df.columns and 'ap_hi' in df.columns:
            df['bmi_systolic_interaction'] = df['bmi'] * df['ap_hi']
        
        print("Etkileşim özellikleri oluşturuldu.")
        return df
    
    def create_polynomial_features(self, df, degree=2):
        """Polinom özellikleri oluştur."""
        # Sürekli değişkenler için polinom özellikleri
        continuous_cols = ['age_years', 'bmi', 'ap_hi', 'ap_lo']
        available_cols = [col for col in continuous_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[available_cols])
            
            # Polinom feature isimleri oluştur
            feature_names = poly.get_feature_names_out(available_cols)
            
            # DataFrame'e ekle
            for i, name in enumerate(feature_names):
                if name not in df.columns:  # Sadece yeni özellikleri ekle
                    df[f'poly_{name}'] = poly_features[:, i]
            
            print(f"Polinom özellikleri oluşturuldu (degree={degree}).")
        
        return df
    
    def encode_categorical_variables(self, df):
        """Kategorik değişkenleri encode et."""
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        print("Kategorik değişkenler encode edildi.")
        return df
    
    def scale_continuous_variables(self, df):
        """Sürekli değişkenleri ölçekle."""
        # Orijinal sürekli değişkenleri ölçekle
        available_continuous = [col for col in self.continuous_columns if col in df.columns]
        
        if available_continuous:
            scaled_features = self.scaler.fit_transform(df[available_continuous])
            scaled_df = pd.DataFrame(scaled_features, columns=[f'{col}_scaled' for col in available_continuous])
            
            # Orijinal kolonları koru, ölçeklenmiş versiyonları ekle
            for col in available_continuous:
                df[f'{col}_scaled'] = scaled_df[f'{col}_scaled']
        
        print("Sürekli değişkenler ölçeklendi.")
        return df
    
    def prepare_features(self, df, target_column='cardio'):
        """Feature matrix ve target vector hazırla."""
        if target_column not in df.columns:
            print(f"Hedef değişken '{target_column}' bulunamadı!")
            return None, None
        
        y = df[target_column]
        
        # Çıkarılacak kolonlar
        exclude_columns = ['id', 'age', 'cardio', 'age_group', 'bmi_category', 
                          'bp_category', 'risk_category']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns]
        
        print(f"Feature matrix boyutu: {X.shape}")
        print(f"Hedef değişken boyutu: {y.shape}")
        print(f"Toplam feature sayısı: {len(feature_columns)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Veriyi eğitim ve test setlerine ayır."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Eğitim seti: {X_train.shape}")
        print(f"Test seti: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def advanced_pipeline_with_outliers(self, file_path, sep=';'):
        """Outlier'ları çıkarmadan gelişmiş feature engineering pipeline'ını çalıştır."""
        print("=== GELİŞMİŞ FEATURE ENGINEERING PIPELINE (OUTLIER'LAR İLE) ===\n")
        
        # 1. Veri yükleme
        df = self.load_data(file_path, sep)
        if df is None:
            return None
        
        # 2. Yaş dönüşümü
        df = self.convert_age_to_years(df)
        
        # 3. Eksik değer kontrolü ve temizleme
        missing_counts = self.check_missing_values(df)
        if missing_counts.sum() > 0:
            df = self.clean_missing_values(df, strategy='drop')
        
        # 4. Outlier sayılarını hesapla (çıkarma)
        outlier_counts = self.count_outliers(df)
        
        # 5. Gelişmiş özellikler oluştur
        df = self.create_age_groups(df)
        df = self.create_bmi(df)
        df = self.create_blood_pressure_features(df)
        df = self.create_health_risk_score(df)
        df = self.create_interaction_features(df)
        df = self.create_polynomial_features(df, degree=2)
        
        # 6. Kategorik değişken encoding
        df = self.encode_categorical_variables(df)
        
        # 7. Sürekli değişken ölçekleme
        df = self.scale_continuous_variables(df)
        
        # 8. Feature hazırlama
        X, y = self.prepare_features(df)
        if X is None:
            return None
        
        # 9. Veri ayrımı
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\n=== GELİŞMİŞ FEATURE ENGINEERING TAMAMLANDI (OUTLIER'LAR İLE) ===")
        print(f"Toplam feature sayısı: {X.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'outlier_counts': outlier_counts,
            'data_type': 'advanced_features_with_outliers'
        }
    
    def advanced_pipeline_without_outliers(self, file_path, sep=';'):
        """Outlier'ları çıkararak gelişmiş feature engineering pipeline'ını çalıştır."""
        print("=== GELİŞMİŞ FEATURE ENGINEERING PIPELINE (OUTLIER'LAR ÇIKARILARAK) ===\n")
        
        # 1. Veri yükleme
        df = self.load_data(file_path, sep)
        if df is None:
            return None
        
        # 2. Yaş dönüşümü
        df = self.convert_age_to_years(df)
        
        # 3. Eksik değer kontrolü ve temizleme
        missing_counts = self.check_missing_values(df)
        if missing_counts.sum() > 0:
            df = self.clean_missing_values(df, strategy='drop')
        
        # 4. Outlier sayılarını hesapla ve çıkar
        outlier_counts = self.count_outliers(df)
        df = self.remove_outliers(df)
        
        # 5. Gelişmiş özellikler oluştur
        df = self.create_age_groups(df)
        df = self.create_bmi(df)
        df = self.create_blood_pressure_features(df)
        df = self.create_health_risk_score(df)
        df = self.create_interaction_features(df)
        df = self.create_polynomial_features(df, degree=2)
        
        # 6. Kategorik değişken encoding
        df = self.encode_categorical_variables(df)
        
        # 7. Sürekli değişken ölçekleme
        df = self.scale_continuous_variables(df)
        
        # 8. Feature hazırlama
        X, y = self.prepare_features(df)
        if X is None:
            return None
        
        # 9. Veri ayrımı
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\n=== GELİŞMİŞ FEATURE ENGINEERING TAMAMLANDI (OUTLIER'LAR ÇIKARILARAK) ===")
        print(f"Toplam feature sayısı: {X.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'outlier_counts': outlier_counts,
            'data_type': 'advanced_features_without_outliers'
        }

def main():
    """Ana fonksiyon."""
    # Feature engineering
    fe = AdvancedFeatureEngineer()
    
    # Veri dosyası yolu
    data_path = "data/cardiokaggle.csv"
    
    # Outlier'lı pipeline çalıştır
    print("OUTLIER'LAR İLE:")
    data_with_outliers = fe.advanced_pipeline_with_outliers(data_path)
    
    print("\n" + "="*50)
    
    # Outlier'sız pipeline çalıştır
    print("OUTLIER'LAR ÇIKARILARAK:")
    data_without_outliers = fe.advanced_pipeline_without_outliers(data_path)
    
    if data_with_outliers is not None:
        print(f"\nOutlier'lı veri - Feature sayısı: {data_with_outliers['X_train'].shape[1]}")
    
    if data_without_outliers is not None:
        print(f"Outlier'sız veri - Feature sayısı: {data_without_outliers['X_train'].shape[1]}")

if __name__ == "__main__":
    main() 