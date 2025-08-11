"""
Temizlenmiş veri dosyasını kontrol etmek için test scripti
"""

import pandas as pd
from pathlib import Path

# Temizlenmiş veri dosyasını oku
data_path = Path("data/processed/cardio_cleaned.csv")

print("Temizlenmiş veri dosyası kontrol ediliyor...")
print(f"Dosya yolu: {data_path}")
print(f"Dosya var mı: {data_path.exists()}")

if data_path.exists():
    # Farklı ayırıcılarla okumayı dene
    separators = [';', ',', '\t']
    
    for sep in separators:
        try:
            df = pd.read_csv(data_path, sep=sep)
            print(f"\nAyırıcı '{sep}' ile okuma başarılı:")
            print(f"Boyut: {df.shape}")
            print(f"Kolonlar: {list(df.columns)}")
            print(f"İlk 3 satır:")
            print(df.head(3))
            break
        except Exception as e:
            print(f"Ayırıcı '{sep}' ile okuma başarısız: {e}")
else:
    print("Temizlenmiş veri dosyası bulunamadı!") 