(cardiovask) C:\Users\seda.murutsoy\Desktop\cardiovask\cardiyovask>python src/data/preprocessor.py
================================================================================
PREPROCESSOR TEST - OUTLIER'LAR İLE
================================================================================
================================================================================
KALP KRİZİ RİSK TAHMİN MODELİ - TAM ÖN İŞLEME PIPELINE
================================================================================
Veri başarıyla yüklendi! Boyut: (70050, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'sm
oke', 'alco', 'active', 'cardio']
Yaş dönüşümü tamamlandı. Örnek değerler:
Gün: [18393.0, 20228.0, 18857.0] -> Yıl: [50, 55, 52]

=== DUPLİKE KAYIT KONTROLÜ ===
Toplam duplike kayıt sayısı: 0
Duplike kayıt bulunmamaktadır.

=== EKSİK DEĞER ANALİZİ ===
Toplam eksik değer: 104
Toplam hücre sayısı: 980700
Eksik değer oranı: 0.01%

Kolon bazında eksik değer analizi:
      Kolon  Eksik_Sayı  Eksik_Oranı
        age           8      0.01142
     gender           8      0.01142
     weight           8      0.01142
     height           8      0.01142
      ap_hi           8      0.01142
      ap_lo           8      0.01142
       alco           8      0.01142
cholesterol           8      0.01142
       gluc           8      0.01142
      smoke           8      0.01142
     cardio           8      0.01142
     active           8      0.01142
  age_years           8      0.01142
         id           0      0.00000

=== DOMAIN-SPECIFIC OUTLIER TESPİTİ ===
age: 0 outlier (0.00%)
height: 53 outlier (0.08%)
weight: 7 outlier (0.01%)
ap_hi: 247 outlier (0.35%)
ap_lo: 1034 outlier (1.48%)
gender: 8 outlier (0.01%)
cholesterol: 8 outlier (0.01%)
gluc: 8 outlier (0.01%)
smoke: 8 outlier (0.01%)
alco: 8 outlier (0.01%)
active: 8 outlier (0.01%)
cardio: 8 outlier (0.01%)
Tansiyon kuralı ihlali (ap_hi < ap_lo): 1234 kayıt

Toplam domain-specific outlier: 2631
Veri türleri doğrulanıyor...
Tekrarlanan satırlar kontrol ediliyor...
Tekrarlanan satır sayısı: 0

=== DOMAIN-SPECIFIC OUTLIER TEMİZLEME ===
height: 53 outlier tespit edildi (temizlenmedi)
weight: 7 outlier tespit edildi (temizlenmedi)
ap_hi: 247 outlier tespit edildi (temizlenmedi)
ap_lo: 1034 outlier tespit edildi (temizlenmedi)
gender: 8 geçersiz değer tespit edildi (temizlenmedi)
cholesterol: 8 geçersiz değer tespit edildi (temizlenmedi)
gluc: 8 geçersiz değer tespit edildi (temizlenmedi)
smoke: 8 geçersiz değer tespit edildi (temizlenmedi)
alco: 8 geçersiz değer tespit edildi (temizlenmedi)
active: 8 geçersiz değer tespit edildi (temizlenmedi)
cardio: 8 geçersiz değer tespit edildi (temizlenmedi)
Tansiyon kuralı ihlali: 1234 kayıt tespit edildi (temizlenmedi)
Toplam temizlenen outlier: 2631
Eksik değerler dolduruluyor (strateji: median)...
Veri kalitesi kontrol ediliyor...
=== VERİ KALİTE RAPORU ===
Toplam satır: 70050
Toplam kolon: 14
Eksik değer: 0
Tekrarlanan satır: 0
Bellek kullanımı: 5.21 MB
Nümerik kolon: 9
Kategorik kolon: 5
Veri başarıyla yüklendi! Boyut: (70050, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 
'alco', 'active', 'cardio']

=== TEMİZLEME ÖZETİ ===
Orijinal veri boyutu: (70050, 13)
Temizlenmiş veri boyutu: (70050, 14)
Silinen satır sayısı: 0
Silinen kolon sayısı: -1
Orijinal eksik değer sayısı: 96
Temizleme sonrası eksik değer sayısı: 0

================================================================================
TAM ÖN İŞLEME PIPELINE TAMAMLANDI!
================================================================================

Outlier'lı veri işleme başarılı!
Final veri boyutu: (70050, 14)

================================================================================
PREPROCESSOR TEST - OUTLIER'LAR ÇIKARILARAK
================================================================================
================================================================================
KALP KRİZİ RİSK TAHMİN MODELİ - TAM ÖN İŞLEME PIPELINE
================================================================================
Veri başarıyla yüklendi! Boyut: (70050, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 
'alco', 'active', 'cardio']
Yaş dönüşümü tamamlandı. Örnek değerler:
Gün: [18393.0, 20228.0, 18857.0] -> Yıl: [50, 55, 52]

=== DUPLİKE KAYIT KONTROLÜ ===
Toplam duplike kayıt sayısı: 0
Duplike kayıt bulunmamaktadır.

=== EKSİK DEĞER ANALİZİ ===
Toplam eksik değer: 104
Toplam hücre sayısı: 980700
Eksik değer oranı: 0.01%

Kolon bazında eksik değer analizi:
      Kolon  Eksik_Sayı  Eksik_Oranı
        age           8      0.01142
     gender           8      0.01142
     weight           8      0.01142
     height           8      0.01142
      ap_hi           8      0.01142
      ap_lo           8      0.01142
       alco           8      0.01142
cholesterol           8      0.01142
       gluc           8      0.01142
      smoke           8      0.01142
     cardio           8      0.01142
     active           8      0.01142
  age_years           8      0.01142
         id           0      0.00000

=== DOMAIN-SPECIFIC OUTLIER TESPİTİ ===
age: 0 outlier (0.00%)
height: 53 outlier (0.08%)
weight: 7 outlier (0.01%)
ap_hi: 247 outlier (0.35%)
ap_lo: 1034 outlier (1.48%)
gender: 8 outlier (0.01%)
cholesterol: 8 outlier (0.01%)
gluc: 8 outlier (0.01%)
smoke: 8 outlier (0.01%)
alco: 8 outlier (0.01%)
active: 8 outlier (0.01%)
cardio: 8 outlier (0.01%)
Tansiyon kuralı ihlali (ap_hi < ap_lo): 1234 kayıt

Toplam domain-specific outlier: 2631
Veri türleri doğrulanıyor...
Tekrarlanan satırlar kontrol ediliyor...
Tekrarlanan satır sayısı: 0

=== DOMAIN-SPECIFIC OUTLIER TEMİZLEME ===
height: 53 outlier NaN yapıldı
weight: 7 outlier NaN yapıldı
ap_hi: 247 outlier NaN yapıldı
ap_lo: 1034 outlier NaN yapıldı
gender: 8 geçersiz değer NaN yapıldı
cholesterol: 8 geçersiz değer NaN yapıldı
gluc: 8 geçersiz değer NaN yapıldı
smoke: 8 geçersiz değer NaN yapıldı
alco: 8 geçersiz değer NaN yapıldı
active: 8 geçersiz değer NaN yapıldı
cardio: 8 geçersiz değer NaN yapıldı
Tansiyon kuralı ihlali: 74 kayıt NaN yapıldı
Toplam temizlenen outlier: 1471
Eksik değerler dolduruluyor (strateji: median)...
Veri kalitesi kontrol ediliyor...
=== VERİ KALİTE RAPORU ===
Toplam satır: 70050
Toplam kolon: 14
Eksik değer: 0
Tekrarlanan satır: 0
Bellek kullanımı: 5.21 MB
Nümerik kolon: 9
Kategorik kolon: 5
Veri başarıyla yüklendi! Boyut: (70050, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 
'alco', 'active', 'cardio']
'alco', 'active', 'cardio']

=== TEMİZLEME ÖZETİ ===
Orijinal veri boyutu: (70050, 13)
Temizlenmiş veri boyutu: (70050, 14)
Silinen satır sayısı: 0
Silinen kolon sayısı: -1
Orijinal eksik değer sayısı: 96
Temizleme sonrası eksik değer sayısı: 0

================================================================================
TAM ÖN İŞLEME PIPELINE TAMAMLANDI!
================================================================================

Outlier'lar çıkarılarak veri işleme başarılı!
Final veri boyutu: (70050, 14)