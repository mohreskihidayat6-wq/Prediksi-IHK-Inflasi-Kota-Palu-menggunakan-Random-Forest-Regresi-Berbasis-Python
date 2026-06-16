# ============================================================
#   PREDIKSI INDEKS HARGA KONSUMEN (IHK) KOTA PALU
#   Algoritma  : Random Forest Regression
#   Dataset    : Data Real BPS Kota Palu (Jan 2011 – Des 2023)
#   Sumber     : palukota.bps.go.id | sulteng.bps.go.id
#   Universitas: Universitas Tadulako
#   Prodi      : Sistem Informasi
#   Matkul     : Statistika & Probabilitas
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv('dataset_IHK_palu_ml.csv')

# Konversi kolom ke numerik
fitur = [
    'Bahan Makanan',
    'Makanan jadi, Minuman, Rokok, dan Tembakau',
    'Perumahan, Air, Listrik, Gas dan Bahan Bakar',
    'Sandang',
    'Kesehatan',
    'Pendidikan, Rekreasi, dan Olahraga',
    'Transportasi, Komunikasi, dan Jasa Keuangan'
]
target = 'Umum'

for col in fitur + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=fitur + [target]).reset_index(drop=True)

X = df[fitur]
y = df[target]

print("=" * 65)
print("   PREDIKSI IHK KOTA PALU – RANDOM FOREST REGRESSION")
print("   Sumber Data : BPS Kota Palu (Data Real)")
print("   Periode     : Januari 2011 – Desember 2023")
print("=" * 65)

# ============================================================
# 2. EKSPLORASI DATA
# ============================================================

print(f"\n[1] INFORMASI DATASET:")
print(f"    Jumlah data  : {len(df)} baris")
print(f"    Periode      : {df['Periode'].iloc[0]} s/d {df['Periode'].iloc[-1]}")
print(f"    Jumlah fitur : {len(fitur)} kelompok pengeluaran")
print(f"    Target       : IHK Umum (IHK Total) Kota Palu")

print(f"\n[2] SAMPEL DATASET (5 baris pertama):")
print(df[['Periode'] + fitur[:3] + [target]].head().to_string(index=False))

print(f"\n[3] STATISTIK DESKRIPTIF IHK UMUM:")
print(f"    Nilai Minimum : {y.min():.2f}")
print(f"    Nilai Maximum : {y.max():.2f}")
print(f"    Rata-rata     : {y.mean():.2f}")
print(f"    Std Deviasi   : {y.std():.2f}")
print(f"    Median        : {y.median():.2f}")

# ============================================================
# 3. SPLIT DATA TRAINING & TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"\n[4] PEMBAGIAN DATA (80:20):")
print(f"    Data Training : {len(X_train)} data — {df['Periode'].iloc[0]} s/d {df['Periode'].iloc[len(X_train)-1]}")
print(f"    Data Testing  : {len(X_test)} data  — {df['Periode'].iloc[len(X_train)]} s/d {df['Periode'].iloc[-1]}")

# ============================================================
# 4. MEMBANGUN MODEL RANDOM FOREST REGRESSION
# ============================================================

model = RandomForestRegressor(
    n_estimators      = 200,
    max_depth         = 6,
    min_samples_split = 3,
    random_state      = 42
)
model.fit(X_train, y_train)

print(f"\n[5] PARAMETER MODEL RANDOM FOREST:")
print(f"    n_estimators      : 200 (jumlah pohon keputusan)")
print(f"    max_depth         : 6   (kedalaman maksimum pohon)")
print(f"    min_samples_split : 3   (minimum sampel untuk split)")
print(f"    random_state      : 42")

# ============================================================
# 5. PREDIKSI & EVALUASI
# ============================================================

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

if r2 >= 0.90:
    ket = "Sangat Baik"
elif r2 >= 0.75:
    ket = "Baik"
elif r2 >= 0.50:
    ket = "Cukup"
else:
    ket = "Kurang"

print(f"\n[6] HASIL EVALUASI MODEL:")
print(f"    MAE  (Mean Absolute Error)      : {mae:.4f}")
print(f"    MSE  (Mean Squared Error)       : {mse:.4f}")
print(f"    RMSE (Root Mean Squared Error)  : {rmse:.4f}")
print(f"    R²   (Koefisien Determinasi)    : {r2:.4f}  → {ket}")

# ============================================================
# 6. PERBANDINGAN AKTUAL VS PREDIKSI
# ============================================================

hasil = pd.DataFrame({
    'Periode'      : df['Periode'].iloc[len(X_train):].values,
    'IHK Aktual'   : y_test.values.round(2),
    'IHK Prediksi' : y_pred.round(2),
    'Selisih'      : (y_test.values - y_pred).round(2)
})

print(f"\n[7] PERBANDINGAN AKTUAL vs PREDIKSI ({len(hasil)} data testing):")
print(hasil.to_string(index=False))

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

label_pendek = {
    'Bahan Makanan'                                   : 'Bahan Makanan',
    'Makanan jadi, Minuman, Rokok, dan Tembakau'      : 'Makanan Jadi & Rokok',
    'Perumahan, Air, Listrik, Gas dan Bahan Bakar'    : 'Perumahan & Listrik',
    'Sandang'                                         : 'Sandang',
    'Kesehatan'                                       : 'Kesehatan',
    'Pendidikan, Rekreasi, dan Olahraga'              : 'Pendidikan & Rekreasi',
    'Transportasi, Komunikasi, dan Jasa Keuangan'     : 'Transportasi & Komun.',
}

importances = pd.Series(
    model.feature_importances_, index=fitur
).sort_values(ascending=False)

print(f"\n[8] FEATURE IMPORTANCE (Kontribusi Kelompok Pengeluaran):")
for k, v in importances.items():
    bar = "█" * int(v * 60)
    print(f"    {label_pendek[k]:<25} : {v:.4f}  {bar}")

# ============================================================
# 8. VISUALISASI
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    'Prediksi IHK Kota Palu – Random Forest Regression\n'
    'Data Real BPS Kota Palu | Jan 2011 – Des 2023',
    fontsize=13, fontweight='bold'
)

# Plot 1: Tren IHK Seluruh Periode
ax1 = axes[0, 0]
ax1.plot(range(len(df)), y.values, color='steelblue', linewidth=1.8, label='IHK Aktual')
ax1.axvline(x=len(X_train), color='red', linestyle='--', linewidth=1.2, label='Batas Train/Test')
xticks = list(range(0, len(df), 12))
ax1.set_xticks(xticks)
ax1.set_xticklabels([df['Periode'].iloc[i] for i in xticks], rotation=45, fontsize=8)
ax1.set_title('Tren IHK Umum Kota Palu (2011–2023)')
ax1.set_xlabel('Periode')
ax1.set_ylabel('IHK')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Aktual vs Prediksi (Testing)
ax2 = axes[0, 1]
test_idx = range(len(X_train), len(df))
ax2.plot(test_idx, y_test.values, color='steelblue', linewidth=1.8,
         label='Aktual', marker='o', markersize=4)
ax2.plot(test_idx, y_pred, color='tomato', linewidth=1.8,
         label='Prediksi', linestyle='--', marker='s', markersize=4)
xticks2 = list(test_idx)[::3]
ax2.set_xticks(xticks2)
ax2.set_xticklabels([df['Periode'].iloc[i] for i in xticks2], rotation=45, fontsize=8)
ax2.set_title(f'Aktual vs Prediksi (Data Testing)\nR² = {r2:.4f}  |  RMSE = {rmse:.4f}')
ax2.set_xlabel('Periode')
ax2.set_ylabel('IHK')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Scatter Aktual vs Prediksi
ax3 = axes[1, 0]
ax3.scatter(y_test, y_pred, color='steelblue', alpha=0.8, edgecolors='white', s=70)
mn = min(y_test.min(), y_pred.min()) - 1
mx = max(y_test.max(), y_pred.max()) + 1
ax3.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Garis Ideal')
ax3.set_title('Scatter Plot: Aktual vs Prediksi')
ax3.set_xlabel('IHK Aktual')
ax3.set_ylabel('IHK Prediksi')
ax3.legend(fontsize=9)
ax3.text(0.05, 0.90, f'R² = {r2:.4f}', transform=ax3.transAxes,
         fontsize=11, color='darkgreen', fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Feature Importance
ax4 = axes[1, 1]
labels_p = [label_pendek[k] for k in importances.index]
colors   = plt.cm.Blues(np.linspace(0.35, 0.9, len(importances)))
ax4.barh(labels_p[::-1], importances.values[::-1], color=colors[::-1], edgecolor='white')
ax4.set_title('Feature Importance – Kelompok Pengeluaran')
ax4.set_xlabel('Tingkat Kepentingan')
for i, v in enumerate(importances.values[::-1]):
    ax4.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
ax4.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('grafik_IHK_palu_real.png', dpi=150, bbox_inches='tight')
print("\n[9] Grafik disimpan: grafik_IHK_palu_real.png")

# ============================================================
# 9. CONTOH PREDIKSI DATA BARU
# ============================================================

print(f"\n[10] CONTOH PREDIKSI DATA BARU:")
data_baru = pd.DataFrame([{
    'Bahan Makanan'                                  : 120.5,
    'Makanan jadi, Minuman, Rokok, dan Tembakau'     : 130.2,
    'Perumahan, Air, Listrik, Gas dan Bahan Bakar'   : 115.8,
    'Sandang'                                        : 108.4,
    'Kesehatan'                                      : 112.0,
    'Pendidikan, Rekreasi, dan Olahraga'             : 118.6,
    'Transportasi, Komunikasi, dan Jasa Keuangan'    : 116.3
}])
prediksi_baru = model.predict(data_baru)[0]
print(f"    Input  : Bahan Makanan=120.5, Makan Jadi=130.2, Perumahan=115.8")
print(f"             Sandang=108.4, Kesehatan=112.0, Pendidikan=118.6, Transportasi=116.3")
print(f"    Output : Prediksi IHK Umum = {prediksi_baru:.2f}")

print("\n" + "=" * 65)
print("    Selesai. Model Random Forest berhasil dijalankan.")
print("    Sumber data: BPS Kota Palu (Data Real)")
print("=" * 65)
