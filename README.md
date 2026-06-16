Prediksi Indeks Harga Konsumen (IHK) Kota Palu Menggunakan Random Forest Regression

Deskripsi Proyek

Proyek ini merupakan implementasi **Machine Learning** untuk memprediksi nilai **Indeks Harga Konsumen (IHK) Kota Palu** menggunakan algoritma **Random Forest Regression** berbasis Python.

Dataset yang digunakan adalah data **REAL** bersumber dari **Badan Pusat Statistik (BPS) Kota Palu**, mencakup data IHK bulanan dari **Januari 2011 hingga Desember 2023** (155 data) berdasarkan 7 kelompok pengeluaran dengan tahun dasar **2012=100** dan **2018=100**.

**Sumber Data:** BPS Kota Palu — [palukota.bps.go.id](https://palukota.bps.go.id)  
**Tabel IHK:** [sulteng.bps.go.id](https://sulteng.bps.go.id/id/statistics-table/1/NDc4IzE=/indeks-harga-konsumen--ihk--kota-palu--menurut-kelompok--pengeluaran--------.html)

---

Tujuan Penelitian

1. Memprediksi nilai IHK total (IHK Umum) Kota Palu berdasarkan 7 kelompok pengeluaran
2. Menganalisis kelompok pengeluaran yang paling berpengaruh terhadap perubahan IHK
3. Mengevaluasi performa model menggunakan metrik MAE, MSE, RMSE, dan R²

---

Struktur Folder

```
Prediksi-IHK-Kota-Palu/
│
├── 📄 prediksi_IHK_palu_rf.py          # Script utama Random Forest Regression
├── 📊 dataset_IHK_palu_ml.csv          # Dataset real BPS Kota Palu (155 baris)
├── 🖼️ grafik_IHK_palu_real.png         # Visualisasi hasil prediksi
├── 📄 requirements.txt                  # Daftar library Python
├── 📄 .gitignore                        # File yang diabaikan Git
└── 📄 README.md                         # Dokumentasi proyek
```

---

Library yang Digunakan

| Library | Versi | Fungsi |
|---|---|---|
| `numpy` | ≥ 1.24 | Operasi numerik dan array |
| `pandas` | ≥ 2.0 | Manipulasi dan analisis data |
| `scikit-learn` | ≥ 1.3 | Model Random Forest & evaluasi |
| `matplotlib` | ≥ 3.7 | Visualisasi grafik |

---

Variabel Dataset

| No | Variabel | Keterangan | Tipe |
|---|---|---|---|
| 1 | `Bahan Makanan` | IHK kelompok bahan makanan | Fitur |
| 2 | `Makanan jadi, Minuman, Rokok, dan Tembakau` | IHK kelompok makanan jadi | Fitur |
| 3 | `Perumahan, Air, Listrik, Gas dan Bahan Bakar` | IHK kelompok perumahan | Fitur |
| 4 | `Sandang` | IHK kelompok pakaian | Fitur |
| 5 | `Kesehatan` | IHK kelompok kesehatan | Fitur |
| 6 | `Pendidikan, Rekreasi, dan Olahraga` | IHK kelompok pendidikan | Fitur |
| 7 | `Transportasi, Komunikasi, dan Jasa Keuangan` | IHK kelompok transportasi | Fitur |
| 8 | **`Umum`** | **IHK Total Kota Palu (Target)** | **Target** |

---

Parameter Model

```python
RandomForestRegressor(
    n_estimators      = 200,    # Jumlah pohon keputusan
    max_depth         = 6,      # Kedalaman maksimum setiap pohon
    min_samples_split = 3,      # Minimum sampel untuk split node
    random_state      = 42      # Reproduksibilitas hasil
)
```

---

Hasil Evaluasi Model

| Metrik | Nilai | Keterangan |
|---|---|---|
| **MAE** | 2.2302 | Rata-rata kesalahan absolut |
| **MSE** | 6.0798 | Rata-rata kuadrat kesalahan |
| **RMSE** | 2.4657 | Akar kuadrat dari MSE |
| **R²** | 0.5124 | Koefisien determinasi (Cukup) |

> **Pembagian Data:** 80% Training (Jan 2011 – Mei 2021) | 20% Testing (Jun 2021 – Des 2023)

---

Feature Importance

| Ranking | Kelompok Pengeluaran | Nilai Kepentingan |
|---|---|---|
| 🥇 1 | Perumahan, Air, Listrik, Gas & Bahan Bakar | 0.3355 (33.55%) |
| 🥈 2 | Sandang | 0.1895 (18.95%) |
| 🥉 3 | Pendidikan, Rekreasi, dan Olahraga | 0.1698 (16.98%) |
| 4 | Makanan Jadi, Minuman, Rokok & Tembakau | 0.1220 (12.20%) |
| 5 | Bahan Makanan | 0.0912 (9.12%) |
| 6 | Kesehatan | 0.0705 (7.05%) |
| 7 | Transportasi, Komunikasi & Jasa Keuangan | 0.0214 (2.14%) |

---

Informasi Mahasiswa

| | |
|---|---|
| **Nama** | Moh Reski Hidayat] |
| **NIM** | [F5212520091] |
| **Program Studi** | Sistem Informasi |
| **Fakultas** | Teknik |
| **Universitas** | Universitas Tadulako |
| **Mata Kuliah** | Statistika & Probabilitas |
| **Tahun Akademik** | 2025/2026 |

---

Referensi

- BPS Kota Palu. (2023). *Indeks Harga Konsumen Menurut Kelompok Pengeluaran*. Diakses dari [palukota.bps.go.id](https://palukota.bps.go.id)
- BPS Provinsi Sulawesi Tengah. (2023). *IHK Kota Palu Menurut Kelompok Pengeluaran*. Diakses dari [sulteng.bps.go.id](https://sulteng.bps.go.id)
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12*, 2825–2830.

---

Lisensi

Proyek ini dibuat untuk keperluan akademik. Data bersumber dari BPS Kota Palu yang bersifat publik.
